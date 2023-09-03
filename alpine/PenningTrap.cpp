// Penning Trap
//   Usage:
//     srun ./PenningTrap
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny       = No. cell-centered points in the y-direction
//     nz       = No. cell-centered points in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT, CG, P3M, and OPEN supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./PenningTrap 128 128 128 10000 300 FFT 0.01 --overallocate 1.0 --info 10
//

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

template <typename T>
struct Newton1D {
    double tol   = 1e-12;
    int max_iter = 20;
    double pi    = Kokkos::numbers::pi_v<double>;

    T mu, sigma, u;

    KOKKOS_INLINE_FUNCTION Newton1D() {}

    KOKKOS_INLINE_FUNCTION Newton1D(const T& mu_, const T& sigma_, const T& u_)
        : mu(mu_)
        , sigma(sigma_)
        , u(u_) {}

    KOKKOS_INLINE_FUNCTION ~Newton1D() {}

    KOKKOS_INLINE_FUNCTION T f(T& x) {
        T F;
        F = Kokkos::erf((x - mu) / (sigma * Kokkos::sqrt(2.0))) - 2 * u + 1;
        return F;
    }

    KOKKOS_INLINE_FUNCTION T fprime(T& x) {
        T Fprime;
        Fprime = (1 / sigma) * Kokkos::sqrt(2 / pi)
                 * Kokkos::exp(-0.5 * (Kokkos::pow(((x - mu) / sigma), 2)));
        return Fprime;
    }

    KOKKOS_FUNCTION
    void solve(T& x) {
        int iterations = 0;
        while ((iterations < max_iter) && (Kokkos::fabs(f(x)) > tol)) {
            x = x - (f(x) / fprime(x));
            iterations += 1;
        }
    }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type x, v;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T mu, sigma, minU, maxU;
    double pi = Kokkos::numbers::pi_v<double>;

    // Initialize all members
    generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, T& mu_, T& sigma_,
                    T& minU_, T& maxU_)
        : x(x_)
        , v(v_)
        , rand_pool(rand_pool_)
        , mu(mu_)
        , sigma(sigma_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            u       = rand_gen.drand(minU[d], maxU[d]);
            x(i)[d] = (Kokkos::sqrt(pi / 2) * (2 * u - 1)) * sigma[d] + mu[d];
            Newton1D<value_type> solver(mu[d], sigma[d], u);
            solver.solve(x(i)[d]);
            v(i)[d] = rand_gen.normal(0.0, 1.0);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

double CDF(const double& x, const double& mu, const double& sigma) {
    double cdf = 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2))));
    return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t<double, Dim>& xvec, const Vector_t<double, Dim>& mu,
           const Vector_t<double, Dim>& sigma, const unsigned Dim) {
    double pdf = 1.0;
    double pi  = Kokkos::numbers::pi_v<double>;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 / (sigma[d] * Kokkos::sqrt(2 * pi)))
               * Kokkos::exp(-0.5 * Kokkos::pow((xvec[d] - mu[d]) / sigma[d], 2));
    }
    return pdf;
}

const char* TestName = "PenningTrap";

int main(int argc, char* argv[]) {
    static_assert(Dim == 3, "Penning trap must be 3D");

    ippl::initialize(argc, argv);
    {
        setSignalHandler();

        Inform msg("PenningTrap");
        Inform msg2all("PenningTrap", INFORM_ALL_NODES);

        auto start            = std::chrono::high_resolution_clock::now();
        Vector_t<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef dumpDataTimer    = IpplTimings::getTimer("dumpData");
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("Solve");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

        IpplTimings::startTimer(mainTimer);

        size_type totalP      = std::atol(argv[4]);
        const unsigned int nt = std::atoi(argv[5]);

        msg << "Penning Trap " << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> rmin   = 0;
        Vector_t<double, Dim> rmax   = 20;
        Vector_t<double, Dim> length = rmax - rmin;

        Vector_t<double, Dim> hr     = length / nr;
        Vector_t<double, Dim> origin = rmin;
        unsigned int nrMax           = 2048;  // Max grid size in our studies
        double dxFinest              = length[0] / nrMax;
        const double dt              = 0.5 * dxFinest;  // size of timestep

        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        double Q           = -1562.5;
        double Bext        = 5.0;
        std::string solver = argv[6];
        P                  = std::make_unique<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

        P->nr_m = nr;

        Vector_t<double, Dim> mu, sd;

        for (unsigned d = 0; d < Dim; d++) {
            mu[d] = 0.5 * length[d] + origin[d];
        }
        sd[0] = 0.15 * length[0];
        sd[1] = 0.05 * length[1];
        sd[2] = 0.20 * length[2];

        P->initializeFields(mesh, FL);

        P->initSolver();
        P->time_m                 = 0.0;
        P->loadbalancethreshold_m = std::atof(argv[7]);

        bool isFirstRepartition;

        if ((P->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            const int nghost               = P->rho_m.getNghost();
            auto rhoview                   = P->rho_m.getView();

            Kokkos::parallel_for(
                "Assign initial rho based on PDF", P->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // local to global index conversion
                    const size_t ig = i + lDom[0].first() - nghost;
                    const size_t jg = j + lDom[1].first() - nghost;
                    const size_t kg = k + lDom[2].first() - nghost;
                    double x        = (ig + 0.5) * hr[0] + origin[0];
                    double y        = (jg + 0.5) * hr[1] + origin[1];
                    double z        = (kg + 0.5) * hr[2] + origin[2];

                    Vector_t<double, Dim> xvec = {x, y, z};

                    rhoview(i, j, k) = PDF(xvec, mu, sd, Dim);
                });

            Kokkos::fence();

            P->initializeORB(FL, mesh);
            P->repartition(FL, mesh, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        msg << "First domain decomposition done" << endl;
        IpplTimings::startTimer(particleCreation);

        typedef ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>::uniform_type RegionLayout_t;
        const RegionLayout_t& RLayout                           = PL.getRegionLayout();
        const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();
        Vector_t<double, Dim> Nr, Dr, minU, maxU;
        int myRank = ippl::Comm->rank();
        for (unsigned d = 0; d < Dim; ++d) {
            Nr[d] = CDF(Regions(myRank)[d].max(), mu[d], sd[d])
                    - CDF(Regions(myRank)[d].min(), mu[d], sd[d]);
            Dr[d]   = CDF(rmax[d], mu[d], sd[d]) - CDF(rmin[d], mu[d], sd[d]);
            minU[d] = CDF(Regions(myRank)[d].min(), mu[d], sd[d]);
            maxU[d] = CDF(Regions(myRank)[d].max(), mu[d], sd[d]);
        }

        double factor             = (Nr[0] * Nr[1] * Nr[2]) / (Dr[0] * Dr[1] * Dr[2]);
        size_type nloc            = (size_type)(factor * totalP);
        size_type Total_particles = 0;

        MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                      ippl::Comm->getCommunicator());

        int rest = (int)(totalP - Total_particles);

        if (ippl::Comm->rank() < rest)
            ++nloc;

        P->create(nloc);
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), P->P.getView(), rand_pool64, mu, sd, minU, maxU));

        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(particleCreation);

        P->q = P->Q_m / totalP;
        msg << "particles created and initial conditions assigned " << endl;
        isFirstRepartition = false;
        // The update after the particle creation is not needed as the
        // particles are generated locally

        IpplTimings::startTimer(DummySolveTimer);
        P->rho_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP, 0, hr);

        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        P->gatherCIC();

        IpplTimings::startTimer(dumpDataTimer);
        P->dumpData();
        P->gatherStatistics(totalP);
        // P->dumpLocalDomains(FL, 0);
        IpplTimings::stopTimer(dumpDataTimer);

        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext), 2)));
        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // Staggered Leap frog or Boris algorithm as per
            // https://www.sciencedirect.com/science/article/pii/S2590055219300526
            // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
            // the analytical matrix inversion which is not complex in this case.
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick
            IpplTimings::startTimer(PTimer);
            auto Rview = P->R.getView();
            auto Pview = P->P.getView();
            auto Eview = P->E.getView();
            double V0  = 30 * length[2];
            Kokkos::parallel_for(
                "Kick1", P->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                    double Eext_x = -(Rview(j)[0] - origin[0] - 0.5 * length[0])
                                    * (V0 / (2 * Kokkos::pow(length[2], 2)));
                    double Eext_y = -(Rview(j)[1] - origin[1] - 0.5 * length[1])
                                    * (V0 / (2 * Kokkos::pow(length[2], 2)));
                    double Eext_z = (Rview(j)[2] - origin[2] - 0.5 * length[2])
                                    * (V0 / (Kokkos::pow(length[2], 2)));

                    Eext_x += Eview(j)[0];
                    Eext_y += Eview(j)[1];
                    Eext_z += Eview(j)[2];

                    Pview(j)[0] += alpha * (Eext_x + Pview(j)[1] * Bext);
                    Pview(j)[1] += alpha * (Eext_y - Pview(j)[0] * Bext);
                    Pview(j)[2] += alpha * Eext_z;
                });
            IpplTimings::stopTimer(PTimer);

            // drift
            IpplTimings::startTimer(RTimer);
            P->R = P->R + dt * P->P;
            IpplTimings::stopTimer(RTimer);

            // Since the particles have moved spatially update them to correct processors
            IpplTimings::startTimer(updateTimer);
            P->update();
            IpplTimings::stopTimer(updateTimer);

            // Domain Decomposition
            if (P->balance(totalP, it + 1)) {
                msg << "Starting repartition" << endl;
                IpplTimings::startTimer(domainDecomposition);
                P->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
                // IpplTimings::startTimer(dumpDataTimer);
                // P->dumpLocalDomains(FL, it+1);
                // IpplTimings::stopTimer(dumpDataTimer);
            }

            // scatter the charge onto the underlying grid
            P->scatterCIC(totalP, it + 1, hr);

            // Field solve
            IpplTimings::startTimer(SolveTimer);
            P->runSolver();
            IpplTimings::stopTimer(SolveTimer);

            // gather E field
            P->gatherCIC();

            // kick
            IpplTimings::startTimer(PTimer);
            auto R2view = P->R.getView();
            auto P2view = P->P.getView();
            auto E2view = P->E.getView();
            Kokkos::parallel_for(
                "Kick2", P->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                    double Eext_x = -(R2view(j)[0] - origin[0] - 0.5 * length[0])
                                    * (V0 / (2 * Kokkos::pow(length[2], 2)));
                    double Eext_y = -(R2view(j)[1] - origin[1] - 0.5 * length[1])
                                    * (V0 / (2 * Kokkos::pow(length[2], 2)));
                    double Eext_z = (R2view(j)[2] - origin[2] - 0.5 * length[2])
                                    * (V0 / (Kokkos::pow(length[2], 2)));

                    Eext_x += E2view(j)[0];
                    Eext_y += E2view(j)[1];
                    Eext_z += E2view(j)[2];

                    P2view(j)[0] =
                        DrInv
                        * (P2view(j)[0]
                           + alpha * (Eext_x + P2view(j)[1] * Bext + alpha * Bext * Eext_y));
                    P2view(j)[1] =
                        DrInv
                        * (P2view(j)[1]
                           + alpha * (Eext_y - P2view(j)[0] * Bext - alpha * Bext * Eext_x));
                    P2view(j)[2] += alpha * Eext_z;
                });
            IpplTimings::stopTimer(PTimer);

            P->time_m += dt;
            IpplTimings::startTimer(dumpDataTimer);
            P->dumpData();
            P->gatherStatistics(totalP);
            IpplTimings::stopTimer(dumpDataTimer);
            msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;

            if (checkSignalHandler()) {
                msg << "Aborting timestepping loop due to signal " << interruptSignalReceived
                    << endl;
                break;
            }
        }

        msg << "Penning Trap: End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
    }
    ippl::finalize();

    return 0;
}
