// Bump on Tail Instability/Two-stream Instability Test
//   Usage:
//     srun ./BumponTailInstability
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./BumponTailInstability 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
//     Change the TestName to TwoStreamInstability or BumponTailInstability
//     in order to simulate the Two stream instability or bump on tail instability
//     cases
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

constexpr bool EnablePhaseDump = false;

template <typename T>
struct Newton1D {
    double tol   = 1e-12;
    int max_iter = 20;
    double pi    = Kokkos::numbers::pi_v<double>;

    T k, delta, u;

    KOKKOS_INLINE_FUNCTION Newton1D() {}

    KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& delta_, const T& u_)
        : k(k_)
        , delta(delta_)
        , u(u_) {}

    KOKKOS_INLINE_FUNCTION ~Newton1D() {}

    KOKKOS_INLINE_FUNCTION T f(T& x) {
        T F;
        F = x + (delta * (Kokkos::sin(k * x) / k)) - u;
        return F;
    }

    KOKKOS_INLINE_FUNCTION T fprime(T& x) {
        T Fprime;
        Fprime = 1 + (delta * Kokkos::cos(k * x));
        return Fprime;
    }

    KOKKOS_FUNCTION
    void solve(T& x) {
        int iterations = 0;
        while (iterations < max_iter && Kokkos::fabs(f(x)) > tol) {
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

    value_type delta, sigma, muBulk, muBeam;
    size_type nlocBulk;

    T k, minU, maxU;

    // Initialize all members
    generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, value_type& delta_, T& k_,
                    value_type& sigma_, value_type& muBulk_, value_type& muBeam_,
                    size_type& nlocBulk_, T& minU_, T& maxU_)
        : x(x_)
        , v(v_)
        , rand_pool(rand_pool_)
        , delta(delta_)
        , sigma(sigma_)
        , muBulk(muBulk_)
        , muBeam(muBeam_)
        , nlocBulk(nlocBulk_)
        , k(k_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        bool isBeam = (i >= nlocBulk);

        value_type muZ = (value_type)(((!isBeam) * muBulk) + (isBeam * muBeam));

        if constexpr (Dim > 1) {
            for (unsigned d = 0; d < Dim - 1; ++d) {
                x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
                v(i)[d] = rand_gen.normal(0.0, sigma);
            }
        }
        v(i)[Dim - 1] = rand_gen.normal(muZ, sigma);

        value_type u  = rand_gen.drand(minU[Dim - 1], maxU[Dim - 1]);
        x(i)[Dim - 1] = u / (1 + delta);
        Newton1D<value_type> solver(k[Dim - 1], delta, u);
        solver.solve(x(i)[Dim - 1]);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

double CDF(const double& x, const double& delta, const double& k, const unsigned& dim) {
    bool isDimZ = (dim == (Dim - 1));
    double cdf  = x + (double)(isDimZ * ((delta / k) * std::sin(k * x)));
    return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t<double, Dim>& xvec, const double& delta,
           const Vector_t<double, Dim>& kw) {
    double pdf = 1.0 * 1.0 * (1.0 + delta * Kokkos::cos(kw[Dim - 1] * xvec[Dim - 1]));
    return pdf;
}

// const char* TestName = "BumponTailInstability";
const char* TestName = "TwoStreamInstability";

template <typename Bunch>
struct PhaseDump {
    void initialize(size_t nr, double domain) {
        ippl::Index I(nr);
        ippl::NDIndex<2> owned(I, I);
        layout = FieldLayout_t<2>(MPI_COMM_WORLD, owned, isParallel);

        Vector_t<double, 2> hx = {domain / nr, 16. / nr};
        Vector_t<double, 2> origin{0, -8};

        mesh = Mesh_t<2>(owned, hx, origin);
        phaseSpace.initialize(mesh, layout);
        if (ippl::Comm->rank() == 0) {
            phaseSpaceBuf.initialize(mesh, layout);
        }
        std::cout << ippl::Comm->rank() << ": " << phaseSpace.getOwned() << std::endl;
    }

    void dump(int it, std::shared_ptr<Bunch> P, bool allDims = false) {
        const auto pcount = P->getLocalNum();
        phase.realloc(pcount);
        auto& Ri = P->R;
        auto& Pi = P->P;
        for (unsigned d = allDims ? 0 : Dim - 1; d < Dim; d++) {
            Kokkos::parallel_for(
                "Copy phase space", pcount, KOKKOS_CLASS_LAMBDA(const size_t i) {
                    phase(i) = {Ri(i)[d], Pi(i)[d]};
                });
            phaseSpace = 0;
            Kokkos::fence();
            scatter(P->q, phaseSpace, phase);
            auto& view = phaseSpace.getView();
            MPI_Reduce(view.data(), phaseSpaceBuf.getView().data(), view.size(), MPI_DOUBLE,
                       MPI_SUM, 0, ippl::Comm->getCommunicator());
            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                fname << "PhaseSpace_t=" << it << "_d=" << d << ".csv";

                Inform out("Phase Dump", fname.str().c_str(), Inform::OVERWRITE, 0);
                phaseSpaceBuf.write(out);

                auto max = phaseSpaceBuf.max();
                auto min = phaseSpaceBuf.min();
                if (max > maxValue) {
                    maxValue = max;
                }
                if (min < minValue) {
                    minValue = min;
                }
            }
            ippl::Comm->barrier();
        }

        MPI_Bcast(&maxValue, 1, MPI_DOUBLE, 0, ippl::Comm->getCommunicator());
        MPI_Bcast(&minValue, 1, MPI_DOUBLE, 0, ippl::Comm->getCommunicator());
    }

    double maxRecorded() const { return maxValue; }
    double minRecorded() const { return minValue; }

private:
    std::array<bool, 2> isParallel = {false, false};
    FieldLayout_t<2> layout;
    Mesh_t<2> mesh;
    Field_t<2> phaseSpace, phaseSpaceBuf;
    ippl::ParticleAttrib<Vector_t<double, 2>> phase;

    double maxValue = 0, minValue = 0;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        setSignalHandler();

        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        int arg = 1;

        auto start = std::chrono::high_resolution_clock::now();
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        };

        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef dumpDataTimer    = IpplTimings::getTimer("dumpData");
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

        IpplTimings::startTimer(mainTimer);

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        msg << TestName << endl << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;

        std::shared_ptr<bunch_type> P;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        Vector_t<double, Dim> kw;
        double sigma, muBulk, muBeam, epsilon, delta;

        if (std::strcmp(TestName, "TwoStreamInstability") == 0) {
            // Parameters for two stream instability as in
            //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
            kw      = 0.5;
            sigma   = 0.1;
            epsilon = 0.5;
            muBulk  = -pi / 2.0;
            muBeam  = pi / 2.0;
            delta   = 0.01;
        } else if (std::strcmp(TestName, "BumponTailInstability") == 0) {
            kw      = 0.21;
            sigma   = 1.0 / std::sqrt(2.0);
            epsilon = 0.1;
            muBulk  = 0.0;
            muBeam  = 4.0;
            delta   = 0.01;
        } else {
            // Default value is two stream instability
            kw      = 0.5;
            sigma   = 0.1;
            epsilon = 0.5;
            muBulk  = -pi / 2.0;
            muBeam  = pi / 2.0;
            delta   = 0.01;
        }

        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax = 2 * pi / kw;

        Vector_t<double, Dim> hr;
        for (unsigned d = 0; d < Dim; d++) {
            hr[d] = rmax[d] / nr[d];
        }
        Vector_t<double, Dim> origin = rmin;
        const double dt              = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));

        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        // Q = -\int\int f dx dv
        double Q           = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        std::string solver = argv[arg++];

        if (solver == "OPEN") {
            throw IpplException("BumpOnTailInstability",
                                "Open boundaries solver incompatible with this simulation!");
        }

        P = std::make_shared<bunch_type>(PL, hr, rmin, rmax, isParallel, Q, solver);

        P->nr_m = nr;

        P->initializeFields(mesh, FL);

        P->initSolver();
        P->time_m                 = 0.0;
        P->loadbalancethreshold_m = std::atof(argv[arg++]);

        bool isFirstRepartition;

        if ((P->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            const int nghost               = P->rho_m.getNghost();
            auto rhoview                   = P->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", P->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = PDF(xvec, delta, kw);
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
            Nr[d] = CDF(Regions(myRank)[d].max(), delta, kw[d], d)
                    - CDF(Regions(myRank)[d].min(), delta, kw[d], d);
            Dr[d]   = CDF(rmax[d], delta, kw[d], d) - CDF(rmin[d], delta, kw[d], d);
            minU[d] = CDF(Regions(myRank)[d].min(), delta, kw[d], d);
            maxU[d] = CDF(Regions(myRank)[d].max(), delta, kw[d], d);
        }

        double factorConf = 1;
        for (unsigned d = 0; d < Dim; d++) {
            factorConf *= Nr[d] / Dr[d];
        }
        double factorVelBulk      = 1.0 - epsilon;
        double factorVelBeam      = 1.0 - factorVelBulk;
        size_type nlocBulk        = (size_type)(factorConf * factorVelBulk * totalP);
        size_type nlocBeam        = (size_type)(factorConf * factorVelBeam * totalP);
        size_type nloc            = nlocBulk + nlocBeam;
        size_type Total_particles = 0;

        MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                      ippl::Comm->getCommunicator());

        int rest = (int)(totalP - Total_particles);

        if (ippl::Comm->rank() < rest) {
            ++nloc;
        }

        P->create(nloc);

        PhaseDump<bunch_type> phase;
        if constexpr (EnablePhaseDump) {
            if (ippl::Comm->size() != 1) {
                msg << "Phase dump only supported on one rank" << endl;
                ippl::Comm->abort();
            }
            phase.initialize(*std::max_element(nr.begin(), nr.end()),
                             *std::max_element(rmax.begin(), rmax.end()));
        }

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));

        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), P->P.getView(), rand_pool64, delta, kw, sigma, muBulk, muBeam,
                      nlocBulk, minU, maxU));
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
        P->dumpBumponTail();
        P->gatherStatistics(totalP);
        // P->dumpLocalDomains(FL, 0);
        IpplTimings::stopTimer(dumpDataTimer);

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            IpplTimings::startTimer(PTimer);
            P->P = P->P - 0.5 * dt * P->E;
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
            P->P = P->P - 0.5 * dt * P->E;
            IpplTimings::stopTimer(PTimer);

            P->time_m += dt;
            IpplTimings::startTimer(dumpDataTimer);
            P->dumpBumponTail();
            P->gatherStatistics(totalP);
            IpplTimings::stopTimer(dumpDataTimer);
            msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;

            if constexpr (EnablePhaseDump) {
                phase.dump(it, P);
            }

            if (checkSignalHandler()) {
                msg << "Aborting timestepping loop due to signal " << interruptSignalReceived
                    << endl;
                break;
            }
        }

        msg << TestName << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

        if constexpr (EnablePhaseDump) {
            // clang-format off
            msg << "--- Phase Space Parameters ---\n"
                << "Resolution: " << *std::max_element(nr.begin(), nr.end()) << "\n"
                << "Domain: " << rmax[Dim - 1] << "\n"
                << "Phase space axis: " << (Dim - 1)
                    << "; range: [" << phase.minRecorded() << ", " << phase.maxRecorded() << "]\n"
                << "Particle count: " << totalP << "\n"
                << "Ranks: " << ippl::Comm->size() << "\n"
                << "Timestep: " << dt << "\n"
                << "------------------------------" << endl;
            // clang-format on
        }
    }
    ippl::finalize();

    return 0;
}
