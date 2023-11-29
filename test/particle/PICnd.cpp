// Test PICnd
//   This test program sets up a simple sine-wave electric field in N dimensions,
//   creates a population of particles with random positions and and velocities,
//   and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation and periodic particle BCs.
//
//   This test also provides a base for load-balancing using a domain-decomposition
//   based on an ORB.
//
//   Usage:
//     srun ./PICnd 128 128 128 10000 10 --info 10
//
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#define str(x)  #x
#define xstr(x) str(x)

// dimension of our positions
#define DIM     3
constexpr unsigned Dim          = DIM;
constexpr const char* PROG_NAME = "PIC" xstr(DIM) "d";

// some typedefs
typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef Mesh_t::DefaultCentering Centering_t;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim>
using Field = ippl::Field<T, Dim, Mesh_t, Centering_t>;

typedef ippl::OrthogonalRecursiveBisection<Field<double, Dim>> ORB;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim> Vector_t;
typedef Field<double, Dim> Field_t;
typedef Field<Vector_t, Dim> VField_t;

double pi = Kokkos::numbers::pi_v<double>;

template <class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    Field<Vector<double, Dim>, Dim> EFD_m;
    Field<double, Dim> EFDMag_m;

    // ORB
    ORB orb;

    Vector<int, Dim> nr_m;

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    std::array<bool, Dim> isParallel_m;

    double Q_m;

public:
    ParticleAttrib<double> qm;                                       // charge-to-mass ratio
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type
        E;  // electric field at particle position

    ChargedParticles(PLayout& pl, Vector_t hr, Vector_t rmin, Vector_t rmax,
                     std::array<bool, Dim> isParallel, double Q)
        : ippl::ParticleBase<PLayout>(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , isParallel_m(isParallel)
        , Q_m(Q) {
        // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
    }

    void setupBCs() { setBCAllPeriodic(); }

    void updateLayout(FieldLayout_t& fl, Mesh_t& mesh) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        this->EFD_m.updateLayout(fl);
        this->EFDMag_m.updateLayout(fl);

        // Update layout with new FieldLayout
        PLayout& layout = this->getLayout();
        layout.updateLayout(fl, mesh);
        IpplTimings::stopTimer(tupdateLayout);
        static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
        IpplTimings::startTimer(tupdatePLayout);
        this->update();
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeORB(FieldLayout_t& fl, Mesh_t& mesh) { orb.initialize(fl, mesh, EFDMag_m); }

    ~ChargedParticles() {}

    void repartition(FieldLayout_t& fl, Mesh_t& mesh) {
        // Repartition the domains
        bool fromAnalyticDensity = false;
        bool res                 = orb.binaryRepartition(this->R, fl, fromAnalyticDensity);

        if (res != true) {
            std::cout << "Could not repartition!" << std::endl;
            return;
        }
        // Update
        this->updateLayout(fl, mesh);
    }

    bool balance(unsigned int totalP) {  //, int timestep = 1) {
        int local = 0;
        std::vector<int> res(ippl::Comm->size());
        double threshold = 1.0;
        double equalPart = (double)totalP / ippl::Comm->size();
        double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
        if (dev > threshold) {
            local = 1;
        }
        MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT, ippl::Comm->getCommunicator());

        for (unsigned int i = 0; i < res.size(); i++) {
            if (res[i] == 1) {
                return true;
            }
        }
        return false;
    }

    void gatherStatistics(unsigned int totalP) {
        ippl::Comm->barrier();
        std::cout << "Rank " << ippl::Comm->rank() << " has "
                  << (double)this->getLocalNum() / totalP * 100.0
                  << " percent of the total particles " << std::endl;
        ippl::Comm->barrier();
    }

    void gatherCIC() {
        // static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        // IpplTimings::startTimer(gatherTimer);
        gather(this->E, EFD_m, this->R);
        // IpplTimings::stopTimer(gatherTimer);
    }

    void scatterCIC(unsigned int totalP, int iteration) {
        // static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        // IpplTimings::startTimer(scatterTimer);
        Inform m("scatter ");
        EFDMag_m = 0.0;
        scatter(qm, EFDMag_m, this->R);
        // IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("CheckCharge");
        IpplTimings::startTimer(sumTimer);
        double Q_grid = EFDMag_m.sum();

        unsigned int Total_particles = 0;
        unsigned int local_particles = this->getLocalNum();

        ippl::Comm->reduce(&local_particles, &Total_particles, 1, std::plus<unsigned int>());

        double rel_error = std::fabs((Q_m - Q_grid) / Q_m);
        m << "Rel. error in charge conservation = " << rel_error << endl;

        if (ippl::Comm->rank() == 0) {
            if (Total_particles != totalP || rel_error > 1e-10) {
                std::cout << "Total particles in the sim. " << totalP << " "
                          << "after update: " << Total_particles << std::endl;
                std::cout << "Total particles not matched in iteration: " << iteration << std::endl;
                std::cout << "Q grid: " << Q_grid << "Q particles: " << Q_m << std::endl;
                std::cout << "Rel. error in charge conservation: " << rel_error << std::endl;
                exit(1);
            }
        }

        IpplTimings::stopTimer(sumTimer);
    }

    void writePerRank() {
        double lq = 0.0, lqm = 0.0;
        Field_t::view_type viewRho               = this->EFDMag_m.getView();
        ParticleAttrib<double>::view_type viewqm = this->qm.getView();
        int nghost                               = this->EFDMag_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_reduce(
            "Particle Charge", ippl::getRangePolicy(viewRho, nghost),
            KOKKOS_LAMBDA(const index_array_type& args, double& val) {
                val += ippl::apply(viewRho, args);
            },
            lq);
        Kokkos::parallel_reduce(
            "Particle QM", viewqm.extent(0),
            KOKKOS_LAMBDA(const int i, double& val) { val += viewqm(i); }, lqm);
    }

    void initFields() {
        static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");
        IpplTimings::startTimer(initFieldsTimer);
        Inform m("initFields ");

        ippl::NDIndex<Dim> domain = EFD_m.getDomain();

        for (unsigned int i = 0; i < Dim; i++) {
            nr_m[i] = domain[i].length();
        }

        double phi0 = 0.1;
        double pi   = Kokkos::numbers::pi_v<double>;
        // scale_fact so that particles move more
        double scale_fact = 1e5;  // 1e6

        Vector_t hr = hr_m;

        typename VField_t::view_type& view = EFD_m.getView();
        const FieldLayout_t& layout        = EFD_m.getLayout();
        const ippl::NDIndex<Dim>& lDom     = layout.getLocalNDIndex();
        const int nghost                   = EFD_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign EFD_m", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t vec = (0.5 + args + lDom.first() - nghost) * hr;

                ippl::apply(view, args)[0] = -scale_fact * 2.0 * pi * phi0;
                for (unsigned d1 = 0; d1 < Dim; d1++) {
                    ippl::apply(view, args)[0] *= Kokkos::cos(2 * ((d1 + 1) % 3) * pi * vec[d1]);
                }
                for (unsigned d = 1; d < Dim; d++) {
                    ippl::apply(view, args)[d] = scale_fact * 4.0 * pi * phi0;
                    for (int d1 = 0; d1 < (int)Dim - 1; d1++) {
                        ippl::apply(view, args)[d] *=
                            Kokkos::sin(2 * ((d1 + 1) % 3) * pi * vec[d1]);
                    }
                }
            });

        EFDMag_m = dot(EFD_m, EFD_m);
        EFDMag_m = sqrt(EFDMag_m);
        IpplTimings::stopTimer(initFieldsTimer);
    }

    void dumpData(int iteration) {
        ParticleAttrib<Vector_t>::view_type& view = P.getView();

        double Energy = 0.0;

        Kokkos::parallel_reduce(
            "Particle Energy", view.extent(0),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(view(i), view(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;

        ippl::Comm->reduce(&Energy, &gEnergy, 1, std::plus<double>());

        Inform csvout(NULL, "data/energy.csv", Inform::APPEND);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        csvout << iteration << " " << gEnergy << endl;

        ippl::Comm->barrier();
    }

    // @param tag
    //        2 -> uniform(0,1)
    //        1 -> normal(0,1)
    //        0 -> gridpoints
    void initPositions(FieldLayout_t& fl, Vector_t& hr, unsigned int nloc, int tag = 2) {
        Inform m("initPositions ");
        typename ippl::ParticleBase<PLayout>::particle_position_type::HostMirror R_host =
            this->R.getHostMirror();

        std::mt19937_64 eng[Dim];
        for (unsigned i = 0; i < Dim; ++i) {
            eng[i].seed(42 + i * Dim);
            eng[i].discard(nloc * ippl::Comm->rank());
        }

        std::mt19937_64 engN[4 * Dim];
        for (unsigned i = 0; i < 4 * Dim; ++i) {
            engN[i].seed(42 + i * Dim);
            engN[i].discard(nloc * ippl::Comm->rank());
        }

        auto dom                = fl.getDomain();
        unsigned int gridpoints = 1;
        for (unsigned d = 0; d < Dim; d++) {
            gridpoints *= dom[d].length();
        }
        if (tag == 0 && nloc * ippl::Comm->size() != gridpoints) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << "Particle count must match gridpoint count to use gridpoint "
                             "locations. Switching to uniform distribution."
                          << std::endl;
            }
            tag = 2;
        }

        if (tag == 0) {
            m << "Positions are set on grid points" << endl;
            int N = fl.getDomain()[0].length();  // this only works for boxes
            const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();
            int size                       = ippl::Comm->size();
            using index_type               = typename ippl::RangePolicy<Dim>::index_type;
            Kokkos::Array<index_type, Dim> begin, end;
            for (unsigned d = 0; d < Dim; d++) {
                begin[d] = 0;
                end[d]   = N;
            }
            end[0] /= size;
            // Loops over particles
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "initPositions", ippl::createRangePolicy(begin, end),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;
                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= N;
                        }
                        l += next / size;
                    }
                    R_host(l) = (0.5 + args + lDom.first()) * hr;
                });

        } else if (tag == 1) {
            m << "Positions follow normal distribution" << endl;
            std::vector<double> mu = {0.5, 0.6, 0.2, 0.5, 0.6, 0.2};
            std::vector<double> sd = {0.75, 0.3, 0.2, 0.75, 0.3, 0.2};
            std::vector<double> states(Dim);

            Vector_t length = 1;

            std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

            double sum_coord = 0.0;
            for (unsigned long long int i = 0; i < nloc; i++) {
                for (unsigned d = 0; d < Dim; d++) {
                    double u1 = dist_uniform(engN[d * 2]);
                    double u2 = dist_uniform(engN[d * 2 + 1]);
                    states[d] =
                        sd[d] * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2) + mu[d];
                    R_host(i)[d] = std::fabs(std::fmod(states[d], length[d]));
                    sum_coord += R_host(i)[d];
                }
            }
        } else {
            double rmin = 0.0, rmax = 1.0;
            m << "Positions follow uniform distribution U(" << rmin << "," << rmax << ")" << endl;
            std::uniform_real_distribution<double> unif(rmin, rmax);
            for (unsigned long int i = 0; i < nloc; i++) {
                for (unsigned d = 0; d < Dim; d++) {
                    R_host(i)[d] = unif(eng[d]);
                }
            }
        }

        // Copy to device
        Kokkos::deep_copy(this->R.getView(), R_host);
    }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(PROG_NAME);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        ippl::Comm->setDefaultOverallocation(3.0);

        int arg = 1;

        int volume = 1;
        ippl::Vector<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            volume *= nr[d] = std::atoi(argv[arg++]);
        }

        // Each rank must have a minimal volume of 8
        if (volume < 8 * ippl::Comm->size()) {
            msg << "!!! Ranks have not enough volume for proper working !!! (Minimal volume per "
                   "rank: "
                   "8)"
                << endl;
        }

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);
        const unsigned int totalP = std::atoi(argv[arg++]);
        const unsigned int nt     = std::atoi(argv[arg++]);

        msg << "Particle test " << PROG_NAME << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t>;

        std::unique_ptr<bunch_type> P;

        Vector_t rmin(0.0);
        Vector_t rmax(1.0);
        // create mesh and layout objects for this problem domain
        Vector_t hr = rmax / nr;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = ippl::Index(nr[d]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        Vector_t origin = rmin;

        const double dt = 0.5 * hr[0];  // size of timestep

        const bool isAllPeriodic = true;
        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t PL(FL, mesh);

        /**PRINT**/
        msg << "FIELD LAYOUT (INITIAL)" << endl;
        msg << FL << endl;

        double Q = 1.0;
        P        = std::make_unique<bunch_type>(PL, hr, rmin, rmax, isParallel, Q);

        unsigned long int nloc = totalP / ippl::Comm->size();

        int rest = (int)(totalP - nloc * ippl::Comm->size());

        if (ippl::Comm->rank() < rest) {
            ++nloc;
        }

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);
        P->create(nloc);
        // Verifying that particles are created
        double totalParticles = 0.0;
        double localParticles = P->getLocalNum();
        ippl::Comm->reduce(&localParticles, &totalParticles, 1, std::plus<double>());
        msg << "Total particles: " << totalParticles << endl;
        P->initPositions(FL, hr, nloc, 2);

        P->qm = P->Q_m / totalP;
        P->P  = 0.0;
        IpplTimings::stopTimer(particleCreation);

        static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");
        IpplTimings::startTimer(UpdateTimer);
        P->update();
        IpplTimings::stopTimer(UpdateTimer);

        msg << "particles created and initial conditions assigned " << endl;

        P->EFD_m.initialize(mesh, FL);
        P->EFDMag_m.initialize(mesh, FL);
        P->initializeORB(FL, mesh);

        // Mass conservation
        // P->writePerRank();

        static IpplTimings::TimerRef domainDecomposition0 = IpplTimings::getTimer("domainDecomp0");
        IpplTimings::startTimer(domainDecomposition0);
        if (P->balance(totalP)) {
            P->repartition(FL, mesh);
        }
        IpplTimings::stopTimer(domainDecomposition0);
        msg << "Balancing finished" << endl;

        // Mass conservation
        // P->writePerRank();

        P->scatterCIC(totalP, 0);
        msg << "scatter done" << endl;

        P->initFields();
        msg << "P->initField() done" << endl;

        // Moving particles one grid cell
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        P->P = 1.0;

        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionUpdate");
            IpplTimings::startTimer(RTimer);
            P->R = P->R + dt * P->P;
            IpplTimings::stopTimer(RTimer);

            IpplTimings::startTimer(UpdateTimer);
            P->update();
            IpplTimings::stopTimer(UpdateTimer);

            // Domain Decomposition
            if (P->balance(totalP)) {
                msg << "Starting repartition" << endl;
                IpplTimings::startTimer(domainDecomposition0);
                P->repartition(FL, mesh);
                IpplTimings::stopTimer(domainDecomposition0);
                // Conservations
                // P->writePerRank();
            }

            // scatter the charge onto the underlying grid
            msg << "Starting scatterCIC" << endl;
            P->scatterCIC(totalP, it + 1);

            // gather the local value of the E field
            P->gatherCIC();

            // advance the particle velocities
            static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityUpdate");
            IpplTimings::startTimer(PTimer);
            P->P = P->P + dt * P->qm * P->E;
            IpplTimings::stopTimer(PTimer);

            P->dumpData(it);

            msg << "Finished iteration " << it << endl;

            P->gatherStatistics(totalP);
        }

        msg << "Particle test " << PROG_NAME << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing" + std::to_string(ippl::Comm->size()) + "r_"
                                       + std::to_string(nr[0]) + "c.dat"));
    }
    ippl::finalize();

    return 0;
}
