// ChargedParticles header file
//   Defines a particle attribute for charged particles to be used in
//   test programs
//
#include "Ippl.h"

#include <csignal>
#include <thread>

#include "Utility/TypeUtils.h"

#include "PoissonSolvers/FFTOpenPoissonSolver.h"
#include "PoissonSolvers/FFTPeriodicPoissonSolver.h"
#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"
#include "PoissonSolvers/PoissonCG.h"

unsigned LoggingPeriod = 1;

// some typedefs
template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim = 3>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim = 3>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim = 3, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

template <typename T = double, unsigned Dim = 3>
using ORB = ippl::OrthogonalRecursiveBisection<Field<double, Dim>, T>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

template <typename T, unsigned Dim = 3>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim = 3, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T = double, unsigned Dim = 3, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

// heFFTe does not support 1D FFTs, so we switch to CG in the 1D case
template <typename T = double, unsigned Dim = 3>
using CGSolver_t = ippl::PoissonCG<Field<T, Dim>, Field_t<Dim>>;

using ippl::detail::ConditionalType, ippl::detail::VariantFromConditionalTypes;

template <typename T = double, unsigned Dim = 3>
using FFTSolver_t = ConditionalType<Dim == 2 || Dim == 3,
                                    ippl::FFTPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using FFTTruncatedGreenSolver_t = ConditionalType<Dim == 3, ippl::FFTTruncatedGreenPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using OpenSolver_t =
    ConditionalType<Dim == 3, ippl::FFTOpenPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using Solver_t = VariantFromConditionalTypes<CGSolver_t<T, Dim>, FFTSolver_t<T, Dim>,
                                             FFTTruncatedGreenSolver_t<T, Dim>, OpenSolver_t<T, Dim>>;

const double pi = Kokkos::numbers::pi_v<double>;

// Test programs have to define this variable for VTK dump purposes
extern const char* TestName;

// Signal handling
int interruptSignalReceived = 0;

/*!
 * Signal handler records the received signal
 * @param signal received signal
 */
void interruptHandler(int signal) {
    interruptSignalReceived = signal;
}

/*!
 * Checks whether a signal was received
 * @return Signal handler was called
 */
bool checkSignalHandler() {
    ippl::Comm->barrier();
    return interruptSignalReceived != 0;
}

/*!
 * Sets up the signal handler
 */
void setSignalHandler() {
    struct sigaction sa;
    sa.sa_handler = interruptHandler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGTERM, &sa, NULL) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGTERM ("
                  << SIGTERM << ")" << std::endl;
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGINT ("
                  << SIGINT << ")" << std::endl;
    }
}

template <typename T>
void dumpVTK(VField_t<T, 3>& E, int nx, int ny, int nz, int iteration, double dx, double dy,
             double dz) {
    typename VField_t<T, 3>::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                       << host_view(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpVTK(Field_t<3>& rho, int nx, int ny, int nz, int iteration, double dx, double dy,
             double dz) {
    typename Field_t<3>::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << TestName << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z) << endl;
            }
        }
    }
}

template <class PLayout, typename T, unsigned Dim = 3>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    VField_t<T, Dim> E_m;
    Field_t<Dim> rho_m;
    Field<T, Dim> phi_m;

    typedef ippl::BConds<Field<T, Dim>, Dim> bc_type;
    bc_type allPeriodic;

    // ORB
    ORB<T, Dim> orb;

    Vector_t<T, Dim> nr_m;

    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    std::string stype_m;
    std::string ptype_m;

    std::array<bool, Dim> isParallel_m;

    double Q_m;

private:
    Solver_t<T, Dim> solver_m;

public:
    double time_m;

    double rhoNorm_m;

    unsigned int loadbalancefreq_m;

    double loadbalancethreshold_m;

public:
    ParticleAttrib<double> q;                 // charge
    typename Base::particle_position_type P;  // particle velocity
    typename Base::particle_position_type E;  // electric field at particle position

    ChargedParticles(PLayout& pl, Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
                     Vector_t<double, Dim> rmax, std::array<bool, Dim> isParallel, double Q,
                     std::string solver)
        : Base(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , stype_m(solver)
        , isParallel_m(isParallel)
        , Q_m(Q) {
        registerAttributes();
        setupBCs();
        setPotentialBCs();
    }

    void setPotentialBCs() {
        // CG requires explicit periodic boundary conditions while the periodic Poisson solver
        // simply assumes them
        if (stype_m == "CG" || stype_m == "PCG") {
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                allPeriodic[i] = std::make_shared<ippl::PeriodicFace<Field<T, Dim>>>(i);
            }
        }
    }

    void registerAttributes() {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }

    ~ChargedParticles() {}

    void setupBCs() { setBCAllPeriodic(); }

    void updateLayout(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh, bool& isFirstRepartition) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        E_m.updateLayout(fl);
        rho_m.updateLayout(fl);
        if (stype_m == "CG" || stype_m == "PCG") {
            this->phi_m.updateLayout(fl);
            phi_m.setFieldBC(allPeriodic);
        }

        // Update layout with new FieldLayout
        PLayout& layout = this->getLayout();
        layout.updateLayout(fl, mesh);
        IpplTimings::stopTimer(tupdateLayout);
        static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
        IpplTimings::startTimer(tupdatePLayout);
        if (!isFirstRepartition) {
            this->update();
        }
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeFields(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& fl) {
        E_m.initialize(mesh, fl);
        rho_m.initialize(mesh, fl);
        if (stype_m == "CG" || stype_m == "PCG") {
            phi_m.initialize(mesh, fl);
            phi_m.setFieldBC(allPeriodic);
        }
    }

    void initializeORB(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh) {
        orb.initialize(fl, mesh, rho_m);
    }

    void repartition(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh, bool& isFirstRepartition) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl, isFirstRepartition);

        if (res != true) {
            std::cout << "Could not repartition!" << std::endl;
            return;
        }
        // Update
        this->updateLayout(fl, mesh, isFirstRepartition);
        if constexpr (Dim == 2 || Dim == 3) {
            if (stype_m == "FFT") {
                std::get<FFTSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
            }
            if constexpr (Dim == 3) {
                if (stype_m == "TG") {
                    std::get<FFTTruncatedGreenSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                } else if (stype_m == "OPEN") {
                    std::get<OpenSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                }
            }
        }
    }

    bool balance(size_type totalP, const unsigned int nstep) {
        if (ippl::Comm->size() < 2) {
            return false;
        }
        if (std::strcmp(TestName, "UniformPlasmaTest") == 0) {
            return (nstep % loadbalancefreq_m == 0);
        } else {
            int local = 0;
            std::vector<int> res(ippl::Comm->size());
            double equalPart = (double)totalP / ippl::Comm->size();
            double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
            if (dev > loadbalancethreshold_m) {
                local = 1;
            }
            MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT,
                          ippl::Comm->getCommunicator());

            for (unsigned int i = 0; i < res.size(); i++) {
                if (res[i] == 1) {
                    return true;
                }
            }
            return false;
        }
    }

    void gatherStatistics(size_type totalP) {
        std::vector<double> imb(ippl::Comm->size());
        double equalPart = (double)totalP / ippl::Comm->size();
        double dev       = (std::abs((double)this->getLocalNum() - equalPart) / totalP) * 100.0;
        ippl::Comm->gather(&dev, imb.data(), 1);

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/LoadBalance_";
            fname << ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(5);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, rank, imbalance percentage" << endl;
            }

            for (int r = 0; r < ippl::Comm->size(); ++r) {
                csvout << time_m << " " << r << " " << imb[r] << endl;
            }
        }

        ippl::Comm->barrier();
    }

    void gatherCIC() { gather(this->E, E_m, this->R); }

    void scatterCIC(size_type totalP, unsigned int iteration, Vector_t<double, Dim>& hrField) {
        Inform m("scatter ");

        rho_m = 0.0;
        scatter(q, rho_m, this->R);

        static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");
        IpplTimings::startTimer(sumTimer);
        double Q_grid = rho_m.sum();

        size_type Total_particles = 0;
        size_type local_particles = this->getLocalNum();

        ippl::Comm->reduce(local_particles, Total_particles, 1, std::plus<size_type>());

        double rel_error = std::fabs((Q_m - Q_grid) / Q_m);
        m << "Rel. error in charge conservation = " << rel_error << endl;

        if (ippl::Comm->rank() == 0) {
            if (Total_particles != totalP || rel_error > 1e-10) {
                m << "Time step: " << iteration << endl;
                m << "Total particles in the sim. " << totalP << " "
                  << "after update: " << Total_particles << endl;
                m << "Rel. error in charge conservation: " << rel_error << endl;
                ippl::Comm->abort();
            }
        }

        double cellVolume =
            std::reduce(hrField.begin(), hrField.end(), 1., std::multiplies<double>());
        rho_m = rho_m / cellVolume;

        rhoNorm_m = norm(rho_m);
        IpplTimings::stopTimer(sumTimer);

        // dumpVTK(rho_m, nr_m[0], nr_m[1], nr_m[2], iteration, hrField[0], hrField[1], hrField[2]);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax_m[d] - rmin_m[d];
            }
            rho_m = rho_m - (Q_m / size);
        }
    }

    void initSolver(const ippl::ParameterList& sp = ippl::ParameterList()) {
        Inform m("solver ");
        if (stype_m == "FFT") {
            initFFTSolver();
        } else if (stype_m == "CG" || stype_m == "PCG") {
            initCGSolver(sp);
        } else if (stype_m == "TG") {
            initTGSolver();
        } else if (stype_m == "OPEN") {
            initOpenSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    void runSolver() {
        if (stype_m == "CG" || stype_m == "PCG") {
            CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(solver_m);
            solver.solve();

            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                fname << "data/";
                fname << stype_m << "_";
                if (stype_m == "PCG") {
                    fname << ptype_m << "_";
                }
                fname << ippl::Comm->size();
                fname << ".csv";

                Inform log(NULL, fname.str().c_str(), Inform::APPEND);
                int iterations = solver.getIterationCount();
                // Assume the dummy solve is the first call
                if (time_m == 0 && iterations == 0) {
                    log << "time,residue,iterations" << endl;
                }
                // Don't print the dummy solve
                if (time_m > 0 || iterations > 0) {
                    log << time_m << "," << solver.getResidue() << "," << iterations << endl;
                }
            }
            ippl::Comm->barrier();
        } else if (stype_m == "FFT") {
            if constexpr (Dim == 2 || Dim == 3) {
                std::get<FFTSolver_t<T, Dim>>(solver_m).solve();
            }
        } else if (stype_m == "TG") {
            if constexpr (Dim == 3) {
                std::get<FFTTruncatedGreenSolver_t<T, Dim>>(solver_m).solve();
            }
        } else if (stype_m == "OPEN") {
            if constexpr (Dim == 3) {
                std::get<OpenSolver_t<T, Dim>>(solver_m).solve();
            }
        } else {
            throw std::runtime_error("Unknown solver type");
        }
    }

    template <typename Solver>
    void initSolverWithParams(const ippl::ParameterList& sp) {
        solver_m.template emplace<Solver>();
        Solver& solver = std::get<Solver>(solver_m);

        solver.mergeParameters(sp);

        solver.setRhs(rho_m);

        if constexpr (std::is_same_v<Solver, CGSolver_t<T, Dim>>) {
            // The CG solver computes the potential directly and
            // uses this to get the electric field
            solver.setLhs(phi_m);
            solver.setGradient(E_m);
        } else {
            // The periodic Poisson solver, Open boundaries solver,
            // and the TG solver compute the electric field directly
            solver.setLhs(E_m);
        }
    }

    void initCGSolver(const ippl::ParameterList& sp_old) {
        ippl::ParameterList sp;
        sp.merge(sp_old);
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);
        std::string solver_type = "";
        if (stype_m == "PCG") {
            solver_type = "preconditioned";
            ptype_m     = sp.get<std::string>("preconditioner_type");
        }
        sp.add("solver", solver_type);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }

    void initFFTSolver() {
        if constexpr (Dim == 2 || Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<FFTSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for FFT solver");
        }
    }

    void initTGSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTTruncatedGreenSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<FFTTruncatedGreenSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for TG solver");
        }
    }

    void initOpenSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", OpenSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);
            sp.add("algorithm", OpenSolver_t<T, Dim>::HOCKNEY);

            initSolverWithParams<OpenSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for OPEN solver");
        }
    }

    void dumpData() {
        auto Pview = P.getView();

        double kinEnergy = 0.0;
        double potEnergy = 0.0;

        rho_m     = dot(E_m, E_m);
        potEnergy = 0.5 * hr_m[0] * hr_m[1] * hr_m[2] * rho_m.sum();

        Kokkos::parallel_reduce(
            "Particle Kinetic Energy", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(kinEnergy));

        kinEnergy *= 0.5;
        double gkinEnergy = 0.0;

        ippl::Comm->reduce(kinEnergy, gkinEnergy, 1, std::plus<double>());

        const int nghostE = E_m.getNghost();
        auto Eview        = E_m.getView();
        Vector_t<T, Dim> normE;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        for (unsigned d = 0; d < Dim; ++d) {
            T temp = 0.0;
            ippl::parallel_reduce(
                "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    T myVal = std::pow(ippl::apply(Eview, args)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<T>(temp));
            T globaltemp = 0.0;
            ippl::Comm->reduce(temp, globaltemp, 1, std::plus<T>());
            normE[d] = std::sqrt(globaltemp);
        }

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2";
                for (unsigned d = 0; d < Dim; d++) {
                    csvout << ", E" << static_cast<char>((Dim <= 3 ? 'x' : '1') + d) << "_norm2";
                }
                csvout << endl;
            }

            csvout << time_m << " " << potEnergy << " " << gkinEnergy << " "
                   << potEnergy + gkinEnergy << " " << rhoNorm_m << " ";
            for (unsigned d = 0; d < Dim; d++) {
                csvout << normE[d] << " ";
            }
            csvout << endl;
        }

        ippl::Comm->barrier();
    }

    typename VField_t<T, Dim>::HostMirror getEMirror() const {
        auto Eview = E_m.getHostMirror();
        updateEMirror(Eview);
        return Eview;
    }

    void updateEMirror(typename VField_t<T, Dim>::HostMirror& mirror) const {
        Kokkos::deep_copy(mirror, E_m.getView());
    }

    void dumpLandau() { dumpLandau(E_m.getView()); }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = E_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Eview, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(Eview, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());
        double fieldEnergy =
            std::reduce(hr_m.begin(), hr_m.end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }

        ippl::Comm->barrier();
    }

    void dumpBumponTail() {
        const int nghostE = E_m.getNghost();
        auto Eview        = E_m.getView();
        double fieldEnergy, EzAmp;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double temp            = 0.0;
        ippl::parallel_reduce(
            "Ex inner product", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                // ippl::apply accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double myVal = std::pow(ippl::apply(Eview, args)[Dim - 1], 2);
                valL += myVal;
            },
            Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());
        fieldEnergy = std::reduce(hr_m.begin(), hr_m.end(), globaltemp, std::multiplies<double>());

        double tempMax = 0.0;
        ippl::parallel_reduce(
            "Ex max norm", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& valL) {
                // ippl::apply accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double myVal = std::fabs(ippl::apply(Eview, args)[Dim - 1]);
                if (myVal > valL) {
                    valL = myVal;
                }
            },
            Kokkos::Max<double>(tempMax));
        EzAmp = 0.0;
        ippl::Comm->reduce(tempMax, EzAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldBumponTail_";
            fname << ippl::Comm->size();
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Ez_field_energy, Ez_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << EzAmp << endl;
        }

        ippl::Comm->barrier();
    }

    void dumpParticleData() {
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror R_host = this->R.getHostMirror();
        typename ParticleAttrib<Vector_t<T, Dim>>::HostMirror P_host = this->P.getHostMirror();
        Kokkos::deep_copy(R_host, this->R.getView());
        Kokkos::deep_copy(P_host, P.getView());
        std::stringstream pname;
        pname << "data/ParticleIC_";
        pname << ippl::Comm->rank();
        pname << ".csv";
        Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, ippl::Comm->rank());
        pcsvout.precision(10);
        pcsvout.setf(std::ios::scientific, std::ios::floatfield);
        pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
        for (size_type i = 0; i < this->getLocalNum(); i++) {
            for (unsigned d = 0; d < Dim; d++) {
                pcsvout << R_host(i)[d] << " ";
            }
            for (unsigned d = 0; d < Dim; d++) {
                pcsvout << P_host(i)[d] << " ";
            }
            pcsvout << endl;
        }
        ippl::Comm->barrier();
    }

    void dumpLocalDomains(const FieldLayout_t<Dim>& fl, const unsigned int step) {
        if (ippl::Comm->rank() == 0) {
            const typename FieldLayout_t<Dim>::host_mirror_type domains = fl.getHostLocalDomains();
            std::ofstream myfile;
            myfile.open("data/domains" + std::to_string(step) + ".txt");
            for (unsigned int i = 0; i < domains.size(); ++i) {
                for (unsigned d = 0; d < Dim; d++) {
                    myfile << domains[i][d].first() << " " << domains[i][d].last() << " ";
                }
                myfile << "\n";
            }
            myfile.close();
        }
        ippl::Comm->barrier();
    }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};
