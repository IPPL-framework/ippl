#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>

#include "Manager/BaseManager.h"

#include <csignal>
#include <thread>

#include "Utility/TypeUtils.h"

#include "Solver/ElectrostaticsCG.h"
#include "Solver/FFTPeriodicPoissonSolver.h"
#include "Solver/FFTPoissonSolver.h"
#include "Solver/P3MSolver.h"


// some typedefs
template <unsigned Dim>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim= 3, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

template <typename T = double, unsigned Dim=3>
using ORB = ippl::OrthogonalRecursiveBisection<Field<double, Dim>, T>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T = double, unsigned Dim=3, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

// heFFTe does not support 1D FFTs, so we switch to CG in the 1D case
template <typename T = double, unsigned Dim=3>
using CGSolver_t = ippl::ElectrostaticsCG<Field<T, Dim>, Field_t<Dim>>;

using ippl::detail::ConditionalType, ippl::detail::VariantFromConditionalTypes;

template <typename T = double, unsigned Dim=3>
using FFTSolver_t = ConditionalType<Dim == 2 || Dim == 3,
                                    ippl::FFTPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::P3MSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using OpenSolver_t =
    ConditionalType<Dim == 3, ippl::FFTPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim=3>
using Solver_t = VariantFromConditionalTypes<CGSolver_t<T, Dim>, FFTSolver_t<T, Dim>,
                                             P3MSolver_t<T, Dim>, OpenSolver_t<T, Dim>>;

class PicManager;

namespace ippl {
    // Define the ParticlesContainer class
    template <class PLayout, typename T, unsigned Dim = 3>
    class ParticlesContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

    public:
        // Constructor, destructor, and other member functions as needed
        ParticlesContainer() { /* constructor code */ }

        ~ParticlesContainer() { /* destructor code */ }
        
        ParticleAttrib<double> q;                 // charge
        typename Base::particle_position_type P;  // particle velocity
        typename Base::particle_position_type E;  // electric field at particle position
        ParticlesContainer(PLayout& pl)
        : Base(pl) {
        registerAttributes();
        setupBCs();
        }
	void registerAttributes() {
		// register the particle attributes
		this->addAttribute(q);
		this->addAttribute(P);
		this->addAttribute(E);
	}
	void setupBCs() { setBCAllPeriodic(); }
    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
    };
}  // namespace ippl

namespace ippl {
    // Define the FieldsContainer class
    template <class FieldLayout_t, typename T, unsigned Dim = 3>
    class FieldsContainer{
    
    public:
        FieldsContainer(Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
                        Vector_t<double, Dim> rmax, ippl::e_dim_tag decomp[Dim], double Q)
            : hr_m(hr), rmin_m(rmin), rmax_m(rmax), Q_m(Q) {
            for (unsigned int i = 0; i < Dim; i++) {
                decomp_m[i] = decomp[i];
            }
     }
    
    VField_t<T, Dim> E_m;
    Field_t<Dim> rho_m;
    Field_t<Dim> phi_m;

    // ORB
    ORB<T, Dim> orb;

    Vector_t<T, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    std::string stype_m;

    double Q_m;
    };
}  // namespace ippl

namespace ippl {
    // Define the FieldSolver class     
    template <class FieldLayout_t, typename T, unsigned Dim = 3>
    class FieldSolver {
    private:
        std::string stype_m; // Declare stype_m as a member variable
        Solver_t<T, Dim> solver_m;
        double time_m;
    public:
    FieldSolver(std::string solver, Solver_t<T, Dim> solver_, PicManager& picManager)
        : stype_m(solver), solver_m(solver_), time_m(picManager.time_m) {}
    
    void initSolver() {
        Inform m("solver ");
        if (stype_m == "FFT") {
            initFFTSolver();
        } else if (stype_m == "CG") {
            initCGSolver();
        } else if (stype_m == "P3M") {
            initP3MSolver();
        } else if (stype_m == "OPEN") {
            initOpenSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    void runSolver() {
        if (stype_m == "CG") {
            CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(solver_m);
            solver.solve();

            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                fname << "data/CG_";
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
        } else if (stype_m == "P3M") {
            if constexpr (Dim == 3) {
                std::get<P3MSolver_t<T, Dim>>(solver_m).solve();
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
    void initSolverWithParams(const ippl::ParameterList& sp,  PicManager& picManager) {
        solver_m.template emplace<Solver>();
        Solver& solver = std::get<Solver>(solver_m);

        solver.mergeParameters(sp);

        solver.setRhs(baseManager.PicManager.rho_m);

        if constexpr (std::is_same_v<Solver, CGSolver_t<T, Dim>>) {
            // The CG solver computes the potential directly and
            // uses this to get the electric field
            solver.setLhs(baseManager::PicManager.phi_m);
            solver.setGradient(baseManager.PicManager.E_m);
        } else {
            // The periodic Poisson solver, Open boundaries solver,
            // and the P3M solver compute the electric field directly
            solver.setLhs(baseManager::PicManager.E_m);
        }
    }

    void initCGSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);

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

    void initP3MSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", P3MSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<P3MSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for P3M solver");
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
    };
}
/*
namespace ippl {
    // Define the ParticlesContainer class
    template <class PLayout, typename T, unsigned Dim = 3>
    class LoadBalancer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

    public:
        // Constructor, destructor, and other member functions as needed
        LoadBalancer() {   }

        ~LoadBalancer() {  }
        
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
      void repartition(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh,
                     ChargedParticles<PLayout, T, Dim>& buffer, bool& isFirstRepartition) {
          // Repartition the domains
          bool res = orb.binaryRepartition(this->R, fl, isFirstRepartition);

          if (res != true) {
              std::cout << "Could not repartition!" << std::endl;
              return;
          }
          // Update
          this->updateLayout(fl, mesh, buffer, isFirstRepartition);
          if constexpr (Dim == 2 || Dim == 3) {
            if (stype_m == "FFT") {
                std::get<FFTSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
            }
            if constexpr (Dim == 3) {
                if (stype_m == "P3M") {
                    std::get<P3MSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                } else if (stype_m == "OPEN") {
                    std::get<OpenSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                }
            }
          }
      }
      void updateLayout(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh,
		              ChargedParticles<PLayout, T, Dim>& buffer, bool& isFirstRepartition) {
		// Update local fields
		static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
		IpplTimings::startTimer(tupdateLayout);
		E_m.updateLayout(fl);
		rho_m.updateLayout(fl);
		if (stype_m == "CG") {
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
		    layout.update(*this, buffer);
		}
		IpplTimings::stopTimer(tupdatePLayout);
       }
       void initializeFields(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& fl) {
		E_m.initialize(mesh, fl);
		rho_m.initialize(mesh, fl);
		if (stype_m == "CG") {
		    phi_m.initialize(mesh, fl);
		    phi_m.setFieldBC(allPeriodic);
		}
	}  
    };
}  // namespace ippl
 */
 
 namespace ippl {

    template <class ParticleContainer, class FieldContainer>
    class PicManager : public BaseManager {
    public:
        PicManager()
            : BaseManager() {}

        virtual ~PicManager() = default;

        virtual void par2grid() = 0;

        virtual void grid2par() = 0;
        
        static double time_m = BaseManager::time_m;

    protected:
     
        std::unique_ptr<FieldsContainer> fcontainer_m;

        std::unique_ptr<ParticlesContainer> pcontainer_m;

        //std::unique_ptr<Stepper> stepper_m;

        //std::unique_ptr<LoadBalancer> loadbalancer_m;

        //std::unique_ptr<FieldSolver<FLayout, T, Dim>> fsolver_m;
    };
}  // namespace ippl
 
#endif

