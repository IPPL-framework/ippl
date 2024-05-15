#ifndef IPPL_FIELD_SOLVER_H
#define IPPL_FIELD_SOLVER_H

#include <memory>
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"

// Define the FieldSolver class
template <typename T, unsigned Dim>
class FieldSolver : public ippl::FieldSolverBase<T, Dim> {
  private:
    Field_t<Dim> *rho_m;
    VField_t<T, Dim> *E_m;
    Field<T, Dim> *phi_m;

  public:
    FieldSolver(std::string solver, Field_t<Dim> *rho, VField_t<T, Dim> *E, Field<T, Dim> *phi)
          : ippl::FieldSolverBase<T, Dim>(solver)
          , rho_m(rho)
          , E_m(E)
          , phi_m(phi) {
            setPotentialBCs();
          }

    ~FieldSolver(){}

    Field_t<Dim> *getRho() const { return rho_m; }
    void setRho(Field_t<Dim> *rho){ rho_m = rho; }

    VField_t<T, Dim> *getE() const { return E_m; }
    void setE(VField_t<T, Dim> *E){ E_m = E; }

    Field<T, Dim> *getPhi() const { return phi_m; }
    void setPhi(Field<T, Dim> *phi){ phi_m = phi; }

    void initSolver() override {
        Inform m("solver ");
        if (this->getStype() == "FFT") {
            initFFTSolver();
        } else if (this->getStype() == "CG") {
            initCGSolver();
        } else if (this->getStype() == "P3M") {
            initP3MSolver();
        } else if (this->getStype() == "OPEN") {
            initOpenSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    void setPotentialBCs() {
        // CG requires explicit periodic boundary conditions while the periodic Poisson solver
        // simply assumes them
        if (this->getStype() == "CG") {
            typedef ippl::BConds<Field<T, Dim>, Dim> bc_type;
            bc_type allPeriodic;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                allPeriodic[i] = std::make_shared<ippl::PeriodicFace<Field<T, Dim>>>(i);
            }
            phi_m->setFieldBC(allPeriodic);
        }
    }

    void runSolver() override {
        if (this->getStype() == "CG") {
            CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(this->getSolver());
            solver.solve();

            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                fname << "data/CG_";
                fname << ippl::Comm->size();
                fname << ".csv";

                Inform log(NULL, fname.str().c_str(), Inform::APPEND);
                int iterations = solver.getIterationCount();
                // Assume the dummy solve is the first call
                if (iterations == 0) {
                    log << "residue,iterations" << endl;
                }
                // Don't print the dummy solve
                if (iterations > 0) {
                    log << solver.getResidue() << "," << iterations << endl;
                }

            }
            ippl::Comm->barrier();
        } else if (this->getStype() == "FFT") {
            if constexpr (Dim == 2 || Dim == 3) {
                std::get<FFTSolver_t<T, Dim>>(this->getSolver()).solve();
            }
        } else if (this->getStype() == "P3M") {
            if constexpr (Dim == 3) {
                std::get<P3MSolver_t<T, Dim>>(this->getSolver()).solve();
            }
        } else if (this->getStype() == "OPEN") {
            if constexpr (Dim == 3) {
                std::get<OpenSolver_t<T, Dim>>(this->getSolver()).solve();
            }
        } else {
            throw std::runtime_error("Unknown solver type");
        }
    }

    template <typename Solver>
    void initSolverWithParams(const ippl::ParameterList& sp) {
        this->getSolver().template emplace<Solver>();
        Solver& solver = std::get<Solver>(this->getSolver());

        solver.mergeParameters(sp);

        solver.setRhs(*rho_m);

        if constexpr (std::is_same_v<Solver, CGSolver_t<T, Dim>>) {
            // The CG solver computes the potential directly and
            // uses this to get the electric field
            solver.setLhs(*phi_m);
            solver.setGradient(*E_m);
        } else {
            // The periodic Poisson solver, Open boundaries solver,
            // and the P3M solver compute the electric field directly
            solver.setLhs(*E_m);
        }
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

    void initCGSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
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
#endif
