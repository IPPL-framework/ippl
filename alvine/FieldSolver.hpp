#ifndef IPPL_FIELD_SOLVER_H
#define IPPL_FIELD_SOLVER_H

#include <memory>
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"

// Define the FieldSolver class
template <typename T, unsigned Dim>
class FieldSolver : public ippl::FieldSolverBase<T, Dim> {
  using omega_field_type = std::conditional<Dim == 2, Field_t<Dim>, VField_t<T,Dim>>::type;
  private:
    omega_field_type *omega_m;

  public:
    FieldSolver(std::string solver, omega_field_type *omega)
          : ippl::FieldSolverBase<T, Dim>(solver)
          , omega_m(omega) {}

    ~FieldSolver(){}

    Field_t<Dim> *getOmega() const { return omega_m; }
    void setOmega(Field_t<Dim> *omega){ omega_m = omega; }

    void initSolver() override {

        Inform m("solver ");
        if (this->getStype() == "FFT") {
            initFFTSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    void runSolver() override {
      if constexpr (Dim == 2) {
        if (this->getStype() == "FFT") {
            std::get<FFTSolver_t<T, Dim>>(this->getSolver()).solve();
        } else {
          throw std::runtime_error("Unknown solver type");
        } 
      } else if constexpr (Dim == 3) {
        //TODO: probably need to run solver three times in this case and do ugly conversion.

      }
    } 

    template <typename Solver>
    void initSolverWithParams(const ippl::ParameterList& sp) {
        this->getSolver().template emplace<Solver>();

        Solver& solver = std::get<Solver>(this->getSolver());

        solver.mergeParameters(sp);

        if constexpr (Dim == 2) {
          solver.setRhs(*omega_m);
        } else if constexpr (Dim == 3) {
          //TODO probably need ugly construct for 3D case here.
        }
    }

    void initFFTSolver() {
        if constexpr (Dim == 2 || Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTSolver_t<T, Dim>::SOL);
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

};
#endif
