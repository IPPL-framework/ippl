#ifndef IPPL_FIELD_SOLVER_H
#define IPPL_FIELD_SOLVER_H

#include <memory>
#include "FieldContainer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"
#include "PoissonSolvers/FFTPeriodicPoissonSolver.h"

template <class fc>
class FieldSolverStrategy {
  public:
    virtual ~FieldSolverStrategy() = default;

    virtual void initSolver(std::shared_ptr<fc> fcontainer);

    virtual void solve(std::shared_ptr<fc> fcontainer);
};

template <typename T>
class TwoDimFFTSolver : public FieldSolverStrategy<FieldContainer<T, 2>> {
    
  public:
    TwoDimFFTSolver() {}
    
    void initSolver(std::shared_ptr<FieldContainer<T, 2>> fcontainer) override {
        ippl::ParameterList sp;
        sp.add("output_type", FFTSolver_t<T, Dim>::SOL);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        solver.mergeParameters(sp);

        solver.setRhs(fcontainer->getOmegaField());
    }

    void solve([[maybe_unused]]std::shared_ptr<FieldContainer<T, 2>> fcontainer) override {
      solver.solve();
    }

  private:
    ippl::FFTPeriodicPoissonSolver<VField_t<T, 2>, Field<T, 2>> solver;
};


#endif
