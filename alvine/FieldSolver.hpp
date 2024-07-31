#include <memory>

template <typename T, unsigned Dim, class FieldContainerType>
class FieldSolver {
public:
    FieldSolver() {}

    ~FieldSolver() = default;

    virtual void initSolver(std::shared_ptr<FieldContainerType> fcontainer);

    virtual void solve(std::shared_ptr<FieldContainerType> fcontainer);
};

template <typename T>
class FieldSolver<T, 2, FieldContainer<T, 2>> {
public:
    FieldSolver() {}

    ~FieldSolver() = default;

    void initSolver(std::shared_ptr<FieldContainer<T, 2>> fc) {
        ippl::ParameterList sp;
        sp.add("output_type", FFTSolver_t<T, Dim>::SOL);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        solver.mergeParameters(sp);

        solver.setRhs(fc->getOmegaField());
    }

    void solve([[maybe_unused]] std::shared_ptr<FieldContainer<T, 2>> fc) { solver.solve(); }

private:
    ippl::FFTPeriodicPoissonSolver<VField_t<T, 2>, Field<T, 2>> solver;
};

template <typename T>
class FieldSolver<T, 3, FieldContainer<T, 3>> {
public:
    FieldSolver() {}

    ~FieldSolver() = default;

    void initSolver(std::shared_ptr<FieldContainer<T, 3>> fc) {
        ippl::ParameterList sp;
        sp.add("max_iterations", 2000);
        sp.add("output_type", CGSolver_t<T, 1>::SOL);

        solver_x.mergeParameters(sp);
        solver_y.mergeParameters(sp);
        solver_z.mergeParameters(sp);

        Field<T, 1> lhsx(fc->getOmegaFieldx().get_mesh(), fc->getOmegaFieldx().getLayout());
        solver_x.setLhs(lhsx);
        lhsx = 0.0;

        Field<T, 1> lhsy(fc->getOmegaFieldy().get_mesh(), fc->getOmegaFieldy().getLayout());
        solver_y.setLhs(lhsy);
        lhsy = 0.0;

        Field<T, 1> lhsz(fc->getOmegaFieldz().get_mesh(), fc->getOmegaFieldx().getLayout());
        solver_z.setLhs(lhsz);
        lhsz = 0.0;

        solver_x.setRhs(fc->getOmegaFieldx());
        solver_y.setRhs(fc->getOmegaFieldy());
        solver_z.setRhs(fc->getOmegaFieldz());
    }

    void solve([[maybe_unused]] std::shared_ptr<FieldContainer<T, 3>> fc) {
        solver_x.solve();
        solver_y.solve();
        solver_z.solve();
    }

private:
    ippl::PoissonCG<Field<T, 1>, Field_t<1>> solver_x;
    ippl::PoissonCG<Field<T, 1>, Field_t<1>> solver_y;
    ippl::PoissonCG<Field<T, 1>, Field_t<1>> solver_z;
};
