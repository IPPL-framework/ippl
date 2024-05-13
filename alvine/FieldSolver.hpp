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
