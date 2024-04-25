#ifndef IPPL_FIELD_SOLVER_BASE_H
#define IPPL_FIELD_SOLVER_BASE_H

#include <memory>

#include "Manager/BaseManager.h"
#include "PoissonSolvers/FFTOpenPoissonSolver.h"
#include "PoissonSolvers/FFTPeriodicPoissonSolver.h"
#include "PoissonSolvers/P3MSolver.h"
#include "PoissonSolvers/PoissonCG.h"

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

template <typename T, unsigned Dim, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T, unsigned Dim, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

template <typename T, unsigned Dim>
using CGSolver_t = ippl::PoissonCG<Field<T, Dim>, Field_t<Dim>>;

using ippl::detail::ConditionalType, ippl::detail::VariantFromConditionalTypes;

template <typename T, unsigned Dim>
using FFTSolver_t = ConditionalType<Dim == 2 || Dim == 3,
                                    ippl::FFTPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T, unsigned Dim>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::P3MSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T, unsigned Dim>
using OpenSolver_t =
    ConditionalType<Dim == 3, ippl::FFTOpenPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T, unsigned Dim>
using Solver_t = VariantFromConditionalTypes<CGSolver_t<T, Dim>, FFTSolver_t<T, Dim>,
                                             P3MSolver_t<T, Dim>, OpenSolver_t<T, Dim>>;

// Define the FieldSolverBase class
namespace ippl {
    template <typename T, unsigned Dim>
    class FieldSolverBase {
    private:
        std::string stype_m;
        Solver_t<T, Dim> solver_m;

    public:
        FieldSolverBase(std::string solver)
            : stype_m(solver) {}

        virtual void initSolver() = 0;

        virtual void runSolver() = 0;

        virtual ~FieldSolverBase() = default;

        std::string& getStype() { return stype_m; }

        void setStype(std::string& solver) { stype_m = solver; }

        Solver_t<T, Dim>& getSolver() { return solver_m; }

        void setSolver(Solver_t<T, Dim>& solver) { solver_m = solver; }
    };
}  // namespace ippl
#endif
