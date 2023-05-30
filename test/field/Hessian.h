#ifndef hessian_h
#define hessian_h

#include "Ippl.h"

template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <unsigned Dim = 3>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim = 3>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

template <unsigned Dim = 3>
using Vector_t = Vector<double, Dim>;

template <unsigned Dim = 3>
using Field_t = Field<double, Dim>;

template <unsigned Dim = 3>
using VField_t = Field<Vector_t<Dim>, Dim>;

template <unsigned Dim = 3>
using Matrix_t = Vector<Vector<double>>;

template <unsigned Dim = 3>
using MField_t = Field<Matrix_t<Dim>, Dim>;

template <unsigned Dim = 3>
using FView_t = typename Field_t<Dim>::view_type;

template <unsigned Dim = 3>
using MView_t = typename MField_t<Dim>::view_type;

enum DiffType {
    Centered,
    Forward,
    Backward,
    CenteredDeriv2
};
enum OpDim {
    X,
    Y,
    Z
};

///////////////////
// Index helpers //
///////////////////

template <class Callable, typename... IdxArgs,
          std::enable_if_t<std::conjunction_v<std::is_same<size_type, IdxArgs>...>, bool> = true>
inline typename Callable::value_type idxApply(Callable& F, IdxArgs... idxargs) {
    return F(idxargs...);
}

template <OpDim D, class Callable, typename... IdxArgs>
inline typename Callable::value_type shiftedIdxApply(Callable& F, size_type shift,
                                                     IdxArgs... idxargs) {
    auto shiftedIdx = std::make_tuple(idxargs...);
    std::get<D>(shiftedIdx) += shift;
    return std::apply(
        [&](auto&&... lambdaArgs) {
            return idxApply(F, std::forward<decltype(lambdaArgs)>(lambdaArgs)...);
        },
        shiftedIdx);
}

// Interface of Stencils
template <OpDim applyDim, unsigned Dim, class Callable>
struct BaseStencil {
    virtual Callable::value_type operator()(Callable& F, Vector_t<Dim>& hInv, size_type i,
                                            size_type j, size_type k) = 0;
};

// Stencil implementing Centered Difference along dimension `applyDim`
template <OpDim applyDim, unsigned Dim, class Callable>
struct CenteredStencil : BaseStencil<applyDim, Dim, Callable> {
    Callable::value_type operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j,
                                    size_type k) {
        return hInv[applyDim] * hInv[applyDim]
               * (shiftedIdxApply<applyDim>(F, -1, i, j, k) - 2.0 * idxApply(F, i, j, k)
                  + shiftedIdxApply<applyDim>(F, 1, i, j, k));
    }
};

// Stencil implementing Forward Difference along dimension `applyDim`
template <OpDim applyDim, unsigned Dim, class Callable>
struct ForwardStencil : BaseStencil<applyDim, Dim, Callable> {
    Callable::value_type operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j,
                                    size_type k) {
        return 0.5 * hInv[applyDim]
               * (-3.0 * idxApply(F, i, j, k) + 4.0 * shiftedIdxApply<applyDim>(F, 1, i, j, k)
                  - shiftedIdxApply<applyDim>(F, 2, i, j, k));
    }
};

// Stencil implementing Backward Difference along dimension `applyDim`
template <OpDim applyDim, unsigned Dim, class Callable>
struct BackwardStencil : BaseStencil<applyDim, Dim, Callable> {
    Callable::value_type operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j,
                                    size_type k) {
        return 0.5 * hInv[applyDim]
               * (3.0 * idxApply(F, i, j, k) - 4.0 * shiftedIdxApply<applyDim>(F, -1, i, j, k)
                  + shiftedIdxApply<applyDim>(F, -2, i, j, k));
    }
};

// Operator taking a Callable to apply the stencil `Stencil` from the left
// Stores the stencil and gridspacing
template <typename T, unsigned Dim, class Callable, class Stencil>
struct OperatorBase {
    typedef T value_type;

    OperatorBase(Vector_t<Dim>& hInv, Stencil& stencil)
        : hInv_m(hInv)
        , stencil_m(stencil) {}

    T stencilOp(Callable& F, size_type i, size_type j, size_type k) {
        return stencil_m(F, hInv_m, i, j, k);
    }

    Vector_t<Dim>& hInv_m;
    Stencil stencil_m;
};

// Operator taking a Callable to apply the stencil `Stencil` from the left
template <typename T, unsigned Dim, class Callable, class Stencil>
struct ChainedOperator : public OperatorBase<T, Dim, Callable, Stencil> {
    ChainedOperator(Callable& leftOp, Vector_t<Dim>& hInv, Stencil& stencil)
        : OperatorBase<T, Dim, Callable, Stencil>(hInv, stencil)
        , leftOp_m(leftOp) {}

    T operator()(size_type i, size_type j, size_type k) {
        return this->stencilOp(leftOp_m, i, j, k);
    }

    Callable& leftOp_m;
};

// Template Specialization for applying stenciil to field directly (right-most operator)
template <unsigned Dim, typename T, class Stencil>
struct ChainedOperator<T, Dim, FView_t<Dim>, Stencil>
    : public OperatorBase<T, Dim, FView_t<Dim>, Stencil> {
    ChainedOperator(FView_t<Dim>& view, Vector_t<Dim>& hInv, Stencil& stencil)
        : OperatorBase<T, Dim, FView_t<Dim>, Stencil>(hInv, stencil)
        , view_m(view) {}

    T operator()(size_type i, size_type j, size_type k) { return this->stencilOp(view_m, i, j, k); }

    FView_t<Dim>& view_m;
};

// TODO
// template <unsigned Dim, typename T, class ReturnType, class DiffOpX, class DiffOpY, class
// DiffOpZ> class GeneralizedHessOp { public:
//     typedef typename Field_t<Dim>::view_type FView_t;

//     // Define typedefs for innermost operators applied to Field<T> as they are identical on each
//     // row
//     typedef ChainedOperator<OpDim::X, Dim, T, DiffX, FView_t> colOpX_t;
//     typedef ChainedOperator<OpDim::Y, Dim, T, DiffY, FView_t> colOpY_t;
//     typedef ChainedOperator<OpDim::Z, Dim, T, DiffZ, FView_t> colOpZ_t;

//     GeneralizedHessOp(const Field_t<Dim>& field, Vector_t<Dim> hInvVector) {}
// };

#endif  // hessian_h
