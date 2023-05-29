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
inline typename Callable::value_type idxApply(const Callable& F, IdxArgs... idxargs) {
    return F(idxargs...);
}

template <OpDim D, class Callable, typename... IdxArgs>
inline typename Callable::value_type shiftedIdxApply(const Callable& F, size_type shift,
                                                     IdxArgs... idxargs) {
    auto shiftedIdx = std::make_tuple(idxargs...);
    std::get<D>(shiftedIdx) += shift;
    return std::apply(
        [&](auto&&... lambdaArgs) {
            return idxApply(F, std::forward<decltype(lambdaArgs)>(lambdaArgs)...);
        },
        shiftedIdx);
}

template <OpDim applyDim, typename T, unsigned Dim, class Callable>
struct BackwardBase {
    typedef T value_type;

    BackwardBase(const FView_t<Dim>& view, const Vector_t<Dim>& hInv)
        : view_m(view)
        , hInv_m(hInv) {}

    T stencilOp(const Callable& F, size_type i, size_type j, size_type k) const {
        return 0.5 * this->hInv_m[applyDim]
               * (3.0 * idxApply(F, i, j, k) - 4.0 * shiftedIdxApply<applyDim>(F, -1, i, j, k)
                  + shiftedIdxApply<applyDim>(F, -2, i, j, k));
    }

    const FView_t<Dim>& view_m;
    const Vector_t<Dim>& hInv_m;
};

template <OpDim applyDim, typename T, unsigned Dim, class Callable>
struct BackwardStencil : public BackwardBase<applyDim, T, Dim, Callable> {
    BackwardStencil(const FView_t<Dim>& view, const Callable& leftOp, const Vector_t<Dim>& hInv)
        : BackwardBase<applyDim, T, Dim, Callable>(view, hInv)
        , leftOp_m(leftOp) {}

    T operator()(size_type i, size_type j, size_type k) const {
        return this->stencilOp(this->leftOp_m, i, j, k);
    }

    const Callable& leftOp_m;
};

template <OpDim applyDim, unsigned Dim, typename T>
struct BackwardStencil<applyDim, T, Dim, FView_t<Dim>>
    : public BackwardBase<applyDim, T, Dim, FView_t<Dim>> {
    BackwardStencil(const FView_t<Dim>& view, const Vector_t<Dim>& hInv)
        : BackwardBase<applyDim, T, Dim, FView_t<Dim>>(view, hInv) {}

    T operator()(size_type i, size_type j, size_type k) const {
        return this->stencilOp(this->view_m, i, j, k);
    }
};

#endif  // hessian_h
