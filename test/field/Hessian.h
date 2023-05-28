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

template <OpDim applyDim, unsigned Dim = 3>
struct StencilBase {
    virtual double operator()(const FView_t<Dim>& fieldView, const Vector_t<Dim>& hInv, size_type i,
                              size_type j, size_type k);
};

template <OpDim applyDim, unsigned Dim = 3>
struct CenteredStencil : StencilBase<applyDim, Dim> {
    double operator()(const FView_t<Dim>& fieldView, const Vector_t<Dim>& hInv, size_type i,
                      size_type j, size_type k) {
        return 0.5 * hInv[applyDim]
               * (-shiftedIdxApply<applyDim>(fieldView, -1, i, j, k)
                  + shiftedIdxApply<applyDim>(fieldView, 1, i, j, k));
    }
};

template <OpDim applyDim, unsigned Dim = 3>
struct BackwardStencil : StencilBase<applyDim, Dim> {
    double operator()(const FView_t<Dim>& fieldView, const Vector_t<Dim>& hInv, size_type i,
                      size_type j, size_type k) {
        return 0.5 * hInv[applyDim]
               * (3.0 * idxApply(fieldView, i, j, k)
                  - 4.0 * shiftedIdxApply<applyDim>(fieldView, -1, i, j, k)
                  + shiftedIdxApply<applyDim>(fieldView, -2, i, j, k));
    }
};

#endif  // hessian_h
