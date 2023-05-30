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

template <OpDim applyDim, typename T, unsigned Dim, class Callable>
struct CenteredStencil {
    T operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j, size_type k) {
        return hInv[applyDim] * hInv[applyDim]
               * (shiftedIdxApply<applyDim>(F, -1, i, j, k) - 2.0 * idxApply(F, i, j, k)
                  + shiftedIdxApply<applyDim>(F, 1, i, j, k));
    }
};

template <OpDim applyDim, typename T, unsigned Dim, class Callable>
struct ForwardStencil {
    T operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j, size_type k) {
        return 0.5 * hInv[applyDim]
               * (-3.0 * idxApply(F, i, j, k) + 4.0 * shiftedIdxApply<applyDim>(F, 1, i, j, k)
                  - shiftedIdxApply<applyDim>(F, 2, i, j, k));
    }
};

template <OpDim applyDim, typename T, unsigned Dim, class Callable>
struct BackwardStencil {
    T operator()(Callable& F, Vector_t<Dim>& hInv, size_type i, size_type j, size_type k) {
        return 0.5 * hInv[applyDim]
               * (3.0 * idxApply(F, i, j, k) - 4.0 * shiftedIdxApply<applyDim>(F, -1, i, j, k)
                  + shiftedIdxApply<applyDim>(F, -2, i, j, k));
    }
};

template <OpDim applyDim, typename T, unsigned Dim, class Callable, class Stencil>
struct OperatorBase {
    typedef T value_type;

    OperatorBase(FView_t<Dim>& view, Vector_t<Dim>& hInv, Stencil& stencil)
        : view_m(view)
        , hInv_m(hInv)
        , stencil_m(stencil) {}

    T stencilOp(Callable& F, size_type i, size_type j, size_type k) {
        return stencil_m(F, hInv_m, i, j, k);
    }

    FView_t<Dim>& view_m;
    Vector_t<Dim>& hInv_m;
    Stencil stencil_m;
};

template <OpDim applyDim, typename T, unsigned Dim, class Callable, class Stencil>
struct ChainedOperator : public OperatorBase<applyDim, T, Dim, Callable, Stencil> {
    ChainedOperator(FView_t<Dim>& view, Callable& leftOp, Vector_t<Dim>& hInv, Stencil& stencil)
        : OperatorBase<applyDim, T, Dim, Callable, Stencil>(view, hInv, stencil)
        , leftOp_m(leftOp) {}

    T operator()(size_type i, size_type j, size_type k) {
        return this->stencilOp(this->leftOp_m, i, j, k);
    }

    Callable& leftOp_m;
};

// Template Specialization for applying stenciil to field directly (right-most operator)
template <OpDim applyDim, unsigned Dim, typename T, class Stencil>
struct ChainedOperator<applyDim, T, Dim, FView_t<Dim>, Stencil>
    : public OperatorBase<applyDim, T, Dim, FView_t<Dim>, Stencil> {
    ChainedOperator(FView_t<Dim>& view, Vector_t<Dim>& hInv, Stencil& stencil)
        : OperatorBase<applyDim, T, Dim, FView_t<Dim>, Stencil>(view, hInv, stencil) {}

    T operator()(size_type i, size_type j, size_type k) {
        return this->stencilOp(this->view_m, i, j, k);
    }
};

// template <unsigned Dim, typename T, class ReturnType, class DiffOpX, class DiffOpY, class
// DiffOpZ> class GeneralizedHessOp { public:
//     typedef typename Field_t<Dim>::view_type FView_t;

//     // Define typedefs for innermost operators applied to Field<T> as they are identical on each
//     // row
//     typedef DiffOpChain<OpDim::X, Dim, T, DiffX, FView_t> colOpX_t;
//     typedef DiffOpChain<OpDim::Y, Dim, T, DiffY, FView_t> colOpY_t;
//     typedef DiffOpChain<OpDim::Z, Dim, T, DiffZ, FView_t> colOpZ_t;

//     GeneralizedHessOp(const Field_t<Dim>& field, Vector_t<Dim> hInvVector) {}
// };

// template <unsigned Dim, typename T, class ReturnType, class DiffOpX, class DiffOpY, class
// DiffOpZ> class GeneralizedHessOp { public:
//     typedef typename Field_t<Dim>::view_type FView_t;

//     // Define typedefs for innermost operators applied to Field<T> as they are identical on each
//     // row
//     typedef DiffOpChain<OpDim::X, Dim, T, DiffX, FView_t> colOpX_t;
//     typedef DiffOpChain<OpDim::Y, Dim, T, DiffY, FView_t> colOpY_t;
//     typedef DiffOpChain<OpDim::Z, Dim, T, DiffZ, FView_t> colOpZ_t;

//     GeneralizedHessOp(const Field_t<Dim>& field, Vector_t<Dim> hInvVector)
//         : GeneralDiffOpInterface<Dim, T, ReturnType>(field, hInvVector)
//         ,
//         // Define Operators of each element of the 3x3 Hessian
//         diff_xx(this->view_m, this->hInvVector_m)
//         , diff_xy(this->view_m, this->hInvVector_m)
//         , diff_xz(this->view_m, this->hInvVector_m)
//         , diff_yx(this->view_m, this->hInvVector_m)
//         , diff_yy(this->view_m, this->hInvVector_m)
//         , diff_yz(this->view_m, this->hInvVector_m)
//         , diff_zx(this->view_m, this->hInvVector_m)
//         , diff_zy(this->view_m, this->hInvVector_m)
//         , diff_zz(this->view_m, this->hInvVector_m) {}

//     // Compute Hessian of specific Index_t `idx`
//     inline ReturnType operator()(size_type i, size_type j, size_type k) const {
//         ReturnType hess_matrix;
//         hess_matrix[0] = {diff_xx(i, j, k), diff_xy(i, j, k), diff_xz(i, j, k)};
//         hess_matrix[1] = {diff_yx(i, j, k), diff_yy(i, j, k), diff_yz(i, j, k)};
//         hess_matrix[2] = {diff_zx(i, j, k), diff_zy(i, j, k), diff_zz(i, j, k)};

//         return hess_matrix;
//     }

// private:
//     // Row 1
//     DiffOpX<OpDim::X, T, Dim, DiffOpX> diff_xx;
//     DiffOpX<OpDim::X, T, Dim, DiffOpY> diff_xy;
//     DiffOpX<OpDim::X, T, Dim, DiffOpZ> diff_xz;

//     // Row 2
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpX_t> diff_yx;
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpY_t> diff_yy;
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpZ_t> diff_yz;

//     // Row 3
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpX_t> diff_zx;
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpY_t> diff_zy;
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpZ_t> diff_zz;
// };

// ///////////////////////////////////////////////
// // Specialization to chain stencil operators //
// ///////////////////////////////////////////////

// template <unsigned Dim, class Callable, class Stencil1D>
// class BaseDiffOp {
// public:
//     typedef Field_t<Dim>::type value_type;
//     typedef typename Field_t<Dim>::view_type FView_t;

//     BaseDiffOp(const FView_t& view, Vector_t<Dim> hInvVector, const Stencil1D& stencilOp)
//         : view_m(view)
//         , hInvVector_m(hInvVector)
//         , stencil_m(stencilOp){};

//     // Applies templated stencil type on specific callable `F`
//     inline value_type stencilOp(const Callable& F, size_type i, size_type j, size_type k) const {
//         return stencil_m(F, hInvVector_m, i, j, k);
//     }

// protected:
//     const FView_t& view_m;
//     Vector_t<Dim> hInvVector_m;
//     const Stencil1D& stencil_m;
// };

// template <unsigned Dim, class Callable, class Stencil1D>
// class DiffOpChain : public BaseDiffOp<Dim, Callable, Stencil1D> {
// public:
//     typedef Field_t<Dim>::type value_type;
//     typedef typename Field_t<Dim>::view_type FView_t;

//     DiffOpChain(const FView_t& view, Vector_t<Dim> hInvVector, const Stencil1D& stencilOp)
//         : BaseDiffOp<Dim, Callable, Stencil1D>(view, hInvVector, stencilOp)
//         , leftOp_m(view, this->hInvVector_m, stencilOp) {}

//     // Specialization to call the stencil operator on the left operator
//     inline value_type operator()(size_type i, size_type j, size_type k) const {
//         return this->template stencilOp(leftOp_m, i, j, k);
//     }

// private:
//     // Need additional callable which might contain other operators
//     const Callable& leftOp_m;
// };

// // Innermost operator acting on the field (template specialization)
// template <unsigned Dim, class Stencil1D>
// class DiffOpChain<Dim, typename Field_t<Dim>::view_type, Stencil1D>
//     : public BaseDiffOp<Dim, typename Field_t<Dim>::view_type, Stencil1D> {
// public:
//     typedef Field_t<Dim>::type value_type;
//     typedef typename Field_t<Dim>::view_type FView_t;

//     DiffOpChain(const FView_t& view, Vector_t<Dim> hInvVector, const Stencil1D& stencilOp)
//         : BaseDiffOp<Dim, FView_t, Stencil1D>(view, hInvVector, stencilOp) {}

//     // Specialization to call the stencil operator on the field
//     inline value_type operator()(size_type i, size_type j, size_type k) const {
//         return this->template stencilOp(this->view_m, i, j, k);
//     }
// };

// Specialization for compact centered stencils of 2nd order
// e.g. $\frac{\partial^2 f}{\partial x^2}$
// template <OpDim D, unsigned Dim, typename T>
// class DiffOpChain<D, Dim, T, Centered,
//                   DiffOpChain<D, Dim, T, Centered, typename Field_t<Dim>::view_type>>
//     : public BaseDiffOp<D, Dim, T, Centered,
//                         DiffOpChain<D, Dim, T, Centered, typename Field_t<Dim>::view_type>> {
// public:
//     typedef T value_type;
//     typedef typename Field_t<Dim>::view_type FView_t;

//     DiffOpChain(const FView_t& view, Vector_t<Dim> hInvVector)
//         : BaseDiffOp<D, Dim, T, Centered,
//                      DiffOpChain<D, Dim, T, Centered, typename Field_t<Dim>::view_type>>(
//             view, hInvVector) {}

//     // Specialization to call the stencil operator on the field
//     inline T operator()(size_type i, size_type j, size_type k) const {
//         return centered_stencil_deriv2<D, T>(this->hInvVector_m[D], this->view_m, i, j, k);
//     }
// };

// Can be used as type for storing a composed operator in STL containers
// template <unsigned Dim, typename T, class ReturnType>
// class GeneralDiffOpInterface {
// public:
//     typedef typename Field_t<Dim>::view_type FView_t;

//     GeneralDiffOpInterface(const Field_t<Dim>& field, Vector_t<Dim> hInvVector)
//         : view_m(field.getView())
//         , hInvVector_m(hInvVector) {}

//     virtual inline ReturnType operator()(size_type i, size_type j, size_type k) const = 0;

// protected:
//     const FView_t& view_m;
//     const Vector_t<Dim> hInvVector_m;
// };

// template <unsigned Dim, typename T, class ReturnType, DiffType DiffX, DiffType DiffY,
//           DiffType DiffZ>
// class GeneralizedHessOp : public GeneralDiffOpInterface<Dim, T, ReturnType> {
// public:
//     typedef typename Field_t<Dim>::view_type FView_t;

//     // Define typedefs for innermost operators applied to Field<T> as they are identical on each
//     // row
//     typedef DiffOpChain<OpDim::X, Dim, T, DiffX, FView_t> colOpX_t;
//     typedef DiffOpChain<OpDim::Y, Dim, T, DiffY, FView_t> colOpY_t;
//     typedef DiffOpChain<OpDim::Z, Dim, T, DiffZ, FView_t> colOpZ_t;

//     GeneralizedHessOp(const Field_t<Dim>& field, Vector_t<Dim> hInvVector)
//         : GeneralDiffOpInterface<Dim, T, ReturnType>(field, hInvVector)
//         ,
//         // Define Operators of each element of the 3x3 Hessian
//         diff_xx(this->view_m, this->hInvVector_m)
//         , diff_xy(this->view_m, this->hInvVector_m)
//         , diff_xz(this->view_m, this->hInvVector_m)
//         , diff_yx(this->view_m, this->hInvVector_m)
//         , diff_yy(this->view_m, this->hInvVector_m)
//         , diff_yz(this->view_m, this->hInvVector_m)
//         , diff_zx(this->view_m, this->hInvVector_m)
//         , diff_zy(this->view_m, this->hInvVector_m)
//         , diff_zz(this->view_m, this->hInvVector_m) {}

//     // Compute Hessian of specific Index_t `idx`
//     inline ReturnType operator()(size_type i, size_type j, size_type k) const {
//         ReturnType hess_matrix;
//         hess_matrix[0] = {diff_xx(i, j, k), diff_xy(i, j, k), diff_xz(i, j, k)};
//         hess_matrix[1] = {diff_yx(i, j, k), diff_yy(i, j, k), diff_yz(i, j, k)};
//         hess_matrix[2] = {diff_zx(i, j, k), diff_zy(i, j, k), diff_zz(i, j, k)};

//         return hess_matrix;
//     }

// private:
//     // Row 1
//     DiffOpChain<OpDim::X, Dim, T, DiffX, colOpX_t> diff_xx;
//     DiffOpChain<OpDim::X, Dim, T, DiffX, colOpY_t> diff_xy;
//     DiffOpChain<OpDim::X, Dim, T, DiffX, colOpZ_t> diff_xz;

//     // Row 2
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpX_t> diff_yx;
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpY_t> diff_yy;
//     DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpZ_t> diff_yz;

//     // Row 3
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpX_t> diff_zx;
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpY_t> diff_zy;
//     DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpZ_t> diff_zz;
// };

#endif  // hessian_h
