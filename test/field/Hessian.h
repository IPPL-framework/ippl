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

//////////////////////////////////////////////////////////////
// Stencil definitions along a template specified dimension //
//////////////////////////////////////////////////////////////

// More stencils can be found at:
// `https://en.wikipedia.org/wiki/Finite_difference_coefficient`

template <OpDim D, typename T, class Callable>
inline T centered_stencil(const T& hInv, const Callable& F, size_type i, size_type j, size_type k) {
    return 0.5 * hInv * (-shiftedIdxApply<D>(F, -1, i, j, k) + shiftedIdxApply<D>(F, 1, i, j, k));
}

// Compact version of the `centered_stencil` for the 2nd derivative along the same dimension
template <OpDim D, typename T, class Callable>
inline T centered_stencil_deriv2(const T& hInv, const Callable& F, size_type i, size_type j,
                                 size_type k) {
    return hInv * hInv
           * (shiftedIdxApply<D>(F, -1, i, j, k) - 2.0 * idxApply(F, i, j, k)
              + shiftedIdxApply<D>(F, 1, i, j, k));
}

template <OpDim D, typename T, class Callable>
inline T forward_stencil(const T& hInv, const Callable& F, size_type i, size_type j, size_type k) {
    return 0.5 * hInv
           * (-3.0 * idxApply(F, i, j, k) + 4.0 * shiftedIdxApply<D>(F, 1, i, j, k)
              - shiftedIdxApply<D>(F, 2, i, j, k));
}

template <OpDim D, typename T, class Callable>
inline T backward_stencil(const T& hInv, const Callable& F, size_type i, size_type j, size_type k) {
    return 0.5 * hInv
           * (3.0 * idxApply(F, i, j, k) - 4.0 * shiftedIdxApply<D>(F, -1, i, j, k)
              + shiftedIdxApply<D>(F, -2, i, j, k));
}

///////////////////////////////////////////////
// Specialization to chain stencil operators //
///////////////////////////////////////////////

template <OpDim D, unsigned Dim, typename T, DiffType Diff, class Callable>
class BaseDiffOp {
public:
    typedef typename Field_t<Dim>::view_type FView_t;

    BaseDiffOp(const FView_t& view, Vector_t<Dim> hInvVector)
        : view_m(view)
        , hInvVector_m(hInvVector){};

    // Applies templated stencil type on specific callable `F`
    inline T stencilOp(const Callable& F, size_type i, size_type j, size_type k) const {
        if constexpr (Diff == DiffType::Centered) {
            return centered_stencil<D, T, Callable>(hInvVector_m[D], F, i, j, k);
        } else if constexpr (Diff == DiffType::Forward) {
            return forward_stencil<D, T, Callable>(hInvVector_m[D], F, i, j, k);
        } else if constexpr (Diff == DiffType::Backward) {
            return backward_stencil<D, T, Callable>(hInvVector_m[D], F, i, j, k);
        } else if constexpr (Diff == DiffType::CenteredDeriv2) {
            return centered_stencil_deriv2<D, T, Callable>(hInvVector_m[D], F, i, j, k);
        }
    }

    inline T operator()(size_type i, size_type j, size_type k) const;

protected:
    const FView_t& view_m;
    Vector_t<Dim> hInvVector_m;
};

template <OpDim D, unsigned Dim, typename T, DiffType Diff, class Callable>
class DiffOpChain : public BaseDiffOp<D, Dim, T, Diff, Callable> {
public:
    typedef T value_type;
    typedef typename Field_t<Dim>::view_type FView_t;

    DiffOpChain(const FView_t& view, Vector_t<Dim> hInvVector)
        : BaseDiffOp<D, Dim, T, Diff, Callable>(view, hInvVector)
        , leftOp_m(view, this->hInvVector_m) {}

    // Specialization to call the stencil operator on the left operator
    inline T operator()(size_type i, size_type j, size_type k) const {
        return this->template stencilOp(leftOp_m, i, j, k);
    }

private:
    // Need additional callable which might contain other operators
    const Callable leftOp_m;
};

// Innermost operator acting on the field (template specialization)
template <OpDim D, unsigned Dim, typename T, DiffType Diff>
class DiffOpChain<D, Dim, T, Diff, typename Field_t<Dim>::view_type>
    : public BaseDiffOp<D, Dim, T, Diff, typename Field_t<Dim>::view_type> {
public:
    typedef T value_type;
    typedef typename Field_t<Dim>::view_type FView_t;

    DiffOpChain(const FView_t& view, Vector_t<Dim> hInvVector)
        : BaseDiffOp<D, Dim, T, Diff, FView_t>(view, hInvVector) {}

    // Specialization to call the stencil operator on the field
    inline T operator()(size_type i, size_type j, size_type k) const {
        return this->template stencilOp(this->view_m, i, j, k);
    }
};

template <unsigned Dim, typename T, class ReturnType>
class GeneralDiffOpInterface {
public:
    typedef typename Field_t<Dim>::view_type FView_t;
    // GeneralDiffOpInterface(GeneralDiffOpInterface<T,ReturnType>&& source) :
    // view_m(std::move(source.view_m)), hInvVector_m(std::move(source.hInvVector_m)) {};

    GeneralDiffOpInterface(const Field_t<Dim>& field, Vector_t<Dim> hInvVector)
        : view_m(field.getView())
        , hInvVector_m(hInvVector) {}

    virtual inline ReturnType operator()(size_type i, size_type j, size_type k) const = 0;

protected:
    const FView_t& view_m;
    const Vector_t<Dim> hInvVector_m;
};

template <unsigned Dim, typename T, class ReturnType, DiffType DiffX, DiffType DiffY,
          DiffType DiffZ>
class GeneralizedHessOp : public GeneralDiffOpInterface<Dim, T, ReturnType> {
public:
    typedef typename Field_t<Dim>::view_type FView_t;

    // Define typedefs for innermost operators applied to Field<T> as they are identical on each
    // row
    typedef DiffOpChain<OpDim::X, Dim, T, DiffX, FView_t> colOpX_t;
    typedef DiffOpChain<OpDim::Y, Dim, T, DiffY, FView_t> colOpY_t;
    typedef DiffOpChain<OpDim::Z, Dim, T, DiffZ, FView_t> colOpZ_t;
    // template<Dim D, DiffType Diff>
    // using CompactDiffOp<D,T,Diff,Fview_t> = diagOp_t;

    // GeneralizedHessOp(GeneralizedHessOp<T,ReturnType,DiffX,DiffY,DiffZ>&& source) = default;

    GeneralizedHessOp(const Field_t<Dim>& field, Vector_t<Dim> hInvVector)
        : GeneralDiffOpInterface<Dim, T, ReturnType>(field, hInvVector)
        ,
        // Define Operators of each element of the 3x3 Hessian
        diff_xx(this->view_m, this->hInvVector_m)
        , diff_xy(this->view_m, this->hInvVector_m)
        , diff_xz(this->view_m, this->hInvVector_m)
        , diff_yx(this->view_m, this->hInvVector_m)
        , diff_yy(this->view_m, this->hInvVector_m)
        , diff_yz(this->view_m, this->hInvVector_m)
        , diff_zx(this->view_m, this->hInvVector_m)
        , diff_zy(this->view_m, this->hInvVector_m)
        , diff_zz(this->view_m, this->hInvVector_m) {}

    // Compute Hessian of specific Index_t `idx`
    inline ReturnType operator()(size_type i, size_type j, size_type k) const {
        ReturnType hess_matrix;
        hess_matrix[0] = {diff_xx(i, j, k), diff_xy(i, j, k), diff_xz(i, j, k)};
        hess_matrix[1] = {diff_yx(i, j, k), diff_yy(i, j, k), diff_yz(i, j, k)};
        hess_matrix[2] = {diff_zx(i, j, k), diff_zy(i, j, k), diff_zz(i, j, k)};

        return hess_matrix;
    }

private:
    // const FView_t &view;

    // Row 1
    DiffOpChain<OpDim::X, Dim, T, DiffX, colOpX_t> diff_xx;
    DiffOpChain<OpDim::X, Dim, T, DiffX, colOpY_t> diff_xy;
    DiffOpChain<OpDim::X, Dim, T, DiffX, colOpZ_t> diff_xz;

    // Row 2
    DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpX_t> diff_yx;
    DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpY_t> diff_yy;
    DiffOpChain<OpDim::Y, Dim, T, DiffY, colOpZ_t> diff_yz;

    // Row 3
    DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpX_t> diff_zx;
    DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpY_t> diff_zy;
    DiffOpChain<OpDim::Z, Dim, T, DiffZ, colOpZ_t> diff_zz;
};

#endif  // hessian_h
