#ifndef hessian_h
#define hessian_h

#include "Ippl.h"

namespace hessOp {

    constexpr unsigned int dim = 3;

    enum DiffType {Centered, Forward, Backward, CenteredDeriv2};
    enum Dim {X, Y, Z};

    // type definitions
    typedef ippl::detail::size_type size_type;

    ///////////////////
    // Index helpers //
    ///////////////////

    template<class Callable, typename... IdxArgs,
                std::enable_if_t<std::conjunction_v<std::is_same<size_type, IdxArgs>...>, bool> =true>
    inline typename Callable::value_type idxApply(const Callable &F, IdxArgs... idxargs){
        return F(idxargs...);
    }

    template<Dim D, class Callable, typename... IdxArgs>
    inline typename Callable::value_type shiftedIdxApply(const Callable &F, size_type shift, IdxArgs... idxargs){
        auto shiftedIdx = std::make_tuple(idxargs...);
        std::get<D>(shiftedIdx) += shift;
        return std::apply([&](auto&&... lambdaArgs){return idxApply(F, std::forward<decltype(lambdaArgs)>(lambdaArgs)...);}, shiftedIdx);
    }

    //////////////////////////////////////////////////////////////
    // Stencil definitions along a template specified dimension //
    //////////////////////////////////////////////////////////////

    // More stencils can be found at:
    // `https://en.wikipedia.org/wiki/Finite_difference_coefficient`

    template<Dim D, typename T, class Callable>
    inline T centered_stencil(const T &hInv, const Callable &F, size_type i, size_type j, size_type k){
        return 0.5 * hInv * (- shiftedIdxApply<D>(F, -1, i,j,k) + shiftedIdxApply<D>(F, 1, i,j,k));
    }

    // Compact version of the `centered_stencil` for the 2nd derivative along the same dimension
    template<Dim D, typename T, class Callable>
    inline T centered_stencil_deriv2(const T &hInv, const Callable &F, size_type i, size_type j, size_type k){
        return hInv * hInv * (shiftedIdxApply<D>(F, -1, i,j,k) - 2.0*idxApply(F, i,j,k) + shiftedIdxApply<D>(F, 1, i,j,k));
    }

    template<Dim D, typename T, class Callable>
    inline T forward_stencil(const T &hInv, const Callable &F, size_type i, size_type j, size_type k){
        return 0.5 * hInv * (-3.0*idxApply(F, i,j,k) + 4.0*shiftedIdxApply<D>(F, 1, i,j,k) - shiftedIdxApply<D>(F, 2, i,j,k));
    }

    template<Dim D, typename T, class Callable>
    inline T backward_stencil(const T &hInv, const Callable &F, size_type i, size_type j, size_type k){
        return 0.5 * hInv * (3.0*idxApply(F, i,j,k) - 4.0*shiftedIdxApply<D>(F, -1, i,j,k) + shiftedIdxApply<D>(F, -2, i,j,k));
    }


    ///////////////////////////////////////////////
    // Specialization to chain stencil operators //
    ///////////////////////////////////////////////

    template<Dim D, typename T, DiffType Diff, class C>
    class BaseDiffOp {
        public:
            typedef ippl::Vector<T,dim> Vector_t;
            typedef ippl::Field<T,dim> Field_t;
            typedef typename Field_t::view_type FView_t;
            typedef T value_type;

            BaseDiffOp(const FView_t &view, Vector_t hInvVector) : view_m(view), hInvVector_m(hInvVector) {};
        
            // Applies templated stencil type on specific callable `F`
            inline T stencilOp(const C &F, size_type i, size_type j, size_type k) const {
                if constexpr      (Diff == DiffType::Centered) { return centered_stencil<D,T,C>(hInvVector_m[D], F, i,j,k); }
                else if constexpr (Diff == DiffType::Forward)  { return forward_stencil<D,T,C>(hInvVector_m[D], F, i,j,k); }
                else if constexpr (Diff == DiffType::Backward) { return backward_stencil<D,T,C>(hInvVector_m[D], F, i,j,k); }
                else if constexpr (Diff == DiffType::CenteredDeriv2) { return centered_stencil_deriv2<D,T,C>(hInvVector_m[D], F, i,j,k); }
            }

        protected:
            const FView_t &view_m;
            Vector_t hInvVector_m;
    };

    template<Dim D, typename T, DiffType Diff, class C>
    class DiffOpChain : public BaseDiffOp<D,T,Diff,C> {
        public: 
            typedef ippl::Vector<T,dim> Vector_t;
            typedef ippl::Field<T,dim> Field_t;
            typedef typename Field_t::view_type FView_t;

            DiffOpChain(const FView_t &view, Vector_t hInvVector) : BaseDiffOp<D,T,Diff,C>(view, hInvVector), leftOp_m(view, this->hInvVector_m) {}
            
            // Specialization to call the stencil operator on the left operator
            inline T operator()(size_type i, size_type j, size_type k) const {
                return this->template stencilOp(leftOp_m, i,j,k);
            }

        private:
            // Need additional callable which might contain other operators
            const C leftOp_m;
    };

    // Innermost operator acting on the field (template specialization)
    template<Dim D, typename T, DiffType Diff>
    class DiffOpChain<D,T,Diff,typename ippl::Field<T,dim>::view_type> : public BaseDiffOp<D,T,Diff,typename ippl::Field<T,dim>::view_type> {
        public: 
            typedef ippl::Vector<T,dim> Vector_t;
            typedef ippl::Field<T,dim> Field_t;
            typedef typename Field_t::view_type FView_t;

            DiffOpChain(const FView_t &view, Vector_t hInvVector) : BaseDiffOp<D,T,Diff,FView_t>(view, hInvVector) {}
            
            // Specialization to call the stencil operator on the field
            inline T operator()(size_type i, size_type j, size_type k) const {
                return this->template stencilOp(this->view_m, i,j,k);
            }
    };

    template<typename T, DiffType DiffX, DiffType DiffY, DiffType DiffZ>
    class GeneralizedHessOp {
        public:
            typedef ippl::Vector<T,dim> Vector_t;
            typedef ippl::Field<T,dim> Field_t;
            typedef ippl::Vector<Vector_t,dim> Matrix_t;
            typedef typename Field_t::view_type FView_t;
            
            // Define typedefs for innermost operators applied to Field<T> as they are identical on each row
            typedef DiffOpChain<Dim::X,T,DiffX,FView_t> colOpX_t;
            typedef DiffOpChain<Dim::Y,T,DiffY,FView_t> colOpY_t;
            typedef DiffOpChain<Dim::Z,T,DiffZ,FView_t> colOpZ_t;

            GeneralizedHessOp(const Field_t &field, Vector_t hInvVector) : view(field.getView()),
                                // Define Operators of each element of the 3x3 Hessian
                                diff_xx(view, hInvVector), diff_xy(view, hInvVector), diff_xz(view, hInvVector),
                                diff_yx(view, hInvVector), diff_yy(view, hInvVector), diff_yz(view, hInvVector),
                                diff_zx(view, hInvVector), diff_zy(view, hInvVector), diff_zz(view, hInvVector) {}
            
            // Compute Hessian of specific Index_t `idx`
            inline Matrix_t operator()(size_type i, size_type j, size_type k) const {
                Matrix_t hess_matrix;
                hess_matrix[0] = {diff_xx(i,j,k), diff_xy(i,j,k), diff_xz(i,j,k)};
                hess_matrix[1] = {diff_yx(i,j,k), diff_yy(i,j,k), diff_yz(i,j,k)};
                hess_matrix[2] = {diff_zx(i,j,k), diff_zy(i,j,k), diff_zz(i,j,k)};
                return hess_matrix;
            }

        private:
            const FView_t &view;

            // Row 1
            DiffOpChain<Dim::X,T,DiffX,colOpX_t> diff_xx;
            DiffOpChain<Dim::X,T,DiffX,colOpY_t> diff_xy;
            DiffOpChain<Dim::X,T,DiffX,colOpZ_t> diff_xz;

            // Row 2
            DiffOpChain<Dim::Y,T,DiffY,colOpX_t> diff_yx;
            DiffOpChain<Dim::Y,T,DiffY,colOpY_t> diff_yy;
            DiffOpChain<Dim::Y,T,DiffY,colOpZ_t> diff_yz;

            // Row 3
            DiffOpChain<Dim::Z,T,DiffZ,colOpX_t> diff_zx;
            DiffOpChain<Dim::Z,T,DiffZ,colOpY_t> diff_zy;
            DiffOpChain<Dim::Z,T,DiffZ,colOpZ_t> diff_zz;
    };

}  // namespace hess_test

#endif // hessian_h
