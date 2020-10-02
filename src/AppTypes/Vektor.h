#ifndef IPPL_VEKTOR_H
#define IPPL_VEKTOR_H

#include "VektorExpressions.h"

#include <initializer_list>

#include <iostream>
#include <iomanip>

namespace ippl {
    template<typename T, unsigned Dim>
    class Vektor : public VektorExpr<T, Vektor<T, Dim>> {
    public:
        typedef T value_t;
        static constexpr unsigned dim = Dim;
    
        KOKKOS_FUNCTION
        Vektor() : Vektor(value_t(0)) { }

        KOKKOS_FUNCTION
        Vektor(const Vektor<T, Dim>&) = default;

        KOKKOS_FUNCTION
        Vektor(const T& val);

        KOKKOS_FUNCTION
        Vektor(const std::initializer_list<T>& list);

        KOKKOS_FUNCTION
        ~Vektor() { }
        

        // Get and Set Operations
        KOKKOS_INLINE_FUNCTION
        value_t& operator[](unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_t operator[](unsigned int i) const;

        KOKKOS_INLINE_FUNCTION
        value_t& operator()(unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_t operator()(unsigned int i) const;

        // Assignment Operators
        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator+=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator-=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator*=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator/=(const VektorExpr<T, E>& rhs);

    private:
        T data_m[Dim];
    };
}

#include "Vektor.hpp"

// //////////////////////////////////////////////////////////////////////
// //
// // Unary Operators
// //
// //////////////////////////////////////////////////////////////////////
//
// //----------------------------------------------------------------------
// // unary operator-
// template<class T, unsigned D>
// inline Vektor<T,D> operator-(const Vektor<T,D> &op)
// {
//   return TSV_MetaUnary< Vektor<T,D> , OpUnaryMinus > :: apply(op);
// }
//
// //----------------------------------------------------------------------
// // unary operator+
// template<class T, unsigned D>
// inline const Vektor<T,D> &operator+(const Vektor<T,D> &op)
// {
//   return op;
// }
//
// //////////////////////////////////////////////////////////////////////
// //
// // Binary Operators
// //
// //////////////////////////////////////////////////////////////////////
//
// //
// // Elementwise operators.
// //
//
// TSV_ELEMENTWISE_OPERATOR(Vektor,operator+,OpAdd)
// TSV_ELEMENTWISE_OPERATOR(Vektor,operator-,OpSubtract)
// TSV_ELEMENTWISE_OPERATOR(Vektor,operator*,OpMultipply)
// TSV_ELEMENTWISE_OPERATOR(Vektor,operator/,OpDivide)
// TSV_ELEMENTWISE_OPERATOR(Vektor,Min,FnMin)
// TSV_ELEMENTWISE_OPERATOR(Vektor,Max,FnMax)
//
// //----------------------------------------------------------------------
// // dot product
// //----------------------------------------------------------------------
//
// template < class T1, class T2, unsigned D >
// inline typename PETEBinaryReturn<T1,T2,OpMultipply>::type
// dot(const Vektor<T1,D> &lhs, const Vektor<T2,D> &rhs)
// {
//   return TSV_MetaDot< Vektor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
// }
//
// //----------------------------------------------------------------------
// // cross product
// //----------------------------------------------------------------------
//
// template < class T1, class T2, unsigned D >
// inline Vektor<typename PETEBinaryReturn<T1,T2,OpMultipply>::type,D>
// cross(const Vektor<T1,D> &lhs, const Vektor<T2,D> &rhs)
// {
//   return TSV_MetaCross< Vektor<T1,D> , Vektor<T2,D> > :: apply(lhs,rhs);
// }
//
// //----------------------------------------------------------------------
// // I/O
// template<class T, unsigned D>
// inline std::ostream& operator<<(std::ostream& out, const Vektor<T,D>& rhs)
// {
//   std::streamsize sw = out.width();
//   out << std::setw(1);
//   if (D >= 1) {
//     out << "( ";
//     for (unsigned int i=0; i<D - 1; i++)
//       out << std::setw(sw) << rhs[i] << " , ";
//     out << std::setw(sw) << rhs[D - 1] << " )";
//   } else {
//     out << "( " << std::setw(sw) << rhs[0] << " )";
//   }
//
//   return out;
// }

#endif // IPPL_VEKTOR_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
