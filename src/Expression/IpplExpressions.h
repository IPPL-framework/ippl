#ifndef IPPL_EXPRESSIONS_H
#define IPPL_EXPRESSIONS_H

// #include "Expression/IpplExpressionTraits.h"

#include <functional>

namespace ippl {
    template<typename E, size_t N = sizeof(E)>
    struct Expression {
        KOKKOS_INLINE_FUNCTION
        auto operator[](size_t i) const {
            return static_cast<const E&>(*this)[i];
        }
    };


    template <typename E, size_t N = sizeof(E)>
    struct CapturedExpression {
        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        auto operator()(Args... args) const {
            return reinterpret_cast<const E&>(*this)(args...);
        }

        char buffer[N];
    };


    template<typename T>
    struct ExprType {
        typedef T value_type;
    };


    /*
     * Scalar Expressions
     *
     */
    template<typename T>
    struct Scalar : public Expression<Scalar<T>, sizeof(T)> {
        typedef T value_t;

        Scalar(value_t val) : val_m(val) { }

        KOKKOS_INLINE_FUNCTION
        value_t operator[](size_t /*i*/) const {
            return val_m;
        }

        template<typename ...Args>                                          \
        KOKKOS_INLINE_FUNCTION
        auto operator()(Args... /*args*/) const {
            return val_m;
        }

    private:
        value_t val_m;
    };



    #define DefineScalarType(type)          \
    template<>                              \
    struct ExprType<type> {                 \
        typedef Scalar<type> value_type;    \
    };


    DefineScalarType(double)
    DefineScalarType(float)
    DefineScalarType(short)
    DefineScalarType(int)
}


#include "Expression/IpplOperations.h"

#endif
