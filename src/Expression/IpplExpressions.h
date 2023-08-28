//
// File IpplExpressions.h
//   Expression Templates classes.
//
#ifndef IPPL_EXPRESSIONS_H
#define IPPL_EXPRESSIONS_H

#include <type_traits>

namespace ippl {
    namespace detail {
        /*!
         * @file IpplExpressions.h
         *
         * Expression class which should be Kokkos-aware
         * need to inherit from Expression.
         */

        /*!
         * Basic expression class for BareField, Vector and Scalar.
         * Expression classes need to inherit from this with the
         * CRTP (curiously recursive template pattern) design
         * pattern.
         */
        template <typename E, size_t N = sizeof(E)>
        struct Expression {
            constexpr static unsigned dim = E::dim;

            /*!
             * Access single element of the expression
             */
            KOKKOS_INLINE_FUNCTION auto operator[](size_t i) const {
                return static_cast<const E&>(*this)[i];
            }
        };

        /*!
         * This expression is only used to allocate
         * enough memory for the kernel on the device.
         * It is instantiated in the assignment operator
         * of the BareField class.
         */
        template <typename E, size_t N = sizeof(E)>
        struct CapturedExpression {
            constexpr static unsigned dim = E::dim;

            template <typename... Args>
            KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {
                static_assert(sizeof...(Args) == dim || dim == 0);
                return reinterpret_cast<const E&>(*this)(args...);
            }

            char buffer[N];
        };

        /*!
         * Expression for intrinsic data types. They are both regular expressions
         * and field expressions.
         */
        template <typename T>
        struct Scalar : public Expression<Scalar<T>, sizeof(T)> {
            typedef T value_type;
            constexpr static unsigned dim = 0;

            KOKKOS_FUNCTION
            Scalar(value_type val)
                : val_m(val) {}

            /*!
             * Access the scalar value with single index.
             * This is used for binary operations between
             * Scalar and Vector.
             */
            KOKKOS_INLINE_FUNCTION value_type operator[](size_t /*i*/) const { return val_m; }

            /*!
             * Access the scalar value with multiple indices.
             * This is used for binary operations between
             * Scalar and BareField, Scalar and BareField,
             * and Scalar and Field.
             */
            template <typename... Args>
            KOKKOS_INLINE_FUNCTION auto operator()(Args... /*args*/) const {
                return val_m;
            }

        private:
            value_type val_m;
        };

        template <typename T>
        struct isExpression : std::false_type {};

        template <typename T>
        struct isExpression<Scalar<T>> : std::true_type {};

    }  // namespace detail
}  // namespace ippl

#include "Expression/IpplOperations.h"

#endif
