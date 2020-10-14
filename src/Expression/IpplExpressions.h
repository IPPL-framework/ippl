//
// File IpplExpressions.h
//   Expression Templates classes.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_EXPRESSIONS_H
#define IPPL_EXPRESSIONS_H

namespace ippl {
    /*!
     * @file IpplExpressions.h
     *
     * Expression class which should be Kokkos-aware
     * need to inherit from Expression.
     */

    /*!
     * Basic expression class for LField, Vector and Scalar.
     * Expression classes need to inherit from this with the
     * CRTP (curiously recursive template pattern) design
     * pattern.
     */
    template<typename E, size_t N = sizeof(E)>
    struct Expression {
        /*!
         * Access single element of the expression
         */
        KOKKOS_INLINE_FUNCTION
        auto operator[](size_t i) const {
            return static_cast<const E&>(*this)[i];
        }
    };


    /*!
     * Basic field expression class for Field and BareField.
     * FieldExpression classes need to inherit from this with the
     * CRTP (curiously recursive template pattern) design
     * pattern.
     */
    template<typename E>
    struct FieldExpression {
        /*!
         * Access single element of the field expression
         */
        auto operator[](size_t i) const {
            return static_cast<const E&>(*this)[i];
        }
    };

    namespace detail {
        /*!
         * This expression is only used to allocate
         * enough memory for the kernel on the device.
         * It is instantiated in the assignment operator
         * of the LField class.
         */
        template <typename E, size_t N = sizeof(E)>
        struct CapturedExpression {
            template<typename ...Args>
            KOKKOS_INLINE_FUNCTION
            auto operator()(Args... args) const {
                return reinterpret_cast<const E&>(*this)(args...);
            }

            char buffer[N];
        };


        /*!
         * Expression for intrinsic data types. They are both regular expressions
         * and field expressions.
         */
        template<typename T>
        struct Scalar : public Expression<Scalar<T>, sizeof(T)>
                      , public FieldExpression<Scalar<T>>
        {
            typedef T value_type;


            KOKKOS_FUNCTION
            Scalar(value_type val) : val_m(val) { }

            /*!
             * Access the scalar value with single index.
             * This is used for binary operations between
             * Scalar and Vector.
             */
            KOKKOS_INLINE_FUNCTION
            value_type operator[](size_t /*i*/) const {
                return val_m;
            }

            /*!
             * Access the scalar value with multiple indices.
             * This is used for binary operations between
             * Scalar and LField, Scalar and BareField,
             * and Scalar and Field.
             */
            template<typename ...Args>
            KOKKOS_INLINE_FUNCTION
            auto operator()(Args... /*args*/) const {
                return val_m;
            }

        private:
            value_type val_m;
        };
    }
}


#include "Expression/IpplOperations.h"
#include "Expression/IpplFieldOperations.h"

#endif
