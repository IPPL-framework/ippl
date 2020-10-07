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


    template<typename E>
    struct FieldExpression {
        auto operator[](size_t i) const {
            return static_cast<const E&>(*this)[i];
        }
    };


    /*
     * Scalar Expressions
     *
     */
    template<typename T>
    struct Scalar : public Expression<Scalar<T>, sizeof(T)>
                  , public FieldExpression<Scalar<T>>
    {
        typedef T value_t;

        KOKKOS_FUNCTION
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
}


#include "Expression/IpplOperations.h"
#include "Expression/IpplFieldOperations.h"

#endif
