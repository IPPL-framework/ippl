#ifndef KOKKOS_FIELD_EXPR_H
#define KOKKOS_FIELD_EXPR_H

#include <Kokkos_Core.hpp>

template <typename E>
class FieldExpr {

public:
    KOKKOS_INLINE_FUNCTION
    double operator()(size_t i) const {
        return static_cast<const E&>(*this)(i);
    }
};

#endif