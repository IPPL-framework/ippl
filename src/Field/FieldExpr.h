#ifndef KOKKOS_FIELD_EXPR_H
#define KOKKOS_FIELD_EXPR_H

#include <Kokkos_Core.hpp>

template <typename T, typename E, size_t N = sizeof(E)>
class FieldExpr {

public:
    KOKKOS_INLINE_FUNCTION
    T operator()(size_t i) const {
        return static_cast<const E&>(*this)(i);
    }
};



template <class T, typename E, size_t N = sizeof(E)>
class LFieldCaptureExpr {

public:
    template<typename ...Args>
    KOKKOS_INLINE_FUNCTION
    T operator()(Args... args) const {
      return reinterpret_cast<const E&>(*this)(args...);
    }
    char buffer[N];
};

#endif