#include <Kokkos_Core.hpp>

#include <chrono>
#include <initializer_list>
#include <iostream>
#include <typeinfo>

template <typename E>
class Expression {
public:
    KOKKOS_INLINE_FUNCTION double operator[](size_t i) const {
        return static_cast<E const&>(*this)[i];
    }
};

template <typename T, unsigned D>
class Vector : public Expression<Vector<T, D>> {
public:
    KOKKOS_FUNCTION
    Vector() {}

    KOKKOS_FUNCTION
    Vector(T val) {
        for (unsigned i = 0; i < D; ++i) {
            data_m[i] = val;
        }
    }

    KOKKOS_FUNCTION
    Vector(const std::initializer_list<T>& l) {
        int i = 0;
        for (auto a : l) {
            data_m[i] = a;
            ++i;
        }
    }

    KOKKOS_FUNCTION
    T operator[](const int i) const { return data_m[i]; }

    KOKKOS_FUNCTION
    T& operator[](const int i) { return data_m[i]; }

    KOKKOS_FUNCTION
    ~Vector() {}

    template <typename E>
    KOKKOS_FUNCTION Vector& operator=(Expression<E> const& expr) {
        for (unsigned i = 0; i < D; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }

private:
    T data_m[D];
};

template <typename E1, typename E2>
class VecSum : public Expression<VecSum<E1, E2>> {
    E1 const _u;
    E2 const _v;

public:
    KOKKOS_FUNCTION
    VecSum(E1 const& u, E2 const& v)
        : _u(u)
        , _v(v) {}

    KOKKOS_INLINE_FUNCTION double operator[](size_t i) const { return _u[i] + _v[i]; }
};

template <typename E1, typename E2>
KOKKOS_FUNCTION VecSum<E1, E2> operator+(E1 const& u, E2 const& v) {
    return VecSum<E1, E2>(u, v);
}

int main() {
    constexpr int dim = 50000;

    typedef Vector<double, dim> vector_type;

    vector_type x(1.0), y(2.0), z(0.0);

    auto start = std::chrono::high_resolution_clock::now();

    z = x + y + x + x;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time.count() << std::endl;

    return 0;
}
