#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>

#define third

template <typename E>
class VectorExpr {
public:
    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const {
        return static_cast<E const&>(*this)[i];
    }
};

template <typename T, unsigned D>
class Vector : public VectorExpr<Vector<T, D>> {
public:
    KOKKOS_FUNCTION
    Vector() {
    }

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
    T operator[](const int i) const {
        return data_m[i];
    }

    KOKKOS_FUNCTION
    T& operator[](const int i) {
        return data_m[i];
    }

    KOKKOS_FUNCTION
    ~Vector() {
    }

    template <typename E>
    KOKKOS_FUNCTION Vector& operator=(VectorExpr<E> const& expr) {
        for (unsigned i = 0; i < D; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }

private:
    T data_m[D];
};

template <typename E1, typename E2>
class VecSum : public VectorExpr<VecSum<E1, E2>> {
    E1 const _u;
    E2 const _v;

public:
    KOKKOS_FUNCTION
    VecSum(E1 const& u, E2 const& v) : _u(u), _v(v) {
    }

    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const {
        return _u[i] + _v[i];
    }
};

template <typename E1, typename E2>
KOKKOS_FUNCTION VecSum<E1, E2> operator+(VectorExpr<E1> const& u, VectorExpr<E2> const& v) {
    return VecSum<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
}

/*
 * Cross product: First implementation
 *
 */
#ifdef first
template <class T1, class T2>
struct meta_cross {};

template <class T1, class T2>
struct meta_cross<Vector<T1, 3>, Vector<T2, 3>> {
    KOKKOS_INLINE_FUNCTION
    static Vector<double, 3> apply(const Vector<T1, 3>& a, const Vector<T2, 3>& b) {
        Vector<double, 3> c;
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
        return c;
    }
};

template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION Vector<T1, D> cross(const Vector<T1, D>& lhs, const Vector<T2, D>& rhs) {
    return meta_cross<Vector<T1, D>, Vector<T2, D>>::apply(lhs, rhs);
}
#endif

#ifdef second
/*
 * Cross product: Second implementation
 *
 */
template <typename E1, typename E2>
struct meta_cross : public VectorExpr<meta_cross<E1, E2>> {
    KOKKOS_FUNCTION
    meta_cross(E1 const& a, E2 const& b) {
        _result[0] = a[1] * b[2] - a[2] * b[1];
        _result[1] = a[2] * b[0] - a[0] * b[2];
        _result[2] = a[0] * b[1] - a[1] * b[0];
    }

    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const {
        return _result[i];
    }

private:
    E1 _result;
};

template <typename E1, typename E2>
KOKKOS_INLINE_FUNCTION meta_cross<E1, E2> cross(
    const VectorExpr<E1>& lhs, const VectorExpr<E2>& rhs) {
    return meta_cross<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
}
#endif

#ifdef third
/*
 * Cross product: third implementation
 *
 */
template <typename E1, typename E2>
struct meta_cross : public VectorExpr<meta_cross<E1, E2>> {
    KOKKOS_FUNCTION
    meta_cross(E1 const& u, E2 const& v) : _u(u), _v(v) {
    }

    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const {
        size_t j = (i + 1) % 3;
        size_t k = (i + 2) % 3;
        return _u[j] * _v[k] - _u[k] * _v[j];
    }

private:
    E1 const _u;
    E2 const _v;
};

template <typename E1, typename E2>
KOKKOS_INLINE_FUNCTION meta_cross<E1, E2> cross(
    const VectorExpr<E1>& lhs, const VectorExpr<E2>& rhs) {
    return meta_cross<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
}
#endif

/*
 * Dot product
 *
 */
template <typename E1, typename E2>
struct meta_dot : public VectorExpr<meta_dot<E1, E2>> {
    KOKKOS_FUNCTION
    meta_dot(E1 const& u, E2 const& v) : _u(u), _v(v) {
    }

    KOKKOS_INLINE_FUNCTION
    double operator()() const {
        return _u[0] * _v[0] + _u[1] * _v[1] + _u[2] * _v[2];
    }

private:
    E1 const& _u;
    E2 const& _v;
};

template <typename E1, typename E2>
KOKKOS_INLINE_FUNCTION double  // meta_dot<E1, E2>
dot(const VectorExpr<E1>& lhs, const VectorExpr<E2>& rhs) {
    return meta_dot<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs))();
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int length = 10;

        typedef Vector<double, 3> vector_type;

        typedef Kokkos::View<vector_type*> vector_field_type;

        vector_field_type vfield("vfield", length);

        Kokkos::parallel_for(
            "assign", length, KOKKOS_LAMBDA(const int i) {
                vfield(i) = {1.0, 2.0, 3.0};
            });

        vector_field_type wfield("wfield", length);

        Kokkos::parallel_for(
            "assign", length, KOKKOS_LAMBDA(const int i) {
                wfield(i) = {4.0, -5.0, 6.0};
            });

        vector_field_type vvfield("vvfield", length);
        Kokkos::parallel_for(
            "assign", length,
            KOKKOS_LAMBDA(const int i) { vvfield(i) = cross(vfield(i), wfield(i)) + wfield(i); });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vvfield);
        Kokkos::deep_copy(host_view, vvfield);

        for (int i = 0; i < length; ++i) {
            std::cout << host_view(i)[0] << " " << host_view(i)[1] << " " << host_view(i)[2]
                      << std::endl;
        }

        typedef Kokkos::View<double*> scalar_field_type;
        scalar_field_type sfield("sfield", length);

        Kokkos::parallel_for(
            "assign", length,
            KOKKOS_LAMBDA(const int i) { sfield(i) = dot(wfield(i), wfield(i)); });

        Kokkos::fence();

        scalar_field_type::HostMirror host_sview = Kokkos::create_mirror_view(sfield);
        Kokkos::deep_copy(host_sview, sfield);

        for (int i = 0; i < length; ++i) {
            std::cout << host_sview(i) << std::endl;
        }
    }
    Kokkos::finalize();

    return 0;
}
