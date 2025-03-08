#ifndef MATRIX_H
#define MATRIX_H
#include <Kokkos_Core.hpp>

#include "Vector.h"
namespace ippl {
    /**
     * @brief Column major
     * 
     * @tparam T Scalar type
     * @tparam m Number of rows
     * @tparam n Number of columns
     */
    template <typename T, int m, int n>
    struct matrix {
        
        ippl::Vector<ippl::Vector<T, m>, n> data;
        constexpr static bool squareMatrix = (m == n && m > 0);

        constexpr KOKKOS_INLINE_FUNCTION matrix(T diag)
            requires(m == n)
        {
            for (unsigned i = 0; i < n; i++) {
                for (unsigned j = 0; j < n; j++) {
                    data[i][j] = diag * T(i == j);
                }
            }
        }
        constexpr matrix() = default;
        KOKKOS_INLINE_FUNCTION constexpr static matrix zero() {
            matrix<T, m, n> ret;
            for (unsigned i = 0; i < n; i++) {
                for (unsigned j = 0; j < n; j++) {
                    ret.data[i][j] = 0;
                }
            }
            return ret;
        };
        KOKKOS_INLINE_FUNCTION void setZero() {
            for (unsigned i = 0; i < n; i++) {
                for (unsigned j = 0; j < n; j++) {
                    data[i][j] = 0;
                }
            }
        }

        KOKKOS_INLINE_FUNCTION T operator()(int i, int j) const noexcept { return data[j][i]; }
        KOKKOS_INLINE_FUNCTION T& operator()(int i, int j) noexcept { return data[j][i]; }
        template <typename O>
        KOKKOS_INLINE_FUNCTION matrix<O, m, n> cast() const noexcept {
            matrix<O, m, n> ret;
            for (unsigned i = 0; i < n; i++) {
                ret.data[i] = data[i].template cast<O>();
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator+(const matrix<T, m, n>& other) const {
            matrix<T, m, n> result;
            for (int i = 0; i < n; ++i) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        // Implement matrix subtraction
        KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator-(const matrix<T, m, n>& other) const {
            matrix<T, m, n> result;
            for (int i = 0; i < n; ++i) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }
        KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator*(const T& factor) const {
            matrix<T, m, n> result;
            for (int i = 0; i < n; ++i) {
                result.data[i] = data[i] * factor;
            }
            return result;
        }
        KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator/(const T& divisor) const {
            matrix<T, m, n> result;
            for (int i = 0; i < n; ++i) {
                result.data[i] = data[i] / divisor;
            }
            return result;
        }

        // Implement matrix-vector multiplication
        template <unsigned int other_m>
        KOKKOS_INLINE_FUNCTION ippl::Vector<T, m> operator*(
            const ippl::Vector<T, other_m>& vec) const {
            static_assert((int)other_m == n);
            ippl::Vector<T, m> result(0);
            for (int i = 0; i < n; ++i) {
                for(int j = 0;j < m;i++){
                    //This could be a vector operation, but you never know with ippl::Vector
                    result[j] += vec[i] * data[i][j];
                }
            }
            return result;
        }
        template <int otherm, int othern>
            requires(n == otherm)
        KOKKOS_INLINE_FUNCTION matrix<T, m, othern> operator*(
            const matrix<T, otherm, othern>& otherMat) const noexcept {
            matrix<T, m, othern> ret(0);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < othern; j++) {
                    for (int k = 0; k < n; k++) {
                        ret(i, j) += (*this)(i, k) * otherMat(k, j);
                    }
                }
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION void addCol(int i, int j, T alpha = 1.0) {
            data[j] += data[i] * alpha;
        }
        KOKKOS_INLINE_FUNCTION matrix<T, m, n> inverse() const noexcept
            requires(squareMatrix)
        {
            constexpr int N = m;

            matrix<T, m, n> ret(1.0);
            matrix<T, m, n> dis(*this);

            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    T alpha = -dis(i, j) / dis(i, i);
                    dis.addCol(i, j, alpha);
                    dis(i, j) = 0;
                    ret.addCol(i, j, alpha);
                }
            }
            for (int i = N - 1; i >= 0; i--) {
                for (int j = i - 1; j >= 0; j--) {
                    T alpha = -dis(i, j) / dis(i, i);
                    dis.addCol(i, j, alpha);
                    dis(i, j) = 0;
                    ret.addCol(i, j, alpha);
                }
            }
            for (int i = 0; i < N; i++) {
                T d     = dis(i, i);
                T oneod = T(1) / d;
                dis.data[i] *= oneod;
                ret.data[i] *= oneod;
            }

            return ret;
        }

        template <typename stream_t>
        friend stream_t& operator<<(stream_t& str, const matrix<T, m, n>& mat) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    str << mat.data[j][i] << " ";
                }
                str << "\n";
            }
            return str;
        }
    };
    /**
     * @brief Computes the outer product l r^T of two vectors
     * 
     * @tparam T Scalar type
     * @tparam N Output rows
     * @tparam M Output cols
     * @param l left operand
     * @param r right operand 
     * @return KOKKOS_INLINE_FUNCTION the matrix l r^T 
     */
    template <typename T, unsigned N, unsigned M>
    KOKKOS_INLINE_FUNCTION matrix<T, N, N> outer_product(const ippl::Vector<T, N>& l,
                                                         const ippl::Vector<T, M>& r) {
        matrix<T, N, M> ret;
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < M; j++) {
                ret.data[j][i] = l[i] * r[j];
            }
        }
        return ret;
    }
}  // namespace ippl
#endif