#include "Ippl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "KokkosBlas3_gemm.hpp"
#include "gtest/gtest.h"

#ifdef IPPL_TEST_HOST_EIGENANALYSIS
#include "KokkosKernels_config.h"
#ifdef IPPL_TEST_HOST_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#endif

namespace {
    constexpr int Dimension = 4;
    using Matrix            = std::array<std::array<double, Dimension>, Dimension>;

    // OPALX's transverse two-plane rotation convention; tunes are in turns.
    Matrix rotations(double tune1, double tune2, double radius1 = 1, double radius2 = 1) {
        Matrix matrix{};
        for (int plane = 0; plane < 2; ++plane) {
            const double angle  = 2 * Kokkos::numbers::pi * (plane ? tune2 : tune1);
            const double radius = plane ? radius2 : radius1;
            const int i         = 2 * plane;
            matrix[i][i] = matrix[i + 1][i + 1] = radius * std::cos(angle);
            matrix[i][i + 1]                    = radius * std::sin(angle);
            matrix[i + 1][i]                    = -radius * std::sin(angle);
        }
        return matrix;
    }

    // Exercise GEMM in the configured execution/memory space, then explicitly
    // transfer Q A Q^T to host memory for host-only eigenanalysis.
    Matrix coupled(const Matrix& matrix) {
        using View = Kokkos::View<double**>;
        View a("map", Dimension, Dimension), q("coupling", Dimension, Dimension);
        View work("work", Dimension, Dimension), result("result", Dimension, Dimension);
        auto hostA     = Kokkos::create_mirror_view(a);
        auto hostQ     = Kokkos::create_mirror_view(q);
        const double c = std::cos(0.6), s = std::sin(0.6);
        const Matrix rotation{{{c, 0, s, 0}, {0, c, 0, s}, {-s, 0, c, 0}, {0, -s, 0, c}}};
        for (int i = 0; i < Dimension; ++i)
            for (int j = 0; j < Dimension; ++j) {
                hostA(i, j) = matrix[i][j];
                hostQ(i, j) = rotation[i][j];
            }
        Kokkos::deep_copy(a, hostA);
        Kokkos::deep_copy(q, hostQ);
        KokkosBlas::gemm("N", "N", 1.0, q, a, 0.0, work);
        KokkosBlas::gemm("N", "T", 1.0, work, q, 0.0, result);
        auto hostResult = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
        Matrix output{};
        for (int i = 0; i < Dimension; ++i)
            for (int j = 0; j < Dimension; ++j)
                output[i][j] = hostResult(i, j);
        return output;
    }

    TEST(KokkosKernels, CoupledMapGemm) {
        const auto matrix = rotations(0.155, 0.31);
        const auto result = coupled(matrix);
        const double c = std::cos(0.6), s = std::sin(0.6);
        // A few operations on unit-scale entries: 1e-14 accommodates FMA differences.
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                EXPECT_NEAR(result[i][j], c * c * matrix[i][j] + s * s * matrix[i + 2][j + 2],
                            1e-14);
                EXPECT_NEAR(result[i + 2][j + 2],
                            s * s * matrix[i][j] + c * c * matrix[i + 2][j + 2], 1e-14);
                EXPECT_NEAR(result[i][j + 2], c * s * (matrix[i + 2][j + 2] - matrix[i][j]), 1e-14);
                EXPECT_NEAR(result[i + 2][j], result[i][j + 2], 1e-14);
            }
    }

#ifdef IPPL_TEST_HOST_EIGENANALYSIS
    void checkEigenpairs(const Matrix& input, std::vector<std::complex<double>> expected) {
        using View   = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        using Vector = Kokkos::View<double*, Kokkos::HostSpace>;
        View a("map", Dimension, Dimension), left("left", Dimension, Dimension);
        View right("right", Dimension, Dimension);
        Vector real("real", Dimension), imaginary("imaginary", Dimension);
        for (int i = 0; i < Dimension; ++i)
            for (int j = 0; j < Dimension; ++j)
                a(i, j) = input[i][j];
        // Kokkos Kernels 5.2's experimental eigen header does not compile with GCC 15
        // (invalid Householder template call). Use its configured host TPL directly,
        // retaining LAPACKE's INFO, with contiguous column-major host views.
        Kokkos::deep_copy(real, std::numeric_limits<double>::quiet_NaN());
        Kokkos::deep_copy(imaginary, std::numeric_limits<double>::quiet_NaN());
        Kokkos::deep_copy(right, std::numeric_limits<double>::quiet_NaN());
        ASSERT_EQ(
            LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V', Dimension, a.data(), Dimension, real.data(),
                          imaginary.data(), left.data(), Dimension, right.data(), Dimension),
            0);
        // Small well-conditioned maps should have O(epsilon) errors; allow 1e-12
        // for variation among LAPACK providers without changing existing tolerances.
        constexpr double Tolerance = 1e-12;
        for (int k = 0; k < Dimension; ++k) {
            ASSERT_TRUE(std::isfinite(real(k)));
            ASSERT_TRUE(std::isfinite(imaginary(k)));
            const std::complex<double> lambda(real(k), imaginary(k));
            auto match = std::min_element(expected.begin(), expected.end(), [&](auto x, auto y) {
                return std::abs(x - lambda) < std::abs(y - lambda);
            });
            ASSERT_NE(match, expected.end());
            EXPECT_LT(std::abs(*match - lambda), Tolerance);
            expected.erase(match);
            // LAPACK packs each conjugate pair into adjacent real columns.
            if (imaginary(k) > 0) {
                ASSERT_LT(k + 1, Dimension);
                EXPECT_NEAR(real(k + 1), real(k), Tolerance);
                EXPECT_NEAR(imaginary(k + 1), -imaginary(k), Tolerance);
            }
            if (imaginary(k) < 0) {
                ASSERT_GT(k, 0);
                ASSERT_GT(imaginary(k - 1), 0);
            }
            std::array<std::complex<double>, Dimension> vector{};
            for (int i = 0; i < Dimension; ++i) {
                vector[i] = imaginary(k) == 0 ? std::complex<double>(right(i, k), 0)
                            : imaginary(k) > 0
                                ? std::complex<double>(right(i, k), right(i, k + 1))
                                : std::complex<double>(right(i, k - 1), -right(i, k));
            }
            double norm = 0, residual = 0, matrixNorm = 0;
            for (int i = 0; i < Dimension; ++i) {
                std::complex<double> av = 0;
                for (int j = 0; j < Dimension; ++j) {
                    av += input[i][j] * vector[j];
                    matrixNorm += input[i][j] * input[i][j];
                }
                norm += std::norm(vector[i]);
                residual += std::norm(av - lambda * vector[i]);
            }
            ASSERT_GT(norm, 0);
            EXPECT_LT(std::sqrt(residual)
                          / ((std::sqrt(matrixNorm) + std::abs(lambda)) * std::sqrt(norm)),
                      Tolerance);
        }
    }

    std::vector<std::complex<double>> spectrum(double q1, double q2, double r1 = 1, double r2 = 1) {
        const auto first  = std::polar(r1, 2 * Kokkos::numbers::pi * q1);
        const auto second = std::polar(r2, 2 * Kokkos::numbers::pi * q2);
        return {first, std::conj(first), second, std::conj(second)};
    }

    TEST(KokkosKernels, StableRotationsAndConjugateBranches) {
        checkEigenpairs(rotations(0.155, 0.89), spectrum(0.155, 0.89));
    }

    TEST(KokkosKernels, CoupledStableModes) {
        checkEigenpairs(coupled(rotations(0.155, 0.31)), spectrum(0.155, 0.31));
    }

    TEST(KokkosKernels, RealAndComplexInstabilities) {
        auto matrix  = rotations(0.15, 0.3);
        matrix[0][0] = 2;
        matrix[1][1] = 0.5;
        matrix[0][1] = matrix[1][0] = 0;
        auto expected               = spectrum(0.15, 0.3);
        expected[0]                 = 2;
        expected[1]                 = 0.5;
        checkEigenpairs(matrix, expected);
        checkEigenpairs(coupled(rotations(0.2, 0.2, 1.1, 1 / 1.1)),
                        spectrum(0.2, 0.2, 1.1, 1 / 1.1));
        checkEigenpairs(rotations(0.2, 0.3, 0.9, 0.8), spectrum(0.2, 0.3, 0.9, 0.8));
    }

    TEST(KokkosKernels, NearIntegerAndNeutralModes) {
        checkEigenpairs(rotations(1e-8, 0.3), spectrum(1e-8, 0.3));
        Matrix identity{};
        for (int i = 0; i < Dimension; ++i)
            identity[i][i] = 1;
        checkEigenpairs(identity, {1, 1, 1, 1});
        identity[0][0] = identity[1][1] = -1;
        checkEigenpairs(identity, {-1, -1, 1, 1});
    }
#endif
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
