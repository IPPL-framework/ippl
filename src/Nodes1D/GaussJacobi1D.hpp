#ifndef IPPL_NODES1D_GAUSS_JACOBI_1D_HPP
#define IPPL_NODES1D_GAUSS_JACOBI_1D_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace ippl {
namespace nodes1d {
namespace detail {

    [[noreturn]] inline void throwDuplicateNodes(const char* context, std::size_t n,
                                                 std::size_t i, std::size_t j, double xi,
                                                 double xj) {
        std::ostringstream os;
        os << context << ": duplicate node at indices " << i << " and " << j << " (n=" << n
           << ", x[" << i << "]=" << xi << ", x[" << j << "]=" << xj << ")";
        throw std::runtime_error(os.str());
    }

    template <typename Scalar>
    void assertDistinctSortedNodes(const char* context, const Scalar* nodes, std::size_t n,
                                   Scalar sepTol = Scalar(1e-14)) {
        for (std::size_t i = 1; i < n; ++i) {
            const Scalar gap = nodes[i] - nodes[i - 1];
            if (gap <= sepTol) {
                throwDuplicateNodes(context, n, i - 1, i, static_cast<double>(nodes[i - 1]),
                                    static_cast<double>(nodes[i]));
            }
        }
    }

    template <typename Scalar>
    bool isDuplicateAmong(const Scalar* nodes, std::size_t count, Scalar z, Scalar sepTol) {
        for (std::size_t j = 0; j < count; ++j) {
            if (Kokkos::abs(z - nodes[j]) <= sepTol) {
                return true;
            }
        }
        return false;
    }

    template <typename Scalar>
    Scalar jacobiCompanionDiagonal(std::size_t k, Scalar alpha, Scalar beta) {
        const Scalar ab = alpha + beta;
        if (k == 0) {
            return (beta - alpha) / (Scalar(2) + ab);
        }
        if (ab == Scalar(0)) {
            return Scalar(0);
        }
        return (beta * beta - alpha * alpha)
               / ((Scalar(2) * k + ab) * (Scalar(2) * k + ab + Scalar(2)));
    }

    template <typename Scalar>
    Scalar jacobiCompanionSubdiag(std::size_t k, Scalar alpha, Scalar beta) {
        const Scalar ab = alpha + beta;
        const Scalar kk = static_cast<Scalar>(k);
        Scalar val      = Scalar(2) / (Scalar(2) * kk + ab)
                     * Kokkos::sqrt((kk + alpha) * (kk + beta) / (Scalar(2) * kk + ab + Scalar(1)));
        if (k != 1) {
            val *= Kokkos::sqrt(kk * (kk + ab) / (Scalar(2) * kk + ab - Scalar(1)));
        }
        return val;
    }

    /**
     * @brief Independent Tricomi-style asymptotic for the k-th ascending Jacobi root.
     * k = 0 … n−1; k=0 near −1, k=n−1 near +1.
     */
    template <typename Scalar>
    Scalar asymptoticJacobiRoot(std::size_t k, std::size_t n, Scalar alpha, Scalar beta) {
        const Scalar t =
            Kokkos::numbers::pi_v<Scalar>
            * (Scalar(2) * static_cast<Scalar>(k) + beta + Scalar(1))
            / (Scalar(2) * static_cast<Scalar>(n) + alpha + beta + Scalar(2));
        return -Kokkos::cos(t);
    }

    /**
     * @brief Evaluate P_n^{(α,β)}(x) and its derivative via Bonnet recurrence (host).
     */
    template <typename Scalar>
    void evalJacobiAndDeriv(std::size_t n, Scalar alpha, Scalar beta, Scalar x, Scalar& p,
                            Scalar& dp) {
        if (n == 0) {
            p  = Scalar(1);
            dp = Scalar(0);
            return;
        }
        const Scalar alfbet = alpha + beta;
        Scalar p0           = Scalar(1);
        Scalar p1           = (alpha - beta + (Scalar(2) + alfbet) * x) / Scalar(2);
        if (n == 1) {
            p  = p1;
            dp = (Scalar(2) + alfbet) / Scalar(2);
            return;
        }
        for (std::size_t j = 2; j <= n; ++j) {
            const Scalar temp = Scalar(2) * static_cast<Scalar>(j) + alfbet;
            const Scalar a =
                Scalar(2) * static_cast<Scalar>(j) * (static_cast<Scalar>(j) + alfbet) * (temp - Scalar(2));
            const Scalar b =
                (temp - Scalar(1))
                * (alpha * alpha - beta * beta + temp * (temp - Scalar(2)) * x);
            const Scalar c = Scalar(2) * (static_cast<Scalar>(j) - Scalar(1) + alpha)
                             * (static_cast<Scalar>(j) - Scalar(1) + beta) * temp;
            const Scalar pn = (b * p1 - c * p0) / a;
            p0              = p1;
            p1              = pn;
        }
        p = p1;
        // P_n' = n/(2x) * ... use recurrence identity:
        // (1-x^2) P' = n/2 * ((α-β-(2n+α+β)x) P_n + 2(n+α)(n+β)/(2n+α+β) P_{n-1} wait —
        // SciPy: df = 0.5*(n+α+β+1)*P_{n-1}^{(α+1,β+1)}
        // From Newton weight path:
        const Scalar temp = Scalar(2) * static_cast<Scalar>(n) + alfbet;
        dp = (static_cast<Scalar>(n) * (alpha - beta - temp * x) * p1
              + Scalar(2) * (static_cast<Scalar>(n) + alpha) * (static_cast<Scalar>(n) + beta) * p0)
             / (temp * (Scalar(1) - x * x));
    }

    /**
     * @brief Implicit QL for symmetric tridiagonal (d,e) with eigenvectors in Z (columns).
     * d: diagonal (n); e: subdiagonal (n), e[0] unused, e[1..n-1] = off-diagonals.
     * Based on Numerical Recipes tqli / Golub–Van Loan.
     */
    template <typename Scalar>
    void tridiagonalQL(std::vector<Scalar>& d, std::vector<Scalar>& e, std::vector<Scalar>& z,
                       std::size_t n) {
        const Scalar eps = Scalar(2e-16);
        for (std::size_t i = 1; i < n; ++i) {
            e[i - 1] = e[i];
        }
        e[n - 1] = Scalar(0);

        for (std::size_t l = 0; l < n; ++l) {
            std::size_t iter = 0;
            for (;;) {
                std::size_t m = l;
                for (; m < n - 1; ++m) {
                    const Scalar dd = Kokkos::abs(d[m]) + Kokkos::abs(d[m + 1]);
                    if (Kokkos::abs(e[m]) + dd == dd) {
                        break;
                    }
                }
                if (m == l) {
                    break;
                }
                if (++iter > 100 * n) {
                    throw std::runtime_error(
                        "nodes1d::tridiagonalQL: failed to converge (Golub–Welsch)");
                }

                Scalar g = (d[l + 1] - d[l]) / (Scalar(2) * e[l]);
                Scalar r = Kokkos::sqrt(g * g + Scalar(1));
                g        = d[m] - d[l] + e[l] / (g + (g >= Scalar(0) ? r : -r));
                Scalar s = Scalar(1);
                Scalar c = Scalar(1);
                Scalar p = Scalar(0);
                for (std::size_t i = m - 1; i + 1 > l; --i) {
                    const Scalar f = s * e[i];
                    const Scalar b = c * e[i];
                    if (Kokkos::abs(f) >= Kokkos::abs(g)) {
                        c        = g / f;
                        r        = Kokkos::sqrt(c * c + Scalar(1));
                        e[i + 1] = f * r;
                        s        = Scalar(1) / r;
                        c *= s;
                    } else {
                        s        = f / g;
                        r        = Kokkos::sqrt(s * s + Scalar(1));
                        e[i + 1] = g * r;
                        c        = Scalar(1) / r;
                        s *= c;
                    }
                    g        = d[i + 1] - p;
                    r        = (d[i] - g) * s + Scalar(2) * c * b;
                    p        = s * r;
                    d[i + 1] = g + p;
                    g        = c * r - b;

                    for (std::size_t k = 0; k < n; ++k) {
                        const Scalar f2 = z[k * n + (i + 1)];
                        z[k * n + (i + 1)] = s * z[k * n + i] + c * f2;
                        z[k * n + i]       = c * z[k * n + i] - s * f2;
                    }
                    if (i == l) {
                        break;
                    }
                }
                d[l] -= p;
                e[l]     = g;
                e[m]     = Scalar(0);
                (void)eps;
            }
        }
    }

    template <typename Scalar>
    void polishJacobiRoot(std::size_t n, Scalar alpha, Scalar beta, Scalar& x) {
        Scalar p, dp;
        evalJacobiAndDeriv(n, alpha, beta, x, p, dp);
        if (Kokkos::abs(dp) > Scalar(0)) {
            x -= p / dp;
        }
        // Clamp to open interval for weight formulas.
        const Scalar lim = Scalar(1) - Scalar(1e-15);
        if (x > lim) {
            x = lim;
        }
        if (x < -lim) {
            x = -lim;
        }
    }

    template <typename Scalar>
    void finalizeGolubWelschResults(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes,
                                    Scalar* weights, const std::vector<Scalar>& evals,
                                    const std::vector<Scalar>& evecs, const std::vector<std::size_t>& perm) {
        const Scalar mu0 =
            Kokkos::pow(Scalar(2), alpha + beta + Scalar(1))
            * Kokkos::exp(Kokkos::lgamma(alpha + Scalar(1)) + Kokkos::lgamma(beta + Scalar(1))
                          - Kokkos::lgamma(alpha + beta + Scalar(2)));

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t j = perm[i];
            Scalar x            = evals[j];
            polishJacobiRoot(n, alpha, beta, x);
            nodes[i]       = x;
            const Scalar v0 = evecs[0 * n + j];
            weights[i]     = mu0 * v0 * v0;
        }

        Scalar wsum = Scalar(0);
        for (std::size_t i = 0; i < n; ++i) {
            wsum += weights[i];
        }
        if (wsum > Scalar(0)) {
            const Scalar scale = mu0 / wsum;
            for (std::size_t i = 0; i < n; ++i) {
                weights[i] *= scale;
            }
        }

        if (alpha == beta) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t j = n - 1 - i;
                if (i > j) {
                    break;
                }
                const Scalar xsym = Scalar(0.5) * (nodes[i] - nodes[j]);
                const Scalar wsym = Scalar(0.5) * (weights[i] + weights[j]);
                nodes[i]          = xsym;
                nodes[j]          = -xsym;
                weights[i] = weights[j] = wsym;
            }
        }
    }

    template <typename Scalar>
    void gaussJacobiMu0SingleNode(Scalar alpha, Scalar beta, Scalar* nodes, Scalar* weights) {
        nodes[0]   = (beta - alpha) / (alpha + beta + Scalar(2));
        weights[0] = Kokkos::pow(Scalar(2), alpha + beta + Scalar(1))
                     * Kokkos::exp(Kokkos::lgamma(alpha + Scalar(1))
                                   + Kokkos::lgamma(beta + Scalar(1))
                                   - Kokkos::lgamma(alpha + beta + Scalar(2)));
    }

    /**
     * @brief Cyclic Jacobi eigenvalue algorithm for a dense symmetric matrix (O(n³)).
     * On entry A is destroyed; on exit diag(A) holds eigenvalues, V holds orthonormal
     * eigenvectors as columns (V is n*n row-major).
     */
    template <typename Scalar>
    void denseSymmetricEigenJacobi(std::vector<Scalar>& A, std::vector<Scalar>& V, std::size_t n,
                                   std::size_t maxSweeps = 64) {
        V.assign(n * n, Scalar(0));
        for (std::size_t i = 0; i < n; ++i) {
            V[i * n + i] = Scalar(1);
        }

        const Scalar tol = Scalar(2e-16);
        for (std::size_t sweep = 0; sweep < maxSweeps; ++sweep) {
            Scalar off = Scalar(0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i + 1; j < n; ++j) {
                    off += Kokkos::abs(A[i * n + j]);
                }
            }
            if (off <= tol) {
                return;
            }

            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = p + 1; q < n; ++q) {
                    const Scalar apq = A[p * n + q];
                    if (Kokkos::abs(apq) <= tol) {
                        continue;
                    }
                    const Scalar app = A[p * n + p];
                    const Scalar aqq = A[q * n + q];
                    const Scalar tau = (aqq - app) / (Scalar(2) * apq);
                    const Scalar t =
                        (tau >= Scalar(0))
                            ? Scalar(1) / (tau + Kokkos::sqrt(Scalar(1) + tau * tau))
                            : Scalar(-1) / (-tau + Kokkos::sqrt(Scalar(1) + tau * tau));
                    const Scalar c = Scalar(1) / Kokkos::sqrt(Scalar(1) + t * t);
                    const Scalar s = t * c;

                    A[p * n + p] = app - t * apq;
                    A[q * n + q] = aqq + t * apq;
                    A[p * n + q] = Scalar(0);
                    A[q * n + p] = Scalar(0);

                    for (std::size_t r = 0; r < n; ++r) {
                        if (r == p || r == q) {
                            continue;
                        }
                        const Scalar arp = A[r * n + p];
                        const Scalar arq = A[r * n + q];
                        A[r * n + p] = A[p * n + r] = c * arp - s * arq;
                        A[r * n + q] = A[q * n + r] = s * arp + c * arq;
                    }

                    for (std::size_t r = 0; r < n; ++r) {
                        const Scalar vrp = V[r * n + p];
                        const Scalar vrq = V[r * n + q];
                        V[r * n + p]     = c * vrp - s * vrq;
                        V[r * n + q]     = s * vrp + c * vrq;
                    }
                }
            }
        }
    }

    template <typename Scalar>
    void computeGaussJacobiGolubWelschDense(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes,
                                            Scalar* weights) {
        if (n == 1) {
            gaussJacobiMu0SingleNode(alpha, beta, nodes, weights);
            return;
        }

        std::vector<Scalar> A(n * n, Scalar(0));
        for (std::size_t i = 0; i < n; ++i) {
            A[i * n + i] = jacobiCompanionDiagonal(i, alpha, beta);
        }
        for (std::size_t k = 1; k < n; ++k) {
            const Scalar b = jacobiCompanionSubdiag(k, alpha, beta);
            A[(k - 1) * n + k] = b;
            A[k * n + (k - 1)] = b;
        }

        std::vector<Scalar> V;
        denseSymmetricEigenJacobi(A, V, n);

        std::vector<Scalar> evals(n);
        for (std::size_t i = 0; i < n; ++i) {
            evals[i] = A[i * n + i];
        }
        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) {
            perm[i] = i;
        }
        std::sort(perm.begin(), perm.end(),
                  [&](std::size_t i, std::size_t j) { return evals[i] < evals[j]; });

        finalizeGolubWelschResults(n, alpha, beta, nodes, weights, evals, V, perm);
    }

    template <typename Scalar>
    void computeGaussJacobiGolubWelsch(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes,
                                       Scalar* weights) {
        if (n == 1) {
            gaussJacobiMu0SingleNode(alpha, beta, nodes, weights);
            return;
        }

        std::vector<Scalar> d(n), e(n, Scalar(0)), z(n * n, Scalar(0));
        for (std::size_t i = 0; i < n; ++i) {
            d[i]         = jacobiCompanionDiagonal(i, alpha, beta);
            z[i * n + i] = Scalar(1);
        }
        for (std::size_t k = 1; k < n; ++k) {
            e[k] = jacobiCompanionSubdiag(k, alpha, beta);
        }

        tridiagonalQL(d, e, z, n);

        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) {
            perm[i] = i;
        }
        std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) { return d[i] < d[j]; });

        finalizeGolubWelschResults(n, alpha, beta, nodes, weights, d, z, perm);
    }

    template <typename Scalar>
    Scalar lehrFEMInitialGuess(std::size_t i, std::size_t n, Scalar alpha, Scalar beta,
                               const Scalar* ascending_nodes_so_far) {
        // LehrFEM formulas were written for a descending-from-+1 scan. Map ascending index
        // i to that convention via previously accepted ascending nodes when available.
        Scalar r1, r2, r3;
        Scalar z = (i > 0) ? ascending_nodes_so_far[i - 1] : Scalar(0);

        if (i == 0) {
            const Scalar an = alpha / static_cast<Scalar>(n);
            const Scalar bn = beta / static_cast<Scalar>(n);
            r1 = (1.0 + alpha) * (2.78 / (4.0 + n * n) + 0.768 * an / static_cast<Scalar>(n));
            r2 = 1.0 + 1.48 * an + 0.96 * bn + 0.452 * an * an + 0.83 * an * bn;
            z  = -(1.0 - r1 / r2);  // mirror LehrFEM's first (+1)-side guess to −1 side
        } else if (i == 1) {
            r1 = (4.1 + beta) / ((1.0 + beta) * (1.0 + 0.156 * beta));
            r2 = 1.0 + 0.06 * (n - 8.0) * (1.0 + 0.12 * beta) / static_cast<Scalar>(n);
            r3 = 1.0 + 0.012 * alpha * (1.0 + 0.25 * Kokkos::abs(beta)) / static_cast<Scalar>(n);
            z += (1.0 + z) * r1 * r2 * r3 * Scalar(0.25);
            if (z > Scalar(0.9)) {
                z = asymptoticJacobiRoot(i, n, alpha, beta);
            }
        } else {
            z = asymptoticJacobiRoot(i, n, alpha, beta);
        }
        return z;
    }

    template <typename Scalar>
    bool newtonOneRootBounded(std::size_t n, Scalar alpha, Scalar beta, Scalar& z, Scalar lo,
                              Scalar hi, std::size_t maxNewtonIterations,
                              std::size_t minNewtonIterations, Scalar& p_out, Scalar& dp_out,
                              Scalar& pnm1_out, Scalar& temp_out) {
        const Scalar tolerance = Scalar(2e-16);
        const Scalar alfbet    = alpha + beta;
        const Scalar softLo    = lo + Scalar(1e-15);
        const Scalar softHi    = hi - Scalar(1e-15);
        if (!(softHi > softLo)) {
            return false;
        }
        if (z <= softLo) {
            z = softLo + Scalar(0.1) * (softHi - softLo);
        }
        if (z >= softHi) {
            z = softHi - Scalar(0.1) * (softHi - softLo);
        }

        Scalar a, b, c, p1, p2, p3, pp, temp, z1;
        std::size_t its = 1;
        do {
            temp = 2.0 + alfbet;
            p1   = (alpha - beta + temp * z) / 2.0;
            p2   = 1.0;
            for (std::size_t j = 2; j <= n; ++j) {
                p3   = p2;
                p2   = p1;
                temp = 2 * j + alfbet;
                a    = 2 * j * (j + alfbet) * (temp - 2.0);
                b    = (temp - 1.0) * (alpha * alpha - beta * beta + temp * (temp - 2.0) * z);
                c    = 2.0 * (j - 1 + alpha) * (j - 1 + beta) * temp;
                p1   = (b * p2 - c * p3) / a;
            }
            const Scalar one_m_z2 = Scalar(1) - z * z;
            if (Kokkos::abs(one_m_z2) < Scalar(1e-30)) {
                return false;
            }
            pp = (n * (alpha - beta - temp * z) * p1 + 2.0 * (n + alpha) * (n + beta) * p2)
                 / (temp * one_m_z2);
            z1 = z;
            z  = z1 - p1 / pp;
            if (z < softLo) {
                z = softLo;
            }
            if (z > softHi) {
                z = softHi;
            }
            if (its > minNewtonIterations && Kokkos::abs(z - z1) <= tolerance
                && Kokkos::abs(p1) <= Scalar(1e-12)) {
                p_out    = p1;
                dp_out   = pp;
                pnm1_out = p2;
                temp_out = temp;
                return true;
            }
            ++its;
        } while (its <= maxNewtonIterations);
        p_out    = p1;
        dp_out   = pp;
        pnm1_out = p2;
        temp_out = temp;
        return false;
    }

    template <typename Scalar>
    Scalar brentJacobiRoot(std::size_t n, Scalar alpha, Scalar beta, Scalar lo, Scalar hi,
                           std::size_t maxIter = 200) {
        Scalar a = lo;
        Scalar b = hi;
        Scalar fa, dfa, fb, dfb;
        evalJacobiAndDeriv(n, alpha, beta, a, fa, dfa);
        evalJacobiAndDeriv(n, alpha, beta, b, fb, dfb);
        if (fa * fb > Scalar(0)) {
            throw std::runtime_error("nodes1d::brentJacobiRoot: bracket does not change sign");
        }
        if (Kokkos::abs(fa) < Kokkos::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
        Scalar c = a;
        Scalar fc = fa;
        bool mflag = true;
        Scalar d = Scalar(0);
        for (std::size_t it = 0; it < maxIter; ++it) {
            if (fb == Scalar(0) || Kokkos::abs(b - a) < Scalar(2e-16)) {
                return b;
            }
            Scalar s;
            if (fa != fc && fb != fc) {
                s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            const bool cond1 = (s < (Scalar(3) * a + b) / Scalar(4) || s > b);
            const bool cond2 = mflag && Kokkos::abs(s - b) >= Kokkos::abs(b - c) / Scalar(2);
            const bool cond3 = !mflag && Kokkos::abs(s - b) >= Kokkos::abs(c - d) / Scalar(2);
            const bool cond4 = mflag && Kokkos::abs(b - c) < Scalar(2e-16);
            const bool cond5 = !mflag && Kokkos::abs(c - d) < Scalar(2e-16);
            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s     = (a + b) / Scalar(2);
                mflag = true;
            } else {
                mflag = false;
            }
            Scalar fs, dfs;
            evalJacobiAndDeriv(n, alpha, beta, s, fs, dfs);
            d  = c;
            c  = b;
            fc = fb;
            if (fa * fs < Scalar(0)) {
                b  = s;
                fb = fs;
            } else {
                a  = s;
                fa = fs;
            }
            if (Kokkos::abs(fa) < Kokkos::abs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
        }
        return b;
    }

    template <typename Scalar>
    bool findSignChange(std::size_t n, Scalar alpha, Scalar beta, Scalar lo, Scalar hi, Scalar& a,
                        Scalar& b, int samples = 128) {
        Scalar fa, dfa;
        evalJacobiAndDeriv(n, alpha, beta, lo, fa, dfa);
        Scalar prev = lo;
        Scalar fprev = fa;
        for (int s = 1; s <= samples; ++s) {
            const Scalar x =
                lo + (hi - lo) * (static_cast<Scalar>(s) / static_cast<Scalar>(samples));
            Scalar fx, dfx;
            evalJacobiAndDeriv(n, alpha, beta, x, fx, dfx);
            if (fprev * fx <= Scalar(0)) {
                a = prev;
                b = x;
                return true;
            }
            prev  = x;
            fprev = fx;
        }
        return false;
    }

    template <typename Scalar>
    Scalar pickGuess(InitialGuessType kind, std::size_t k, std::size_t n, Scalar alpha, Scalar beta,
                     const Scalar* ascending_nodes) {
        switch (kind) {
            case InitialGuessType::Asymptotic:
                return asymptoticJacobiRoot(k, n, alpha, beta);
            case InitialGuessType::Chebyshev:
                return chebyshevNode<Scalar>(k, n);
            case InitialGuessType::LehrFEM:
                return lehrFEMInitialGuess(k, n, alpha, beta, ascending_nodes);
            default:
                return asymptoticJacobiRoot(k, n, alpha, beta);
        }
    }

    template <typename Scalar>
    void collectJacobiRootsBySignChange(std::size_t n, Scalar alpha, Scalar beta,
                                        std::vector<Scalar>& roots, int samples = 512) {
        roots.clear();
        const Scalar glo = Scalar(-1) + Scalar(1e-14);
        const Scalar ghi = Scalar(1) - Scalar(1e-14);
        Scalar prev = glo;
        Scalar fprev, df;
        evalJacobiAndDeriv(n, alpha, beta, prev, fprev, df);
        for (int s = 1; s <= samples; ++s) {
            const Scalar x =
                glo + (ghi - glo) * (static_cast<Scalar>(s) / static_cast<Scalar>(samples));
            Scalar fx;
            evalJacobiAndDeriv(n, alpha, beta, x, fx, df);
            if (fprev * fx <= Scalar(0) && roots.size() < n) {
                Scalar a = prev;
                Scalar b = x;
                // Shrink tiny brackets at endpoints if needed.
                if (b > a) {
                    roots.push_back(brentJacobiRoot(n, alpha, beta, a, b));
                }
            }
            prev  = x;
            fprev = fx;
        }
    }

    template <typename Scalar>
    void computeGaussJacobiNewton(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes,
                                  Scalar* weights, std::size_t maxNewtonIterations,
                                  std::size_t minNewtonIterations, InitialGuessType initialGuess) {
        const Scalar sepTol = Scalar(1e-13);
        const Scalar alfbet = alpha + beta;
        const Scalar mu0 =
            Kokkos::exp(Kokkos::lgamma(alpha + n) + Kokkos::lgamma(beta + n)
                        - Kokkos::lgamma(static_cast<Scalar>(n) + 1.)
                        - Kokkos::lgamma(static_cast<Scalar>(n) + alfbet + 1.0))
            * Kokkos::pow(Scalar(2), alfbet);
        const Scalar glo = Scalar(-1) + Scalar(1e-14);
        const Scalar ghi = Scalar(1) - Scalar(1e-14);

        std::vector<InitialGuessType> ladder;
        ladder.push_back(initialGuess);
        if (initialGuess != InitialGuessType::Asymptotic) {
            ladder.push_back(InitialGuessType::Asymptotic);
        }
        if (initialGuess != InitialGuessType::Chebyshev) {
            ladder.push_back(InitialGuessType::Chebyshev);
        }

        std::vector<Scalar> cand;
        cand.reserve(n * ladder.size());
        for (std::size_t k = 0; k < n; ++k) {
            for (InitialGuessType kind : ladder) {
                Scalar z = pickGuess(kind, k, n, alpha, beta, nodes);
                Scalar p1, pp, p2, temp;
                if (!newtonOneRootBounded(n, alpha, beta, z, glo, ghi, maxNewtonIterations,
                                          minNewtonIterations, p1, pp, p2, temp)) {
                    continue;
                }
                if (Kokkos::abs(p1) > Scalar(1e-10)) {
                    continue;
                }
                cand.push_back(z);
            }
        }

        std::sort(cand.begin(), cand.end());
        std::vector<Scalar> unique;
        for (Scalar z : cand) {
            if (unique.empty() || z - unique.back() > sepTol) {
                unique.push_back(z);
            }
        }

        if (unique.size() != n) {
            // Robust fallback: Brent on every sign-change of P_n on [-1,1].
            collectJacobiRootsBySignChange(n, alpha, beta, unique);
            std::sort(unique.begin(), unique.end());
            // Unique-ify again.
            std::vector<Scalar> u2;
            for (Scalar z : unique) {
                if (u2.empty() || z - u2.back() > sepTol) {
                    u2.push_back(z);
                }
            }
            unique.swap(u2);
        }

        if (unique.size() != n) {
            throw std::runtime_error(
                "nodes1d::computeGaussJacobi (Newton): expected " + std::to_string(n)
                + " distinct roots, found " + std::to_string(unique.size()));
        }

        for (std::size_t i = 0; i < n; ++i) {
            nodes[i] = unique[i];
            Scalar z = nodes[i], p1, pp, p2, temp;
            newtonOneRootBounded(n, alpha, beta, z, glo, ghi, maxNewtonIterations,
                                 minNewtonIterations, p1, pp, p2, temp);
            weights[i] = mu0 * temp / (pp * p2);
        }
        assertDistinctSortedNodes("nodes1d::computeGaussJacobi (Newton)", nodes, n, sepTol);

        if (alpha == beta) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t j = n - 1 - i;
                if (i > j) {
                    break;
                }
                const Scalar xsym = Scalar(0.5) * (nodes[i] - nodes[j]);
                const Scalar wsym = Scalar(0.5) * (weights[i] + weights[j]);
                nodes[i]          = xsym;
                nodes[j]          = -xsym;
                weights[i] = weights[j] = wsym;
            }
        }
    }

}  // namespace detail

    template <typename Scalar>
    void computeGaussJacobi(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes, Scalar* weights,
                            std::size_t maxNewtonIterations, std::size_t minNewtonIterations,
                            InitialGuessType initialGuess, RootFinderMethod method) {
        assert(n >= 1);
        assert(alpha > -1.0);
        assert(beta > -1.0);
        assert(nodes != nullptr && weights != nullptr);

        if (method == RootFinderMethod::Newton) {
            detail::computeGaussJacobiNewton(n, alpha, beta, nodes, weights, maxNewtonIterations,
                                             minNewtonIterations, initialGuess);
            return;
        }
        if (method == RootFinderMethod::DenseGolubWelsch) {
            detail::computeGaussJacobiGolubWelschDense(n, alpha, beta, nodes, weights);
            return;
        }
        detail::computeGaussJacobiGolubWelsch(n, alpha, beta, nodes, weights);
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
