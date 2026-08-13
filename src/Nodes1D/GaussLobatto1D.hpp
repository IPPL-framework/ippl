/**
 * @file GaussLobatto1D.hpp
 * @brief Implementation of computeGaussLobatto (Golub-Welsch and Newton backends).
 * @internal Included by GaussLobatto1D.h; not for direct use.
 *
 * Host-only implementation: uses std::vector, exceptions, and serial root-finding.
 * Not suitable for Kokkos parallel_for or device kernels.
 */
#ifndef IPPL_NODES1D_GAUSS_LOBATTO_1D_HPP
#define IPPL_NODES1D_GAUSS_LOBATTO_1D_HPP

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Nodes1D/GaussJacobi1D.h"

namespace ippl {
namespace nodes1d {
/** @internal Helpers for GLL node/weight computation. */
namespace detail {

    /** @internal Evaluate P_n(x) and P_n'(x) by Legendre three-term recurrence. */
    template <typename Scalar>
    void legendreAndDeriv(std::size_t n, Scalar x, Scalar& p, Scalar& dp) {
        if (n == 0) {
            p  = Scalar(1);
            dp = Scalar(0);
            return;
        }
        if (n == 1) {
            p  = x;
            dp = Scalar(1);
            return;
        }
        Scalar p0 = Scalar(1);
        Scalar p1 = x;
        for (std::size_t k = 2; k <= n; ++k) {
            const Scalar pk =
                ((2 * static_cast<Scalar>(k) - 1) * x * p1 - (static_cast<Scalar>(k) - 1) * p0)
                / static_cast<Scalar>(k);
            p0 = p1;
            p1 = pk;
        }
        p                = p1;
        const Scalar pm1 = p0;
        dp               = static_cast<Scalar>(n) * (x * p - pm1) / (x * x - Scalar(1));
    }

    /**
     * @internal Form GLL weights from nodes: boundary 2/(n*(n-1)),
     * interior 2/(n*(n-1)*P_{n-1}(x_i)^2). Requires nodes[0]=-1, nodes[n-1]=+1.
     */
    template <typename Scalar>
    void fillGLLWeights(std::size_t n, Scalar* nodes, Scalar* weights) {
        const Scalar boundary = Scalar(2) / (static_cast<Scalar>(n) * static_cast<Scalar>(n - 1));
        weights[0]            = boundary;
        weights[n - 1]        = boundary;
        const Scalar denomFactor = static_cast<Scalar>(n) * static_cast<Scalar>(n - 1);
        for (std::size_t i = 1; i < n - 1; ++i) {
            const Scalar p = evalLegendre(n - 1, nodes[i]);
            weights[i]     = Scalar(2) / (denomFactor * p * p);
        }
    }

    /**
     * @internal Brent root finder for P_deg'(x) on bracket (lo, hi).
     * @throws std::runtime_error if the bracket does not change sign.
     */
    template <typename Scalar>
    Scalar brentLegendreDerivRoot(std::size_t deg, Scalar lo, Scalar hi, std::size_t maxIter = 200) {
        // Find root of P'_deg on (lo, hi).
        auto f = [&](Scalar x) {
            Scalar p, dp;
            legendreAndDeriv(deg, x, p, dp);
            return dp;
        };
        Scalar a = lo;
        Scalar b = hi;
        Scalar fa = f(a);
        Scalar fb = f(b);
        if (fa * fb > Scalar(0)) {
            throw std::runtime_error("nodes1d::brentLegendreDerivRoot: no sign change");
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
            const Scalar fs = f(s);
            d               = c;
            c               = b;
            fc              = fb;
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

    /**
     * @internal Golub-Welsch GLL: fix +/-1, Jacobi (1,1) interiors, then fillGLLWeights.
     * @param dense If true, use dense companion eigen-solve for interior nodes.
     */
    template <typename Scalar>
    void computeGaussLobattoGolubWelsch(std::size_t n, Scalar* nodes, Scalar* weights,
                                        bool dense) {
        nodes[0]     = Scalar(-1);
        nodes[n - 1] = Scalar(1);

        if (n == 2) {
            weights[0] = Scalar(1);
            weights[1] = Scalar(1);
            return;
        }

        std::vector<Scalar> interior(n - 2);
        std::vector<Scalar> iw(n - 2);
        if (dense) {
            computeGaussJacobiGolubWelschDense(n - 2, Scalar(1), Scalar(1), interior.data(),
                                               iw.data());
        } else {
            computeGaussJacobiGolubWelsch(n - 2, Scalar(1), Scalar(1), interior.data(), iw.data());
        }
        for (std::size_t i = 0; i < n - 2; ++i) {
            nodes[i + 1] = interior[i];
        }

        fillGLLWeights(n, nodes, weights);
        assertDistinctSortedNodes(dense ? "nodes1d::computeGaussLobatto (DenseGolubWelsch)"
                                        : "nodes1d::computeGaussLobatto (GolubWelsch)",
                                  nodes, n);
    }

    /**
     * @internal Newton on P_{n-1}' for interior GLL nodes (sequential left-to-right).
     *
     * For each interior j = 0 .. n-3: try Asymptotic Jacobi (1,1), then Chebyshev-Lobatto
     * cosine, then Brent on P_{n-1}' in (lo, hi) where lo is the previous accepted interior
     * (or just above -1) and hi is derived from asymptotic guesses. Convergence requires
     * at least minNewtonIterations steps, matching computeGaussJacobi.
     *
     * @throws std::runtime_error on bracket failure or duplicate interiors.
     */
    template <typename Scalar>
    void computeGaussLobattoNewton(std::size_t n, Scalar* nodes, Scalar* weights,
                                   std::size_t maxNewtonIterations,
                                   std::size_t minNewtonIterations) {
        nodes[0]     = Scalar(-1);
        nodes[n - 1] = Scalar(1);

        if (n == 2) {
            weights[0] = Scalar(1);
            weights[1] = Scalar(1);
            return;
        }

        const std::size_t deg    = n - 1;
        const Scalar tol         = Scalar(2e-16);
        const Scalar sepTol      = Scalar(1e-13);
        const std::size_t maxIts = (maxNewtonIterations < 40) ? 40 : maxNewtonIterations;
        const std::size_t nInt   = n - 2;

        // Interior index j = 0..nInt-1 maps to ascending nodes[1..n-2].
        for (std::size_t j = 0; j < nInt; ++j) {
            const Scalar lo =
                (j == 0) ? Scalar(-1) + Scalar(1e-14) : nodes[j] + sepTol;
            Scalar hi = Scalar(1) - Scalar(1e-14);
            if (j + 1 < nInt) {
                const Scalar nextAsym = asymptoticJacobiRoot(j + 1, nInt, Scalar(1), Scalar(1));
                const Scalar curAsym  = asymptoticJacobiRoot(j, nInt, Scalar(1), Scalar(1));
                hi = Scalar(0.5) * (curAsym + nextAsym);
                if (hi <= lo) {
                    hi = Scalar(1) - Scalar(1e-14);
                }
            }

            Scalar guesses[2] = {
                asymptoticJacobiRoot(j, nInt, Scalar(1), Scalar(1)),
                -Kokkos::cos(Kokkos::numbers::pi_v<Scalar> * static_cast<Scalar>(j + 1)
                             / static_cast<Scalar>(deg)),
            };

            bool accepted = false;
            Scalar z      = Scalar(0);
            for (Scalar guess : guesses) {
                z = guess;
                if (z <= lo || z >= hi) {
                    z = Scalar(0.5) * (lo + hi);
                }
                bool converged = false;
                for (std::size_t it = 0; it < maxIts; ++it) {
                    Scalar p, dp;
                    legendreAndDeriv(deg, z, p, dp);
                    const Scalar denom = (Scalar(1) - z * z);
                    if (Kokkos::abs(denom) < Scalar(1e-30)) {
                        break;
                    }
                    const Scalar ddp =
                        (Scalar(2) * z * dp - static_cast<Scalar>(deg * (deg + 1)) * p) / denom;
                    const Scalar delta = dp / ddp;
                    z -= delta;
                    if (z < lo) {
                        z = lo;
                    }
                    if (z > hi) {
                        z = hi;
                    }
                    if (it >= minNewtonIterations && Kokkos::abs(delta) <= tol) {
                        converged = true;
                        break;
                    }
                }
                if (!converged || z <= lo || z >= hi || isDuplicateAmong(nodes + 1, j, z, sepTol)) {
                    continue;
                }
                accepted = true;
                break;
            }

            if (!accepted) {
                // Sample P' for a sign change on (lo, hi), then Brent.
                auto fprime = [&](Scalar x) {
                    Scalar p, dp;
                    legendreAndDeriv(deg, x, p, dp);
                    return dp;
                };
                Scalar blo = lo;
                Scalar bhi = hi;
                Scalar fprev = fprime(blo);
                bool found = false;
                for (int s = 1; s <= 128; ++s) {
                    const Scalar x =
                        lo + (hi - lo) * (static_cast<Scalar>(s) / Scalar(128));
                    const Scalar fx = fprime(x);
                    if (fprev * fx <= Scalar(0)) {
                        blo   = lo + (hi - lo) * (static_cast<Scalar>(s - 1) / Scalar(128));
                        bhi   = x;
                        found = true;
                        break;
                    }
                    fprev = fx;
                }
                if (!found) {
                    blo = lo;
                    bhi = Scalar(1) - Scalar(1e-14);
                    fprev = fprime(blo);
                    for (int s = 1; s <= 256; ++s) {
                        const Scalar x =
                            blo + (bhi - blo) * (static_cast<Scalar>(s) / Scalar(256));
                        const Scalar fx = fprime(x);
                        if (fprev * fx <= Scalar(0)) {
                            const Scalar left =
                                blo + (bhi - blo) * (static_cast<Scalar>(s - 1) / Scalar(256));
                            blo   = left;
                            bhi   = x;
                            found = true;
                            break;
                        }
                        fprev = fx;
                    }
                }
                if (!found) {
                    throw std::runtime_error(
                        "nodes1d::computeGaussLobatto (Newton): could not bracket interior j="
                        + std::to_string(j) + ", n=" + std::to_string(n));
                }
                z = brentLegendreDerivRoot(deg, blo, bhi);
                if (z <= lo || isDuplicateAmong(nodes + 1, j, z, sepTol)) {
                    throwDuplicateNodes("nodes1d::computeGaussLobatto (Newton)", n, j + 1, j,
                                        static_cast<double>(z),
                                        static_cast<double>(nodes[j]));
                }
            }
            nodes[j + 1] = z;
        }

        assertDistinctSortedNodes("nodes1d::computeGaussLobatto (Newton)", nodes, n, sepTol);
        fillGLLWeights(n, nodes, weights);
    }

}  // namespace detail

    /** @copydoc computeGaussLobatto */
    template <typename Scalar>
    void computeGaussLobatto(std::size_t n, Scalar* nodes, Scalar* weights,
                             std::size_t maxNewtonIterations, std::size_t minNewtonIterations,
                             RootFinderMethod method) {
        assert(n >= 2);
        assert(nodes != nullptr && weights != nullptr);

        if (method == RootFinderMethod::Newton) {
            detail::computeGaussLobattoNewton(n, nodes, weights, maxNewtonIterations,
                                              minNewtonIterations);
            return;
        }
        detail::computeGaussLobattoGolubWelsch(n, nodes, weights,
                                               method == RootFinderMethod::DenseGolubWelsch);
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
