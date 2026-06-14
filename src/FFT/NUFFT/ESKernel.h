/*!
 * @file ESKernel.h
 * @brief Exponential-of-Semicircle (ES) spreading kernel for NUFFT.
 *
 * The header exposes:
 *   - @c es_kernel_eval_wN<T> device-friendly polynomial evaluators for
 *     each precomputed kernel half-width N (3..15).
 *   - @c ESKernel<T> a runtime selector that picks the smallest width
 *     achieving the requested tolerance and dispatches to the matching
 *     polynomial.
 *
 * Coefficients are minimax polynomials in t=x^2 with magic literals encoded
 * as hex floating-point so they are exact across compilers.
 */
#ifndef IPPL_NUFFT_ES_KERNEL_H
#define IPPL_NUFFT_ES_KERNEL_H

#include <Kokkos_Core.hpp>

#include <cmath>

namespace ippl {
    namespace nufft {

        // ============================================================
        // ES Kernel polynomial evaluation for each width
        // Coefficients are inlined to work in device code
        // ============================================================

        /*!
         * @brief Polynomial evaluator for the ES kernel at width @c w=4
         *        (beta = 9.2, degree = 5, max error < 3.53e-4).
         *
         * @tparam T Floating-point precision (float / double).
         * @param  x Argument in [0, 1] (the kernel is even, so callers pass |x|).
         * @return Polynomial approximation of @c exp(beta*(sqrt(1-x^2)-1)).
         */
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w4(T x) {
            const T t = x * x;
            T r       = T(-0x1.7a61695c211e2p0);
            r         = r * t + T(0x1.7741c5626c40bp2);
            r         = r * t + T(-0x1.3ccc694d94a47p3);
            r         = r * t + T(0x1.22ccba2dfd6a5p3);
            r         = r * t + T(-0x1.24a8469892c07p2);
            r         = r * t + T(0x1.ffd1c0e2c6db8p-1);
            return r;
        }

        // w = 5, beta = 11.5, degree = 7, error < 1.74e-5
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w5(T x) {
            const T t = x * x;
            T r       = T(-0x1.53ef7dfa4cddap0);
            r         = r * t + T(0x1.c1971511f9acbp2);
            r         = r * t + T(-0x1.0ed3008908634p4);
            r         = r * t + T(0x1.8a1fbea0250d2p4);
            r         = r * t + T(-0x1.7b28be8632757p4);
            r         = r * t + T(0x1.e1574df952c56p3);
            r         = r * t + T(-0x1.6fd9a0a46e05ap2);
            r         = r * t + T(0x1.fffdb8228ab29p-1);
            return r;
        }

        // w = 6, beta = 13.8, degree = 9, error < 8.50e-7
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w6(T x) {
            const T t = x * x;
            T r       = T(-0x1.2290d68e62bcfp0);
            r         = r * t + T(0x1.d9edacd03ff7ep2);
            r         = r * t + T(-0x1.6afaba0b81c05p4);
            r         = r * t + T(0x1.5e4576519fd7ep5);
            r         = r * t + T(-0x1.dd17e73ccea97p5);
            r         = r * t + T(0x1.ddcc59a8e52adp5);
            r         = r * t + T(-0x1.5d0a435b3428ep5);
            r         = r * t + T(0x1.612ebf7fd3207p4);
            r         = r * t + T(-0x1.b996b31b5ba67p2);
            r         = r * t + T(0x1.ffffe37649033p-1);
            return r;
        }

        // w = 7, beta = 16.1, degree = 10, error < 4.52e-7
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w7(T x) {
            const T t = x * x;
            T r       = T(0x1.0d43fb941e8cp1);
            r         = r * t + T(-0x1.d07f762f83e4cp3);
            r         = r * t + T(0x1.76aa8fe35ec4ep5);
            r         = r * t + T(-0x1.7b643526b5bfcp6);
            r         = r * t + T(0x1.0ff008e09251bp7);
            r         = r * t + T(-0x1.23a681af8ca75p7);
            r         = r * t + T(0x1.da7c21b010748p6);
            r         = r * t + T(-0x1.1eb0a0c2150d6p6);
            r         = r * t + T(0x1.e6250a3d3d121p4);
            r         = r * t + T(-0x1.0198abec8a4eep3);
            r         = r * t + T(0x1.fffff0d9a4617p-1);
            return r;
        }

        // w = 8, beta = 18.4, degree = 12, error < 2.33e-8
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w8(T x) {
            const T t = x * x;
            T r       = T(0x1.d5cd3d519edbcp0);
            r         = r * t + T(-0x1.da1032722476ep3);
            r         = r * t + T(0x1.c5cddddf485b9p5);
            r         = r * t + T(-0x1.1533735f3e975p7);
            r         = r * t + T(0x1.e9cf3a5cff485p7);
            r         = r * t + T(-0x1.4e105fc4aa621p8);
            r         = r * t + T(0x1.6a81ebd6683f4p8);
            r         = r * t + T(-0x1.3a18274b0a1e3p8);
            r         = r * t + T(0x1.ab15412d571b6p7);
            r         = r * t + T(-0x1.b70af68ed9b3fp6);
            r         = r * t + T(0x1.4028006f3c69bp5);
            r         = r * t + T(-0x1.26665564ca36p3);
            r         = r * t + T(0x1.ffffff37dec0bp-1);
            return r;
        }

        // w = 9, beta = 20.7, degree = 14, error < 1.20e-9
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w9(T x) {
            const T t = x * x;
            T r       = T(0x1.90982143e4ae3p0);
            r         = r * t + T(-0x1.cd7480dda6f42p3);
            r         = r * t + T(0x1.fcc7e2c6cba0dp5);
            r         = r * t + T(-0x1.694ede76783d8p7);
            r         = r * t + T(0x1.7752cfcde75a6p8);
            r         = r * t + T(-0x1.31e18cfaaedc8p9);
            r         = r * t + T(0x1.968a12dc2f151p9);
            r         = r * t + T(-0x1.bf8cb2e1ddc74p9);
            r         = r * t + T(0x1.97659066136f6p9);
            r         = r * t + T(-0x1.2e19d0407765dp9);
            r         = r * t + T(0x1.6374318a4696bp8);
            r         = r * t + T(-0x1.3e98a86653c26p7);
            r         = r * t + T(0x1.97ca274772c9fp5);
            r         = r * t + T(-0x1.4b33320a95fccp3);
            r         = r * t + T(0x1.fffffff5b40bp-1);
            return r;
        }

        // w = 10, beta = 23.0, degree = 16, error < 6.20e-11
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w10(T x) {
            const T t = x * x;
            T r       = T(0x1.5017876c93c63p0);
            r         = r * t + T(-0x1.b266030e0096ap3);
            r         = r * t + T(0x1.0e462bf649f05p6);
            r         = r * t + T(-0x1.b3a48c1d9288p7);
            r         = r * t + T(0x1.025a87a6770c9p9);
            r         = r * t + T(-0x1.e4e7636a1af58p9);
            r         = r * t + T(0x1.780fc81f87388p10);
            r         = r * t + T(-0x1.ed395a8d098afp10);
            r         = r * t + T(0x1.1373eb09a6122p11);
            r         = r * t + T(-0x1.04dade35c1013p11);
            r         = r * t + T(0x1.9d913dd2b0b9ep10);
            r         = r * t + T(-0x1.0d05b6008f472p10);
            r         = r * t + T(0x1.1733f862d30f6p9);
            r         = r * t + T(-0x1.bbb54324b711ap7);
            r         = r * t + T(0x1.f9fffe1fb327ap5);
            r         = r * t + T(-0x1.6fffffec5ff52p3);
            r         = r * t + T(0x1.ffffffff77951p-1);
            return r;
        }

        // w = 11, beta = 25.3, degree = 17, error < 3.20e-11
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w11(T x) {
            const T t = x * x;
            T r       = T(-0x1.59009930566c2p1);
            r         = r * t + T(0x1.cd02dcaa304dcp4);
            r         = r * t + T(-0x1.27b6792d84287p7);
            r         = r * t + T(0x1.e93b8598d8732p8);
            r         = r * t + T(-0x1.2827b4bce0869p10);
            r         = r * t + T(0x1.1a3cd50b8ebb6p11);
            r         = r * t + T(-0x1.bb6bf72d60cf3p11);
            r         = r * t + T(0x1.27231ed89e683p12);
            r         = r * t + T(-0x1.5123d0d1023adp12);
            r         = r * t + T(0x1.4ad53cddfcaf1p12);
            r         = r * t + T(-0x1.14cc6d44fbd4ap12);
            r         = r * t + T(0x1.856985c4590bep11);
            r         = r * t + T(-0x1.c3591519272cep10);
            r         = r * t + T(0x1.a2f8d49451968p9);
            r         = r * t + T(-0x1.2af4cfa9b5757p8);
            r         = r * t + T(0x1.33651e1b6dc6p6);
            r         = r * t + T(-0x1.94ccccc16875dp3);
            r         = r * t + T(0x1.ffffffffb99d9p-1);
            return r;
        }

        // w = 12, beta = 27.6, degree = 19, error < 1.65e-12
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w12(T x) {
            const T t = x * x;
            T r       = T(-0x1.279db73ec08b8p1);
            r         = r * t + T(0x1.b36c753c2aa7dp4);
            r         = r * t + T(-0x1.3537a8d5dacp7);
            r         = r * t + T(0x1.1c4ad5335b3c1p9);
            r         = r * t + T(-0x1.7febf4612070ap10);
            r         = r * t + T(0x1.99bccfa2b45ffp11);
            r         = r * t + T(-0x1.6a7e6841cdd94p12);
            r         = r * t + T(0x1.1229b3df0dde7p13);
            r         = r * t + T(-0x1.68b72ed61ef3ap13);
            r         = r * t + T(0x1.9f70e8e221323p13);
            r         = r * t + T(-0x1.a21e19c0501eap13);
            r         = r * t + T(0x1.6d0860e26b0f4p13);
            r         = r * t + T(-0x1.11406ebe54d37p13);
            r         = r * t + T(0x1.594e7a8cbab51p12);
            r         = r * t + T(-0x1.68bd26f203d21p11);
            r         = r * t + T(0x1.2ed3db062aa1p10);
            r         = r * t + T(-0x1.8820826cc8dc9p8);
            r         = r * t + T(0x1.6f147ad4fda94p6);
            r         = r * t + T(-0x1.b9999998e030ap3);
            r         = r * t + T(0x1.fffffffffc5dcp-1);
            return r;
        }

        // w = 13, beta = 29.9, degree = 21, error < 8.48e-14
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w13(T x) {
            const T t = x * x;
            T r       = T(-0x1.f5b0db6099cd3p0);
            r         = r * t + T(0x1.93556b5289c77p4);
            r         = r * t + T(-0x1.39b591914c767p7);
            r         = r * t + T(0x1.3cc43c9c9dd5p9);
            r         = r * t + T(-0x1.d6e0b91c3f952p10);
            r         = r * t + T(0x1.1530d489678cfp12);
            r         = r * t + T(-0x1.0f58cc3b57c6fp13);
            r         = r * t + T(0x1.c8810a3e2b99cp13);
            r         = r * t + T(-0x1.50d1cdb6d5abcp14);
            r         = r * t + T(0x1.b8757eaf1ebap14);
            r         = r * t + T(-0x1.ffa40c1c3b2a6p14);
            r         = r * t + T(0x1.07216db3d2af2p15);
            r         = r * t + T(-0x1.dc0307c1e538p14);
            r         = r * t + T(0x1.76f66049cee1ap14);
            r         = r * t + T(-0x1.fc0d2356bef94p13);
            r         = r * t + T(0x1.2354fa3750e5cp13);
            r         = r * t + T(-0x1.14f973d66e8afp12);
            r         = r * t + T(0x1.a85e5aa22f6bap10);
            r         = r * t + T(-0x1.f6e308d0ea5eap8);
            r         = r * t + T(0x1.b00e1479f5c26p6);
            r         = r * t + T(-0x1.de6666665ae9p3);
            r         = r * t + T(0x1.ffffffffffd04p-1);
            return r;
        }

        // w = 14, beta = 32.2, degree = 22, error < 4.69e-14
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w14(T x) {
            const T t = x * x;
            T r       = T(0x1.0354efce10e2dp2);
            r         = r * t + T(-0x1.ae25418562cf8p5);
            r         = r * t + T(0x1.58827d3735653p8);
            r         = r * t + T(-0x1.65245773d1613p10);
            r         = r * t + T(0x1.0f606d2c65324p12);
            r         = r * t + T(-0x1.44f69b9ed1fe3p13);
            r         = r * t + T(0x1.41dfc1b9c2d3ap14);
            r         = r * t + T(-0x1.10df39932e40cp15);
            r         = r * t + T(0x1.9536924063e52p15);
            r         = r * t + T(-0x1.0b15d6e06f48p16);
            r         = r * t + T(0x1.3a48e2051a128p16);
            r         = r * t + T(-0x1.49fabd3679aeep16);
            r         = r * t + T(0x1.33b8d92983ebfp16);
            r         = r * t + T(-0x1.f9f712f3c39a1p15);
            r         = r * t + T(0x1.6b15557642d58p15);
            r         = r * t + T(-0x1.c13113d9b8e7fp14);
            r         = r * t + T(0x1.d7772ec3bd135p13);
            r         = r * t + T(-0x1.9b2d52745b9d6p12);
            r         = r * t + T(0x1.21a51b7b71831p11);
            r         = r * t + T(-0x1.3c60dfe4bed5dp9);
            r         = r * t + T(0x1.f651eb8483554p6);
            r         = r * t + T(-0x1.019999999621ep4);
            r         = r * t + T(0x1.ffffffffffe5bp-1);
            return r;
        }

        // w = 15, beta = 34.5, degree = 24, error < 2.46e-15
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval_w15(T x) {
            const T t = x * x;
            T r       = T(0x1.bcc3fd37452d5p1);
            r         = r * t + T(-0x1.8eb424f00a28dp5);
            r         = r * t + T(0x1.5a19d25b45029p8);
            r         = r * t + T(-0x1.85c332c280f8dp10);
            r         = r * t + T(0x1.424ce09ce2f3ap12);
            r         = r * t + T(-0x1.a49341a23526dp13);
            r         = r * t + T(0x1.c69265bf4a9d7p14);
            r         = r * t + T(-0x1.a5637540ea295p15);
            r         = r * t + T(0x1.57650b877caaep16);
            r         = r * t + T(-0x1.f3de53de15289p16);
            r         = r * t + T(0x1.47a7b42596f6ep17);
            r         = r * t + T(-0x1.83c339a13039ep17);
            r         = r * t + T(0x1.9d812c92ca63fp17);
            r         = r * t + T(-0x1.8b81cbd899304p17);
            r         = r * t + T(0x1.51124b72fffedp17);
            r         = r * t + T(-0x1.fbb1cbaa2b528p16);
            r         = r * t + T(0x1.4e6e67a82c9d4p16);
            r         = r * t + T(-0x1.7c8f736d5aefp15);
            r         = r * t + T(0x1.70230df3fd36ap14);
            r         = r * t + T(-0x1.28835f02a5b9bp13);
            r         = r * t + T(0x1.829acbfacfe6bp11);
            r         = r * t + T(-0x1.87a0ffff89722p9);
            r         = r * t + T(0x1.20effffffa61ap7);
            r         = r * t + T(-0x1.13ffffffffc9fp4);
            r         = r * t + T(0x1.fffffffffffebp-1);
            return r;
        }

        // ============================================================
        // Runtime width dispatch
        // ============================================================

        /**
         * @brief Runtime width dispatch for ES kernel evaluation
         * @param x Evaluation point in [0, 1]
         * @param w Kernel width (4-15)
         * @return Approximation to exp(beta * (sqrt(1-x^2) - 1))
         */
        template <typename T>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval(T x, int w) {
            switch (w) {
                case 4:
                    return es_kernel_eval_w4(x);
                case 5:
                    return es_kernel_eval_w5(x);
                case 6:
                    return es_kernel_eval_w6(x);
                case 7:
                    return es_kernel_eval_w7(x);
                case 8:
                    return es_kernel_eval_w8(x);
                case 9:
                    return es_kernel_eval_w9(x);
                case 10:
                    return es_kernel_eval_w10(x);
                case 11:
                    return es_kernel_eval_w11(x);
                case 12:
                    return es_kernel_eval_w12(x);
                case 13:
                    return es_kernel_eval_w13(x);
                case 14:
                    return es_kernel_eval_w14(x);
                case 15:
                    return es_kernel_eval_w15(x);
                default:
                    // Fallback to exact evaluation for unsupported widths
                    return Kokkos::exp(T(2.30) * w * (Kokkos::sqrt(T(1) - x * x) - T(1)));
            }
        }

        // ============================================================
        // Compile-time width dispatch (template version)
        // ============================================================

        /*!
         * @brief Compile-time width dispatch for ES kernel evaluation.
         *
         * The compiler can fold a single branch when @p W is known, removing
         * the runtime switch in @c es_kernel_eval(T,int).
         *
         * @tparam T Floating-point precision.
         * @tparam W Kernel width (4..15); other values fall back to exact eval.
         * @param  x Argument in [0, 1].
         */
        template <typename T, int W>
        KOKKOS_INLINE_FUNCTION T es_kernel_eval(T x) {
            if constexpr (W == 4) {
                return es_kernel_eval_w4(x);
            } else if constexpr (W == 5) {
                return es_kernel_eval_w5(x);
            } else if constexpr (W == 6) {
                return es_kernel_eval_w6(x);
            } else if constexpr (W == 7) {
                return es_kernel_eval_w7(x);
            } else if constexpr (W == 8) {
                return es_kernel_eval_w8(x);
            } else if constexpr (W == 9) {
                return es_kernel_eval_w9(x);
            } else if constexpr (W == 10) {
                return es_kernel_eval_w10(x);
            } else if constexpr (W == 11) {
                return es_kernel_eval_w11(x);
            } else if constexpr (W == 12) {
                return es_kernel_eval_w12(x);
            } else if constexpr (W == 13) {
                return es_kernel_eval_w13(x);
            } else if constexpr (W == 14) {
                return es_kernel_eval_w14(x);
            } else if constexpr (W == 15) {
                return es_kernel_eval_w15(x);
            } else {
                // Fallback to exact evaluation for unsupported widths
                return Kokkos::exp(T(2.30) * W * (Kokkos::sqrt(T(1) - x * x) - T(1)));
            }
        }

        // ============================================================
        // ESKernel class
        // ============================================================

        /**
         * @brief Exponential-of-Semicircle (ES) spreading kernel.
         *
         * The ES kernel is defined as:
         *   phi(x) = exp(beta * (sqrt(1 - x^2) - 1))  for |x| < 1
         *          = 0                                  otherwise
         *
         * Width (w) and beta are derived from a user-chosen error tolerance.
         *
         * @tparam T Floating point type (float or double)
         */
        template <typename T = double>
        class ESKernel {
        public:
            static constexpr bool has_width_template = true;
            // Upper bound on the runtime width; matches the precomputed
            // ES-kernel polynomial expansions (w = 4..15) used by NativeNUFFT.
            static constexpr int max_width = 15;
            using value_type = T;

            static constexpr T default_tol = T(1e-10);
            static constexpr T beta_factor = T(2.30);

            /**
             * @brief Construct kernel with given tolerance.
             * @param tol Error tolerance for NUFFT accuracy
             */
            KOKKOS_INLINE_FUNCTION explicit ESKernel(T tol = default_tol)
                : w_(static_cast<int>(Kokkos::ceil(Kokkos::log10(T(1.0) / tol))) + 1)
                , beta_(beta_factor * w_)
                , tol_(tol) {}

            /**
             * @brief Construct kernel with explicit width and beta.
             * @param width Kernel width (number of grid points)
             * @param beta Beta parameter controlling decay
             */
            KOKKOS_INLINE_FUNCTION ESKernel(int width, T beta)
                : w_(width)
                , beta_(beta)
                , tol_(default_tol) {}

            /**
             * @brief Evaluate the ES kernel at position x in [-1, 1].
             * @param x Normalized position
             * @return Normalized kernel value
             */
            KOKKOS_INLINE_FUNCTION T operator()(T x) const {
                x = Kokkos::abs(x);
                return x >= T(1.0) ? T(0.0) : es_kernel_eval(x, w_);
            }

            /*!
             * @brief Evaluate the ES kernel using compile-time width @c W.
             * @tparam W Kernel width matching one of the precomputed polynomials.
             */
            template<int W>
            KOKKOS_INLINE_FUNCTION T eval(T x) const {
                return x >= T(1.0) ? T(0.0) : es_kernel_eval<T, W>(x);
            }

            //! @return Runtime kernel half-width @c w (in grid points).
            KOKKOS_INLINE_FUNCTION int width() const { return w_; }
            //! @return Decay parameter @c beta = beta_factor * w.
            KOKKOS_INLINE_FUNCTION T beta() const { return beta_; }
            //! @return Target relative error tolerance.
            KOKKOS_INLINE_FUNCTION T tol() const { return tol_; }

        private:
            int w_;
            T beta_;
            T tol_;
        };

    }  // namespace NUFFT
}  // namespace ippl


#endif  // IPPL_NUFFT_ES_KERNEL_H
