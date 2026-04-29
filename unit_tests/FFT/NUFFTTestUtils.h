//
// NUFFTTestUtils.h
//   Shared utilities for NUFFT unit tests (dim-agnostic for Type-1 and Type-2).
//
#ifndef IPPL_NUFFT_TEST_UTILS_H
#define IPPL_NUFFT_TEST_UTILS_H

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Random.hpp>

#include "Utility/ParameterList.h"

namespace ippl {
    namespace test {

        template <typename T, class PLayout>
        struct Bunch : public ippl::ParticleBase<PLayout> {
            Bunch(PLayout& playout)
                : ippl::ParticleBase<PLayout>(playout) {
                this->addAttribute(Q);
            }

            using charge_container_type = ippl::ParticleAttrib<T>;
            charge_container_type Q;
        };

        // MPI datatype helper for SUM/MAX over scalar T
        template <typename T>
        constexpr MPI_Datatype mpiDatatypeFor() {
            if constexpr (std::is_same_v<T, float>)
                return MPI_FLOAT;
            else
                return MPI_DOUBLE;
        }

        template <unsigned Dim>
        class IndexUtils {
        public:
            static bool isOwnedLocally(const ippl::NDIndex<Dim>& lDom,
                                       const ippl::Vector<int, Dim>& globalIdx) {
                for (unsigned d = 0; d < Dim; ++d) {
                    if (globalIdx[d] < lDom[d].first() || globalIdx[d] > lDom[d].last()) {
                        return false;
                    }
                }
                return true;
            }

            static ippl::Vector<int, Dim> globalToLocal(const ippl::NDIndex<Dim>& lDom,
                                                        const ippl::Vector<int, Dim>& globalIdx,
                                                        int nghost) {
                ippl::Vector<int, Dim> localIdx;
                for (unsigned d = 0; d < Dim; ++d) {
                    localIdx[d] = globalIdx[d] - lDom[d].first() + nghost;
                }
                return localIdx;
            }

            // Centered k in [-N/2, N/2-1] -> corner-DC k in [0, N-1]
            // (or [0, 2*N-1] when use_upsampling).
            static ippl::Vector<int, Dim> centeredToCornerDC(const ippl::Vector<int, Dim>& kVec,
                                                             const ippl::Vector<int, Dim>& nModes,
                                                             bool useUpsampling = false) {
                ippl::Vector<int, Dim> cornerIdx;
                for (unsigned d = 0; d < Dim; ++d) {
                    if (kVec[d] >= 0) {
                        cornerIdx[d] = kVec[d];
                    } else {
                        cornerIdx[d] = (useUpsampling ? 2 : 1) * nModes[d] + kVec[d];
                    }
                }
                return cornerIdx;
            }
        };

        // Read field at a Dim-dependent local index. Indices must be local + ghost-shifted.
        template <typename FieldView, unsigned Dim>
        KOKKOS_INLINE_FUNCTION auto readFieldAt(const FieldView& field,
                                                const ippl::Vector<int, Dim>& localIdx) {
            if constexpr (Dim == 3) {
                return field(localIdx[0], localIdx[1], localIdx[2]);
            } else if constexpr (Dim == 2) {
                return field(localIdx[0], localIdx[1]);
            } else {
                return field(localIdx[0]);
            }
        }

        template <typename T, unsigned Dim>
        class DFTReference {
        public:
            // Type-1 single mode: f_k = sum_j Q_j * exp(-i k . x_j)
            template <typename PosView, typename ChargeView>
            static Kokkos::complex<T> computeType1ModeLocal(const PosView& R, const ChargeView& Q,
                                                            const ippl::Vector<int, Dim>& kVec,
                                                            const ippl::Vector<T, Dim>& hx,
                                                            const ippl::Vector<int, Dim>& nModes,
                                                            size_t nloc) {
                const T pi = Kokkos::numbers::pi_v<T>;
                Kokkos::complex<T> dftLocal(0.0, 0.0);
                const Kokkos::complex<T> imag = {0.0, 1.0};

                Kokkos::parallel_reduce(
                    "DFT_Type1_Local", nloc,
                    KOKKOS_LAMBDA(const size_t idx, Kokkos::complex<T>& val) {
                        T arg = 0.0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            arg += (2 * pi / (hx[d] * nModes[d])) * kVec[d] * R(idx)[d];
                        }
                        val += (Kokkos::cos(arg) - imag * Kokkos::sin(arg)) * Q(idx);
                    },
                    Kokkos::Sum<Kokkos::complex<T>>(dftLocal));

                return dftLocal;
            }

            template <typename PosView, typename ChargeView>
            static Kokkos::complex<T> computeType1Mode(const PosView& R, const ChargeView& Q,
                                                       const ippl::Vector<int, Dim>& kVec,
                                                       const ippl::Vector<T, Dim>& hx,
                                                       const ippl::Vector<int, Dim>& nModes,
                                                       size_t nloc) {
                auto dftLocal = computeType1ModeLocal(R, Q, kVec, hx, nModes, nloc);
                T sendBuf[2]  = {dftLocal.real(), dftLocal.imag()};
                T recvBuf[2]  = {0.0, 0.0};
                MPI_Allreduce(sendBuf, recvBuf, 2, mpiDatatypeFor<T>(), MPI_SUM,
                              ippl::Comm->getCommunicator());
                return Kokkos::complex<T>(recvBuf[0], recvBuf[1]);
            }

            // Type-2 at single particle position: q(x) = sum_k f_k * exp(+i k . x)
            // Iterates the local field domain; outer caller does an MPI reduction.
            template <typename FieldView>
            static Kokkos::complex<T> computeType2ValueLocal(const FieldView& field,
                                                             const ippl::Vector<T, Dim>& testPos,
                                                             const ippl::NDIndex<Dim>& lDom,
                                                             const ippl::Vector<T, Dim>& /*hx*/,
                                                             const ippl::Vector<int, Dim>& nModes,
                                                             int nghost) {
                Kokkos::complex<T> dftLocal(0.0, 0.0);
                const Kokkos::complex<T> imag = {0.0, 1.0};

                ippl::Vector<int, Dim> localStart, localExtent;
                for (unsigned d = 0; d < Dim; ++d) {
                    localStart[d]  = lDom[d].first();
                    localExtent[d] = lDom[d].length();
                }

                auto testPosD = testPos;
                auto nModesD  = nModes;

                if constexpr (Dim == 3) {
                    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
                    Kokkos::parallel_reduce(
                        "DFT_Type2_Local",
                        mdrange_type({0, 0, 0},
                                     {localExtent[0], localExtent[1], localExtent[2]}),
                        KOKKOS_LAMBDA(const int li, const int lj, const int lk,
                                      Kokkos::complex<T>& val) {
                            int gi  = li + localStart[0];
                            int gj  = lj + localStart[1];
                            int gk  = lk + localStart[2];
                            int kc0 = (gi < nModesD[0] / 2 ? gi : gi - nModesD[0]);
                            int kc1 = (gj < nModesD[1] / 2 ? gj : gj - nModesD[1]);
                            int kc2 = (gk < nModesD[2] / 2 ? gk : gk - nModesD[2]);
                            T arg   = kc0 * testPosD[0] + kc1 * testPosD[1] + kc2 * testPosD[2];
                            auto fk = field(li + nghost, lj + nghost, lk + nghost);
                            val += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
                        },
                        Kokkos::Sum<Kokkos::complex<T>>(dftLocal));
                } else if constexpr (Dim == 2) {
                    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
                    Kokkos::parallel_reduce(
                        "DFT_Type2_Local",
                        mdrange_type({0, 0}, {localExtent[0], localExtent[1]}),
                        KOKKOS_LAMBDA(const int li, const int lj, Kokkos::complex<T>& val) {
                            int gi  = li + localStart[0];
                            int gj  = lj + localStart[1];
                            int kc0 = (gi < nModesD[0] / 2 ? gi : gi - nModesD[0]);
                            int kc1 = (gj < nModesD[1] / 2 ? gj : gj - nModesD[1]);
                            T arg   = kc0 * testPosD[0] + kc1 * testPosD[1];
                            auto fk = field(li + nghost, lj + nghost);
                            val += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
                        },
                        Kokkos::Sum<Kokkos::complex<T>>(dftLocal));
                } else {
                    Kokkos::parallel_reduce(
                        "DFT_Type2_Local", localExtent[0],
                        KOKKOS_LAMBDA(const int li, Kokkos::complex<T>& val) {
                            int gi  = li + localStart[0];
                            int kc0 = (gi < nModesD[0] / 2 ? gi : gi - nModesD[0]);
                            T arg   = kc0 * testPosD[0];
                            auto fk = field(li + nghost);
                            val += (Kokkos::cos(arg) + imag * Kokkos::sin(arg)) * fk;
                        },
                        Kokkos::Sum<Kokkos::complex<T>>(dftLocal));
                }

                return dftLocal;
            }

            template <typename FieldView>
            static Kokkos::complex<T> computeType2Value(const FieldView& field,
                                                        const ippl::Vector<T, Dim>& testPos,
                                                        const ippl::NDIndex<Dim>& lDom,
                                                        const ippl::Vector<T, Dim>& hx,
                                                        const ippl::Vector<int, Dim>& nModes,
                                                        int nghost) {
                auto dftLocal = computeType2ValueLocal(field, testPos, lDom, hx, nModes, nghost);
                T sendBuf[2]  = {dftLocal.real(), dftLocal.imag()};
                T recvBuf[2]  = {0.0, 0.0};
                MPI_Allreduce(sendBuf, recvBuf, 2, mpiDatatypeFor<T>(), MPI_SUM,
                              ippl::Comm->getCommunicator());
                return Kokkos::complex<T>(recvBuf[0], recvBuf[1]);
            }
        };

        struct NUFFTParams {
            template <typename T>
            static ippl::ParameterList createNativeParams(
                T tolerance, bool useUpsampling, const std::string& spreadMethod = "atomic",
                const std::string& gatherMethod = "atomic_sort") {
                ippl::ParameterList params;
                params.add("tolerance", tolerance);
                params.add("use_upsampled_inputs", useUpsampling);
                params.add("use_finufft", false);
                params.add("use_kokkos_nufft", false);
                params.add("spread_method", spreadMethod);
                params.add("gather_method", gatherMethod);
                params.add("sort", true);
                params.add("tile_size_3d", 6);
                params.add("z_tiles", 1);

#ifdef ENABLE_GPU_NUFFT
                params.add("gpu_method", 1);
                params.add("gpu_sort", 0);
                params.add("gpu_kerevalmeth", 1);
#else
                params.add("spread_kerevalmeth", 1);
                params.add("spread_sort", 2);
                params.add("nthreads", 0);
#endif

                return params;
            }

#ifdef ENABLE_FINUFFT
            template <typename T>
            static ippl::ParameterList createFinufftParams(T tolerance, bool useUpsampling) {
                ippl::ParameterList params;
                params.add("tolerance", tolerance);
                params.add("use_upsampled_inputs", useUpsampling);
                params.add("use_finufft", true);
                params.add("use_finufft_defaults", false);
                params.add("use_kokkos_nufft", false);

#ifdef ENABLE_GPU_NUFFT
                params.add("gpu_method", 1);
                params.add("gpu_sort", 0);
                params.add("gpu_kerevalmeth", 1);
#else
                params.add("spread_kerevalmeth", 1);
                params.add("spread_sort", 2);
                params.add("nthreads", 0);
#endif

                return params;
            }
#endif
        };

        template <typename T>
        struct ErrorMetrics {
            T absError;
            T relError;

            static ErrorMetrics compute(const Kokkos::complex<T>& expected,
                                        const Kokkos::complex<T>& actual) {
                ErrorMetrics metrics;
                T expectedMag    = Kokkos::abs(expected);
                T diff           = Kokkos::abs(expected - actual);
                metrics.absError = diff;
                metrics.relError = (expectedMag > 0) ? (diff / expectedMag) : diff;
                return metrics;
            }
        };

    }  // namespace test
}  // namespace ippl

#endif  // IPPL_NUFFT_TEST_UTILS_H
