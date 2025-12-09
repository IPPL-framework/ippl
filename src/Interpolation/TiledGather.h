#ifndef IPPL_TILED_GATHER_H
#define IPPL_TILED_GATHER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            __device__ inline double shfl_down(double val, int offs) {
#ifdef KOKKOS_ENABLE_CUDA
                return __shfl_down_sync(0xffffffff, val, offs);
#endif
#ifdef KOKKOS_ENABLE_HIP
                return __shfl_down(val, offs);
#endif
            }
#ifdef KOKKOS_ENABLE_CUDA
            using device_exec     = Kokkos::Cuda;
            using device_memspace = Kokkos::CudaSpace;
#endif
#ifdef KOKKOS_ENABLE_HIP
            using device_exec     = Kokkos::HIP;
            using device_memspace = Kokkos::HIPSpace;
#endif

            /**
             * @brief Kernel for 3D tiled gather (interpolation) with warp-level
             * parallelism
             *
             * Template parameters:
             * @tparam w The kernel width (compile-time constant for optimization)
             * @tparam RealType The floating point type (float or double)
             * @tparam PositionViewType The position view type
             * @tparam FieldViewType The field view type
             * @tparam KernelType The kernel function type
             * @tparam ValueType The type of values being gathered
             */
            template <int w, typename RealType, typename PositionViewType, typename PermuteViewType,
                      typename FieldViewType, typename KernelType, typename ValueType>
            __global__ void tiled_gather_3d_cuda_kernel(
                PositionViewType x,        // Particle positions in PHYSICAL coordinates [-pi, pi]
                PermuteViewType permute,   // Permutation array
                FieldViewType field_view,  // Input field with ghosts (LOCAL view)
                Kokkos::View<ValueType*, device_memspace> output,  // Output values
                int64_t n_points, int nghost,
                Vector<int, 3>
                    n_grid_global,  // GLOBAL grid dimensions (for coordinate transformation)
                Vector<int, 3> n_grid_local,  // LOCAL grid dimensions (owned by this rank)
                Vector<int, 3> local_offset,  // First global index of local domain
                RealType inv_hw,              // 1 / half_width for kernel scaling
                KernelType kernel, bool add_to_attribute) {
                constexpr bool is_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
                constexpr int w3          = w * w * w;
#ifdef KOKKOS_ENABLE_CUDA
                constexpr int WARP_SIZE = 32;
#elif defined(KOKKOS_ENABLE_HIP)
                constexpr int WARP_SIZE = 64;
#endif

                // Determine which particle this warp is processing
                const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
                if (warp_id >= n_points)
                    return;

                const int lane_id       = threadIdx.x % WARP_SIZE;
                const int warp_in_block = threadIdx.x / WARP_SIZE;

                // Apply permutation to get actual particle index
                const int particle_idx = permute(warp_id);

                // Helper to access position component
                auto get_pos = [&](int d) -> RealType {
                    if constexpr (std::is_same_v<PositionViewType,
                                                 Kokkos::View<RealType* [3], device_memspace>>) {
                        return x(particle_idx, d);
                    } else {
                        return x(particle_idx)[d];
                    }
                };

                // Shared memory: kernel values for each warp in the block
                constexpr int max_warps_per_block = 256 / WARP_SIZE;
                __shared__ RealType ker_shared[max_warps_per_block][3 * w];

                // Transform positions to grid coordinates using helper function
                RealType pos[3];
                int idx0_global[3];

                for (int d = 0; d < 3; ++d) {
                    pos[d] = scale_to_grid_indices(get_pos(d), n_grid_global[d]);
                    idx0_global[d] =
                        grid_point_to_grid_idx(pos[d], n_grid_global[d], w) - (w - 1) / 2;
                }

                // Collaboratively compute kernel values within the warp
                for (int i = lane_id; i < 3 * w; i += WARP_SIZE) {
                    int d = i / w;
                    int k = i % w;

                    ker_shared[warp_in_block][i] =
                        kernel((pos[d] - static_cast<RealType>(idx0_global[d] + k)) * inv_hw);
                }
#ifdef KOKKOS_ENABLE_CUDA
                __syncwarp();
#endif
#ifdef KOKKOS_ENABLE_HIP
                __syncthreads();
#endif

                // Determine grid element type
                using grid_element_type = std::remove_reference_t<decltype(field_view(0, 0, 0))>;
                constexpr bool grid_is_complex =
                    std::is_same_v<grid_element_type, Kokkos::complex<RealType>>;
                constexpr bool value_is_complex = is_complex;

                // Each thread accumulates its portion
                grid_element_type thread_sum(0);

                for (int linear_idx = lane_id; linear_idx < w3; linear_idx += WARP_SIZE) {
                    const int i = linear_idx % w;
                    const int j = (linear_idx / w) % w;
                    const int k = linear_idx / (w * w);

                    // Compute global grid indices
                    int gi_global = idx0_global[0] + i;
                    int gj_global = idx0_global[1] + j;
                    int gk_global = idx0_global[2] + k;

                    // Convert to local indices
                    int gi_local = gi_global - local_offset[0];
                    int gj_local = gj_global - local_offset[1];
                    int gk_local = gk_global - local_offset[2];

#ifndef NDEBUG
                    assert(gi_local >= -nghost && gi_local < n_grid_local[0] + nghost);
                    assert(gj_local >= -nghost && gj_local < n_grid_local[1] + nghost);
                    assert(gk_local >= -nghost && gk_local < n_grid_local[2] + nghost);
#endif

                    const RealType kernel_val = ker_shared[warp_in_block][i]
                                                * ker_shared[warp_in_block][w + j]
                                                * ker_shared[warp_in_block][2 * w + k];

                    // Access field using LOCAL indices + ghost offset
                    thread_sum +=
                        field_view(gi_local + nghost, gj_local + nghost, gk_local + nghost)
                        * kernel_val;
                }

                // Warp-level reduction using shuffle operations
                if constexpr (grid_is_complex && !value_is_complex) {
                    // Grid is complex but output is real - extract real part
                    RealType res_real = thread_sum.real();
                    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                        // res_real += __shfl_down_sync(0xffffffff, res_real, offset);
                        res_real += shfl_down(res_real, offset);
                    }

                    if (lane_id == 0) {
                        if (add_to_attribute) {
                            output(particle_idx) += res_real;
                        } else {
                            output(particle_idx) = res_real;
                        }
                    }
                } else if constexpr (is_complex) {
                    // Both complex
                    RealType res_real = thread_sum.real();
                    RealType res_imag = thread_sum.imag();

                    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                        // res_real += __shfl_down_sync(0xffffffff, res_real, offset);
                        // res_imag += __shfl_down_sync(0xffffffff, res_imag, offset);
                        res_real += shfl_down(res_real, offset);
                        res_imag += shfl_down(res_imag, offset);
                    }

                    if (lane_id == 0) {
                        if (add_to_attribute) {
                            output(particle_idx) += ValueType(res_real, res_imag);
                        } else {
                            output(particle_idx) = ValueType(res_real, res_imag);
                        }
                    }
                } else {
                    // Both real
                    RealType res = thread_sum;
                    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                        // res += __shfl_down_sync(0xffffffff, res, offset);
                        res += shfl_down(res, offset);
                    }

                    if (lane_id == 0) {
                        if (add_to_attribute) {
                            output(particle_idx) += res;
                        } else {
                            output(particle_idx) = res;
                        }
                    }
                }
            }

            /**
             * @brief Kernel launcher dispatcher for different kernel widths
             */
            template <int W, int MaxW>
            struct CudaGatherDispatcher {
                template <typename RealType, typename PositionViewType, typename PermuteViewType,
                          typename FieldViewType, typename KernelType, typename ValueType>
                static void dispatch_3d(int w, size_t n, PositionViewType x,
                                        PermuteViewType permute, FieldViewType field_view,
                                        Kokkos::View<ValueType*, device_memspace> output,
                                        int nghost, Vector<int, 3> n_grid_global,
                                        Vector<int, 3> n_grid_local, Vector<int, 3> local_offset,
                                        RealType inv_hw, const KernelType& kernel,
                                        bool add_to_attribute) {
                    if constexpr (W <= MaxW) {
                        if (w == W) {
#ifdef KOKKOS_ENABLE_CUDA
                            constexpr int WARP_SIZE       = 32;
                            constexpr int warps_per_block = 8;
#elif defined(KOKKOS_ENABLE_HIP)
                            constexpr int WARP_SIZE = 64;
                            constexpr int warps_per_block = 4;
#endif
                            int block_size                = warps_per_block * WARP_SIZE;
                            int num_warps                 = n;
                            int grid_size = (num_warps + warps_per_block - 1) / warps_per_block;


                            tiled_gather_3d_cuda_kernel<W, RealType, PositionViewType,
                                                        PermuteViewType, FieldViewType, KernelType,
                                                        ValueType><<<grid_size, block_size>>>(
                                x, permute, field_view, output, n, nghost, n_grid_global,
                                n_grid_local, local_offset, inv_hw, kernel, add_to_attribute);
                        } else {
                            CudaGatherDispatcher<W + 1, MaxW>::template dispatch_3d<
                                RealType, PositionViewType, PermuteViewType, FieldViewType,
                                KernelType, ValueType>(w, n, x, permute, field_view, output, nghost,
                                                       n_grid_global, n_grid_local, local_offset,
                                                       inv_hw, kernel, add_to_attribute);
                        }
                    } else {
                        throw std::runtime_error("Kernel width exceeds maximum supported width");
                    }
                }
            };
#endif  // KOKKOS_ENABLE_CUDA

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILED_GATHER_H