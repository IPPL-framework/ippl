#ifndef IPPL_TILED_GATHER_H
#define IPPL_TILED_GATHER_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace ippl {
namespace Interpolation {
namespace detail {

#ifdef KOKKOS_ENABLE_CUDA
    /**
     * @brief Generic CUDA kernel for 3D tiled gather (interpolation) with warp-level parallelism
     *
     * This is a generic implementation that works with any kernel function.
     * Optimized for CUDA with:
     * - Each warp (32 threads) processes one particle
     * - Kernel values stored in shared memory for reuse
     * - Warp shuffle operations for efficient reduction
     *
     * Template parameters:
     * @tparam w The kernel width (compile-time constant for optimization)
     * @tparam RealType The floating point type (float or double)
     * @tparam FieldViewType The field view type (generic, can be any Kokkos::View)
     * @tparam KernelType The kernel function type
     * @tparam ValueType The type of values being gathered (can be scalar or complex)
     */
    template<int w, typename RealType, typename FieldViewType, typename KernelType, typename ValueType>
    __global__ void tiled_gather_3d_cuda_kernel(
        Kokkos::View<RealType*[3], Kokkos::CudaSpace> x_sorted,  // Particle positions in GRID coordinates [0, n_grid)
        FieldViewType field_view,  // Input field with ghosts
        Kokkos::View<ValueType*, Kokkos::CudaSpace> c_sorted,  // Output values
        int64_t n_points,
        int hw,  // half width
        int nghost,
        int n0, int n1, int n2,  // Grid dimensions
        RealType inv_hw,  // 1 / half_width for kernel scaling
        KernelType kernel) {

        constexpr bool odd = w % 2 == 1;
        constexpr bool is_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
        using size_type = typename Kokkos::CudaSpace::size_type;
        constexpr int w3 = w * w * w;
        constexpr int WARP_SIZE = 32;

        // Determine which particle this warp is processing
        const size_type warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
        if (warp_id >= n_points) return;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_in_block = threadIdx.x / WARP_SIZE;

        // Shared memory: kernel values for each warp in the block
        constexpr int max_warps_per_block = 256 / WARP_SIZE;
        __shared__ RealType ker_shared[max_warps_per_block][3 * w];

        // Particle position in grid coordinates [0, n_grid)
        RealType sx = x_sorted(warp_id, 0);
        size_type idx0_0 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        sx = x_sorted(warp_id, 1);
        size_type idx0_1 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        sx = x_sorted(warp_id, 2);
        size_type idx0_2 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        // Collaboratively compute kernel values within the warp
        for (int i = lane_id; i < 3 * w; i += WARP_SIZE) {
            int d = i / w;
            int k = i % w;

            auto sx_loc = x_sorted(warp_id, d);
            auto idx0_loc = (d == 2) ? idx0_2 : (d == 1) ? idx0_1 : idx0_0;

            ker_shared[warp_in_block][i] =
                kernel((sx_loc - static_cast<RealType>(idx0_loc + k)) * inv_hw);
        }
        __syncwarp();

        // Each thread in warp accumulates its portion of the 3D grid
        ValueType thread_sum(0);

        for (int linear_idx = lane_id; linear_idx < w3; linear_idx += WARP_SIZE) {
            const int i = linear_idx % w;
            const int j = (linear_idx / w) % w;
            const int k = linear_idx / (w * w);

            // Compute grid indices with periodic wrapping
            int gi = idx0_0 + i;
            int gj = idx0_1 + j;
            int gk = idx0_2 + k;

            if (gi < 0) gi += n0;
            else if (gi >= n0) gi -= n0;
            if (gj < 0) gj += n1;
            else if (gj >= n1) gj -= n1;
            if (gk < 0) gk += n2;
            else if (gk >= n2) gk -= n2;

            const RealType kernel_val = ker_shared[warp_in_block][i] *
                                       ker_shared[warp_in_block][w + j] *
                                       ker_shared[warp_in_block][2 * w + k];

            thread_sum += field_view(gi + nghost, gj + nghost, gk + nghost) * kernel_val;
        }

        // Warp-level reduction using shuffle operations
        if constexpr (is_complex) {
            RealType res_real = thread_sum.real();
            RealType res_imag = thread_sum.imag();

            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                res_real += __shfl_down_sync(0xffffffff, res_real, offset);
                res_imag += __shfl_down_sync(0xffffffff, res_imag, offset);
            }

            // First lane in warp writes result
            if (lane_id == 0) {
                c_sorted(warp_id) = ValueType(res_real, res_imag);
            }
        } else {
            RealType res = thread_sum;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                res += __shfl_down_sync(0xffffffff, res, offset);
            }

            if (lane_id == 0) {
                c_sorted(warp_id) = res;
            }
        }
    }

    /**
     * @brief Kernel launcher dispatcher for different kernel widths
     * Uses template recursion to dispatch to the correct kernel width at runtime
     */
    template<int W, int MaxW>
    struct CudaGatherDispatcher {
        template<typename RealType, typename FieldViewType, typename KernelType, typename ValueType>
        static void dispatch_3d(
            int w, size_t n,
            Kokkos::View<RealType*[3], Kokkos::CudaSpace> x_sorted,
            FieldViewType field_view,
            Kokkos::View<ValueType*, Kokkos::CudaSpace> c_sorted,
            int hw, int nghost, int n0, int n1, int n2, RealType inv_hw,
            const KernelType& kernel) {

            if constexpr (W <= MaxW) {
                if (w == W) {
                    constexpr int WARP_SIZE = 32;
                    constexpr int warps_per_block = 8;  // 256 threads per block
                    int block_size = warps_per_block * WARP_SIZE;
                    int num_warps = n;
                    int grid_size = (num_warps + warps_per_block - 1) / warps_per_block;

                    tiled_gather_3d_cuda_kernel<W, RealType, FieldViewType, KernelType, ValueType>
                        <<<grid_size, block_size>>>(
                            x_sorted, field_view, c_sorted,
                            n, hw, nghost, n0, n1, n2, inv_hw, kernel);
                } else {
                    CudaGatherDispatcher<W + 1, MaxW>::template dispatch_3d<RealType, FieldViewType, KernelType, ValueType>(
                        w, n, x_sorted, field_view, c_sorted,
                        hw, nghost, n0, n1, n2, inv_hw, kernel);
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
