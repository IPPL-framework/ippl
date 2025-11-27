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
     * @tparam PositionViewType The position view type
     * @tparam FieldViewType The field view type (generic, can be any Kokkos::View)
     * @tparam KernelType The kernel function type
     * @tparam ValueType The type of values being gathered (can be scalar or complex)
     */
    template<int w, typename RealType, typename PositionViewType, typename PermuteViewType, typename FieldViewType, typename KernelType, typename ValueType>
    __global__ void tiled_gather_3d_cuda_kernel(
        PositionViewType x,  // Particle positions in PHYSICAL coordinates [-pi, pi]
        PermuteViewType permute,  // Permutation array
        FieldViewType field_view,  // Input field with ghosts
        Kokkos::View<ValueType*, Kokkos::CudaSpace> output,  // Output values (written at permuted indices)
        int64_t n_points,
        int hw,  // half width
        int nghost,
        int n0, int n1, int n2,  // Grid dimensions
        RealType inv_hw,  // 1 / half_width for kernel scaling
        KernelType kernel,
        bool add_to_attribute) {

        constexpr bool odd = w % 2 == 1;
        constexpr bool is_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
        using size_type = typename PermuteViewType::value_type;
        constexpr int w3 = w * w * w;
        constexpr int WARP_SIZE = 32;
        constexpr RealType inv_two_pi = RealType(0.5) / RealType(3.14159265358979323846);

        // Determine which particle this warp is processing
        const size_type warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
        if (warp_id >= n_points) return;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_in_block = threadIdx.x / WARP_SIZE;

        // Apply permutation to get actual particle index
        const size_type particle_idx = permute(warp_id);

        // Helper to access position component - works with both View<T*[3]> and View<Vector<T,3>*>
        auto get_pos = [&](int d) -> RealType {
            if constexpr (std::is_same_v<PositionViewType, Kokkos::View<RealType*[3], Kokkos::CudaSpace>>) {
                return x(particle_idx, d);
            } else {
                return x(particle_idx)[d];
            }
        };

        // Transform from physical coordinates [-pi, pi] to grid coordinates [0, n_grid)
        auto transform_coord = [&](RealType pos_phys, int grid_size) -> RealType {
            RealType k = pos_phys * inv_two_pi;
            k = k - Kokkos::floor(k);
            return k * grid_size;
        };

        // Shared memory: kernel values for each warp in the block
        constexpr int max_warps_per_block = 256 / WARP_SIZE;
        __shared__ RealType ker_shared[max_warps_per_block][3 * w];

        // Particle positions in grid coordinates [0, n_grid)
        RealType sx = transform_coord(get_pos(0), n0);
        size_type idx0_0 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        sx = transform_coord(get_pos(1), n1);
        size_type idx0_1 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        sx = transform_coord(get_pos(2), n2);
        size_type idx0_2 = odd ? llround(sx) - hw : static_cast<size_type>(sx) + 1 - hw;

        // Collaboratively compute kernel values within the warp
        for (int i = lane_id; i < 3 * w; i += WARP_SIZE) {
            int d = i / w;
            int k = i % w;

            RealType sx_loc;
            size_type idx0_loc;
            int grid_size_loc;

            if (d == 0) {
                sx_loc = transform_coord(get_pos(0), n0);
                idx0_loc = idx0_0;
            } else if (d == 1) {
                sx_loc = transform_coord(get_pos(1), n1);
                idx0_loc = idx0_1;
            } else {
                sx_loc = transform_coord(get_pos(2), n2);
                idx0_loc = idx0_2;
            }

            ker_shared[warp_in_block][i] =
                kernel((sx_loc - static_cast<RealType>(idx0_loc + k)) * inv_hw);
        }
        __syncwarp();

        // Determine if grid is complex (ValueType might be real even if grid is complex)
        using grid_element_type = std::remove_reference_t<decltype(field_view(0, 0, 0))>;
        constexpr bool grid_is_complex = std::is_same_v<grid_element_type, Kokkos::complex<RealType>>;
        constexpr bool value_is_complex = is_complex;

        // Each thread in warp accumulates its portion of the 3D grid
        // thread_sum should match grid type
        grid_element_type thread_sum(0);

        for (int linear_idx = lane_id; linear_idx < w3; linear_idx += WARP_SIZE) {
            const int i = linear_idx % w;
            const int j = (linear_idx / w) % w;
            const int k = linear_idx / (w * w);

            // Compute grid indices
            int gi = idx0_0 + i;
            int gj = idx0_1 + j;
            int gk = idx0_2 + k;

            const RealType kernel_val = ker_shared[warp_in_block][i] *
                                       ker_shared[warp_in_block][w + j] *
                                       ker_shared[warp_in_block][2 * w + k];

            thread_sum += field_view(gi + nghost, gj + nghost, gk + nghost) * kernel_val;
        }

        // Warp-level reduction using shuffle operations

        if constexpr (grid_is_complex && !value_is_complex) {
            // Grid is complex but output is real - extract real part
            RealType res_real = thread_sum.real();
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                res_real += __shfl_down_sync(0xffffffff, res_real, offset);
            }

            // First lane in warp writes result using permuted index
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
                res_real += __shfl_down_sync(0xffffffff, res_real, offset);
                res_imag += __shfl_down_sync(0xffffffff, res_imag, offset);
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
                res += __shfl_down_sync(0xffffffff, res, offset);
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
     * Uses template recursion to dispatch to the correct kernel width at runtime
     */
    template<int W, int MaxW>
    struct CudaGatherDispatcher {
        template<typename RealType, typename PositionViewType, typename PermuteViewType, typename FieldViewType, typename KernelType, typename ValueType>
        static void dispatch_3d(
            int w, size_t n,
            PositionViewType x,
            PermuteViewType permute,
            FieldViewType field_view,
            Kokkos::View<ValueType*, Kokkos::CudaSpace> output,
            int hw, int nghost, int n0, int n1, int n2, RealType inv_hw,
            const KernelType& kernel,
            bool add_to_attribute) {

            if constexpr (W <= MaxW) {
                if (w == W) {
                    constexpr int WARP_SIZE = 32;
                    constexpr int warps_per_block = 8;  // 256 threads per block
                    int block_size = warps_per_block * WARP_SIZE;
                    int num_warps = n;
                    int grid_size = (num_warps + warps_per_block - 1) / warps_per_block;

                    tiled_gather_3d_cuda_kernel<W, RealType, PositionViewType, PermuteViewType, FieldViewType, KernelType, ValueType>
                        <<<grid_size, block_size>>>(
                            x, permute, field_view, output,
                            n, hw, nghost, n0, n1, n2, inv_hw, kernel, add_to_attribute);
                } else {
                    CudaGatherDispatcher<W + 1, MaxW>::template dispatch_3d<RealType, PositionViewType, PermuteViewType, FieldViewType, KernelType, ValueType>(
                        w, n, x, permute, field_view, output,
                        hw, nghost, n0, n1, n2, inv_hw, kernel, add_to_attribute);
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
