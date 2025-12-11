#ifndef IPPL_TILED_GATHER_H
#define IPPL_TILED_GATHER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>

#include "InterpolationUtil.h"

namespace ippl {
    namespace Interpolation {
        namespace detail {

            /**
             * @brief Tiled gather functor for 3D grid-to-particle operations
             *
             * One team processes one particle. The team uses scratch memory to store
             * the 1D kernel values in x, y, and z directions, and then performs a
             * team-parallel sum over the W^3 stencil.
             *
             * Template parameters:
             * @tparam W           Kernel width (compile-time constant)
             * @tparam RealType    Floating point type (float or double)
             * @tparam ExecSpace   Kokkos execution space
             * @tparam KernelType  Kernel functor type
             * @tparam ValueType   Type of values being gathered (scalar or complex)
             * @tparam FieldViewType     Type of the grid/field view
             * @tparam PositionViewType  Type of the position view (View<T*[3]> or
             * View<Vector<T,3>*>)
             * @tparam PermuteViewType   Type of the permutation view
             */
            template <int W, typename RealType, typename ExecSpace, typename KernelType,
                      typename ValueType, typename FieldViewType, typename PositionViewType,
                      typename PermuteViewType>
            struct TiledGatherFunctor3D {
                using real_type    = RealType;
                using value_type   = ValueType;
                using exec_space   = ExecSpace;
                using memory_space = typename ExecSpace::memory_space;
                using size_type    = typename memory_space::size_type;

                using team_policy   = Kokkos::TeamPolicy<ExecSpace>;
                using team_member   = typename team_policy::member_type;
                using scratch_space = typename ExecSpace::scratch_memory_space;

                using scratch_real_view = Kokkos::View<real_type*, scratch_space,
                                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

                // Input
                PositionViewType x;  // particle positions in physical coordinates [-pi, pi]
                PermuteViewType permute;
                FieldViewType field_view;  // grid with ghosts (LOCAL view)

                // Output
                Kokkos::View<ValueType*, memory_space> output;

                // Parameters
                size_t n_points;
                int nghost;
                Vector<int, 3> n_grid_global;  // GLOBAL grid dimensions
                Vector<int, 3> n_grid_local;   // LOCAL grid dimensions
                Vector<int, 3> local_offset;   // first global index of local domain
                real_type inv_hw;              // 1 / half-width
                KernelType kernel;
                bool add_to_attribute;

                // Compile-time constants
                static constexpr int w  = W;
                static constexpr int w3 = W * W * W;

                // Helper to access position component - works for View<T*[3]> and
                // View<Vector<T,3>*>
                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i, d)) {
                    return pos(i, d);  // View<T*[3]>
                }

                template <typename PosView>
                KOKKOS_INLINE_FUNCTION static auto get_component(const PosView& pos, size_type i,
                                                                 int d) -> decltype(pos(i)[d]) {
                    return pos(i)[d];  // View<Vector<T,3>*>
                }

                KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
                    const size_type particle_team = team.league_rank();
                    if (particle_team >= n_points) {
                        return;
                    }

                    const size_type particle_idx = permute(particle_team);
                    assert(particle_idx >= 0 && particle_idx < n_points);

                    using grid_element_type =
                        std::remove_reference_t<decltype(field_view(0, 0, 0))>;
                    constexpr bool grid_is_complex =
                        std::is_same_v<grid_element_type, Kokkos::complex<real_type>>;
                    constexpr bool value_is_complex =
                        std::is_same_v<ValueType, Kokkos::complex<real_type>>;

                    real_type pos[3];
                    int idx0_global[3];

                    for (int d = 0; d < 3; ++d) {
                        const real_type phys = get_component(x, particle_idx, d);
                        pos[d]               = scale_to_grid_indices(phys, n_grid_global[d]);
                        idx0_global[d] =
                            grid_point_to_grid_idx(pos[d], n_grid_global[d], w) - (w - 1) / 2;
                    }

                    // Allocate scratch for kernel_x, kernel_y, kernel_z (3*W entries)
                    scratch_real_view ker(team.team_scratch(0), 3 * W);
                    real_type* kernel_x = &ker(0);
                    real_type* kernel_y = &ker(W);
                    real_type* kernel_z = &ker(2 * W);

                    // Let one thread per team precompute kernel values into scratch
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, W), [&](int i) {
                        kernel_x[i] =
                            kernel((pos[0] - static_cast<real_type>(idx0_global[0] + i)) * inv_hw);
                        kernel_y[i] =
                            kernel((pos[1] - static_cast<real_type>(idx0_global[1] + i)) * inv_hw);
                        kernel_z[i] =
                            kernel((pos[2] - static_cast<real_type>(idx0_global[2] + i)) * inv_hw);
                    });

                    // Team-parallel reduction over W^3 stencil
                    ValueType particle_sum = ValueType(0);

                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, w3),
                        [&](int linear_idx, ValueType& local_sum) {
                            const int i = linear_idx % W;
                            const int j = (linear_idx / W) % W;
                            const int k = linear_idx / (W * W);

                            int gi_global = idx0_global[0] + i;
                            int gj_global = idx0_global[1] + j;
                            int gk_global = idx0_global[2] + k;

                            int gi_local = gi_global - local_offset[0];
                            int gj_local = gj_global - local_offset[1];
                            int gk_local = gk_global - local_offset[2];

#ifndef NDEBUG
                            // LOCAL domain (including ghosts)
                            assert(gi_local >= -nghost
                                   && gi_local < static_cast<int>(n_grid_local[0]) + nghost);
                            assert(gj_local >= -nghost
                                   && gj_local < static_cast<int>(n_grid_local[1]) + nghost);
                            assert(gk_local >= -nghost
                                   && gk_local < static_cast<int>(n_grid_local[2]) + nghost);
#endif

                            const real_type kernel_val = kernel_x[i] * kernel_y[j] * kernel_z[k];

                            const auto grid_val =
                                field_view(gi_local + nghost, gj_local + nghost, gk_local + nghost);

                            if constexpr (grid_is_complex && !value_is_complex) {
                                // Complex grid -> real output: take real part
                                local_sum += static_cast<ValueType>(grid_val.real() * kernel_val);
                            } else if constexpr (value_is_complex && grid_is_complex) {
                                // Complex grid -> complex output
                                local_sum += ValueType(grid_val.real() * kernel_val,
                                                       grid_val.imag() * kernel_val);
                            } else if constexpr (value_is_complex && !grid_is_complex) {
                                // Real grid -> complex output (imag part zero)
                                local_sum += ValueType(grid_val * kernel_val, 0.0);
                            } else {
                                // Real grid -> real output
                                local_sum += static_cast<ValueType>(grid_val * kernel_val);
                            }
                        },
                        particle_sum);

                    // One thread per team writes the result
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        if (add_to_attribute) {
                            output(particle_idx) += particle_sum;
                        } else {
                            output(particle_idx) = particle_sum;
                        }
                    });
                }
            };

            inline int get_default_team_size() {
#ifdef KOKKOS_ENABLE_CUDA
                return 32;
#elif defined(KOKKOS_ENABLE_HIP)
                return 64;
#endif
                return 4;
            }

            /**
             * @brief Dispatcher for gather with different kernel widths
             * Uses template recursion to dispatch to the correct compile-time width.
             */
            template <int W, int MaxW>
            struct TiledGatherDispatcher {
                template <typename RealType, typename ExecSpace, typename KernelType,
                          typename ValueType, typename FieldViewType, typename PositionViewType,
                          typename PermuteViewType>
                static void dispatch_3d(
                    int w, size_t n_points, PositionViewType x, PermuteViewType permute,
                    FieldViewType field_view,
                    Kokkos::View<ValueType*, typename ExecSpace::memory_space> output, int nghost,
                    Vector<int, 3> n_grid_global, Vector<int, 3> n_grid_local,
                    Vector<int, 3> local_offset, RealType inv_hw, const KernelType& kernel,
                    bool add_to_attribute, int team_size = get_default_team_size()) {
                    if constexpr (W <= MaxW) {
                        if (w == W) {
                            using functor_type =
                                TiledGatherFunctor3D<W, RealType, ExecSpace, KernelType, ValueType,
                                                     FieldViewType, PositionViewType,
                                                     PermuteViewType>;

                            functor_type functor{
                                x,      permute,         field_view,   output,       n_points,
                                nghost, n_grid_global,   n_grid_local, local_offset, inv_hw,
                                kernel, add_to_attribute};

                            using team_policy = Kokkos::TeamPolicy<ExecSpace>;
                            team_policy policy(n_points, team_size);

                            // Scratch memory: 3*W reals per team (kernel_x/y/z)
                            const size_t scratch_size  = 3 * W;
                            const size_t scratch_bytes = scratch_size * sizeof(RealType);
                            policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

                            Kokkos::parallel_for("tiled_gather_3d", policy, functor);
                        } else {
                            TiledGatherDispatcher<W + 1, MaxW>::template dispatch_3d<
                                RealType, ExecSpace, KernelType, ValueType, FieldViewType,
                                PositionViewType, PermuteViewType>(
                                w, n_points, x, permute, field_view, output, nghost, n_grid_global,
                                n_grid_local, local_offset, inv_hw, kernel, add_to_attribute,
                                team_size);
                        }
                    } else {
                        throw std::runtime_error("Kernel width exceeds maximum supported width");
                    }
                }
            };

        }  // namespace detail
    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILED_GATHER_H
