/*!
 * @file GatherArgumentsBase.h
 * @brief Common type bundle and shared argument struct for gather functors.
 */
#ifndef IPPL_GATHER_ARGUMENTS_BASE_H
#define IPPL_GATHER_ARGUMENTS_BASE_H

#include "Types/IpplTypes.h"
#include "Types/Vector.h"

namespace ippl::Interpolation::detail {

    /*!
     * @struct GatherTypes
     * @brief Type bundle propagated through every gather implementation.
     */
    template <unsigned Dim_, typename RealType_, typename KernelType_, typename GridViewType_,
              typename PositionViewType_, typename ValuesViewType_>
    struct GatherTypes {
        static constexpr unsigned Dim = Dim_;
        using RealType         = RealType_;
        using KernelType       = KernelType_;
        using GridViewType     = GridViewType_;
        using PositionViewType = PositionViewType_;
        using ValuesViewType   = ValuesViewType_;
        using ValueType        = typename ValuesViewType_::value_type;
        using memory_space     = typename GridViewType_::memory_space;
        using execution_space  = typename PositionViewType_::execution_space;
    };

    /*!
     * @struct GatherArgumentsBase
     * @brief CRTP base bundling per-call inputs (views, mesh info, kernel)
     *        consumed by the gather functors.
     */
    template <typename Derived, typename Types>
    struct GatherArgumentsBase {
        static constexpr unsigned Dim = Types::Dim;
        using RealType         = typename Types::RealType;
        using KernelType       = typename Types::KernelType;
        using GridViewType     = typename Types::GridViewType;
        using PositionViewType = typename Types::PositionViewType;
        using ValuesViewType   = typename Types::ValuesViewType;
        using memory_space     = typename Types::memory_space;

        PositionViewType x;
        ValuesViewType values;
        GridViewType grid;
        int nghost;
        KernelType kernel;
        Vector<int, Dim> n_grid;
        Vector<int, Dim> n_grid_local;
        Vector<int, Dim> local_offset;
        Vector<RealType, Dim> origin;
        Vector<RealType, Dim> invdx;
        RealType inv_hw;
        bool add_to_attribute;

    protected:
        template <typename Field, typename Positions, typename Values, typename Kernel>
        void initBase(const Field& field, const Positions& positions, Values& vals,
                      const Kernel& k, bool add_to) {
            x      = positions.getView();
            values = vals.getView();
            grid   = field.getView();
            nghost = field.getNghost();
            kernel = k;
            add_to_attribute = add_to;

            const auto& layout = field.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();
            const auto& gDom   = layout.getDomain();

            for (unsigned d = 0; d < Dim; ++d) {
                n_grid[d]       = gDom[d].length();
                n_grid_local[d] = lDom[d].length();
                local_offset[d] = lDom[d].first();
            }

            const auto& mesh = field.get_mesh();
            origin = mesh.getOrigin();
            invdx  = RealType(1) / mesh.getMeshSpacing();
            inv_hw = RealType(2) / k.width();
        }
    };

    // Reuse BinningResult from scatter
    template <typename MemorySpace>
    struct GatherBinningResult {
        // Particle counts can exceed INT_MAX in 3D, so the per-particle id
        // and per-bin offset views use ippl::detail::size_type (== size_t).
        // Keeping them as size_type also avoids a Kokkos View value-type
        // mismatch with SortBuffer / Binning, where std::size_t and
        // uint64_t are different types on platforms like macOS.
        Kokkos::View<ippl::detail::size_type*, MemorySpace> permute;
        Kokkos::View<ippl::detail::size_type*, MemorySpace> bin_offsets;
        Vector<int, 3> num_tiles;

        operator bool() const { return permute.data() != nullptr; }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GATHER_ARGUMENTS_BASE_H
