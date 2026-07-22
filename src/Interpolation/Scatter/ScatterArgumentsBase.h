/*!
 * @file ScatterArgumentsBase.h
 * @brief Common type bundle and shared argument struct for scatter functors.
 */
#ifndef IPPL_SCATTER_ARGUMENTS_BASE_H
#define IPPL_SCATTER_ARGUMENTS_BASE_H

#include "Types/Vector.h"

namespace ippl::Interpolation::detail {

    /*!
     * @struct ScatterTypes
     * @brief Type bundle propagated through every scatter implementation.
     *
     * Centralizes the dimension, real precision, kernel, and view types so
     * each individual scatter functor can be parameterized on a single
     * @c Types parameter.
     */
    template <unsigned Dim_, typename RealType_, typename KernelType_, typename GridViewType_,
              typename PositionViewType_, typename ValuesViewType_>
    struct ScatterTypes {
        static constexpr unsigned Dim = Dim_;
        using RealType                = RealType_;
        using KernelType              = KernelType_;
        using GridViewType            = GridViewType_;
        using PositionViewType        = PositionViewType_;
        using ValuesViewType          = ValuesViewType_;
        using ValueType               = typename ValuesViewType_::value_type;
        using memory_space            = typename GridViewType_::memory_space;
        using execution_space         = typename PositionViewType_::execution_space;

        static_assert(std::is_same_v<memory_space,
                                     typename PositionViewType_::memory_space>,
                      "Scatter: field grid and particle positions must live in the same "
                      "memory space");
        static_assert(std::is_same_v<memory_space,
                                     typename ValuesViewType_::memory_space>,
                      "Scatter: field grid and particle values must live in the same "
                      "memory space");
    };

    /*!
     * @struct ScatterArgumentsBase
     * @brief CRTP base bundling per-call inputs (views, mesh info, kernel)
     *        consumed by the scatter functors.
     *
     * Concrete derived structs add algorithm-specific tuning fields
     * (tile sizes, team layouts, ...) and forward shared initialization to
     * @c initBase.
     */
    template <typename Derived, typename Types>
    struct ScatterArgumentsBase {
        static constexpr unsigned Dim = Types::Dim;
        using RealType                = typename Types::RealType;
        using KernelType              = typename Types::KernelType;
        using GridViewType            = typename Types::GridViewType;
        using PositionViewType        = typename Types::PositionViewType;
        using ValuesViewType          = typename Types::ValuesViewType;
        using memory_space            = typename Types::memory_space;

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
        size_t n_particles;

    protected:
        template <typename Field, typename Positions, typename Values, typename Kernel>
        void initBase(Field& field, const Positions& positions, const Values& vals,
                      const Kernel& k) {
            x      = positions.getView();
            values = vals.getView();
            grid   = field.getView();
            nghost = field.getNghost();
            kernel = k;
            n_particles = positions.getParticleCount();

            const auto& layout = field.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();
            const auto& gDom   = layout.getDomain();

            for (unsigned d = 0; d < Dim; ++d) {
                n_grid[d]       = gDom[d].length();
                n_grid_local[d] = lDom[d].length();
                local_offset[d] = lDom[d].first();
            }

            const auto& mesh = field.get_mesh();
            origin           = mesh.getOrigin();
            invdx            = RealType(1) / mesh.getMeshSpacing();
            inv_hw           = RealType(2) / k.width();
        }
    };

}  // namespace ippl::Interpolation::detail

#endif
