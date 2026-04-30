#ifndef IPPL_GATHER_H
#define IPPL_GATHER_H

#include "Utility/IpplException.h"

#include "Interpolation/Binning.h"
#include "Interpolation/Gather/AtomicGather.h"
#include "Interpolation/Gather/GatherArgumentsBase.h"
#include "Interpolation/Gather/GatherConfig.h"
#include "Interpolation/WidthDispatcher.h"
#include "Particle/ParticleAttrib.h"

namespace ippl {

    namespace Interpolation::detail {
        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        struct DeduceGatherTypes {
            using FieldTr = ippl::detail::FieldTraits<std::decay_t<FieldType>>;
            using PosTr   = ippl::detail::AttribTraits<std::decay_t<PositionsType>>;
            using ValTr   = ippl::detail::AttribTraits<std::decay_t<ValuesType>>;

            // Use the kernel's own value_type for the geometric real-precision
            // computations (origin, invdx, weights). When the position attribute
            // is float but the mesh is double, taking RealType from the position
            // would silently downcast the mesh spacing and lose precision; the
            // kernel knows the correct working precision and the legacy CIC
            // path uses the mesh type explicitly via Kernel = LinearKernel<T>.
            using RealType = typename std::decay_t<Kernel>::value_type;

            using type = GatherTypes<FieldTr::dim, RealType, std::decay_t<Kernel>,
                                     typename FieldTr::view_type, typename PosTr::view_type,
                                     typename ValTr::view_type>;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        using DeducedGatherTypes =
            typename DeduceGatherTypes<Kernel, FieldType, PositionsType, ValuesType>::type;

    }  // namespace Interpolation::detail

    template <typename Kernel, unsigned Dim>
    class Gather {
    public:
        Gather(const Kernel& kernel, const Interpolation::GatherConfig<Dim>& config = {})
            : kernel_m(kernel)
            , config_m(config) {}

        template <typename ValueT, typename FieldT, class Mesh, class Centering,
                  class... ViewArgs, typename ParticleT, class... PosProps, class... ValProps>
        void operator()(Field<FieldT, Dim, Mesh, Centering, ViewArgs...>& field,
                        const ParticleAttrib<Vector<ParticleT, Dim>, PosProps...>& positions,
                        ParticleAttrib<ValueT, ValProps...>& values) {
            using Types =
                Interpolation::detail::DeducedGatherTypes<Kernel, decltype(field),
                                                          decltype(positions), decltype(values)>;

            switch (config_m.method) {
                case Interpolation::GatherMethod::Atomic:
                    dispatch<Interpolation::detail::AtomicGather, Types, false>(field, positions, values);
                    break;
                case Interpolation::GatherMethod::AtomicSort:
                    dispatch<Interpolation::detail::AtomicGather, Types, true>(field, positions, values);
                    break;
                default:
                    throw IpplException("Gather", "Unknown GatherMethod");
            }
        }

    private:
        template <template <int, typename, bool> class Impl, typename Types, bool UseSorting,
                  typename Field, typename Positions, typename Values>
        void dispatch(Field& field, const Positions& positions, Values& values) {
            using memory_space = typename Types::memory_space;

            Interpolation::detail::GatherBinningResult<memory_space> binning;

            // Check requires_binning using W=1 (trait doesn't depend on W)
            if constexpr (Impl<1, Types, UseSorting>::requires_binning) {
                binning = performBinning<Types>(positions, field);
            } else if (config_m.do_binning()) {
                binning = performBinning<Types>(positions, field);
            }

            const int width          = kernel_m.width();
            const size_t n_particles = positions.getParticleCount();

            // The halo must be valid before any stencil reads it, and is
            // independent of the runtime kernel width — fill once outside
            // the WidthDispatcher.
            field.fillHalo();

            Interpolation::WidthDispatcher<1, std::decay_t<Kernel>::max_width>::dispatch(width, [&]<int W>() {
                auto args = Impl<W, Types, UseSorting>::Arguments::create(field, positions, values, kernel_m,
                                                                           config_m, binning);
                Impl<W, Types, UseSorting> functor(std::move(args));
                functor.run(n_particles);
            });
        }

        template <typename Types, typename Positions, typename Field>
        auto performBinning(const Positions& positions, const Field& field) {
            using memory_space = typename Types::memory_space;

            auto [permute, bin_offsets, num_tiles] =
                Interpolation::detail::bin_particles(positions, field.getLayout(), field.get_mesh(),
                                                     config_m.get_tile_size(), kernel_m.width());

            return Interpolation::detail::GatherBinningResult<memory_space>{permute, bin_offsets,
                                                                            num_tiles};
        }

        Kernel kernel_m;
        Interpolation::GatherConfig<Dim> config_m;
    };

}  // namespace ippl

#endif  // IPPL_GATHER_H