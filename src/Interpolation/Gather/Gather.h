#ifndef IPPL_GATHER_H
#define IPPL_GATHER_H

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
            using VecTr   = ippl::detail::VectorTraits<typename PosTr::value_type>;

            using type = GatherTypes<FieldTr::dim, typename VecTr::real_type, std::decay_t<Kernel>,
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
                case Interpolation::GatherMethod::Tiled:
                    // TODO: Implement tiled, fallback to atomic
                    dispatch<Interpolation::detail::AtomicGather, Types, false>(field, positions, values);
                    break;
                case Interpolation::GatherMethod::Native:
                    // TODO: Implement native, fallback to atomic
                    dispatch<Interpolation::detail::AtomicGather, Types, false>(field, positions, values);
                    break;
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

            Interpolation::WidthDispatcher<1, 14>::dispatch(width, [&]<int W>() {
                auto args = Impl<W, Types, UseSorting>::Arguments::create(field, positions, values, kernel_m,
                                                                           config_m, binning);
                Impl<W, Types, UseSorting> functor(std::move(args));
                field.fillHalo();
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