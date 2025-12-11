//
// Class ParticleAttrib
//   Templated class for all particle attribute classes.
//
//   This templated class is used to represent a single particle attribute.
//   An attribute is one data element within a particle object, and is
//   stored as a Kokkos::View. This class stores the type information for the
//   attribute, and provides methods to create and destroy new items, and
//   to perform operations involving this attribute with others.
//
//   ParticleAttrib is the primary element involved in expressions for
//   particles (just as BareField is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
//
#include "Ippl.h"

#include "Communicate/DataTypes.h"

#include "Utility/BufferView.h"
#include "Utility/IpplTimings.h"

#include "FFT/FFT.h"
#include "Interpolation/AtomicGather.h"
#include "Interpolation/AtomicScatter.h"
#include "Interpolation/AtomicSortGather.h"
#include "Interpolation/Binning.h"
#include "Interpolation/OutputFocusedScatter.h"
#include "Interpolation/ScatterConfig.h"
#include "Interpolation/TiledGather.h"
#include "Interpolation/TiledKokkosGather.h"
#include "Interpolation/TiledScatter.h"
#include "Particle/ParticleSort.h"
#include "Particle/SortBuffer.h"

namespace ippl {

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_type n) {
        size_type required = *(this->localNum_mp) + n;
        if (this->size() < required) {
            int overalloc = Comm->getDefaultOverallocation();
            this->realloc(required * overalloc);
        }
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const hash_type& deleteIndex,
                                                   const hash_type& keepIndex,
                                                   size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::destroy()", policy_type(0, invalidCount),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                dview_m(deleteIndex(i)) = dview_m(keepIndex(i));
            });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(const hash_type& hash) {
        auto size = hash.extent(0);
        if (buf_m.extent(0) < size) {
            int overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(buf_m, size * overalloc);
        }

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::pack()", policy_type(0, size),
            KOKKOS_CLASS_LAMBDA(const size_t i) { buf_m(i) = dview_m(hash(i)); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::unpack(size_type nrecvs) {
        auto size          = dview_m.extent(0);
        size_type required = *(this->localNum_mp) + nrecvs;
        if (size < required) {
            int overalloc = Comm->getDefaultOverallocation();
            this->resize(required * overalloc);
        }

        size_type count   = *(this->localNum_mp);
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()", policy_type(0, nrecvs),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(count + i) = buf_m(i); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(T x) {
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = x; });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename E, size_t N>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(
        detail::Expression<E, N> const& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = expr_(i); });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename Field, class PT, typename policy_type>
    void ParticleAttrib<T, Properties...>::scatter(
        Field& f, const ParticleAttrib<Vector<PT, Field::dim>, Properties...>& pp,
        policy_type iteration_policy, hash_type hash_array) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        // using policy_type = Kokkos::RangePolicy<execution_space>;
        const bool useHashView = hash_array.extent(0) > 0;
        if (useHashView && (iteration_policy.end() > hash_array.extent(0))) {
            Inform m("scatter");
            m << "Hash array was passed to scatter, but size does not match iteration policy."
              << endl;
            ippl::Comm->abort();
        }
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", iteration_policy, KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // map index to possible hash_map
                size_t mapped_idx = useHashView ? hash_array(idx) : idx;

                // find nearest grid point
                vector_type l                        = (pp(mapped_idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // scatter
                const value_type& val = dview_m(mapped_idx);
                detail::scatterToField(std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi,
                                       args, val);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const bool addToAttribute) {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        f.fillHalo();
        IpplTimings::stopTimer(fillHaloTimer);

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);
        const typename Field::view_type view = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::gather", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // gather
                value_type gathered = detail::gatherFromField(
                    std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi, args);
                if (addToAttribute) {
                    dview_m(idx) += gathered;
                } else {
                    dview_m(idx) = gathered;
                }
            });
        IpplTimings::stopTimer(gatherTimer);
    }

    // Kernel-based scatter
    template <typename T, class... Properties>
    template <typename Field, typename P2, typename Kernel>
    void ParticleAttrib<T, Properties...>::scatter_kernel(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const Kernel& kernel, const Interpolation::ScatterConfig& config) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef scatterKernelTimer = IpplTimings::getTimer("scatterKernel");
        static IpplTimings::TimerRef scatterKernelSortTimer = IpplTimings::getTimer("scatterKernelSort");
        IpplTimings::startTimer(scatterKernelTimer);

        using view_type     = typename Field::view_type;
        view_type full_view = f.getView();
        const int nghost    = f.getNghost();

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const NDIndex<Dim>& gDom       = layout.getDomain();

        const int w = kernel.width();
        // const int hw              = w / 2;
        // const bool odd            = (w & 1);
        const PositionType inv_hw = PositionType(2.0) / w;

        // Use GLOBAL grid dimensions for coordinate scaling
        Vector<int, Dim> ngrid_global;
        Vector<int, Dim> ngrid_local;
        Vector<int, Dim> local_offset;
        for (unsigned d = 0; d < Dim; ++d) {
            ngrid_global[d] = gDom[d].length();
            ngrid_local[d]  = lDom[d].length();
            local_offset[d] = lDom[d].first();
        }

        using policy_type       = Kokkos::RangePolicy<execution_space>;
        const size_t nParticles = *(this->localNum_mp);

        // Dispatch based on spread method
        if ((config.method == Interpolation::ScatterMethod::Tiled
             || config.method == Interpolation::ScatterMethod::OutputFocused)
            && Dim == 3) {
            IpplTimings::startTimer(scatterKernelSortTimer);
            using size_type = typename execution_space::memory_space::size_type;

            // Prepare grid dimensions - use GLOBAL for coordinate transform, LOCAL for binning
            Kokkos::Array<int, 3> n_grid_global_arr;
            Kokkos::Array<int, 3> n_grid_local_arr;
            Kokkos::Array<int, 3> local_offset_arr;
            Kokkos::Array<int, 3> tile_size_arr;
            for (unsigned d = 0; d < 3; ++d) {
                n_grid_global_arr[d] = ngrid_global[d];
                n_grid_local_arr[d]  = ngrid_local[d];
                local_offset_arr[d]  = local_offset[d];
                tile_size_arr[d]     = config.tile_size_3d;
            }

            auto pp_view = pp.getView();

            // Calculate number of tiles (based on LOCAL grid)
            Kokkos::Array<int, 3> num_tiles;
            for (unsigned d = 0; d < 3; ++d) {
                num_tiles[d] = (ngrid_local[d] + config.tile_size_3d - 1) / config.tile_size_3d + 1;
            }
            auto total_tiles = size_t(num_tiles[0]) * size_t(num_tiles[1]) * size_t(num_tiles[2]);

            // Sort particles by tile (using local binning)

            // Kokkos::View<size_type*, typename execution_space::memory_space> permute;
            // Kokkos::View<size_type*, typename execution_space::memory_space> bin_offsets;
            auto &buf_handler = detail::getDefaultSortBufferManager<memory_space>();
            buf_handler.ensureCapacity(std::max(nParticles, total_tiles + 1));
            auto& permute = buf_handler.indices();
            auto& bin_offsets = buf_handler.indicesSorted();

            Interpolation::detail::bin_sort_3d<PositionType, decltype(pp_view), execution_space>(
                pp_view, n_grid_global_arr, n_grid_local_arr, local_offset_arr, tile_size_arr, w,
                permute, bin_offsets, nParticles);
            Kokkos::fence();
            IpplTimings::stopTimer(scatterKernelSortTimer);

            // Dispatch to templated scatter functor based on kernel width
            constexpr int MaxW = 20;
            if (config.method == Interpolation::ScatterMethod::Tiled) {
                Interpolation::detail::ScatterDispatcher<1, MaxW>::template dispatch_3d<
                    PositionType, execution_space, Kernel, T, view_type, decltype(pp_view),
                    decltype(permute), decltype(bin_offsets)>(
                    w, bin_offsets, permute, pp_view, dview_m, full_view, n_grid_global_arr,
                    n_grid_local_arr, local_offset_arr, num_tiles, config.tile_size_3d,
                    config.tile_size_3d, config.tile_size_3d, config.z_tiles, nghost, inv_hw,
                    kernel, config.team_size);
            } else if (config.method == Interpolation::ScatterMethod::OutputFocused) {
                Interpolation::detail::OutputFocusedScatterDispatcher<
                    1, MaxW>::template dispatch_3d<PositionType, execution_space, Kernel, T,
                                                   view_type, decltype(pp_view), decltype(permute),
                                                   decltype(bin_offsets)>(
                    w, bin_offsets, permute, pp_view, dview_m, full_view, n_grid_global_arr,
                    n_grid_local_arr, local_offset_arr, num_tiles, config.tile_size_3d,
                    config.tile_size_3d, config.tile_size_3d, config.z_tiles, nghost, inv_hw,
                    kernel, config.team_size);
            }
            Kokkos::fence();
            IpplTimings::stopTimer(scatterKernelTimer);
            return;
        }

        // Fall back to atomic implementation using generic functor
        auto pp_view = pp.getView();

        // Use AtomicScatterFunctor with T (real values) scattering to complex field
        Interpolation::detail::AtomicScatterFunctor<Dim, PositionType, execution_space, Kernel, T,
                                                    view_type, decltype(pp_view)>
            scatter_functor{.x            = pp_view,
                            .values       = dview_m,
                            .grid         = full_view,
                            .n_grid       = {ngrid_global[0], ngrid_global[1], ngrid_global[2]},
                            .n_grid_local = {ngrid_local[0], ngrid_local[1], ngrid_local[2]},
                            .local_offset = {local_offset[0], local_offset[1], local_offset[2]},
                            .w            = w,
                            .nghost       = nghost,
                            .inv_hw       = inv_hw,
                            .kernel       = kernel};

        Kokkos::parallel_for("atomic_scatter", policy_type(0, nParticles), scatter_functor);
        Kokkos::fence();
        IpplTimings::stopTimer(scatterKernelTimer);
    }

    // Kernel-based gather (new generalized interface)
    template <typename T, class... Properties>
    template <typename Field, typename P2, typename Kernel>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp,
        const Kernel& kernel, bool addToAttribute, const Interpolation::GatherConfig& config) {
        constexpr unsigned Dim                   = Field::dim;
        using PositionType                       = typename Field::Mesh_t::value_type;
        using complex_type                       = typename Field::value_type;
        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);

        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        f.fillHalo();
        IpplTimings::stopTimer(fillHaloTimer);

        static IpplTimings::TimerRef gatherKernelTimer = IpplTimings::getTimer("gatherKernel");
        IpplTimings::startTimer(gatherKernelTimer);

        using view_type     = typename Field::view_type;
        view_type full_view = f.getView();
        const int nghost    = f.getNghost();

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const NDIndex<Dim>& gDom       = layout.getDomain();

        const int w               = kernel.width();
        const int hw              = w / 2;
        const bool odd            = (w & 1);
        const PositionType inv_hw = PositionType(2.0) / w;

        // Use GLOBAL grid dimensions for coordinate scaling
        Vector<int, Dim> ngrid_global;
        Vector<int, Dim> ngrid_local;
        Vector<int, Dim> local_offset;
        for (unsigned d = 0; d < Dim; ++d) {
            ngrid_global[d] = gDom[d].length();
            ngrid_local[d]  = lDom[d].length();
            local_offset[d] = lDom[d].first();
        }

        using policy_type       = Kokkos::RangePolicy<execution_space>;
        const size_t nParticles = *(this->localNum_mp);

        // Dispatch based on method
        if constexpr (Dim == 3) {
            if (config.method == Interpolation::GatherMethod::Native
                || config.method == Interpolation::GatherMethod::Tiled
                || config.method == Interpolation::GatherMethod::AtomicSort) {
                // Use optimized tiled interpolation with warp-level parallelism
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
                if constexpr (
#ifdef KOKKOS_ENABLE_CUDA
                    std::is_same_v<execution_space, Kokkos::Cuda>
#endif
#ifdef KOKKOS_ENABLE_HIP
                        std::is_same_v<execution_space, Kokkos::HIP>
#endif
                ) {
                    // Verify field has sufficient ghost cells for kernel width
                    if (nghost < hw) {
                        throw std::runtime_error(
                            "Field ghost cells (nghost=" + std::to_string(nghost)
                            + ") insufficient for kernel width (" + std::to_string(w)
                            + "). Need nghost >= " + std::to_string(hw));
                    }

                    static IpplTimings::TimerRef gatherSortTimer =
                        IpplTimings::getTimer("gatherKernelSort");
                    IpplTimings::startTimer(gatherSortTimer);

                    using memory_space = typename execution_space::memory_space;

                    // Sort particles by Morton code
                    auto x_view = pp.getView();
                    // Kokkos::View<size_t*, memory_space> permute("permute", nParticles);
                    // detail::SortBufferManager<memory_space>& buf_manager = detail::getDefaultSortBufferManager<memory_space>();
                    // buf_manager.ensureCapacity(nParticles);
                    // auto &permute = buf_manager.indicesSorted();

                    Vector<PositionType, 3> origin;
                    Vector<PositionType, 3> invdx;
                    Vector<size_t, 3> ngrid_vec;
                    for (unsigned d = 0; d < 3; ++d) {
                        origin[d]    = -M_PI;
                        invdx[d]     = ngrid_global[d] / (PositionType(2.0) * M_PI);
                        ngrid_vec[d] = ngrid_global[d];
                    }

                    // detail::sortParticles<3, execution_space, PositionType>(
                    //     x_view, permute, origin, invdx, ngrid_vec, nParticles);
                    //
                    // Kokkos::fence();
                    Kokkos::Array<size_type, 3> num_tiles;
                    for (unsigned d = 0; d < 3; ++d) {
                        num_tiles[d] = (ngrid_local[d] + config.tile_size_3d - 1) / config.tile_size_3d + 1;
                    }
                    auto total_tiles = size_t(num_tiles[0]) * size_t(num_tiles[1]) * size_t(num_tiles[2]);

                    auto &buf_handler = detail::getDefaultSortBufferManager<memory_space>();
                    buf_handler.ensureCapacity(std::max(nParticles, total_tiles + 1));
                    auto& permute = buf_handler.indices();
                    auto& bin_offsets = buf_handler.indicesSorted();
                    Kokkos::Array<int, 3> n_grid_global_arr;
                    Kokkos::Array<int, 3> n_grid_local_arr;
                    Kokkos::Array<int, 3> local_offset_arr;
                    Kokkos::Array<int, 3> tile_size_arr;
                    for (unsigned d = 0; d < 3; ++d) {
                        n_grid_global_arr[d] = ngrid_global[d];
                        n_grid_local_arr[d]  = ngrid_local[d];
                        local_offset_arr[d]  = local_offset[d];
                        tile_size_arr[d]     = config.tile_size_3d;
                    }

                    Interpolation::detail::bin_sort_3d<PositionType, decltype(x_view), execution_space>(
                        x_view, n_grid_global_arr, n_grid_local_arr, local_offset_arr, tile_size_arr, w,
                        permute, bin_offsets, nParticles);
                    Kokkos::fence();

                    IpplTimings::stopTimer(gatherSortTimer);

                    // Dispatch to specialized kernel
                    constexpr int MaxW = 20;
                    int n0 = ngrid_global[0], n1 = ngrid_global[1], n2 = ngrid_global[2];
                    if (config.method == Interpolation::GatherMethod::Native) {
                        Interpolation::detail::CudaGatherDispatcher<1, MaxW>::template dispatch_3d<
                            PositionType, decltype(x_view), decltype(permute), decltype(full_view),
                            Kernel, T>(w, nParticles, x_view, permute, full_view, dview_m, nghost,
                                       ngrid_global, ngrid_local, local_offset, inv_hw, kernel,
                                       addToAttribute);
                    } else if (config.method == Interpolation::GatherMethod::Tiled) {
                        Interpolation::detail::TiledGatherDispatcher<1, MaxW>::template dispatch_3d<
                            PositionType, execution_space, Kernel, T, decltype(full_view),
                            decltype(x_view), decltype(permute)>(
                            w, nParticles, x_view, permute, full_view, dview_m, nghost,
                            ngrid_global, ngrid_local, local_offset, inv_hw, kernel,
                            addToAttribute);
                    } else if (config.method == Interpolation::GatherMethod::AtomicSort) {
                        Interpolation::detail::AtomicSortGatherFunctor<
                            3, PositionType, execution_space, Kernel, T, view_type,
                            decltype(pp.getView()), decltype(permute)>
                            gather_functor{
                                .x            = pp.getView(),
                                .grid         = full_view,
                                .permute      = permute,
                                .values       = dview_m,
                                .n_grid       = {ngrid_global[0], ngrid_global[1], ngrid_global[2]},
                                .n_grid_local = {ngrid_local[0], ngrid_local[1], ngrid_local[2]},
                                .local_offset = {local_offset[0], local_offset[1], local_offset[2]},
                                .w            = w,
                                .nghost       = nghost,
                                .inv_hw       = inv_hw,
                                .add_to_attribute = addToAttribute,
                                .kernel           = kernel};

                        Kokkos::parallel_for("atomic_gather", policy_type(0, nParticles),
                                             gather_functor);
                    }

                    Kokkos::fence();
                } else
#endif
                {
                    // Fallback to atomic for non-CUDA - use generic functor
                    // Pass position view directly - no copy needed
                    auto pp_view = pp.getView();

                    // Use AtomicGatherFunctor - value_type is T (real), will extract real part from
                    // complex grid
                    Interpolation::detail::AtomicGatherFunctor<
                        3, PositionType, execution_space, Kernel, T, view_type, decltype(pp_view)>
                        gather_functor{
                            .x                = pp_view,
                            .grid             = full_view,
                            .values           = dview_m,
                            .n_grid           = {ngrid_global[0], ngrid_global[1], ngrid_global[2]},
                            .n_grid_local     = {ngrid_local[0], ngrid_local[1], ngrid_local[2]},
                            .local_offset     = {local_offset[0], local_offset[1], local_offset[2]},
                            .w                = w,
                            .nghost           = nghost,
                            .inv_hw           = inv_hw,
                            .add_to_attribute = addToAttribute,
                            .kernel           = kernel};

                    Kokkos::parallel_for("atomic_gather", policy_type(0, nParticles),
                                         gather_functor);
                }
            } else {
                // Atomic method (default) - use generic functor
                // Pass position view directly - no copy needed
                auto pp_view = pp.getView();

                // Use AtomicGatherFunctor - value_type is T (real), will extract real part from
                // complex grid
                Interpolation::detail::AtomicGatherFunctor<3, PositionType, execution_space, Kernel,
                                                           T, view_type, decltype(pp_view)>
                    gather_functor{
                        .x                = pp_view,
                        .grid             = full_view,
                        .values           = dview_m,
                        .n_grid           = {ngrid_global[0], ngrid_global[1], ngrid_global[2]},
                        .n_grid_local     = {ngrid_local[0], ngrid_local[1], ngrid_local[2]},
                        .local_offset     = {local_offset[0], local_offset[1], local_offset[2]},
                        .w                = w,
                        .nghost           = nghost,
                        .inv_hw           = inv_hw,
                        .add_to_attribute = addToAttribute,
                        .kernel           = kernel};

                Kokkos::parallel_for("atomic_gather", policy_type(0, nParticles), gather_functor);
            }
        } else {
            // Currently not implemented
        }

        Kokkos::fence();
        IpplTimings::stopTimer(gatherKernelTimer);
        IpplTimings::stopTimer(gatherTimer);
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::applyPermutation(const hash_type& permutation) {
        const auto view = this->getView();
        const auto size = this->getParticleCount();

        view_type temp("copy", size);

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "Copy to temp", policy_type(0, size),
            KOKKOS_LAMBDA(const size_type& i) { temp(permutation(i)) = view(i); });

        Kokkos::fence();

        Kokkos::deep_copy(Kokkos::subview(view, Kokkos::make_pair<size_type, size_type>(0, size)),
                          temp);
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::internalCopy(const hash_type& indices) {
        auto copySize = indices.size();
        create(copySize);

        auto view       = this->getView();
        const auto size = this->getParticleCount();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "Copy to temp", policy_type(0, copySize),
            KOKKOS_LAMBDA(const size_type& i) { view(size + i) = view(i); });

        Kokkos::fence();
    }

    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::scatterPIFNUFFT(
        Field<FT, Dim, M, C>& f, Field<ST, Dim, M, C>& Sk,
        const ParticleAttrib<Vector<PT, Dim>, Properties...>& pp,
        FFT<NUFFTransform, Field<ST, Dim, M, C>>* nufft, const MPI_Comm& spaceComm) const {
        static IpplTimings::TimerRef scatterPIFNUFFTTimer =
            IpplTimings::getTimer("ScatterPIFNUFFT");
        IpplTimings::startTimer(scatterPIFNUFFTTimer);

        auto q = *this;

        typename Field<FT, Dim, M, C>::uniform_type tempField;

        FieldLayout<Dim>& layout = f.getLayout();
        M& mesh                  = f.get_mesh();

        tempField.initialize(mesh, layout);

        tempField = 0.0;

        nufft->transform(pp, q, tempField);

        using view_type                                 = typename Field<FT, Dim, M, C>::view_type;
        view_type fview                                 = f.getView();
        view_type viewLocal                             = tempField.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost                                = f.getNghost();

        IpplTimings::stopTimer(scatterPIFNUFFTTimer);

        // int nRanksSpace;
        // MPI_Comm_size(spaceComm, &nRanksSpace);

        // static IpplTimings::TimerRef scatterAllReducePIFTimer =
        //     IpplTimings::getTimer("scatterAllReducePIF");
        // IpplTimings::startTimer(scatterAllReducePIFTimer);
        // if (nRanksSpace > 1) {
        //     // Cray MPI has problems reducing complex data type GPU-aware so do this trick to
        //     // speed up
        //     double* raw_ptr_viewLocal = reinterpret_cast<double*>(viewLocal.data());
        //     double* raw_ptr_fview     = reinterpret_cast<double*>(fview.data());
        //     int viewSize              = fview.extent(0) * fview.extent(1) * fview.extent(2);
        //     // MPI_Allreduce(viewLocal.data(), fview.data(), viewSize,
        //     //               MPI_C_DOUBLE_COMPLEX, MPI_SUM, spaceComm);
        //     MPI_Allreduce(raw_ptr_viewLocal, raw_ptr_fview, 2 * viewSize, MPI_DOUBLE, MPI_SUM,
        //                   spaceComm);

        //} else {
        Kokkos::deep_copy(fview, viewLocal);
        //}
        // IpplTimings::stopTimer(scatterAllReducePIFTimer);

        IpplTimings::startTimer(scatterPIFNUFFTTimer);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "Multiply with shape functions",
            mdrange_type(
                {nghost, nghost, nghost},
                {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                fview(i, j, k) *= Skview(i, j, k);
            });

        IpplTimings::stopTimer(scatterPIFNUFFTTimer);
    }

    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, class FT, class ST, class PT>
    void ParticleAttrib<T, Properties...>::gatherPIFNUFFT(
        Field<FT, Dim, M, C>& f, Field<ST, Dim, M, C>& Sk,
        const ParticleAttrib<Vector<PT, Dim>, Properties...>& pp,
        FFT<NUFFTransform, Field<ST, Dim, M, C>>* nufft, ParticleAttrib<PT, Properties...>& q) {
        static IpplTimings::TimerRef gatherPIFNUFFTTimer = IpplTimings::getTimer("GatherPIFNUFFT");
        IpplTimings::startTimer(gatherPIFNUFFTTimer);

        typename Field<FT, Dim, M, C>::uniform_type tempField;

        FieldLayout<Dim>& layout = f.getLayout();
        M& mesh                  = f.get_mesh();
        const auto& lDom         = layout.getLocalNDIndex();

        tempField.initialize(mesh, layout);

        using view_type                                 = typename Field<FT, Dim, M, C>::view_type;
        using vector_type                               = typename M::vector_type;
        view_type fview                                 = f.getView();
        view_type tempview                              = tempField.getView();
        auto qview                                      = q.getView();
        typename Field<ST, Dim, M, C>::view_type Skview = Sk.getView();
        const int nghost                                = f.getNghost();
        const vector_type& dx                           = mesh.getMeshSpacing();
        const auto& domain                              = layout.getDomain();
        vector_type Len;
        Vector<int, Dim> N;

        for (unsigned d = 0; d < Dim; ++d) {
            N[d]   = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        double pi                    = std::acos(-1.0);
        Kokkos::complex<double> imag = {0.0, 1.0};
        size_t Np                    = *(this->localNum_mp);

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Gather NUFFT",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    Vector<int, 3> iVec = {i, j, k};
                    for (unsigned d = 0; d < Dim; ++d) {
                        iVec[d] = iVec[d] - nghost + lDom[d].first();
                    }
                    Vector<double, 3> kVec;

                    double Dr = 0.0;
                    for (size_t d = 0; d < Dim; ++d) {
                        bool shift = (iVec[d] > (N[d] / 2));
                        kVec[d]    = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                        // kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                        Dr += kVec[d] * kVec[d];
                    }

                    tempview(i, j, k) = fview(i, j, k);

                    bool isNotZero = (Dr != 0.0);
                    double factor  = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0)));

                    tempview(i, j, k) *= -Skview(i, j, k) * (imag * kVec[gd] * factor);
                });

            nufft->transform(pp, q, tempField);

            Kokkos::parallel_for(
                "Assign E gather NUFFT", Np,
                KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i)[gd] = qview(i); });
        }

        IpplTimings::stopTimer(gatherPIFNUFFTTimer);
    }

    /*
     * Non-class function
     *
     */

    /**
     * @brief Non-class interface for scattering particle attribute data onto a field.
     *
     * This overload preserves legacy functionality by providing a default iteration policy.
     * It calls the member scatter() with a default Kokkos::RangePolicy.
     *
     * @note The default behaviour is to scatter all particles without any custom index mapping.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @tparam policy_type (Default: `Kokkos::RangePolicy<typename Field::execution_space>`)
     * @param attrib The particle attribute to scatter.
     * @param f The field onto which the data is scattered.
     * @param pp The ParticleAttrib representing particle positions.
     */
    template <typename Attrib1, typename Field, typename Attrib2,
              typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.scatter(f, pp, policy_type(0, attrib.getParticleCount()));
    }

    /**
     * @brief Non-class interface for scattering with a custom iteration policy and optional index
     * array.
     *
     * This overload allows the caller to specify a custom `Kokkos::range_policy` and an optional
     * `ippl::hash_type` array. It forwards the parameters to the member scatter() function.
     *
     * @note See ParticleAttrib::scatter() for more information on the custom iteration
     * functionality.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @tparam policy_type (Default: `Kokkos::RangePolicy<typename Field::execution_space>`)
     * @param attrib The particle attribute to scatter.
     * @param f The field onto which the data is scattered.
     * @param pp The ParticleAttrib representing particle positions.
     * @param iteration_policy A custom `Kokkos::range_policy` defining the iteration range.
     * @param hash_array An optional `ippl::hash_type` array for index mapping.
     */
    template <typename Attrib1, typename Field, typename Attrib2,
              typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp,
                        policy_type iteration_policy, typename Attrib1::hash_type hash_array = {}) {
        attrib.scatter(f, pp, iteration_policy, hash_array);
    }

    /**
     * @brief Non-class interface for gathering field data into a particle attribute.
     *
     * This interface calls the member ParticleAttrib::gather() function with the provided
     * parameters and preserving legacy behavior by assigning `addToAttribute` a default value.
     *
     * @note See ParticleAttrib::gather() for more information on the behavior of `addToAttribute`.
     *
     * @tparam Attrib1 The type of the particle attribute.
     * @tparam Field The type of the field.
     * @tparam Attrib2 The type of the particle position attribute.
     * @param attrib The particle attribute to gather data into.
     * @param f The field from which data is gathered.
     * @param pp The ParticleAttrib representing particle positions.
     * @param addToAttribute If true, the gathered field value is added to the current attribute
     * value; otherwise, the attribute value is overwritten.
     */
    template <typename Attrib1, typename Field, typename Attrib2>
    inline void gather(Attrib1& attrib, Field& f, const Attrib2& pp,
                       const bool addToAttribute = false) {
        attrib.gather(f, pp, addToAttribute);
    }

    template <typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4,
              class... Properties>
    inline void gatherPIFNUFFT(ParticleAttrib<P1, Properties...>& attrib, Field<P2, Dim, M, C>& f,
                               Field<P3, Dim, M, C>& Sk,
                               const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp,
                               ippl::FFT<ippl::NUFFTransform, Field<P3, Dim, M, C>>* nufft,
                               ParticleAttrib<P4, Properties...>& q) {
        attrib.gatherPIFNUFFT(f, Sk, pp, nufft, q);
    }

    template <typename P1, unsigned Dim, class M, class C, typename P2, typename P3, typename P4,
              class... Properties>
    inline void scatterPIFNUFFT(const ParticleAttrib<P1, Properties...>& attrib,
                                Field<P2, Dim, M, C>& f, Field<P3, Dim, M, C>& Sk,
                                const ParticleAttrib<Vector<P4, Dim>, Properties...>& pp,
                                FFT<NUFFTransform, Field<P3, Dim, M, C>>* nufft,
                                const MPI_Comm& spaceComm = MPI_COMM_WORLD) {
        attrib.scatterPIFNUFFT(f, Sk, pp, nufft, spaceComm);
    }

#define DefineParticleReduction(fun, name, op, MPI_Op)            \
    template <typename T, class... Properties>                    \
    T ParticleAttrib<T, Properties...>::name() {                  \
        T temp            = 0.0;                                  \
        using policy_type = Kokkos::RangePolicy<execution_space>; \
        Kokkos::parallel_reduce(                                  \
            "fun", policy_type(0, *(this->localNum_mp)),          \
            KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {        \
                T myVal = dview_m(i);                             \
                op;                                               \
            },                                                    \
            Kokkos::fun<T>(temp));                                \
        T globaltemp = 0.0;                                       \
        Comm->allreduce(temp, globaltemp, 1, MPI_Op<T>());        \
        return globaltemp;                                        \
    }

    DefineParticleReduction(Sum, sum, valL += myVal, std::plus)
    DefineParticleReduction(Max, max, if (myVal > valL) valL = myVal, std::greater)
    DefineParticleReduction(Min, min, if (myVal < valL) valL = myVal, std::less)
    DefineParticleReduction(Prod, prod, valL *= myVal, std::multiplies)
}  // namespace ippl
