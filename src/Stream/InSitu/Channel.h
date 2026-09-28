/**
 * @file Channel.h
 * @brief Mesh-centric visualization channel types for Catalyst in-situ export.
 *
 * A Channel is the unit of data export: one Conduit channel per mesh (with
 * multiple field arrays) or per particle container.  Each channel owns
 * persistent host staging buffers allocated once at registration and reused
 * on every Execute, eliminating the per-step allocation/free churn of the
 * previous flat-registry design.
 */
#ifndef IPPL_CHANNEL_H
#define IPPL_CHANNEL_H

#include "Ippl.h"

#include <array>
#include <catalyst.hpp>
#include <conduit.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(MPI_VERSION)
#include <mpi.h>
#endif

namespace ippl {

    // ==================================================================
    // Update policy
    // ==================================================================

    enum class UpdatePolicy {
        Always,  ///< Copy device->host every Execute (default for dynamic data)
        Static,  ///< Copy once at registration; never refreshed
        Manual,  ///< Copy only when refresh(label) is called
    };

    // ==================================================================
    // Geometry policy (viz-only)
    // ==================================================================

    enum class GeometryPolicy {
        Dynamic,  ///< Re-read mesh origin/spacing every Execute (default;
                  ///<   bunch may have moved)
        Static,   ///< Mesh origin/spacing set once at init, never re-read.
                  ///<   Zero per-step geometry cost.
    };

    // ==================================================================
    // Transform data (shared with CatalystAdaptor)
    // ==================================================================

    struct TransformData {
        ippl::Vector<double, 3> origin{};
        std::array<ippl::Vector<double, 3>, 3> rows{};
        ippl::Vector<double, 3> invOrigin{};
        std::array<ippl::Vector<double, 3>, 3> invRows{};
    };

    // Forward declarations
    class Channel;
    class MeshFieldArrayBase;
    template <unsigned Dim>
    class MeshChannelT;
    class ParticleChannel;
    class MeshChannelHandle;
    class ParticleChannelHandle;

    // ==================================================================
    // Channel base
    // ==================================================================

    /**
     * @brief Abstract base for a visualization export channel.
     *
     * Each concrete channel builds its Conduit node tree once in init() and
     * copies data into persistent staging buffers every execute().
     */
    class Channel {
    public:
        virtual ~Channel() = default;

        virtual void init(conduit_cpp::Node& root, int rank)                            = 0;
        virtual void execute(conduit_cpp::Node& root, int cycle, double time, int rank) = 0;

        /// Find a field array by name (mesh channels only).
        /// Returns nullptr for particle channels or unknown names.
        virtual MeshFieldArrayBase* findArray(const std::string& /*name*/) const { return nullptr; }

        /// Returns "mesh" for mesh channels, "multimesh" for particle channels.
        /// Used by CatalystAdaptor to pass channel type metadata to scripts.
        virtual const char* conduitType() const = 0;

        const std::string& name() const noexcept { return name_m; }

    protected:
        explicit Channel(std::string name)
            : name_m(std::move(name)) {}
        std::string name_m;
    };

    // ==================================================================
    // Type-erased field array on a mesh channel
    // ==================================================================

    /**
     * @brief Base class for a field array attached to a MeshChannel.
     *
     * Concrete instances are templated on the field type and own a persistent
     * host mirror allocated once at init().
     */
    class MeshFieldArrayBase {
    public:
        virtual ~MeshFieldArrayBase() = default;

        virtual const std::string& arrayName() const = 0;
        virtual UpdatePolicy policy() const          = 0;
        virtual void setPolicy(UpdatePolicy p)       = 0;

        /// Allocate host mirror, copy device->host, set Conduit external pointers.
        /// Called once during MeshChannel::init().
        virtual void initConduit(conduit_cpp::Node& data, const std::string& topoName, int rank,
                                 Inform& info) = 0;

        /// Deep-copy device->host into the existing mirror.
        /// Called every Execute for dynamic (Always/Manual-dirty) arrays.
        virtual void refresh(int rank, Inform& info) = 0;

        bool needsRefresh() const { return policy_m == UpdatePolicy::Always || dirty_m; }

        void markDirty() { dirty_m = (policy_m == UpdatePolicy::Manual); }
        void clearDirty() { dirty_m = false; }

    protected:
        UpdatePolicy policy_m = UpdatePolicy::Always;
        bool dirty_m          = false;
    };

    // ==================================================================
    // MeshChannel<Dim> — one Conduit channel per mesh, multiple field arrays
    // ==================================================================

    /**
     * @brief A visualization channel that publishes one mesh with multiple
     * field arrays.
     *
     * Shared per-mesh data (coordset, topology, RankID, ghost masks) is
     * allocated once.  Each field array owns its own persistent host mirror.
     *
     * @tparam Dim Mesh dimension (1, 2, or 3).
     */
    template <unsigned Dim>
    class MeshChannelT : public Channel {
    public:
        using Mesh_t   = UniformCartesian<double, Dim>;
        using Layout_t = FieldLayout<Dim>;

        MeshChannelT(const std::string& name, const Mesh_t& mesh, const Layout_t& layout,
                     int nghost, bool useGhostMasks,
                     GeometryPolicy geometryPolicy = GeometryPolicy::Dynamic,
                     const std::string& basePath = "");

        ~MeshChannelT() override = default;

        template <typename T, class... ViewArgs>
        MeshChannelT& addArray(const std::string& arrayName,
                               const Field<T, Dim, ViewArgs...>& field,
                               UpdatePolicy policy = UpdatePolicy::Always);

        /// Internal: add a pre-constructed type-erased array.
        void addArray(std::unique_ptr<MeshFieldArrayBase> array) {
            arrays_m.push_back(std::move(array));
        }

        MeshFieldArrayBase* findArray(const std::string& arrayName) const override;
        const char* conduitType() const override { return "mesh"; }

        void init(conduit_cpp::Node& root, int rank) override;
        void execute(conduit_cpp::Node& root, int cycle, double time, int rank) override;

    private:
        void buildCoordset(conduit_cpp::Node& data, int rank);
        void buildRankID(conduit_cpp::Node& data, int rank);
        bool dimsChanged() const;

        const Mesh_t& mesh_m;
        const Layout_t& layout_m;
        int nghost_m;
        bool useGhostMasks_m;
        GeometryPolicy geometryPolicy_m = GeometryPolicy::Dynamic;
        std::string basePath_m;

        std::vector<std::unique_ptr<MeshFieldArrayBase>> arrays_m;

        // Shared per-mesh arrays (allocated once)
        using RankViewCells_t = Kokkos::View<int***, Kokkos::HostSpace>;
        RankViewCells_t rankIdCells_m;
        size_t localNumCells_m = 0;

        // Ghost mask cache (keyed by mesh+layout+nghost, reused across steps)
        using HostMaskView1D_t =
            Kokkos::View<unsigned char*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        HostMaskView1D_t ghostMask_m;
        bool ghostMaskReady_m = false;

        // Cached dimensions for change detection
        long cachedNX_m = -1, cachedNY_m = -1, cachedNZ_m = -1;
    };

    // ==================================================================
    // ParticleChannel — one Conduit channel per particle container
    // ==================================================================

    /**
     * @brief A visualization channel that publishes a particle container as a
     * Conduit multimesh with optional bunch-frame transform.
     *
     * All host mirrors (R, ID, attributes, iota, rank_id) are allocated once
     * and reused on every execute.  The transform is published as a
     * transform block (compatible with existing scripts).
     *
     * The type-specific work (attribute iteration, Conduit multimesh setup,
     * D2H copies) is delegated to function objects set by
     * CatalystAdaptor::addParticleChannel<T>.  This keeps ParticleChannel
     * non-templated while preserving full type safety at registration time.
     */
    class ParticleChannel : public Channel {
        friend class CatalystAdaptor;

    public:
        using InitFn    = std::function<void(conduit_cpp::Node& channel, int rank, Inform& info)>;
        using ExecuteFn = std::function<void(conduit_cpp::Node& channel, int rank, Inform& info)>;

        ParticleChannel(const std::string& name, void* pc);
        ~ParticleChannel() override = default;

        void setInitFn(InitFn fn) { initFn_m = std::move(fn); }
        void setExecuteFn(ExecuteFn fn) { execFn_m = std::move(fn); }

        void setTransform(const TransformData& td);
        const char* conduitType() const override { return "multimesh"; }
        bool hasTransform() const { return transform_m.has_value(); }
        const std::optional<TransformData>& transform() const { return transform_m; }

        void init(conduit_cpp::Node& root, int rank) override;
        void execute(conduit_cpp::Node& root, int cycle, double time, int rank) override;

    private:
        void* pc_m;  ///< Type-erased ParticleBaseBase*

        std::optional<TransformData> transform_m;
        InitFn initFn_m;
        ExecuteFn execFn_m;

        // Persistent staging (allocated once at init, reallocated on grow)
        Kokkos::View<int64_t*, Kokkos::HostSpace> iota_m{"iota", 0};
        Kokkos::View<int*, Kokkos::HostSpace> rankId_m{"rank_id", 0};
        size_t lastLocalNum_m = static_cast<size_t>(-1);
    };

    // ==================================================================
    // Handle classes
    // ==================================================================

    /**
     * @brief Lightweight handle returned by addMeshChannel().
     * Allows chaining addArray() calls.
     */
    class MeshChannelHandle {
    public:
        explicit MeshChannelHandle(std::shared_ptr<void> ch, int dim)
            : channel_m(std::move(ch))
            , dim_m(dim) {}
        explicit operator bool() const { return channel_m != nullptr; }
        int dim() const { return dim_m; }

        template <typename T, unsigned Dim, class... ViewArgs>
        MeshChannelHandle& addArray(const std::string& arrayName,
                                    const Field<T, Dim, ViewArgs...>& field,
                                    UpdatePolicy policy = UpdatePolicy::Always);

    private:
        std::shared_ptr<void> channel_m;
        int dim_m;
    };

    /**
     * @brief Lightweight handle returned by addParticleChannel().
     */
    class ParticleChannelHandle {
    public:
        explicit ParticleChannelHandle(ParticleChannel* ch)
            : channel_m(ch) {}
        explicit operator bool() const { return channel_m != nullptr; }

        ParticleChannelHandle& setTransform(const TransformData& td) {
            if (channel_m)
                channel_m->setTransform(td);
            return *this;
        }

    private:
        ParticleChannel* channel_m;
    };

}  // namespace ippl

#include "Stream/InSitu/Channel.hpp"

#endif  // IPPL_CHANNEL_H
