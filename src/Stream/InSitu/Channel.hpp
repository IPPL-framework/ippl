#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <sstream>

#include "Utility/IpplException.h"

#include "Stream/InSitu/Channel.h"

namespace ippl {

    // ==================================================================
    // Type-erased field array (templated implementation)
    // ==================================================================

    namespace detail {

        template <typename T, unsigned Dim, class... ViewArgs>
        class MeshFieldArrayImpl : public MeshFieldArrayBase {
        public:
            using Field_type   = Field<T, Dim, ViewArgs...>;
            using DeviceView_t = typename Field_type::view_type;
            using HostView_t   = Kokkos::View<typename DeviceView_t::data_type, Kokkos::LayoutLeft,
                                              Kokkos::HostSpace>;

            explicit MeshFieldArrayImpl(const std::string& arrayName, const Field_type& field)
                : name_m(arrayName)
                , field_m(field) {}

            const std::string& arrayName() const override { return name_m; }
            UpdatePolicy policy() const override { return policy_m; }
            void setPolicy(UpdatePolicy p) override { policy_m = p; }

            void initConduit(conduit_cpp::Node& data, const std::string& topoName, int rank,
                             Inform& info) override {
                const auto& Layout_      = field_m.getLayout();
                const size_t nGhost      = field_m.getNghost();
                const auto LocalNDIndex_ = Layout_.getLocalNDIndex();

                const size_t nx = LocalNDIndex_[0].length();
                size_t ny       = 1;
                size_t nz       = 1;
                if constexpr (Dim >= 2)
                    ny = LocalNDIndex_[1].length();
                if constexpr (Dim >= 3)
                    nz = LocalNDIndex_[2].length();

                // Allocate persistent host mirror (interior only, no ghosts)
                if constexpr (Dim == 1) {
                    hostMirror_m = HostView_t("host_" + name_m, nx);
                } else if constexpr (Dim == 2) {
                    hostMirror_m = HostView_t("host_" + name_m, nx, ny);
                } else {
                    hostMirror_m = HostView_t("host_" + name_m, nx, ny, nz);
                }

                nghostStored_m = nGhost;
                dimsStored_m   = {nx, ny, nz};

                // First device->host copy
                copyInterior(nGhost, nx, ny, nz);

                // Set Conduit field node (external pointers into persistent mirror)
                auto fields     = data["fields"];
                auto field_node = fields[name_m];

                field_node["association"].set_string("element");
                field_node["topology"].set_string(topoName);
                field_node["volume_dependent"].set_string("false");

                const auto n_elems = hostMirror_m.size();
                if constexpr (std::is_scalar_v<T>) {
                    field_node["values"].set_external(hostMirror_m.data(), n_elems);
                } else if constexpr (is_vector_v<T>) {
                    using elem_t        = std::remove_pointer_t<decltype(hostMirror_m.data())>;
                    const size_t stride = sizeof(elem_t);
                    if (n_elems > 0) {
                        field_node["values/x"].set_external(&hostMirror_m.data()[0][0], n_elems, 0,
                                                            stride);
                        if constexpr (T::dim >= 2)
                            field_node["values/y"].set_external(&hostMirror_m.data()[0][1], n_elems,
                                                                0, stride);
                        if constexpr (T::dim >= 3)
                            field_node["values/z"].set_external(&hostMirror_m.data()[0][2], n_elems,
                                                                0, stride);
                    } else {
                        using component_type = typename T::value_type;
                        field_node["values/x"].set_external(static_cast<component_type*>(nullptr),
                                                            0);
                        if constexpr (T::dim >= 2)
                            field_node["values/y"].set_external(
                                static_cast<component_type*>(nullptr), 0);
                        if constexpr (T::dim >= 3)
                            field_node["values/z"].set_external(
                                static_cast<component_type*>(nullptr), 0);
                    }
                }

                info << level4 << "  MeshChannel: initialized array '" << name_m << "' (" << nx
                     << "x" << ny << "x" << nz << ")" << endl;
            }

            void refresh(int /*rank*/, Inform& /*info*/) override {
                if (policy_m == UpdatePolicy::Static)
                    return;
                copyInterior(nghostStored_m, dimsStored_m[0], dimsStored_m[1], dimsStored_m[2]);
                clearDirty();
            }

        private:
            std::string name_m;
            const Field_type& field_m;
            HostView_t hostMirror_m;
            size_t nghostStored_m = 0;
            std::array<size_t, 3> dimsStored_m{0, 0, 0};

            void copyInterior(size_t nGhost, size_t nx, size_t ny, size_t nz) {
                const auto& fullDeviceView = field_m.getView();
                // Copy full device view to host, then extract interior into
                // the persistent mirror.  The full mirror is temporary; only
                // the interior buffer persists across steps.
                auto hostMirrorFull =
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fullDeviceView);
                if constexpr (Dim == 1) {
                    const auto r0 = Kokkos::make_pair(nGhost, nGhost + nx);
                    Kokkos::deep_copy(hostMirror_m, Kokkos::subview(hostMirrorFull, r0));
                } else if constexpr (Dim == 2) {
                    const auto r0 = Kokkos::make_pair(nGhost, nGhost + nx);
                    const auto r1 = Kokkos::make_pair(nGhost, nGhost + ny);
                    Kokkos::deep_copy(hostMirror_m, Kokkos::subview(hostMirrorFull, r0, r1));
                } else {
                    const auto r0 = Kokkos::make_pair(nGhost, nGhost + nx);
                    const auto r1 = Kokkos::make_pair(nGhost, nGhost + ny);
                    const auto r2 = Kokkos::make_pair(nGhost, nGhost + nz);
                    Kokkos::deep_copy(hostMirror_m, Kokkos::subview(hostMirrorFull, r0, r1, r2));
                }
            }
        };

    }  // namespace detail

    // ==================================================================
    // MeshChannelHandle::addArray
    // ==================================================================

    template <typename T, unsigned Dim, class... ViewArgs>
    MeshChannelHandle& MeshChannelHandle::addArray(const std::string& arrayName,
                                                   const Field<T, Dim, ViewArgs...>& field,
                                                   UpdatePolicy policy) {
        if (dim_m != int(Dim)) {
            throw IpplException("MeshChannelHandle::addArray",
                                "Field dimension does not match mesh channel dimension");
        }
        auto* ch = static_cast<MeshChannelT<Dim>*>(channel_m.get());
        auto array =
            std::make_unique<detail::MeshFieldArrayImpl<T, Dim, ViewArgs...>>(arrayName, field);
        array->setPolicy(policy);
        ch->addArray(std::move(array));
        return *this;
    }

    // ==================================================================
    // MeshChannelT<Dim> implementation
    // ==================================================================

    template <unsigned Dim>
    MeshChannelT<Dim>::MeshChannelT(const std::string& name, const Mesh_t& mesh,
                                    const Layout_t& layout, int nghost, bool useGhostMasks,
                                    GeometryPolicy geometryPolicy, const std::string& basePath)
        : Channel(name)
        , mesh_m(mesh)
        , layout_m(layout)
        , nghost_m(nghost)
        , useGhostMasks_m(useGhostMasks)
        , geometryPolicy_m(geometryPolicy)
        , basePath_m(basePath) {}

    template <unsigned Dim>
    template <typename T, class... ViewArgs>
    MeshChannelT<Dim>& MeshChannelT<Dim>::addArray(const std::string& arrayName,
                                                   const Field<T, Dim, ViewArgs...>& field,
                                                   UpdatePolicy policy) {
        auto array =
            std::make_unique<detail::MeshFieldArrayImpl<T, Dim, ViewArgs...>>(arrayName, field);
        array->setPolicy(policy);
        arrays_m.push_back(std::move(array));
        return *this;
    }

    template <unsigned Dim>
    MeshFieldArrayBase* MeshChannelT<Dim>::findArray(const std::string& arrayName) const {
        for (auto& a : arrays_m) {
            if (a->arrayName() == arrayName)
                return a.get();
        }
        return nullptr;
    }

    template <unsigned Dim>
    bool MeshChannelT<Dim>::dimsChanged() const {
        const auto LocalNDIndex_ = layout_m.getLocalNDIndex();
        const long nx            = long(LocalNDIndex_[0].length());
        long ny = 1, nz = 1;
        if constexpr (Dim >= 2)
            ny = long(LocalNDIndex_[1].length());
        if constexpr (Dim >= 3)
            nz = long(LocalNDIndex_[2].length());
        return nx != cachedNX_m || ny != cachedNY_m || nz != cachedNZ_m;
    }

    template <unsigned Dim>
    void MeshChannelT<Dim>::buildCoordset(conduit_cpp::Node& data, int rank) {
        const auto LocalNDIndex_ = layout_m.getLocalNDIndex();
        const auto Origin_       = mesh_m.getOrigin();
        const auto Spacing_      = mesh_m.getMeshSpacing();

        const size_t extra        = useGhostMasks_m ? size_t(2 * nghost_m) : 0;
        const size_t index_offset = useGhostMasks_m ? size_t(nghost_m) : 0;
        const int dims_n          = 1;  // points = cells + 1

        const size_t nx = LocalNDIndex_[0].length() + extra;
        size_t ny       = 1;
        size_t nz       = 1;
        if constexpr (Dim >= 2)
            ny = LocalNDIndex_[1].length() + extra;
        if constexpr (Dim >= 3)
            nz = LocalNDIndex_[2].length() + extra;

        const double Ox =
            Origin_[0] + (double(int(LocalNDIndex_[0].first()) - int(index_offset))) * Spacing_[0];
        double Oy = 0.0, Oz = 0.0;
        if constexpr (Dim >= 2) {
            Oy = Origin_[1]
                 + (double(int(LocalNDIndex_[1].first()) - int(index_offset))) * Spacing_[1];
        }
        if constexpr (Dim >= 3) {
            Oz = Origin_[2]
                 + (double(int(LocalNDIndex_[2].first()) - int(index_offset))) * Spacing_[2];
        }

        data["coordsets/cart_uniform_coords/type"].set_string("uniform");
        data["topologies/fmesh_topo/type"].set_string("uniform");
        data["topologies/fmesh_topo/coordset"].set_string("cart_uniform_coords");

        {
            data["coordsets/cart_uniform_coords/dims/i"].set(nx + dims_n);
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set(Ox);
            data["topologies/fmesh_topo/origin/x"].set(Ox);
        }
        if constexpr (Dim >= 2) {
            data["coordsets/cart_uniform_coords/dims/j"].set(ny + dims_n);
            data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
            data["coordsets/cart_uniform_coords/origin/y"].set(Oy);
            data["topologies/fmesh_topo/origin/y"].set(Oy);
        }
        if constexpr (Dim >= 3) {
            data["coordsets/cart_uniform_coords/dims/k"].set(nz + dims_n);
            data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
            data["coordsets/cart_uniform_coords/origin/z"].set(Oz);
            data["topologies/fmesh_topo/origin/z"].set(Oz);
        }

        // Cache dims
        cachedNX_m      = long(LocalNDIndex_[0].length());
        cachedNY_m      = Dim >= 2 ? long(LocalNDIndex_[1].length()) : 1;
        cachedNZ_m      = Dim >= 3 ? long(LocalNDIndex_[2].length()) : 1;
        localNumCells_m = nx * ny * nz;
    }

    template <unsigned Dim>
    void MeshChannelT<Dim>::buildRankID(conduit_cpp::Node& data, int rank) {
        const auto LocalNDIndex_ = layout_m.getLocalNDIndex();
        const size_t extra       = useGhostMasks_m ? size_t(2 * nghost_m) : 0;

        const size_t nx = LocalNDIndex_[0].length() + extra;
        size_t ny       = 1;
        size_t nz       = 1;
        if constexpr (Dim >= 2)
            ny = LocalNDIndex_[1].length() + extra;
        if constexpr (Dim >= 3)
            nz = LocalNDIndex_[2].length() + extra;

        const size_t localNumCells = nx * ny * nz;

        // Allocate (or reuse) RankID view
        if (rankIdCells_m.extent(0) != nx || rankIdCells_m.extent(1) != ny
            || rankIdCells_m.extent(2) != nz) {
            rankIdCells_m = RankViewCells_t("rank_id_cells", nx, ny, nz);
        }

        if (localNumCells > 0) {
            using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
            Kokkos::MDRangePolicy<HostExecSpace, Kokkos::Rank<3>> host_policy({0, 0, 0},
                                                                              {nx, ny, nz});
            // Host-only execution space: use a plain C++ lambda to avoid the
            // NVCC restriction on extended __host__ __device__ lambdas inside
            // private member functions.
            Kokkos::parallel_for("fill_rank_ids_channel", host_policy,
                                 [&](const int i, const int j, const int k) {
                                     rankIdCells_m(i, j, k) = rank;
                                 });
        }

        auto fields    = data["fields"];
        auto rankField = fields["RankID"];
        rankField["association"].set_string("element");
        rankField["topology"].set_string("fmesh_topo");
        rankField["volume_dependent"].set_string("false");
        if (localNumCells > 0) {
            rankField["values"].set_external(rankIdCells_m.data(), localNumCells);
        } else {
            rankField["values"].set_external(static_cast<int*>(nullptr), 0);
        }
        data["metadata/vtk_fields/RankID/attribute_type"].set_string("ProcessIds");
    }

    template <unsigned Dim>
    void MeshChannelT<Dim>::init(conduit_cpp::Node& root, int rank) {
        // When basePath_m is set, this mesh is a block inside a parent
        // multimesh channel (catalyst/channels/<basePath>/data/<name>).
        // Otherwise it is a top-level channel (catalyst/channels/<name>).
        auto data = basePath_m.empty()
                        ? root["catalyst/channels/" + name_m]["data"]
                        : root["catalyst/channels/" + basePath_m]["data/" + name_m];
        if (basePath_m.empty()) {
            root["catalyst/channels/" + name_m]["type"].set_string("mesh");
        } else {
            // Each block in a multimesh needs its own type field.
            // Also add the assembly entry so the parent knows about this block.
            data["type"].set_string("mesh");
            root["catalyst/channels/" + basePath_m]["assembly/" + name_m] = name_m;
        }

        // Build shared coordset + topology
        buildCoordset(data, rank);

        // Build shared RankID
        buildRankID(data, rank);

        // Initialize each field array (allocates persistent mirror, sets external)
        for (auto& array : arrays_m) {
            array->initConduit(data, "fmesh_topo", rank,
                               *std::make_unique<Inform>("MeshChannel").get());
        }
    }

    template <unsigned Dim>
    void MeshChannelT<Dim>::execute(conduit_cpp::Node& root, int cycle, double time, int rank) {
        // Update Catalyst state
        auto state = root["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        auto data = basePath_m.empty()
                        ? root["catalyst/channels/" + name_m]["data"]
                        : root["catalyst/channels/" + basePath_m]["data/" + name_m];

        // When basePath_m is set, the parent particle channel may have
        // reset this node (on particle count change).  If the node is
        // empty (no type field), re-initialize from scratch, including
        // the assembly entry on the parent.
        if (!basePath_m.empty() && !data.has_path("type")) {
            data["type"].set_string("mesh");
            root["catalyst/channels/" + basePath_m]["assembly/" + name_m] = name_m;
            buildCoordset(data, rank);
            buildRankID(data, rank);
            for (auto& array : arrays_m) {
                array->initConduit(data, "fmesh_topo", rank,
                                   *std::make_unique<Inform>("MeshChannel").get());
            }
        }
        // Check for mesh repartition
        else if (dimsChanged()) {
            // Rebuild coordset + RankID, re-init all arrays.
            // When basePath_m is set, the parent particle channel may have
            // reset this node (on particle count change), so re-set type.
            if (!basePath_m.empty()) {
                data["type"].set_string("mesh");
            }
            buildCoordset(data, rank);
            buildRankID(data, rank);
            for (auto& array : arrays_m) {
                array->initConduit(data, "fmesh_topo", rank,
                                   *std::make_unique<Inform>("MeshChannel").get());
            }
        } else if (geometryPolicy_m == GeometryPolicy::Dynamic) {
            // Update coordset spacing/origin (bunch may have moved)
            const auto Origin_        = mesh_m.getOrigin();
            const auto Spacing_       = mesh_m.getMeshSpacing();
            const auto LocalNDIndex_  = layout_m.getLocalNDIndex();
            const size_t index_offset = useGhostMasks_m ? size_t(nghost_m) : 0;

            const double Ox =
                Origin_[0]
                + (double(int(LocalNDIndex_[0].first()) - int(index_offset))) * Spacing_[0];
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set(Ox);
            data["topologies/fmesh_topo/origin/x"].set(Ox);
            if constexpr (Dim >= 2) {
                const double Oy =
                    Origin_[1]
                    + (double(int(LocalNDIndex_[1].first()) - int(index_offset))) * Spacing_[1];
                data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
                data["coordsets/cart_uniform_coords/origin/y"].set(Oy);
                data["topologies/fmesh_topo/origin/y"].set(Oy);
            }
            if constexpr (Dim >= 3) {
                const double Oz =
                    Origin_[2]
                    + (double(int(LocalNDIndex_[2].first()) - int(index_offset))) * Spacing_[2];
                data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
                data["coordsets/cart_uniform_coords/origin/z"].set(Oz);
                data["topologies/fmesh_topo/origin/z"].set(Oz);
            }
        }
        // When GeometryPolicy::Static, skip geometry update entirely.

        // Refresh dynamic field arrays
        {
            auto info = std::make_unique<Inform>("MeshChannel");
            for (auto& array : arrays_m) {
                if (array->needsRefresh()) {
                    array->refresh(rank, *info);
                }
            }
        }
    }

    // ==================================================================
    // ParticleChannel implementation
    // ==================================================================

    inline ParticleChannel::ParticleChannel(const std::string& name, void* pc)
        : Channel(name)
        , pc_m(pc) {}

    inline void ParticleChannel::setTransform(const TransformData& td) {
        transform_m = td;
    }

    inline void ParticleChannel::init(conduit_cpp::Node& root, int rank) {
        auto channel = root["catalyst/channels/" + name_m];
        channel["type"].set_string("multimesh");

        // The init function (set by CatalystAdaptor::addParticleChannel<T>)
        // handles block assembly, coordset, topology, R/ID/attribute mirrors,
        // and the optional transform block.
        if (initFn_m) {
            Inform info("ParticleChannel");
            initFn_m(channel, rank, info);
        }
    }

    inline void ParticleChannel::execute(conduit_cpp::Node& root, int cycle, double time,
                                         int rank) {
        auto state = root["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // The execute function (set by CatalystAdaptor::addParticleChannel<T>)
        // deep-copies R/ID/attributes into persistent mirrors and updates the
        // transform block if present.
        auto channel = root["catalyst/channels/" + name_m];
        if (execFn_m) {
            Inform info("ParticleChannel");
            execFn_m(channel, rank, info);
        }
    }

}  // namespace ippl
