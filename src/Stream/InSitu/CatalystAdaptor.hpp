#pragma once
#include "Ippl.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Stream/InSitu/CatalystAdaptor.h"
#include "Stream/InSitu/CatalystVisitors.h"
#include "Stream/Registry/VisRegistryRuntime.h"

namespace {
    /**
     * @brief Device functor for zeroing the interior of a ghost-mask field.
     *
     * Using a named functor instead of a lambda avoids the NVCC restriction that
     * extended __host__ __device__ lambdas cannot appear inside private/protected
     * member functions.
     */
    template <typename View>
    struct ZeroGhostMaskInterior {
        View view;

        KOKKOS_INLINE_FUNCTION void operator()(const int i) const { view(i) = 0; }

        KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j) const { view(i, j) = 0; }

        KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
            view(i, j, k) = 0;
        }
    };
}  // namespace

namespace ippl {

    // ==============================================================================================
    //  HELPERS
    //  =====================================================================================
    // ==============================================================================================
    void CatalystAdaptor::setNodeScript(conduit_cpp::Node nodePath, const std::string envVar,
                                        const std::filesystem::path defaultFilePath) {
        const char* filePathEnv = std::getenv(envVar.c_str());
        std::filesystem::path filePath;
        if (filePathEnv && std::filesystem::exists(filePathEnv)) {
            catalystInfo_m << level4 << "::Initialize()::setNodeScripts(...):\n"
                           << "                Using " << envVar << " from environment:\n"
                           << "                " << filePathEnv << endl;
            filePath = filePathEnv;
        } else {
            catalystInfo_m << level4 << "::Initialize()::setNodeScripts(...): No valid " << envVar
                           << " set.\n"
                           << "                Using default:\n"
                           << "                " << defaultFilePath << endl;
            filePath = defaultFilePath;
        }
        nodePath.set(filePath.string());
    }

}  // namespace ippl

// =====================================================================================
// STEERING:
// =====================================================================================
#include "Stream/InSitu/CatalystAdaptorSteering.hpp"
// =====================================================================================

namespace ippl {
    ///////////////////////////////////////////////////////////////
    // Note:
    // this does not really need to be a separate function call,
    // May we should just inline this into the execute functions ...
    ///////////////////////////////////////////////////////////////
    void CatalystAdaptor::fetchResults() {
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results_m));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
    }

    // =====================================================================================
    // Channel-based Initialize / Execute
    // =====================================================================================

    void CatalystAdaptor::Initialize(const std::shared_ptr<VisRegistryRuntime>& steerReg) {
        if (!visEnabled_m)
            return;

        catalystInfo_m
            << level4
            << "::Initialize() START============================================================= 0"
            << endl;

        int all_ready = 1;
#if defined(MPI_VERSION)
        MPI_Allreduce(MPI_IN_PLACE, &all_ready, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        catalystInfo_m << level4 << "::Initialize() Ranks ready for catalyst int: " << all_ready
                       << " ranks" << endl;

        steerRegistry_m = steerReg;

        const int fcomm       = MPI_Comm_c2f(MPI_COMM_WORLD);
        const int64_t fcomm64 = static_cast<int64_t>(fcomm);
        node_m["catalyst/mpi_comm"].set(fcomm64);

        setNodeScript(node_m["catalyst/scripts/script/filename"], "CATALYST_PIPELINE_PATH",
                      resourceDir_m / "pipeline_default.py");
        conduit_cpp::Node args = node_m["catalyst/scripts/script/args"];

        // Pass channel names from registered channels.  Mesh blocks nested
        // inside a parent multimesh are reported as "<parent>/<block>" so
        // scripts can locate them within the parent's producer.
        args.append().set_string("--channel_names");
        for (const auto& ch : channels_m) {
            const std::string parent = ch->basePath();
            args.append().set_string(parent.empty() ? ch->name() : parent + "/" + ch->name());
        }

        // Pass channel types so scripts can dispatch without name-pattern matching
        args.append().set_string("--channel_types");
        for (const auto& ch : channels_m) {
            args.append().set_string(ch->conduitType());
        }

        // Ghost layers included in each channel's export (0 when stripped)
        args.append().set_string("--channel_ghosts");
        for (const auto& ch : channels_m) {
            args.append().set_string(std::to_string(ch->ghostLayers()));
        }

        args.append().set_string("--verbosity");
        args.append().set_string(std::to_string(catalystInfo_m.getOutputLevel()));

        args.append().set_string("--VTKextract");
        args.append().set_string(catalystVtk_m);

        args.append().set_string("--png");
        args.append().set_string(catalystPng_m);

        // Directory for extracts written by the pipeline script
        args.append().set_string("--extracts_output_dir");
        args.append().set_string((outputDir_m / "vtk_extracts").string());

        if (experimentName_m) {
            args.append().set_string("--experiment_name");
            args.append().set_string(*experimentName_m);
        }

        args.append().set_string("--live");
        args.append().set_string(catalystLive_m);

        args.append().set_string("--steer");
        args.append().set_string(catalystSteer_m);

        args.append().set_string("--steer_channel_names");

        // Proxy writer (unchanged from old path)
        std::filesystem::path proxyPath = outputDir_m / "catalyst_proxy.xml";
        bool useExistingProxy           = false;
        if (const char* proxyPathEnv = std::getenv("IPPL_CATALYST_PROXY_PATH");
            proxyPathEnv && *proxyPathEnv) {
            proxyPath        = proxyPathEnv;
            useExistingProxy = true;
        } else if (const char* legacyProxyPathEnv = std::getenv("CATALYST_PROXYS_PATH");
                   legacyProxyPathEnv && *legacyProxyPathEnv) {
            proxyPath        = legacyProxyPathEnv;
            useExistingProxy = true;
            catalystWarn_m << "CATALYST_PROXYS_PATH is deprecated; use "
                              "IPPL_CATALYST_PROXY_PATH instead."
                           << endl;
        }

        std::string cfgYaml;
        if (const char* cfg_env = std::getenv("IPPL_PROXY_CONFIG_YAML")) {
            if (std::filesystem::exists(cfg_env)) {
                cfgYaml = std::string(cfg_env);
            } else {
                catalystInfo_m << level4
                               << "::Initialize() IPPL_PROXY_CONFIG_YAML set but file not found: '"
                               << cfg_env << "', using default." << endl;
            }
        }
        if (cfgYaml.empty()) {
            auto default_cfgYaml = (resourceDir_m / "proxy_default_config.yaml").string();
            if (std::filesystem::exists(default_cfgYaml)) {
                cfgYaml = std::move(default_cfgYaml);
            }
        }

        proxyWriter_m.initialize(proxyPath, cfgYaml);
        if (steerEnabled_m) {
            SteerInitVisitor steerInitV{*this};
            steerRegistry_m->forEach(steerInitV);
        }
        const bool generateProxy = !useExistingProxy && proxyOption_m != "OFF";
        int proxyReady           = 1;

        if (ippl::Comm->rank() == 0) {
            if (useExistingProxy) {
                proxyReady = std::filesystem::is_regular_file(proxyPath) ? 1 : 0;
            } else if (generateProxy) {
                proxyReady = proxyWriter_m.produceUnified("SteerableParameters_SCALARS",
                                                          "SteerableParameters")
                                 ? 1
                                 : 0;
            } else {
                proxyReady = std::filesystem::is_regular_file(proxyPath) ? 1 : 0;
            }
        }

#if defined(MPI_VERSION)
        MPI_Bcast(&proxyReady, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        if ((useExistingProxy || generateProxy) && !proxyReady) {
            throw IpplException(
                "Stream::InSitu::CatalystAdaptor::Initialize()",
                "Could not create or read Catalyst proxy XML: " + proxyPath.string());
        }

        if (proxyReady) {
            node_m["catalyst/proxies/proxy_/filename"].set(proxyPath.string());
        } else if (steerEnabled_m) {
            catalystWarn_m << "Steering is enabled without a Catalyst proxy XML because "
                              "IPPL_CATALYST_PROXY_OPTION=OFF and no existing proxy was found at "
                           << proxyPath.string() << endl;
        }

        if (proxyOption_m == "PRODUCE_ONLY") {
            throw IpplException("Stream::InSitu::CatalystAdaptor",
                                "write_proxy_only_run: proxy available at " + proxyPath.string());
        }

        catalystInfo_m << level4
                       << "::Initialize()   Printing Conduit `node_m` instance passed to "
                          "catalyst_initialize() =>"
                       << endl;
        catalystInfo_m << level4 << node_m.to_yaml() << endl;

        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node_m));
        if (err != catalyst_status_ok) {
            catalystInfo_m << level4 << "::Initialize()   Catalyst initialization failed." << endl;
            throw IpplException("Stream::InSitu::CatalystAdaptor::Initialize()",
                                "Failed to initialize Catalyst!!!");
        } else {
            catalystInfo_m << level4 << "::Initialize()   Catalyst initialized successfully."
                           << endl;
        }

        // Reset node_m (scripts are consumed by catalyst_initialize).
        // Channel tree will be built on first Execute() and reused thereafter.
        node_m.reset();
        channelsBuilt_m = false;

        catalystInfo_m
            << level4
            << "::Initialize()  DONE============================================================= 1"
            << endl;
    }
    void CatalystAdaptor::Execute(int cycle, double time,
                                  int rank /* default = ippl::Comm->rank() */) {
        if (!visEnabled_m)
            return;

        catalystInfo_m
            << level4
            << "::Execute() START =============================================================== 0"
            << endl;

        static IpplTimings::TimerRef TMRcatalyst_execute =
            IpplTimings::getTimer("catalyst_execute");
        static IpplTimings::TimerRef TMRchannelExecute = IpplTimings::getTimer("channelExecute");
        static IpplTimings::TimerRef TMRexecSteerVisitor =
            IpplTimings::getTimer("execSteerVisitor");

        // Build channel tree on first Execute (after catalyst_initialize
        // consumed the scripts node).
        if (!channelsBuilt_m) {
            for (auto& ch : channels_m) {
                ch->init(node_m, rank);
            }
            channelsBuilt_m = true;

            if (cycle == 0) {
#if defined(MPI_VERSION)
                MPI_Barrier(MPI_COMM_WORLD);
                catalystInfo_m << level4
                               << "::Execute() [rank = 0]  Printing first Conduit Node ==>" << endl;
                if (catalystInfo_m.getOutputLevel() >= 4 && ippl::Comm->rank() == 0)
                    node_m.print();
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
        }

        // Update state
        auto state = node_m["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // Execute channels (deep-copy dynamic data into persistent mirrors)
        IpplTimings::startTimer(TMRchannelExecute);
        for (auto& ch : channels_m) {
            ch->execute(node_m, cycle, time, rank);
        }
        IpplTimings::stopTimer(TMRchannelExecute);

        // Steering forward
        IpplTimings::startTimer(TMRexecSteerVisitor);
        if (steerEnabled_m) {
            SteerForwardVisitor steerV{*this};
            steerRegistry_m->forEach(steerV);
        }
        IpplTimings::stopTimer(TMRexecSteerVisitor);

        // catalyst_execute
        catalystInfo_m << level4 << "::Execute()::catalyst_execute() ==>" << endl;
        IpplTimings::startTimer(TMRcatalyst_execute);
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node_m));
        IpplTimings::stopTimer(TMRcatalyst_execute);

        if (err != catalyst_status_ok) {
            std::cerr << "::Execute()   Failed to execute Catalyst (runtime path): " << err
                      << std::endl;
        }

        // Steering fetch
        if (steerEnabled_m) {
            static IpplTimings::TimerRef TMRfetchResult =
                IpplTimings::getTimer("fetchSteerParameters");
            IpplTimings::startTimer(TMRfetchResult);

            fetchResults();
            SteerFetchVisitor fetchV{*this};
            steerRegistry_m->forEach(fetchV);

            IpplTimings::stopTimer(TMRfetchResult);
        }

        // NO viewRegistry_m.clear(), NO node_m.reset().
        // The Conduit tree and all staging buffers persist across steps.

        catalystInfo_m
            << level4
            << "::Execute()  DONE =============================================================== 1"
            << endl;
    }

    void CatalystAdaptor::Finalize() {
        if (!visEnabled_m)
            return;

        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node_m));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }

    // =====================================================================================
    // New channel-based API
    // =====================================================================================

    template <unsigned Dim>
    MeshChannelHandle CatalystAdaptor::addMeshChannel(const std::string& name,
                                                      const UniformCartesian<double, Dim>& mesh,
                                                      const FieldLayout<Dim>& layout, int nghost,
                                                      bool useGhostMasks) {
        auto ch = std::make_shared<MeshChannelT<Dim>>(name, mesh, layout, nghost, useGhostMasks);
        channels_m.push_back(ch);
        catalystInfo_m << level4 << "::addMeshChannel('" << name << "', Dim=" << Dim
                       << ", nghost=" << nghost << ")" << endl;
        return MeshChannelHandle(std::static_pointer_cast<void>(ch), Dim);
    }

    template <unsigned Dim>
    MeshChannelHandle CatalystAdaptor::addMeshBlock(const std::string& parentName,
                                                    const std::string& blockName,
                                                    const UniformCartesian<double, Dim>& mesh,
                                                    const FieldLayout<Dim>& layout, int nghost,
                                                    bool useGhostMasks,
                                                    GeometryPolicy geometryPolicy) {
        auto ch = std::make_shared<MeshChannelT<Dim>>(blockName, mesh, layout, nghost,
                                                      useGhostMasks, geometryPolicy, parentName);
        channels_m.push_back(ch);
        catalystInfo_m << level4 << "::addMeshBlock(parent='" << parentName << "', block='"
                       << blockName << "', Dim=" << Dim << ", nghost=" << nghost << ", geometry="
                       << (geometryPolicy == GeometryPolicy::Static ? "Static" : "Dynamic") << ")"
                       << endl;
        return MeshChannelHandle(std::static_pointer_cast<void>(ch), Dim);
    }

    template <typename T>
        requires std::derived_from<std::decay_t<T>, ParticleBaseBase>
    ParticleChannelHandle CatalystAdaptor::addParticleChannel(const std::string& name,
                                                              const T& pc) {
        auto ch = std::make_shared<ParticleChannel>(
            name, const_cast<void*>(static_cast<const void*>(&pc)));
        auto* raw = ch.get();

        // Type-specific init function: builds multimesh, allocates mirrors,
        // sets Conduit external pointers.  Called once by Initialize().
        raw->setInitFn([this, name, &pc, raw](conduit_cpp::Node& channel, int rank, Inform& info) {
            const size_t localNum = pc.getLocalNum();
            channel["type"].set_string("multimesh");
            channel["assembly/particles"]   = "particles";
            channel["assembly/boundingbox"] = "boundingbox";
            channel["assembly/transform"]   = "transform";

            auto data      = channel["data/particles"];
            auto data_help = channel["data/boundingbox"];
            data["type"].set_string("mesh");

            // Persistent iota + rank_id
            auto& iota   = const_cast<Kokkos::View<int64_t*, Kokkos::HostSpace>&>(raw->iota_m);
            auto& rankId = const_cast<Kokkos::View<int*, Kokkos::HostSpace>&>(raw->rankId_m);
            if (iota.extent(0) != localNum) {
                Kokkos::resize(iota, localNum);
                Kokkos::resize(rankId, localNum);
            }
            for (size_t i = 0; i < localNum; ++i) {
                iota(i)   = static_cast<int64_t>(i);
                rankId(i) = rank;
            }

            // Coordset + topology
            data["coordsets/p_explicit_coords/type"].set_string("explicit");
            data["topologies/p_unstructured_topo/coordset"].set_string("p_explicit_coords");
            data["topologies/p_unstructured_topo/type"].set_string("unstructured");
            data["topologies/p_unstructured_topo/elements/shape"].set_string("point");
            data["topologies/p_unstructured_topo/elements/connectivity"].set_external(iota.data(),
                                                                                      localNum);

            // R host mirror (persistent)
            using RAttrib_t = std::remove_reference_t<decltype(pc.R)>;
            using RHost_t   = typename RAttrib_t::host_mirror_type;
            static thread_local RHost_t RHost;
            if (RHost.extent(0) != localNum) {
                Kokkos::resize(RHost, localNum);
            }
            Kokkos::deep_copy(RHost, pc.R.getView());
            viewRegistry_m.set(name + "_R", RHost);

            constexpr unsigned ParticleDim = particle_dim_v<T>;
            static constexpr size_t R_stride =
                sizeof(std::remove_pointer_t<decltype(RHost.data())>);

            if (localNum > 0) {
                data["coordsets/p_explicit_coords/values/x"].set_external(&RHost.data()[0][0],
                                                                          localNum, 0, R_stride);
                if constexpr (ParticleDim >= 2) {
                    data["coordsets/p_explicit_coords/values/y"].set_external(
                        &RHost.data()[0][1], localNum, 0, R_stride);
                }
                if constexpr (ParticleDim >= 3) {
                    data["coordsets/p_explicit_coords/values/z"].set_external(
                        &RHost.data()[0][2], localNum, 0, R_stride);
                }
            }

            // RankID field
            auto fields    = data["fields"];
            auto rankField = fields["RankID"];
            rankField["association"].set_string("vertex");
            rankField["topology"].set_string("p_unstructured_topo");
            rankField["volume_dependent"].set_string("false");
            if (localNum > 0) {
                rankField["values"].set_external(rankId.data(), localNum);
            } else {
                rankField["values"].set_external(static_cast<int*>(nullptr), 0);
            }
            data["metadata/vtk_fields/RankID/attribute_type"].set_string("ProcessIds");

            // ID field (if enabled)
            if constexpr (T::EnableIDs) {
                using IDAttrib_t = std::remove_reference_t<decltype(pc.ID)>;
                using IDHost_t   = typename IDAttrib_t::host_mirror_type;
                static thread_local IDHost_t IDHost;
                if (IDHost.extent(0) != localNum) {
                    Kokkos::resize(IDHost, localNum);
                }
                Kokkos::deep_copy(IDHost, pc.ID.getView());
                viewRegistry_m.set(name + "_ID", IDHost);

                auto idField = fields["ParticleIDs"];
                idField["association"].set_string("vertex");
                idField["topology"].set_string("p_unstructured_topo");
                idField["volume_dependent"].set_string("false");
                if (localNum > 0) {
                    idField["values"].set_external(IDHost.data(), localNum);
                } else {
                    using id_value_t = typename IDHost_t::value_type;
                    idField["values"].set_external(static_cast<id_value_t*>(nullptr), 0);
                }
                data["metadata/vtk_fields/ParticleIDs/attribute_type"].set_string("GlobalIds");
            }

            // Position field (references RHost)
            auto R_field = fields["position"];
            R_field["association"].set_string("vertex");
            R_field["topology"].set_string("p_unstructured_topo");
            R_field["volume_dependent"].set_string("false");
            if (localNum > 0) {
                R_field["values/x"].set_external(&RHost.data()[0][0], localNum, 0, R_stride);
                if constexpr (ParticleDim >= 2) {
                    R_field["values/y"].set_external(&RHost.data()[0][1], localNum, 0, R_stride);
                }
                if constexpr (ParticleDim >= 3) {
                    R_field["values/z"].set_external(&RHost.data()[0][2], localNum, 0, R_stride);
                }
            }

            // User-defined attributes (persistent mirrors via prepareConduitNode)
            pc.forAllAttributes([&]<typename Attributes>(const Attributes& atts) {
                for (auto* attribute : atts) {
                    if (!pc.isBuiltinAttribute(attribute)) {
                        attribute->prepareConduitNode(localNum, fields, info);
                    }
                }
            });

            // Helper block (bounding box)
            if constexpr (has_getRegionLayout_v<typename T::Layout_t>) {
                using RLayout_t         = typename T::Layout_t::RegionLayout_t;
                using NDRegion_t        = typename RLayout_t::NDRegion_t;
                constexpr unsigned dim_ = T::Layout_t::dim;
                const NDRegion_t ndr    = pc.getLayout().getRegionLayout().getDomain();

                data_help["coordsets/bound_helper_coords/type"].set_string("uniform");
                data_help["topologies/bound_helper_topo/coordset"].set_string(
                    "bound_helper_coords");
                data_help["topologies/bound_helper_topo/type"].set_string("uniform");
                {
                    data_help["coordsets/bound_helper_coords/dims/i"].set(2);
                    data_help["coordsets/bound_helper_coords/spacing/dx"].set(ndr[0].max()
                                                                              - ndr[0].min());
                    data_help["coordsets/bound_helper_coords/origin/x"].set(ndr[0].min());
                }
                if constexpr (dim_ >= 2) {
                    data_help["coordsets/bound_helper_coords/dims/j"].set(2);
                    data_help["coordsets/bound_helper_coords/spacing/dy"].set(ndr[1].max()
                                                                              - ndr[1].min());
                    data_help["coordsets/bound_helper_coords/origin/y"].set(ndr[1].min());
                }
                if constexpr (dim_ >= 3) {
                    data_help["coordsets/bound_helper_coords/dims/k"].set(2);
                    data_help["coordsets/bound_helper_coords/spacing/dz"].set(ndr[2].max()
                                                                              - ndr[2].min());
                    data_help["coordsets/bound_helper_coords/origin/z"].set(ndr[2].min());
                }
            }

            // Transform block (if registered via setTransform)
            if (raw->hasTransform()) {
                const auto& td = *raw->transform();
                auto data_tf   = channel["data/transform"];
                data_tf["type"].set_string("mesh");
                data_tf["coordsets/tf_coords/type"].set_string("explicit");
                data_tf["coordsets/tf_coords/values/x"].set(0.0);
                data_tf["coordsets/tf_coords/values/y"].set(0.0);
                data_tf["coordsets/tf_coords/values/z"].set(0.0);
                data_tf["topologies/tf_topo/coordset"].set_string("tf_coords");
                data_tf["topologies/tf_topo/type"].set_string("unstructured");
                data_tf["topologies/tf_topo/elements/shape"].set_string("point");
                data_tf["topologies/tf_topo/elements/connectivity"].set(0);

                auto setVec3Field = [&](const std::string& fieldName,
                                        const ippl::Vector<double, 3>& v) {
                    auto f = data_tf[std::string("fields/") + fieldName];
                    f["association"].set_string("vertex");
                    f["topology"].set_string("tf_topo");
                    f["volume_dependent"].set_string("false");
                    f["values/x"].set(v[0]);
                    f["values/y"].set(v[1]);
                    f["values/z"].set(v[2]);
                };
                setVec3Field("BunchOrigin", td.origin);
                setVec3Field("BunchRotRow0", td.rows[0]);
                setVec3Field("BunchRotRow1", td.rows[1]);
                setVec3Field("BunchRotRow2", td.rows[2]);
                setVec3Field("BunchInvOrigin", td.invOrigin);
                setVec3Field("BunchInvRotRow0", td.invRows[0]);
                setVec3Field("BunchInvRotRow1", td.invRows[1]);
                setVec3Field("BunchInvRotRow2", td.invRows[2]);
            }

            raw->lastLocalNum_m = localNum;
            info << level4 << "  ParticleChannel '" << name << "' initialized (" << localNum
                 << " particles)" << endl;
        });

        // Type-specific execute function: deep-copy into persistent mirrors.
        // Called every Execute().
        raw->setExecuteFn(
            [this, name, &pc, raw](conduit_cpp::Node& channel, int rank, Inform& info) {
                const size_t localNum = pc.getLocalNum();

                // If particle count changed, re-init the whole channel
                // (re-allocate mirrors, re-set external pointers).
                if (localNum != raw->lastLocalNum_m) {
                    catalystInfo_m << level4 << "  ParticleChannel '" << name
                                   << "' count changed: " << raw->lastLocalNum_m << " -> "
                                   << localNum << ", re-initializing" << endl;
                    channel.reset();
                    raw->initFn_m(channel, rank, info);
                    return;
                }

                // Deep-copy R into persistent mirror
                using RAttrib_t = std::remove_reference_t<decltype(pc.R)>;
                using RHost_t   = typename RAttrib_t::host_mirror_type;
                auto* RHost     = viewRegistry_m.find<RHost_t>(name + "_R");
                if (RHost) {
                    Kokkos::deep_copy(*RHost, pc.R.getView());
                }

                // Deep-copy ID (if enabled)
                if constexpr (T::EnableIDs) {
                    using IDHost_t =
                        typename std::remove_reference_t<decltype(pc.ID)>::host_mirror_type;
                    auto* IDHost = viewRegistry_m.find<IDHost_t>(name + "_ID");
                    if (IDHost) {
                        Kokkos::deep_copy(*IDHost, pc.ID.getView());
                    }
                }

                // Deep-copy all user-defined attributes
                pc.forAllAttributes([&]<typename Attributes>(const Attributes& atts) {
                    for (auto* attribute : atts) {
                        if (!pc.isBuiltinAttribute(attribute)) {
                            attribute->updateConduitData();
                        }
                    }
                });

                // Update transform block values (if present)
                if (raw->hasTransform()) {
                    const auto& td    = *raw->transform();
                    auto data_tf      = channel["data/transform"];
                    auto setVec3Field = [&](const std::string& fieldName,
                                            const ippl::Vector<double, 3>& v) {
                        data_tf[std::string("fields/") + fieldName + "/values/x"].set(v[0]);
                        data_tf[std::string("fields/") + fieldName + "/values/y"].set(v[1]);
                        data_tf[std::string("fields/") + fieldName + "/values/z"].set(v[2]);
                    };
                    setVec3Field("BunchOrigin", td.origin);
                    setVec3Field("BunchRotRow0", td.rows[0]);
                    setVec3Field("BunchRotRow1", td.rows[1]);
                    setVec3Field("BunchRotRow2", td.rows[2]);
                    setVec3Field("BunchInvOrigin", td.invOrigin);
                    setVec3Field("BunchInvRotRow0", td.invRows[0]);
                    setVec3Field("BunchInvRotRow1", td.invRows[1]);
                    setVec3Field("BunchInvRotRow2", td.invRows[2]);
                }
            });

        channels_m.push_back(ch);
        catalystInfo_m << level4 << "::addParticleChannel('" << name
                       << "', type=" << typeid(T).name() << ")" << endl;
        return ParticleChannelHandle(raw);
    }

    void CatalystAdaptor::refresh(const std::string& label) {
        if (!visEnabled_m)
            return;
        for (auto& ch : channels_m) {
            auto* arr = ch->findArray(label);
            if (arr) {
                arr->markDirty();
                Inform info("CatalystAdaptor");
                arr->refresh(ippl::Comm->rank(), info);
                return;
            }
        }
        catalystWarn_m << "::refresh() label not found: '" << label << "'" << endl;
    }

}  // namespace ippl
