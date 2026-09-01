#include "Stream/IO/ConduitIOWriterRelay.h"

#include "conduit.hpp"
#include "conduit_relay_io.hpp"

#if defined(IPPL_CONDUIT_MPI_IO)
#include "conduit_relay_mpi_io_blueprint.hpp"
#endif

namespace ippl::conduit_io {

namespace {

void fillDomainState(conduit::Node& dom, int cycle, double time, int rank) {
    dom["state/cycle"]     = cycle;
    dom["state/time"]      = time;
    dom["state/domain_id"] = rank;
}

} // namespace

#if defined(IPPL_CONDUIT_MPI_IO)

void saveMeshMpi(const std::string& base_path, int cycle, double time, int rank,
                 const std::vector<ChannelExport>& channels, MPI_Comm comm) {
    conduit::Node opts;
    opts["file_style"]      = "multi_file";
    opts["number_of_files"] = 1;
    opts["suffix"]          = "none";

    for (const auto& channel : channels) {
        conduit::Node mesh;
        if (!channel.data_json.empty()) {
            conduit::Node& dom = mesh.append();
            dom.parse(channel.data_json, "json");
            fillDomainState(dom, cycle, time, rank);
        }

        conduit::relay::mpi::io::blueprint::save_mesh(mesh, base_path, "hdf5", opts, comm);
    }
}

#endif

void saveMeshSerial(const std::string& hdf5_path, int cycle, double time, int rank,
                    const std::vector<ChannelExport>& channels) {
    conduit::Node io_root;
    fillDomainState(io_root, cycle, time, rank);

    conduit_index_t written = 0;
    for (const auto& channel : channels) {
        if (channel.data_json.empty()) {
            continue;
        }
        io_root[channel.name].parse(channel.data_json, "json");
        ++written;
    }

    if (written == 0) {
        return;
    }

    conduit::relay::io::save(io_root, hdf5_path, "hdf5");
}

} // namespace ippl::conduit_io
