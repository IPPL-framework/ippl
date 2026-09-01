#include "Stream/IO/ConduitIOWriter.h"

#include "Stream/IO/ConduitIOWriterRelay.h"
#include "Utility/Inform.h"

#include <catalyst_conduit.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <optional>
#include <vector>

#if defined(IPPL_CONDUIT_MPI_IO)
#include <mpi.h>
#endif

namespace ippl {

namespace {

bool hdf5Enabled() {
    const char* v = std::getenv("IPPL_CONDUIT_HDF5");
    return v && std::string(v) == "ON";
}

bool mpiIoEnabled() {
#if defined(IPPL_CONDUIT_MPI_IO)
    const char* v = std::getenv("IPPL_CONDUIT_HDF5_MPI");
    if (!v) {
        return true;
    }
    return std::string(v) == "ON";
#else
    return false;
#endif
}

int writeEvery() {
    const char* v = std::getenv("IPPL_CONDUIT_HDF5_EVERY");
    if (!v) {
        return 1;
    }
    try {
        return std::max(1, std::stoi(v));
    } catch (...) {
        return 1;
    }
}

std::filesystem::path outputDirectory() {
    if (const char* p = std::getenv("IPPL_CONDUIT_HDF5_PATH")) {
        if (p[0] != '\0') {
            return std::filesystem::path(p);
        }
    }
    return std::filesystem::path("conduit_out");
}

std::string cycleTag(int cycle) {
    std::ostringstream oss;
    oss << "cycle_" << std::setw(5) << std::setfill('0') << cycle;
    return oss.str();
}

std::optional<std::string> exportChannelMeshJson(const conduit_cpp::Node& channel) {
    if (!channel.has_path("data")) {
        return std::nullopt;
    }

    std::string mesh_path = "data";
    if (channel.has_path("type")) {
        const std::string type = channel["type"].as_string();
        if (type == "multimesh") {
            mesh_path.clear();
            if (channel.has_path("assembly/main")) {
                const std::string main_block = channel["assembly/main"].as_string();
                const std::string candidate = "data/" + main_block;
                if (channel.has_path(candidate)) {
                    mesh_path = candidate;
                }
            }
            if (mesh_path.empty() && channel.has_path("data/block_main")) {
                mesh_path = "data/block_main";
            }
            if (mesh_path.empty()) {
                return std::nullopt;
            }
        }
    }

    return channel[mesh_path].to_json();
}

std::vector<conduit_io::ChannelExport> exportChannels(const conduit_cpp::Node& channels,
                                                      conduit_index_t n_channels) {
    std::vector<conduit_io::ChannelExport> exports;
    exports.reserve(static_cast<std::size_t>(n_channels));

    for (conduit_index_t i = 0; i < n_channels; ++i) {
        const conduit_cpp::Node& channel = channels.child(i);
        conduit_io::ChannelExport entry;
        entry.name = channel.name();
        if (entry.name.empty()) {
            entry.name = "channel_" + std::to_string(i);
        }
        if (auto mesh_json = exportChannelMeshJson(channel)) {
            entry.data_json = std::move(*mesh_json);
        }
        exports.push_back(std::move(entry));
    }

    return exports;
}

#if defined(IPPL_CONDUIT_MPI_IO)

void saveMpiParallel(const conduit_cpp::Node& channels, conduit_index_t n_channels, int cycle,
                     double time, int rank, const std::filesystem::path& dir, ::Inform& log) {
    const auto exports = exportChannels(channels, n_channels);

    conduit_index_t written = 0;
    for (const auto& entry : exports) {
        if (!entry.data_json.empty()) {
            ++written;
        }
    }

    for (const auto& entry : exports) {
        const auto base = dir / entry.name / cycleTag(cycle);

        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(base.parent_path(), ec);
            if (ec) {
                log << level4 << "::ConduitIOWriter::save() failed to create directory "
                    << base.parent_path().string() << ": " << ec.message() << endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        conduit_io::saveMeshMpi(base.string(), cycle, time, rank, {entry}, MPI_COMM_WORLD);

        log << level4 << "::ConduitIOWriter::save() mpi wrote " << base.string()
            << ".root (+ file_000000.hdf5), channel=" << entry.name
            << (entry.data_json.empty() ? " (empty on this rank)" : "") << endl;
    }

    if (written == 0) {
        log << level4 << "::ConduitIOWriter::save() no channel data on rank " << rank << endl;
    }
}

#endif

void saveSerialPerRank(const conduit_cpp::Node& channels, conduit_index_t n_channels, int cycle,
                       double time, int rank, const std::filesystem::path& dir, ::Inform& log) {
    const auto exports = exportChannels(channels, n_channels);

    conduit_index_t written = 0;
    for (const auto& entry : exports) {
        if (!entry.data_json.empty()) {
            ++written;
        }
    }

    if (written == 0) {
        log << level4 << "::ConduitIOWriter::save() no channel data found, skipping." << endl;
        return;
    }

    std::ostringstream path;
    path << dir.string() << "/" << cycleTag(cycle) << "_rank_" << rank << ".hdf5";

    conduit_io::saveMeshSerial(path.str(), cycle, time, rank, exports);

    log << level4 << "::ConduitIOWriter::save() wrote " << path.str() << " (" << written
        << " channel(s))" << endl;
}

} // namespace

void ConduitIOWriter::saveFromCatalystNode(const conduit_cpp::Node& catalyst_node, int cycle,
                                           double time, int rank, ::Inform& log) {
    if (!hdf5Enabled()) {
        return;
    }

    const int every = writeEvery();
    if (every > 1 && (cycle % every) != 0) {
        return;
    }

    if (!catalyst_node.has_path("catalyst/channels")) {
        log << level4 << "::ConduitIOWriter::save() no catalyst/channels in node, skipping."
            << endl;
        return;
    }

    const conduit_cpp::Node& channels = catalyst_node["catalyst/channels"];
    const conduit_index_t n_channels  = channels.number_of_children();
    if (n_channels == 0) {
        log << level4 << "::ConduitIOWriter::save() empty channels, skipping." << endl;
        return;
    }

    const auto dir = outputDirectory();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        log << level4 << "::ConduitIOWriter::save() failed to create directory " << dir.string()
            << ": " << ec.message() << endl;
        return;
    }

#if defined(IPPL_CONDUIT_MPI_IO)
    if (mpiIoEnabled()) {
        saveMpiParallel(channels, n_channels, cycle, time, rank, dir, log);
        return;
    }
#endif

    saveSerialPerRank(channels, n_channels, cycle, time, rank, dir, log);
}

} // namespace ippl
