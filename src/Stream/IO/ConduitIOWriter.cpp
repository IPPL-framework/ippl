#include "Stream/IO/ConduitIOWriter.h"

#include "Utility/Inform.h"

#include <catalyst_conduit.hpp>

#include "conduit.h"
#include "conduit_node.h"
#include "conduit_relay_io.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace ippl {

namespace {

bool hdf5Enabled() {
    const char* v = std::getenv("IPPL_CONDUIT_HDF5");
    return v && std::string(v) == "ON";
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

void copyBlueprintSubtree(conduit_node* dst, const conduit_cpp::Node& src) {
    conduit_node_update(dst, conduit_cpp::c_node(const_cast<conduit_cpp::Node*>(&src)));
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

    conduit_node* io_root = conduit_node_create();
    conduit_node_set_path_int32(io_root, "state/cycle", cycle);
    conduit_node_set_path_float64(io_root, "state/time", time);
    conduit_node_set_path_int32(io_root, "state/domain_id", rank);

    conduit_index_t written = 0;
    for (conduit_index_t i = 0; i < n_channels; ++i) {
        const conduit_cpp::Node& channel = channels.child(i);
        if (!channel.has_child("data")) {
            continue;
        }

        const std::string name = channel.name();
        if (name.empty()) {
            continue;
        }

        conduit_node* mesh = conduit_node_fetch(io_root, name.c_str());
        copyBlueprintSubtree(mesh, channel["data"]);
        ++written;
    }

    if (written == 0) {
        log << level4 << "::ConduitIOWriter::save() no channel data found, skipping." << endl;
        conduit_node_destroy(io_root);
        return;
    }

    const auto dir = outputDirectory();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        log << level4 << "::ConduitIOWriter::save() failed to create directory " << dir.string()
            << ": " << ec.message() << endl;
        conduit_node_destroy(io_root);
        return;
    }

    std::ostringstream path;
    path << dir.string() << "/" << cycleTag(cycle) << "_rank_" << rank << ".hdf5";

    conduit_relay_io_save(io_root, path.str().c_str(), "hdf5", nullptr);
    conduit_node_destroy(io_root);

    log << level4 << "::ConduitIOWriter::save() wrote " << path.str() << " (" << written
        << " channel(s))" << endl;
}

} // namespace ippl
