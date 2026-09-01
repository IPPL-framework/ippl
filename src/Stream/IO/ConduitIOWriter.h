/**
 * @file ConduitIOWriter.h
 * @brief Serial Conduit Relay HDF5 export from Catalyst Blueprint channels.
 */
#ifndef ConduitIOWriter_h
#define ConduitIOWriter_h

class Inform;

namespace conduit_cpp {
class Node;
}

namespace ippl {

/**
 * @brief Writes per-rank HDF5 files from populated Catalyst channel nodes.
 *
 * Controlled at runtime via:
 *   IPPL_CONDUIT_HDF5=ON|OFF          (default OFF)
 *   IPPL_CONDUIT_HDF5_PATH=<dir>        (default ./conduit_out)
 *   IPPL_CONDUIT_HDF5_EVERY=<N>         (default 1)
 */
class ConduitIOWriter {
public:
    static void saveFromCatalystNode(const conduit_cpp::Node& catalyst_node, int cycle, double time,
                                     int rank, ::Inform& log);
};

} // namespace ippl

#endif
