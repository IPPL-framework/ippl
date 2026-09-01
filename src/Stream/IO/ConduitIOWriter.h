/**
 * @file ConduitIOWriter.h
 * @brief Conduit Relay HDF5 export from Catalyst Blueprint channels.
 */
#ifndef ConduitIOWriter_h
#define ConduitIOWriter_h

class Inform;

namespace conduit_cpp {
class Node;
}

namespace ippl {

/**
 * @brief Writes HDF5 from populated Catalyst channel nodes.
 *
 * Controlled at runtime via:
 *   IPPL_CONDUIT_HDF5=ON|OFF          (default OFF)
 *   IPPL_CONDUIT_HDF5_PATH=<dir>        (default ./conduit_out)
 *   IPPL_CONDUIT_HDF5_EVERY=<N>         (default 1)
 *   IPPL_CONDUIT_HDF5_MPI=ON|OFF        (default ON when built with MPI IO)
 *
 * MPI mode (default): parallel single-file Blueprint HDF5 per channel/cycle via
 *   conduit::relay::mpi::io::blueprint::save_mesh
 * Catalyst multimesh channels (particles): exports assembly/main mesh (block_main).
 * Serial fallback: per-rank files cycle_NNNNN_rank_R.hdf5
 */
class ConduitIOWriter {
public:
    static void saveFromCatalystNode(const conduit_cpp::Node& catalyst_node, int cycle, double time,
                                     int rank, ::Inform& log);
};

} // namespace ippl

#endif
