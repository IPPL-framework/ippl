/**
 * @file ConduitIOWriterRelay.h
 * @brief Full-Conduit Relay HDF5 backend (isolated from Catalyst's bundled Conduit).
 */
#ifndef ConduitIOWriterRelay_h
#define ConduitIOWriterRelay_h

#include <string>
#include <vector>

#if defined(IPPL_CONDUIT_MPI_IO)
#include <mpi.h>
#endif

namespace ippl::conduit_io {

struct ChannelExport {
    std::string name;
    std::string data_json;
};

#if defined(IPPL_CONDUIT_MPI_IO)
void saveMeshMpi(const std::string& base_path, int cycle, double time, int rank,
                 const std::vector<ChannelExport>& channels, MPI_Comm comm);
#endif

void saveMeshSerial(const std::string& hdf5_path, int cycle, double time, int rank,
                    const std::vector<ChannelExport>& channels);

} // namespace ippl::conduit_io

#endif
