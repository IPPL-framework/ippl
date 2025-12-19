#ifndef IPPL_COMMUNICATOR_LOGGING_HPP
#define IPPL_COMMUNICATOR_LOGGING_HPP

#include <vector>

#include "Communicate/LogEntry.h"

namespace ippl::mpi {
    std::vector<char> serializeLogs(const std::vector<LogEntry>& logs);
    std::vector<LogEntry> deserializeLogs(const std::vector<char>& buffer);
}  // namespace ippl::mpi

#endif
