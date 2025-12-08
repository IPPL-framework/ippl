#include "Communicate/CommunicatorLogging.hpp"

#include <fstream>
#include <iomanip>

#include "Utility/Inform.h"

#include "Communicate/Communicator.h"
#include "Communicate/LoggingBufferHandler.h"

namespace ippl::mpi {
    void Communicator::printLogs(const std::string& filename) {
        std::vector<LogEntry> localLogs = gatherLocalLogs();

        std::vector<LogEntry> allLogs;
        if (rank() == 0) {
            allLogs = gatherLogsFromAllRanks(localLogs);
        } else {
            sendLogsToRank0(localLogs);
        }

        if (rank() == 0) {
            writeLogsToFile(allLogs, filename);
        }
    }

    template <class MemorySpace>
    struct is_a_logger : std::false_type {};

    template <class MemorySpace>
    struct is_a_logger<comms::LoggingBufferHandler<MemorySpace> > : std::true_type {};

    std::vector<LogEntry> Communicator::gatherLocalLogs() {
        std::vector<LogEntry> localLogs;
        if constexpr (is_a_logger<buffer_handler_type>::value) {
            buffer_handlers_m->forAll([&](auto& loggingHandler) {
                const auto& logs = loggingHandler.getLogs();
                localLogs.insert(localLogs.end(), logs.begin(), logs.end());
            });
        }
        return localLogs;
    }

    void Communicator::sendLogsToRank0(const std::vector<LogEntry>& localLogs) {
        std::vector<char> buffer = serializeLogs(localLogs);

        int logSize = buffer.size();

        this->send(logSize, 1, 0, 0);
        this->send<char>(buffer.data(), logSize, 0, 0);
    }

    std::vector<LogEntry> Communicator::gatherLogsFromAllRanks(
        const std::vector<LogEntry>& localLogs) {
        std::vector<LogEntry> allLogs = localLogs;

        for (int rank = 1; rank < size_m; ++rank) {
            int logSize;
            Status status;

            this->recv(logSize, 1, rank, 0, status);

            std::vector<char> buffer(logSize);
            this->recv<char>(buffer.data(), logSize, rank, 0, status);

            std::vector<LogEntry> deserializedLogs = deserializeLogs(buffer);
            allLogs.insert(allLogs.end(), deserializedLogs.begin(), deserializedLogs.end());
        }

        return allLogs;
    }

    std::vector<char> serializeLogs(const std::vector<LogEntry>& logs) {
        std::vector<char> buffer;

        for (const auto& logEntry : logs) {
            std::vector<char> serializedEntry = logEntry.serialize();
            buffer.insert(buffer.end(), serializedEntry.begin(), serializedEntry.end());
        }

        return buffer;
    }

    std::vector<LogEntry> deserializeLogs(const std::vector<char>& buffer) {
        std::vector<LogEntry> logs;
        size_t offset = 0;

        while (offset < buffer.size()) {
            LogEntry logEntry = LogEntry::deserialize(buffer, offset);

            logs.push_back(logEntry);

            offset += logEntry.serialize().size();
        }
        return logs;
    }

    void Communicator::writeLogsToFile(const std::vector<LogEntry>& allLogs,
                                       const std::string& filename) {
        Inform logFile(0, filename.c_str(), Inform::OVERWRITE, 0);
        logFile.setOutputLevel(1);

        logFile << "Timestamp,Method,Rank,MemorySpace,usedSize,FreeSize,Parameters" << endl;

        for (const auto& log : allLogs) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 log.timestamp.time_since_epoch())
                                 .count();

            logFile << timestamp << "," << log.methodName << "," << log.rank << ","
                    << log.memorySpace << "," << log.usedSize << "," << log.freeSize;

            logFile << ",\"";
            bool first = true;
            for (const auto& [key, value] : log.parameters) {
                if (!first) {
                    logFile << "; ";
                }
                logFile << key << ": " << value;
                first = false;
            }
            logFile << "\"" << endl;
        }

        logFile.flush();
    }
}  // namespace ippl::mpi
