#include "Communicate/Communicator.h"
#include "Communicate/LogEntry.h"
#include "Communicate/CommunicatorLogging.hpp"

#include <iomanip> 
#include <fstream>

namespace ippl {
    namespace mpi {
        void Communicator::printLogs() {
            std::vector<LogEntry> localLogs = gatherLocalLogs();

            std::vector<LogEntry> allLogs;
            if (rank() == 0) {
                allLogs = gatherLogsFromAllRanks(localLogs);
            } else {
                sendLogsToRank0(localLogs);
            }

            if (rank() == 0) {
                writeLogsToFile(allLogs);
            }

        }

        std::vector<LogEntry> Communicator::gatherLocalLogs() {
            std::vector<LogEntry> localLogs;

            buffer_handlers_m.forAll([&](auto& loggingHandler) {
                const auto& logs = loggingHandler.getLogs();
                localLogs.insert(localLogs.end(), logs.begin(), logs.end());
            });

            return localLogs;
        }

        void Communicator::sendLogsToRank0(const std::vector<LogEntry>& localLogs) {
            std::vector<char> buffer = serializeLogs(localLogs);

            int logSize = buffer.size();
            MPI_Send(&logSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            MPI_Send(buffer.data(), logSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }

        std::vector<LogEntry> Communicator::gatherLogsFromAllRanks(const std::vector<LogEntry>& localLogs) {
            std::vector<LogEntry> allLogs = localLogs;

            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

            for (int rank = 1; rank < worldSize; ++rank) {
                int logSize;
                MPI_Recv(&logSize, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<char> buffer(logSize);
                MPI_Recv(buffer.data(), logSize, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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

        void Communicator::writeLogsToFile(const std::vector<LogEntry>& allLogs) {
            std::ofstream logFile("log_entries.csv");

            logFile << "Timestamp,Method,Rank,MemorySpace,AllocatedSize,FreeSize,Parameters\n";

            for (const auto& log : allLogs) {
                auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     log.timestamp.time_since_epoch())
                                     .count();

                logFile << timestamp << "," << log.methodName << "," << log.rank << "," << log.memorySpace << ","
                        << log.allocatedSize << "," << log.freeSize;

                logFile << ",\"";
                bool first = true;
                for (const auto& [key, value] : log.parameters) {
                    if (!first) {
                        logFile << "; ";
                    }
                    logFile << key << ": " << value;
                    first = false;
                }
                logFile << "\"\n";
            }

            logFile.close();
        }

    }  // namespace mpi
}  // namespace ippl
