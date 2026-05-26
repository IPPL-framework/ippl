// Temporary debug instrumentation for session af3f69.
// Writes NDJSON entries to /Users/sona/Desktop/IPPL/.cursor/debug-af3f69.log
// This file is intended to be removed once the debug session is complete.
#ifndef IPPL_DEBUG_LOG_AF3F69_H
#define IPPL_DEBUG_LOG_AF3F69_H

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "Communicate/Communicator.h"

namespace ippl_debug_af3f69 {

    inline std::string logPath() {
        return (std::filesystem::current_path() / "debug-af3f69.log").string();
    }

    inline void writeLine(const std::string& hypothesisId,
                          const std::string& location,
                          const std::string& message,
                          const std::string& dataJson) {
        std::ofstream f(logPath(), std::ios::app);
        if (!f.is_open()) {
            return;
        }
        auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
        int rank = (ippl::Comm) ? ippl::Comm->rank() : -1;
        std::ostringstream line;
        line << "{\"sessionId\":\"af3f69\","
             << "\"timestamp\":" << nowMs << ","
             << "\"rank\":" << rank << ","
             << "\"hypothesisId\":\"" << hypothesisId << "\","
             << "\"location\":\"" << location << "\","
             << "\"message\":\"" << message << "\","
             << "\"data\":" << dataJson << "}\n";
        f << line.str();
    }

}  // namespace ippl_debug_af3f69

#endif  // IPPL_DEBUG_LOG_AF3F69_H
