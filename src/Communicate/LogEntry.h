#ifndef IPPL_LOGENTRY_H
#define IPPL_LOGENTRY_H

#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ippl {

    struct LogEntry {
        std::string methodName;
        std::map<std::string, std::string> parameters;
        size_t usedSize;
        size_t freeSize;
        std::string memorySpace;
        int rank;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

        std::vector<char> serialize() const;
        static LogEntry deserialize(const std::vector<char>& buffer, size_t offset = 0);
    };

    template <typename T>
    void serializeBasicType(std::vector<char>& buffer, const T& value) {
        size_t size = sizeof(T);
        buffer.resize(buffer.size() + size);
        std::memcpy(buffer.data() + buffer.size() - size, &value, size);
    }

    template <typename T>
    T deserializeBasicType(const std::vector<char>& buffer, size_t& offset) {
        T value;
        std::memcpy(&value, buffer.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

}  // namespace ippl

#endif
