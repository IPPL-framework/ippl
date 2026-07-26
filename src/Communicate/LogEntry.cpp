#include "Communicate/LogEntry.h"

namespace ippl {

    void serializeString(std::vector<char>& buffer, const std::string& str) {
        size_t length = str.size();
        serializeBasicType(buffer, length);  // First, serialize the length of the string
        buffer.insert(buffer.end(), str.begin(), str.end());  // Then, serialize the string itself
    }

    std::string deserializeString(const std::vector<char>& buffer, size_t& offset) {
        size_t length =
            deserializeBasicType<size_t>(buffer, offset);  // Get the length of the string
        std::string str(buffer.begin() + offset,
                        buffer.begin() + offset + length);  // Extract the string
        offset += length;
        return str;
    }

    std::vector<char> LogEntry::serialize() const {
        std::vector<char> buffer;

        serializeString(buffer, methodName);
        serializeBasicType(buffer, usedSize);
        serializeBasicType(buffer, freeSize);
        serializeString(buffer, memorySpace);
        serializeBasicType(buffer, rank);

        // Serialize the timestamp (as duration since epoch)
        auto duration = timestamp.time_since_epoch().count();
        serializeBasicType(buffer, duration);

        size_t mapSize = parameters.size();
        serializeBasicType(buffer, mapSize);
        for (const auto& pair : parameters) {
            serializeString(buffer, pair.first);
            serializeString(buffer, pair.second);
        }

        return buffer;
    }

    LogEntry LogEntry::deserializeAdvance(const std::vector<char>& buffer, size_t& offset) {
        LogEntry entry;

        entry.methodName  = deserializeString(buffer, offset);
        entry.usedSize    = deserializeBasicType<size_t>(buffer, offset);
        entry.freeSize    = deserializeBasicType<size_t>(buffer, offset);
        entry.memorySpace = deserializeString(buffer, offset);
        entry.rank        = deserializeBasicType<int>(buffer, offset);

        auto duration   = deserializeBasicType<long long>(buffer, offset);
        entry.timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(
            std::chrono::high_resolution_clock::duration(duration));

        size_t mapSize = deserializeBasicType<size_t>(buffer, offset);
        for (size_t i = 0; i < mapSize; ++i) {
            std::string key       = deserializeString(buffer, offset);
            std::string value     = deserializeString(buffer, offset);
            entry.parameters[key] = value;
        }

        return entry;
    }

    LogEntry LogEntry::deserialize(const std::vector<char>& buffer, size_t offset) {
        return deserializeAdvance(buffer, offset);
    }

}  // namespace ippl
