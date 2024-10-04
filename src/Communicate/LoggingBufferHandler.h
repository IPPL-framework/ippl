#ifndef IPPL_LOGGING_BUFFER_HANDLER_H
#define IPPL_LOGGING_BUFFER_HANDLER_H

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <map>
#include "Communicate/BufferHandler.h"

struct LogEntry {
    std::string methodName;
    std::map<std::string, std::string> parameters;
    size_t allocatedSize;
    size_t freeSize;
    std::string memorySpace;
    int rank;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
};

template <typename MemorySpace>
class LoggingBufferHandler : public IBufferHandler<MemorySpace> {
public:
    using buffer_type = typename IBufferHandler<MemorySpace>::buffer_type;
    using size_type = typename IBufferHandler<MemorySpace>::size_type;

    LoggingBufferHandler(std::shared_ptr<IBufferHandler<MemorySpace>> handler, int rank)
        : handler_(std::move(handler)), rank_(rank) {}

    buffer_type getBuffer(size_type size, double overallocation) override {
        auto buffer = handler_->getBuffer(size, overallocation);
        logMethod("getBuffer", {{"size", std::to_string(size)}, {"overallocation", std::to_string(overallocation)}});
        return buffer;
    }

    void freeBuffer(buffer_type buffer) override {
        handler_->freeBuffer(buffer);
        logMethod("freeBuffer", {});
    }

    void freeAllBuffers() override {
        handler_->freeAllBuffers();
        logMethod("freeAllBuffers", {});
    }

    void deleteAllBuffers() override {
        handler_->deleteAllBuffers();
        logMethod("deleteAllBuffers", {});
    }

    size_type getAllocatedSize() const override {
        return handler_->getAllocatedSize();
    }

    size_type getFreeSize() const override {
        return handler_->getFreeSize();
    }

    const std::vector<LogEntry>& getLogs() const {
        return logEntries_;
    }

private:
    std::shared_ptr<IBufferHandler<MemorySpace>> handler_;
    std::vector<LogEntry> logEntries_;
    int rank_;

    void logMethod(const std::string& methodName, const std::map<std::string, std::string>& parameters) {
        logEntries_.push_back({
            methodName,
            parameters,
            handler_->getAllocatedSize(),
            handler_->getFreeSize(),
            MemorySpace::name(),
            rank_,
            std::chrono::high_resolution_clock::now()
        });
    }
};

#endif
