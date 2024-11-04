#ifndef IPPL_LOGGING_BUFFER_HANDLER_HPP
#define IPPL_LOGGING_BUFFER_HANDLER_HPP

#include <mpi.h>
#include <iostream>

template <typename MemorySpace>
LoggingBufferHandler<MemorySpace>::LoggingBufferHandler(std::shared_ptr<IBufferHandler<MemorySpace>> handler, int rank)
    : handler_(std::move(handler)), rank_(rank) {}

template <typename MemorySpace>
LoggingBufferHandler<MemorySpace>::LoggingBufferHandler() {
    handler_ = std::make_shared<BufferHandler<MemorySpace>>();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
}

template <typename MemorySpace>
typename LoggingBufferHandler<MemorySpace>::buffer_type LoggingBufferHandler<MemorySpace>::getBuffer(
    size_type size, double overallocation) {
    auto buffer = handler_->getBuffer(size, overallocation);
    logMethod("getBuffer", {{"size", std::to_string(size)}, {"overallocation", std::to_string(overallocation)}});
    return buffer;
}

template <typename MemorySpace>
void LoggingBufferHandler<MemorySpace>::freeBuffer(buffer_type buffer) {
    handler_->freeBuffer(buffer);
    logMethod("freeBuffer", {});
}

template <typename MemorySpace>
void LoggingBufferHandler<MemorySpace>::freeAllBuffers() {
    handler_->freeAllBuffers();
    logMethod("freeAllBuffers", {});
}

template <typename MemorySpace>
void LoggingBufferHandler<MemorySpace>::deleteAllBuffers() {
    handler_->deleteAllBuffers();
    logMethod("deleteAllBuffers", {});
}

template <typename MemorySpace>
typename LoggingBufferHandler<MemorySpace>::size_type LoggingBufferHandler<MemorySpace>::getAllocatedSize() const {
    return handler_->getAllocatedSize();
}

template <typename MemorySpace>
typename LoggingBufferHandler<MemorySpace>::size_type LoggingBufferHandler<MemorySpace>::getFreeSize() const {
    return handler_->getFreeSize();
}

template <typename MemorySpace>
const std::vector<LogEntry>& LoggingBufferHandler<MemorySpace>::getLogs() const {
    std::cout << logEntries_.size() << std::endl;
    return logEntries_;
}

template <typename MemorySpace>
void LoggingBufferHandler<MemorySpace>::logMethod(const std::string& methodName, const std::map<std::string, std::string>& parameters) {
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

#endif

