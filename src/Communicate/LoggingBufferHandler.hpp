#ifndef IPPL_LOGGING_BUFFER_HANDLER_HPP
#define IPPL_LOGGING_BUFFER_HANDLER_HPP

#include <iostream>
#include <mpi.h>

namespace ippl::comms {

    template <typename MemorySpace>
    LoggingBufferHandler<MemorySpace>::LoggingBufferHandler(
        std::shared_ptr<BufferHandler<archive_buffer<MemorySpace>, MemorySpace>> handler, int rank)
        : handler_m(std::move(handler))
        , rank_m(rank) {}

    template <typename MemorySpace>
    LoggingBufferHandler<MemorySpace>::LoggingBufferHandler() {
        handler_m = std::make_shared<DefaultBufferHandler<MemorySpace>>();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_m);
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::buffer_type
    LoggingBufferHandler<MemorySpace>::getBuffer(size_type size, double overallocation) {
        auto buffer = handler_m->getBuffer(size, overallocation);
        logMethod("getBuffer", {{"size", std::to_string(size)},
                                {"overallocation", std::to_string(overallocation)}});
        return buffer;
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::freeBuffer(buffer_type buffer) {
        handler_m->freeBuffer(buffer);
        logMethod("freeBuffer", {});
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::freeAllBuffers() {
        handler_m->freeAllBuffers();
        logMethod("freeAllBuffers", {});
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::deleteAllBuffers() {
        handler_m->deleteAllBuffers();
        logMethod("deleteAllBuffers", {});
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::size_type
    LoggingBufferHandler<MemorySpace>::getUsedSize() const {
        return handler_m->getUsedSize();
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::size_type
    LoggingBufferHandler<MemorySpace>::getFreeSize() const {
        return handler_m->getFreeSize();
    }

    template <typename MemorySpace>
    const std::vector<LogEntry>& LoggingBufferHandler<MemorySpace>::getLogs() const {
        return logEntries_m;
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::logMethod(
        const std::string& methodName, const std::map<std::string, std::string>& parameters) {
        logEntries_m.push_back({methodName, parameters, handler_m->getUsedSize(),
                                handler_m->getFreeSize(), MemorySpace::name(), rank_m,
                                std::chrono::high_resolution_clock::now()});
    }

}  // namespace ippl::comms

#endif
