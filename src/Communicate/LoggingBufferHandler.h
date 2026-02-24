#ifndef IPPL_LOGGING_BUFFER_HANDLER_H
#define IPPL_LOGGING_BUFFER_HANDLER_H

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Communicate/BufferHandler.h"
#include "Communicate/LogEntry.h"

namespace ippl::comms {

    /**
     * @class LoggingBufferHandler
     * @brief Decorator class for buffer management that adds logging capabilities to buffer
     * operations.
     *
     * `LoggingBufferHandler` extends the basic functionality of `BufferHandler` by recording
     * detailed logs of buffer operations, such as allocation, deallocation, and resizing.
     * This allows tracking of memory usage patterns and provides a record of buffer activity.
     *
     * @tparam MemorySpace The memory space in which buffers are managed (e.g., HostSpace,
     * CudaSpace).
     *
     * This class is a decorator for `BufferHandler` and does not modify buffer management logic.
     * Instead, it adds logging for monitoring purposes.
     */
    template <typename MemorySpace>
    class LoggingBufferHandler : public BufferHandler<archive_buffer<MemorySpace>, MemorySpace> {
    public:
        using buffer_type =
            typename BufferHandler<archive_buffer<MemorySpace>, MemorySpace>::buffer_type;
        using size_type =
            typename BufferHandler<archive_buffer<MemorySpace>, MemorySpace>::size_type;

        /**
         * @brief Constructs a LoggingBufferHandler with an existing buffer handler.
         * @param handler A shared pointer to an `BufferHandler` instance used for buffer
         * operations.
         * @param rank The MPI rank for logging purposes, used to identify the source of logs.
         */
        LoggingBufferHandler(
            std::shared_ptr<BufferHandler<archive_buffer<MemorySpace>, MemorySpace>> handler,
            int rank);

        /**
         * @brief Default constructor, creates an internal `BufferHandler` for managing buffers.
         * This constructor also initializes the rank by calling `MPI_Comm_rank`.
         */
        LoggingBufferHandler();

        /**
         * @brief Allocates or retrieves a buffer and logs the action.
         *
         * Overrides `BufferHandler::getBuffer`, providing the same buffer allocation behavior
         * while recording an entry in the log with the operation details.
         *
         * @param size Requested size of the buffer.
         * @param overallocation Optional multiplier to allocate extra buffer space.
         * @return A buffer object.
         */
        buffer_type getBuffer(size_type size, double overallocation) override;

        /**
         * @brief Frees a buffer and logs the action.
         *
         * Overrides `BufferHandler::freeBuffer`. Calls `BufferHandler::freeBuffer` and records the
         * operation in the log.
         *
         * @param buffer The buffer to be freed.
         */
        void freeBuffer(buffer_type buffer) override;

        /**
         * @brief Frees all buffers and logs the action.
         *
         * Overrides `BufferHandler::freeAllBuffers`. Calls `BufferHandler::freeAllBuffers` and
         * logs the operation.
         */
        void freeAllBuffers() override;

        /**
         * @brief Deletes all buffers and logs the action.
         *
         * Overrides `BufferHandler::deleteAllBuffers`. Calls `BufferHandler::deleteAllBuffers` and
         * logs the operation.
         */
        void deleteAllBuffers() override;

        /**
         * @brief Retrieves the total size of allocated buffers.
         * @return The size of allocated buffers.
         */
        size_type getUsedSize() const override;

        /**
         * @brief Retrieves the total size of free buffers.
         * @return The size of free buffers.
         */
        size_type getFreeSize() const override;

        /**
         * @brief Retrieves the list of log entries.
         * @return A constant reference to a vector containing log entries.
         */
        const std::vector<LogEntry>& getLogs() const;

    private:
        std::shared_ptr<BufferHandler<archive_buffer<MemorySpace>, MemorySpace>>
            handler_m;                       ///< Internal handler for buffer management.
        std::vector<LogEntry> logEntries_m;  ///< Log entries for buffer operations.
        int rank_m;                          ///< MPI rank for identifying log sources.

        /**
         * @brief Records a method call in the log with its parameters.
         *
         * This method creates a log entry with details of the buffer operation,
         * including the method name, parameters, allocated size, free size,
         * memory space, rank, and timestamp.
         *
         * @param methodName Name of the method being logged (e.g., "getBuffer").
         * @param parameters A map of parameter names and values for the operation.
         */
        void logMethod(const std::string& methodName,
                       const std::map<std::string, std::string>& parameters);
    };
}  // namespace ippl::comms

#include "Communicate/LoggingBufferHandler.hpp"

#endif
