//
// Buffers.hpp
//   Interface for globally accessible buffer factory for communication
//
//   Data sent between MPI ranks has to be stored in a buffer for sending and receiving.
//   To reduce the number of times memory has to be allocated and freed, the buffer
//   factory interface allows buffers to be reused. This is especially relevant on
//   GPUs, as Cuda allocation calls are expensive. To avoid reallocating the buffers
//   in the case that the amount of data to be exchanged increases, when a new buffer
//   is created, an amount of memory greater than the requested size is allocated
//   for the new buffer. The factor by which memory is overallocated is determined by
//   a data member in Communicate, which can be set and queried at runtime. Only new
//   buffers are overallocated. If a buffer is requested with the same ID as a buffer
//   that has been previously allocated, the same buffer will be used. If the requested
//   size exceeds the buffer size, that buffer will be resized to have exactly
//   the requested size.
//
//   Currently, the buffer factory is used for application of periodic boundary
//   conditions; halo cell exchange along faces, edges, and vertices; as well as
//   exchanging particle data between ranks.
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

namespace ippl {

    template <typename T>
    Communicate::buffer_type Communicate::getBuffer(int id, size_type size, double overallocation) {
        size *= sizeof(T);
#if __cplusplus > 201703L
        if (buffers_m.contains(id)) {
#else
        if (buffers_m.find(id) != buffers_m.end()) {
#endif
            buffer_type buf = buffers_m[id];
            if (buf->getBufferSize() < size) {
                buf->reallocBuffer(size);
            }
            return buf;
        }
        buffers_m[id] = std::make_shared<archive_type>(
            (size_type)(size * std::max(overallocation, defaultOveralloc_m)));
        return buffers_m[id];
    }

}  // namespace ippl
