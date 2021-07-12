//
// Buffers.cpp
//   Interface for globally accessible buffer factory for communication
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

#include "Communicate.h"

namespace ippl {

        void Communicate::setDefaultOverallocation(int factor) {
            defaultOveralloc = factor;
        }

        Communicate::buffer_type Communicate::getBuffer(int id, size_type size, int overallocation) {
            #if __cplusplus > 201703L
            if (buffers.contains(id)) {
            #else
            if (buffers.find(id) != buffers.end()) {
            #endif
                buffer_type buf = buffers[id];
                if (buf->getBufferSize() < size) {
                    buf->resizeBuffer(size);
                }
                return buf;
            }
            //overallocation *= 1;
            buffers[id] = std::make_shared<archive_type>(size * std::max(overallocation, defaultOveralloc));
            //buffers[id] = std::make_shared<archive_type>(size * 4);
            return buffers[id];
        }

        void Communicate::deleteBuffer(int id) {
            buffers.erase(id);
        }

        void Communicate::deleteAllBuffers() {
            buffers.clear();
        }

}
