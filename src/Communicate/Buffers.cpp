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

        Communicate::buffer_type Communicate::getBuffer(int id, size_t size) {
            #if __cplusplus > 201703L
            if (buffers.contains(id)) {
            #else
            if (buffers.find(id) != buffers.end()) {
            #endif
                return buffers[id];
            }
            buffers[id] = std::make_shared<archive_type>(size);
            return buffers[id];
        }

        void Communicate::deleteBuffer(int id) {
            buffers.erase(id);
        }

        void Communicate::deleteAllBuffers() {
            buffers.clear();
        }

}
