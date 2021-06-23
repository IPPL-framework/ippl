//
// BufferIDs.h
//   Unique identifiers for buffers
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

#ifndef BUFFER_IDS_H
#define BUFFER_IDS_H

// Periodic boundary conditions
#define IPPL_PERIODIC_BC_SEND 1000
#define IPPL_PERIODIC_BC_RECV 2000

// Halo cells
#define IPPL_HALO_FACE_SEND 3000
#define IPPL_HALO_FACE_RECV 4000

#define IPPL_HALO_EDGE_SEND 5000
#define IPPL_HALO_EDGE_RECV 6000

#define IPPL_HALO_VERTEX_SEND 7000
#define IPPL_HALO_VERTEX_RECV 8000

// Particle spatial layout
#define IPPL_PARTICLE_SEND 9000
#define IPPL_PARTICLE_RECV 10000

#endif
