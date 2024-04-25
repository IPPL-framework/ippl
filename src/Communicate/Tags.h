//
// Enum Tag
//   Defines tags for MPI communiation.
//
#ifndef IPPL_MPI_TAGS_H
#define IPPL_MPI_TAGS_H

namespace ippl {
    namespace mpi {
        namespace tag {
            enum Tag : int {
                // Tag for applying parallel periodic boundary condition.
                BC_PARALLEL_PERIODIC = 0,
                BC_CYCLE             = 5000,

                // Halo cells
                HALO      = 5001,
                HALO_SEND = 15000,
                HALO_RECV = 20000,

                // Special tags used by Particle classes for communication.
                P_SPATIAL_LAYOUT = 10000,
                P_LAYOUT_CYCLE   = 5000,

                // IDs used to identify buffers created using the buffer factory interface
                // Periodic boundary conditions
                PERIODIC_BC_SEND = 10000,
                PERIODIC_BC_RECV = 15000,

                // Particle spatial layout
                PARTICLE_SEND = 20000,
                PARTICLE_RECV = 25000,

                // FFT Poisson Solver
                SOLVER_SEND = 20000,
                SOLVER_RECV = 25000,
                VICO_SEND   = 26000,
                VICO_RECV   = 31000,

                OPEN_SOLVER = 32000,
                VICO_SOLVER = 32001
            };
        }  // namespace tag
    }      // namespace mpi
}  // namespace ippl

#endif
