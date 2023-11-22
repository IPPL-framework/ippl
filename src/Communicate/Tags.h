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
                // special tag used to indicate the program should quit.  The values are
                // arbitrary, but non-zero.
                IPPL_ABORT = 5,  // program should abort()
                IPPL_EXIT  = 6,  // program should exit()

                // tags for reduction
                COMM_REDUCE_SEND    = 10000,
                COMM_REDUCE_RECV    = 11000,
                COMM_REDUCE_SCATTER = 12000,
                COMM_REDUCE_CYCLE   = 1000,

                // tag for applying parallel periodic boundary condition.

                BC_PARALLEL_PERIODIC = 15000,
                BC_CYCLE             = 1000,

                // Field<T,Dim> tags
                HALO       = 100000,
                HALO_CYCLE = 100000,

                F_GUARD_CELLS       = 20000,  // Field::fillGuardCells()
                F_WRITE             = 21000,  // Field::write()
                F_READ              = 22000,  // Field::read()
                F_GEN_ASSIGN        = 23000,  // assign(BareField,BareField)
                F_REPARTITION_BCAST = 24000,  // broadcast in FieldLayout::repartion.
                F_REDUCE_PERP       = 25000,  // reduction in binary load balance.
                F_GETSINGLE         = 26000,  // IndexedBareField::getsingle()
                F_REDUCE            = 27000,  // Reduction in minloc/maxloc
                F_LAYOUT_IO         = 28000,  // Reduction in minloc/maxloc
                F_CYCLE             = 1000,

                // Special tags used by Particle classes for communication.
                P_WEIGHTED_LAYOUT   = 50000,
                P_WEIGHTED_RETURN   = 51000,
                P_WEIGHTED_TRANSFER = 52000,
                P_SPATIAL_LAYOUT    = 53000,
                P_SPATIAL_RETURN    = 54000,
                P_SPATIAL_TRANSFER  = 55000,
                P_SPATIAL_GHOST     = 56000,
                P_SPATIAL_RANGE     = 57000,
                P_RESET_ID          = 58000,
                P_LAYOUT_CYCLE      = 1000,

                // Tags for Ippl setup
                IPPL_MAKE_HOST_MAP = 60000,
                IPPL_CYCLE         = 1000,

                // Tags for Ippl application codes
                IPPL_APP0      = 90000,
                IPPL_APP1      = 91000,
                IPPL_APP2      = 92000,
                IPPL_APP3      = 93000,
                IPPL_APP4      = 94000,
                IPPL_APP5      = 95000,
                IPPL_APP6      = 96000,
                IPPL_APP7      = 97000,
                IPPL_APP8      = 98000,
                IPPL_APP9      = 99000,
                IPPL_APP_CYCLE = 1000,

                // IDs used to identify buffers created using the buffer factory interface
                // Periodic boundary conditions
                PERIODIC_BC_SEND = 1000,
                PERIODIC_BC_RECV = 2000,

                // Halo cells
                HALO_SEND = 100000,
                HALO_RECV = 200000,

                // Particle spatial layout
                PARTICLE_SEND = 9000,
                PARTICLE_RECV = 10000,

                // FFT Poisson Solver
                SOLVER_SEND = 13000,
                SOLVER_RECV = 14000,
                VICO_SEND   = 16000,
                VICO_RECV   = 17000,

                OPEN_SOLVER = 18000,
                VICO_SOLVER = 70000
            };
        }  // namespace tag
    }      // namespace mpi
}  // namespace ippl

#endif
