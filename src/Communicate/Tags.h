// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TAGS_H
#define TAGS_H

/*
 * Tags.h - list of special tags used by each major component in the ippl
 *	library.  When a new general communication cycle (i.e. swapping
 *	boundaries) is added, a new item should be added to this list.
 * BH, 6/19/95
 *
 * Updated for beta, JVWR, 7/27/96
 */

// special tag used to indicate the program should quit.  The values are
// arbitrary, but non-zero.
#define IPPL_ABORT_TAG            5    // program should abort()
#define IPPL_EXIT_TAG             6    // program should exit()


// tag for applying parallel periodic boundary condition.

#define BC_PARALLEL_PERIODIC_TAG 15000
#define BC_TAG_CYCLE              1000

// Field<T,Dim> tags
namespace ippl {
    namespace detail {
        #define HALO_FACE_TAG           20000
        #define HALO_EDGE_TAG           30000
        #define HALO_VERTEX_TAG         40000
        #define HALO_TAG_CYCLE           1000
    }
}


// Special tags used by Particle classes for communication.
#define P_WEIGHTED_LAYOUT_TAG   50000
#define P_WEIGHTED_RETURN_TAG   51000
#define P_WEIGHTED_TRANSFER_TAG 52000
#define P_SPATIAL_LAYOUT_TAG    53000
#define P_SPATIAL_RETURN_TAG    54000
#define P_SPATIAL_TRANSFER_TAG  55000
#define P_SPATIAL_GHOST_TAG     56000
#define P_SPATIAL_RANGE_TAG     57000
#define P_RESET_ID_TAG          58000
#define P_LAYOUT_CYCLE           1000

// Tags for Ippl setup
#define IPPL_MAKE_HOST_MAP_TAG    60000
#define IPPL_TAG_CYCLE             1000

// Tags for Ippl application codes
#define IPPL_PARAREAL_APP    90000
#define IPPL_APP_TAG1    91000
#define IPPL_APP_TAG2    92000
#define IPPL_APP_TAG3    93000
#define IPPL_APP_TAG4    94000
#define IPPL_APP_TAG5    95000
#define IPPL_APP_TAG6    96000
#define IPPL_APP_TAG7    97000
#define IPPL_APP_TAG8    98000
#define IPPL_APP_TAG9    99000
#define IPPL_APP_CYCLE    1000

// IDs used to identify buffers created using the buffer factory interface
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

// FFT Poisson Solver
#define IPPL_SOLVER_SEND 13000
#define IPPL_SOLVER_RECV 14000
#define IPPL_VICO_SEND   16000
#define IPPL_VICO_RECV   17000

#define OPEN_SOLVER_TAG 18000
#define VICO_SOLVER_TAG 70000

#define IPPL_PARAREAL_SEND 19000
#define IPPL_PARAREAL_RECV 21000

#endif // TAGS_H
