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
#define IPPL_RETRANSMIT_TAG       7    // node should resend a message
#define IPPL_MSG_OK_TAG           8    // some messages were sent correctly


// tags for reduction
#define COMM_REDUCE_SEND_TAG    10000
#define COMM_REDUCE_RECV_TAG    11000
#define COMM_REDUCE_SCATTER_TAG 12000
#define COMM_REDUCE_CYCLE        1000


// tag for applying parallel periodic boundary condition.

#define BC_PARALLEL_PERIODIC_TAG 15000
#define BC_TAG_CYCLE              1000

// Field<T,Dim> tags
#define F_GUARD_CELLS_TAG       20000 // Field::fillGuardCells()
#define F_WRITE_TAG             21000 // Field::write()
#define F_READ_TAG              22000 // Field::read()
#define F_GEN_ASSIGN_TAG        23000 // assign(BareField,BareField)
#define F_REPARTITION_BCAST_TAG 24000 // broadcast in FieldLayout::repartion.
#define F_REDUCE_PERP_TAG       25000 // reduction in binary load balance.
#define F_GETSINGLE_TAG         26000 // IndexedBareField::getsingle()
#define F_REDUCE_TAG            27000 // Reduction in minloc/maxloc
#define F_LAYOUT_IO_TAG         28000 // Reduction in minloc/maxloc
#define F_TAG_CYCLE              1000

// Tags for FieldView and FieldBlock
#define FV_2D_TAG               30000 // FieldView::update_2D_data()
#define FV_3D_TAG               31000 // FieldView::update_2D_data()
#define FV_TAG_CYCLE             1000

#define FB_WRITE_TAG            32000 // FieldBlock::write()
#define FB_READ_TAG             33000 // FieldBlock::read()
#define FB_TAG_CYCLE             1000

#define FP_GATHER_TAG           34000 // FieldPrint::print()
#define FP_TAG_CYCLE             1000

// Tags for DiskField
#define DF_MAKE_HOST_MAP_TAG    35000
#define DF_FIND_RECV_NODES_TAG  36000
#define DF_QUERY_TAG            37000
#define DF_READ_TAG             38000
#define DF_OFFSET_TAG           39000
#define DF_READ_META_TAG        40000
#define DF_TAG_CYCLE             1000

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
#define IPPL_APP_TAG0    90000
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

#endif // TAGS_H
