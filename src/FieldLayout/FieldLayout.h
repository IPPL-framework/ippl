//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks. It performs the initial
//   partitioning. The user may request that a particular dimension not be
//   partitioned by flagging that axis as 'SERIAL' (instead of 'PARALLEL').
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_FIELD_LAYOUT_H
#define IPPL_FIELD_LAYOUT_H

#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"

#include <array>
#include <iostream>
#include <vector>

namespace ippl {

    template <unsigned Dim> class FieldLayout;

    template <unsigned Dim>
    std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

    // enumeration used to select serial or parallel axes
    enum e_dim_tag { SERIAL=0, PARALLEL=1 } ;


    template<unsigned Dim>
    class FieldLayout
    {

    public:
        using NDIndex_t = NDIndex<Dim>;
        using view_type = typename detail::ViewType<NDIndex_t, 1>::view_type;
        using host_mirror_type = typename view_type::host_mirror_type;
        using face_neighbor_type = std::array<std::vector<int>, 2 * Dim>;
        using edge_neighbor_type = std::array<std::vector<int>, Dim * (1 << (Dim - 1))>;
        using vertex_neighbor_type = std::array<int, 2 << (Dim - 1)>;
        
        
        struct bound_type {
            // lower bounds (ordering: x, y, z)
            std::array<long, Dim> lo;
            // upper bounds (ordering x, y, z)
            std::array<long, Dim> hi;
        };
        
        
        using face_neighbor_range_type = std::array<std::vector<int>, 2 * Dim>;
        using edge_neighbor_range_type = std::array<std::vector<int>, Dim * (1 << (Dim - 1))>;
        using vertex_neighbor_range_type = std::array<int, 2 << (Dim - 1)>;


        /*!
         * Default constructor, which should only be used if you are going to
         * call 'initialize' soon after (before using in any context)
         */
        FieldLayout();

        FieldLayout(const NDIndex<Dim>& domain, e_dim_tag *p=0, bool isAllPeriodic=false);

        // Destructor: Everything deletes itself automatically ... the base
        // class destructors inform all the FieldLayoutUser's we're going away.
        virtual ~FieldLayout();

        // Initialization functions, only to be called by the user of FieldLayout
        // objects when the FieldLayout was created using the default constructor;
        // otherwise these are only called internally by the various non-default
        // FieldLayout constructors:

        void initialize(const NDIndex<Dim>& domain, e_dim_tag *p=0, bool isAllPeriodic=false);


        // Return the domain.
        const NDIndex<Dim>& getDomain() const { return gDomain_m; }

        // Compare FieldLayouts to see if they represent the same domain; if
        // dimensionalities are different, the NDIndex operator==() will return
        // false:
        template <unsigned Dim2>
        bool operator==(const FieldLayout<Dim2>& x) const {
            return gDomain_m == x.getDomain();
        }

        // for the requested dimension, report if the distribution is
        // SERIAL or PARALLEL
        e_dim_tag getDistribution(unsigned int d) const {
            e_dim_tag retval = PARALLEL;
            if (minWidth_m[d] == (unsigned int) gDomain_m[d].length())
                retval = SERIAL;
            return retval;
        }

        // for the requested dimension, report if the distribution was requested to
        // be SERIAL or PARALLEL
        e_dim_tag getRequestedDistribution(unsigned int d) const {
            return requestedLayout_m[d];
        }

        const NDIndex_t& getLocalNDIndex(int rank = Ippl::Comm->rank()) const;

        const host_mirror_type& getHostLocalDomains() const;

        const view_type& getDeviceLocalDomains() const;

        const face_neighbor_type& getFaceNeighbors() const;

        /*!
         * Get the dimension of the face. It is based on the ordering:
         * x low, x high, y low, y high, z low, z high
         * @returns the dimension of the face
         */
        unsigned int getDimOfFace(unsigned int face) const { return face / 2; }

        const edge_neighbor_type& getEdgeNeighbors() const;

        const vertex_neighbor_type& getVertexNeighbors() const;
        
        const face_neighbor_range_type& getFaceNeighborsSendRange() const;

        const edge_neighbor_range_type& getEdgeNeighborsSendRange() const;

        const vertex_neighbor_range_type& getVertexNeighborsSendRange() const;

        const face_neighbor_range_type& getFaceNeighborsRecvRange() const;

        const edge_neighbor_range_type& getEdgeNeighborsRecvRange() const;

        const vertex_neighbor_range_type& getVertexNeighborsRecvRange() const;

        void findNeighbors(int nghost = 1);

        void findNeighborsAllPeriodicBC(unsigned int d0, 
                                        NDIndex_t& gnd, 
                                        NDIndex_t& nd, 
                                        NDIndex_t& ndNeighbor, 
                                        int nghost, 
                                        int rank);

        void write(std::ostream& = std::cout) const;

        bool isAllPeriodic_m;

        /*!
         * Obtain the bounds to send / receive. The second domain, i.e.,
         * nd2, is grown by nghost cells in each dimension in order to
         * figure out the intersecting cells.
         * @param nd1 either remote or owned domain
         * @param nd2 either remote or owned domain
         * @param offset to map global to local grid point
         * @param nghost number of ghost cells per dimension
         */
        bound_type getBounds(const NDIndex_t& nd1,
                             const NDIndex_t& nd2,
                             const NDIndex_t& offset,
                             int nghost);


    private:
        /*!
         * @param grown the grown domain of myRank
         * @param inersect the intersection between grown and the remote domain
         * @param rank the rank of the remote domain
         */
        void addVertex(const NDIndex_t& grown, const NDIndex_t& intersect, int rank, 
                       const bound_type& rangeSend, const bound_type& rangeRecv);
        
        void addEdge(const NDIndex_t& grown, const NDIndex_t& intersect, int rank,
                     const bound_type& rangeSend, const bound_type& rangeRecv);

        void addFace(const NDIndex_t& grown, const NDIndex_t& intersect, int rank, 
                     const bound_type& rangeSend, const bound_type& rangeRecv);

    private:
        //! Global domain
        NDIndex_t gDomain_m;

        //! Local domains (device view)
        view_type dLocalDomains_m;

        //! Local domains (host mirror view)
        host_mirror_type hLocalDomains_m;

        e_dim_tag requestedLayout_m[Dim];

        unsigned int minWidth_m[Dim];

        /*!
         * This container has length 2*Dim. Each index represents a face
         * (ordering: x low, x high, y low, y high, z low, z high). Each
         * index contains a vector (length is equal to the number of ranks
         * it shares the face with). The values are the ranks sharing the face.
         * An empty vector denotes a physical / mesh boundary.
         */
        face_neighbor_type faceNeighbors_m;

        /*!
         * Neighboring ranks that store the edge values.
         * [(x low,  y low,  z low),  (x high, y low,  z low)]  --> edge 0
         * [(x low,  y high, z low),  (x high, y high, z low)]  --> edge 1
         * [(x low,  y low,  z high), (x high, y low,  z high)] --> edge 2
         * [(x low,  y high, z high), (x high, y high, z high)] --> edge 3
         *
         * [(x low,  y low,  z low),  (x low,  y high, z low)]  --> edge 4
         * [(x high, y low,  z low),  (x high, y high, z low)]  --> edge 5
         * [(x low,  y low,  z high), (x low,  y high, z high)] --> edge 6
         * [(x high, y low,  z high), (x high, y high, z high)] --> edge 7
         *
         * [(x low,  y low,  z low),  (x low,  y low,  z high)] --> edge 8
         * [(x high, y low,  z low),  (x high, y low,  z high)] --> edge 9
         * [(x low,  y high, z low),  (x low,  y high, z high)] --> edge 10
         * [(x high, y high, z low),  (x high, y high, z high)] --> edge 11
         */
        edge_neighbor_type edgeNeighbors_m;

        /*!
         * Neighboring ranks that have the vertex value (corner cell). The value
         * is negative, i.e. -1, if the vertex is on a mesh boundary.
         * x low,  y low,  z low  --> vertex index 0
         * x high, y low,  z low  --> vertex index 1
         * x low,  y high, z low  --> vertex index 2
         * x high, y high, z low  --> vertex index 3
         * x low,  y low,  z high --> vertex index 4
         * x high, y low,  z high --> vertex index 5
         * x low,  y high, z high --> vertex index 6
         * x high, y high, z high --> vertex index 7
         */
        vertex_neighbor_type vertexNeighbors_m;



        void calcWidths();

        face_neighbor_range_type faceNeighborsSendRange_m, faceNeighborsRecvRange_m;
        edge_neighbor_range_type edgeNeighborsSendRange_m, edgeNeighborsRecvRange_m;
        vertex_neighbor_range_type vertexNeighborsSendRange_m, vertexNeighborsRecvRange_m;

    };


    template<unsigned Dim>
    inline
    std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
        f.write(out);
        return out;
    }
}


#include "FieldLayout/FieldLayout.hpp"

#endif
