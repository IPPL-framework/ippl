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

#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "Types/ViewTypes.h"

#include "Index/NDIndex.h"

namespace ippl {

    template <unsigned Dim>
    class FieldLayout;

    template <unsigned Dim>
    std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

    // enumeration used to select serial or parallel axes
    enum e_dim_tag {
        SERIAL   = 0,
        PARALLEL = 1
    };

    // enumeration used to describe a hypercube's relation to
    // a particular axis in a given bounded domain
    enum e_cube_tag {
        UPPER       = 0,
        LOWER       = 1,
        IS_PARALLEL = 2
    };

    namespace detail {
        /*!
         * Counts the hypercubes in a given dimension
         * @param dim the dimension
         * @return 3^n
         */
        constexpr unsigned int countHypercubes(unsigned int dim) {
            unsigned int ret = 1;
            for (unsigned int d = 0; d < dim; d++)
                ret *= 3;
            return ret;
        }

        constexpr unsigned int factorial(unsigned x) {
            return x == 0 ? 1 : x * factorial(x - 1);
        }

        constexpr unsigned int nCr(unsigned a, unsigned b) {
            return factorial(a) / (factorial(b) * factorial(a - b));
        }

        template <unsigned Dim>
        constexpr unsigned int countCubes(unsigned m) {
            return (1 << (Dim - m)) * nCr(Dim, m);
        }

        bool isUpper(unsigned int face);

        unsigned int getFaceDim(unsigned int face);

        template <unsigned Dim>
        unsigned int indexToFace(unsigned int index) {
            // facets are group low/high by axis
            unsigned int axis = index / 2;
            // the digit to subtract is determined by whether the index describes an upper
            // face (even index) or lower face (odd index) and that digit's position is
            // determined by the axis of the face
            unsigned int toRemove = (2 - index % 2) * countHypercubes(axis);
            // start with all 2s (in base 3) and change the correct digit to get the encoded face
            return countHypercubes(Dim) - 1 - toRemove;
        }

        template <
            unsigned Dim, typename... CubeTags,
            typename = std::enable_if_t<sizeof...(CubeTags) == Dim - 1>,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<e_cube_tag, CubeTags>...>>>
        unsigned int getCube(e_cube_tag tag, CubeTags... tags) {
            if constexpr (Dim == 1) {
                return tag;
            } else {
                return tag + 3 * getCube<Dim - 1>(tags...);
            }
        }

        template <size_t... Idx>
        unsigned int getFace_impl(const std::array<e_cube_tag, sizeof...(Idx)>& args,
                                  const std::index_sequence<Idx...>&) {
            return getCube<sizeof...(Idx)>(args[Idx]...);
        }

        template <unsigned Dim>
        unsigned int getFace(unsigned int axis, e_cube_tag side) {
            std::array<e_cube_tag, Dim> args;
            args.fill(IS_PARALLEL);
            args[axis] = side;
            return getFace_impl(args, std::make_index_sequence<Dim>{});
        }
    }  // namespace detail

    template <unsigned Dim>
    class FieldLayout {
    public:
        using NDIndex_t        = NDIndex<Dim>;
        using view_type        = typename detail::ViewType<NDIndex_t, 1>::view_type;
        using host_mirror_type = typename view_type::host_mirror_type;

        struct bound_type {
            // lower bounds (ordering: x, y, z, ...)
            std::array<long, Dim> lo;
            // upper bounds (ordering: x, y, z, ...)
            std::array<long, Dim> hi;

            /*!
             * Compute the size of the region described by the bounds
             * @return Product of the axial dimensions of the region
             */
            long size() const {
                long total = 1;
                for (unsigned d = 0; d < Dim; d++) {
                    total *= hi[d] - lo[d];
                }
                return total;
            }
        };

        using rank_list   = std::vector<int>;
        using bounds_list = std::vector<bound_type>;

        using neighbor_list       = std::array<rank_list, detail::countHypercubes(Dim) - 1>;
        using neighbor_range_list = std::array<bounds_list, detail::countHypercubes(Dim) - 1>;

        /*!
         * Default constructor, which should only be used if you are going to
         * call 'initialize' soon after (before using in any context)
         */
        FieldLayout();

        FieldLayout(const NDIndex<Dim>& domain, e_dim_tag* p = 0, bool isAllPeriodic = false);

        // Destructor: Everything deletes itself automatically ... the base
        // class destructors inform all the FieldLayoutUser's we're going away.
        virtual ~FieldLayout();

        // Initialization functions, only to be called by the user of FieldLayout
        // objects when the FieldLayout was created using the default constructor;
        // otherwise these are only called internally by the various non-default
        // FieldLayout constructors:

        void initialize(const NDIndex<Dim>& domain, e_dim_tag* p = 0, bool isAllPeriodic = false);

        // Return the domain.
        const NDIndex<Dim>& getDomain() const { return gDomain_m; }

        // Compare FieldLayouts to see if they represent the same domain; if
        // dimensionalities are different, the NDIndex operator==() will return
        // false:
        template <unsigned Dim2>
        bool operator==(const FieldLayout<Dim2>& x) const {
            return gDomain_m == x.getDomain();
        }

        bool operator==(const FieldLayout<Dim>& x) const {
            for (unsigned int i = 0; i < Dim; ++i) {
                if (hLocalDomains_m(Ippl::Comm->rank())[i] != x.getLocalNDIndex()[i])
                    return false;
            }
            return true;
        }

        // for the requested dimension, report if the distribution is
        // SERIAL or PARALLEL
        e_dim_tag getDistribution(unsigned int d) const {
            if (minWidth_m[d] == (unsigned int)gDomain_m[d].length())
                return SERIAL;
            return PARALLEL;
        }

        // for the requested dimension, report if the distribution was requested to
        // be SERIAL or PARALLEL
        e_dim_tag getRequestedDistribution(unsigned int d) const { return requestedLayout_m[d]; }

        const NDIndex_t& getLocalNDIndex(int rank = Ippl::Comm->rank()) const;

        const host_mirror_type getHostLocalDomains() const;

        const view_type getDeviceLocalDomains() const;

        /*!
         * Get a list of all the neighbors, arranged by ternary encoding
         * of the hypercubes
         * @return List of list of neighbor ranks touching each boundary component
         */
        const neighbor_list& getNeighbors() const;

        /*!
         * Get the domain ranges corresponding to regions that should be sent
         * to neighbor ranks
         * @return Ranges to send
         */
        const neighbor_range_list& getNeighborsSendRange() const;

        /*!
         * Get the domain ranges corresponding to regions that should be received
         * from neighbor ranks
         * @return Ranges to receive
         */
        const neighbor_range_list& getNeighborsRecvRange() const;

        /*!
         * Compute the index corresponding to the component opposite the component
         * with the given index, as determined by the ternary encoding for hypercubes
         * @param index index of the known component
         * @return Index of the matching component
         */
        static int getMatchingIndex(int index);

        /*!
         * Recursively finds neighbor ranks for layouts with all periodic boundary
         * conditions
         * @param nghost number of ghost cells
         * @param localDomain the rank's local domain
         * @param grown the local domain, grown by the number of ghost cells
         * @param neighborDomain a candidate neighbor rank's domain
         * @param rank the candidate neighbor's rank
         * @param offsets a dictionary containing offsets along different dimensions
         * @param d0 the dimension from which to start checking (default 0)
         * @param codim the codimension of overlapping regions to check (default 0)
         */
        void findPeriodicNeighbors(const int nghost, const NDIndex<Dim>& localDomain,
                                   NDIndex<Dim>& grown, NDIndex<Dim>& neighborDomain,
                                   const int rank, std::map<unsigned int, int>& offsets,
                                   unsigned d0 = 0, unsigned codim = 0);

        /*!
         * Finds all neighboring ranks based on the field layout
         * @param nghost number of ghost cells (default 1)
         */
        void findNeighbors(int nghost = 1);

        /*!
         * Adds a neighbor to the neighbor list
         * @param gnd the local domain, including ghost cells
         * @param nd the local domain
         * @param ndNeighbor the neighbor rank's domain
         * @param intersect the intersection of the domains
         * @param nghost number of ghost cells
         * @param rank the neighbor's rank
         */
        void addNeighbors(const NDIndex_t& gnd, const NDIndex_t& nd, const NDIndex_t& ndNeighbor,
                          const NDIndex_t& intersect, int nghost, int rank);

        void write(std::ostream& = std::cout) const;

        void updateLayout(const std::vector<NDIndex_t>& domains);

        bool isAllPeriodic_m;

    private:
        /*!
         * Obtain the bounds to send / receive. The second domain, i.e.,
         * nd2, is grown by nghost cells in each dimension in order to
         * figure out the intersecting cells.
         * @param nd1 either remote or owned domain
         * @param nd2 either remote or owned domain
         * @param offset to map global to local grid point
         * @param nghost number of ghost cells per dimension
         */
        bound_type getBounds(const NDIndex_t& nd1, const NDIndex_t& nd2, const NDIndex_t& offset,
                             int nghost);

        int getPeriodicOffset(const NDIndex_t& nd, const unsigned int d, const int k);

    private:
        //! Global domain
        NDIndex_t gDomain_m;

        //! Local domains (device view)
        view_type dLocalDomains_m;

        //! Local domains (host mirror view)
        host_mirror_type hLocalDomains_m;

        e_dim_tag requestedLayout_m[Dim];

        unsigned int minWidth_m[Dim];

        neighbor_list neighbors_m;
        neighbor_range_list neighborsSendRange_m, neighborsRecvRange_m;

        void calcWidths();
    };

    template <unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
        f.write(out);
        return out;
    }
}  // namespace ippl

#include "FieldLayout/FieldLayout.hpp"

#endif
