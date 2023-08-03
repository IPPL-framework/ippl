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
#include "Ippl.h"

#include <cstdlib>
#include <limits>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include "Utility/PAssert.h"

#include "FieldLayout/FieldLayout.h"
#include "Partition/Partitioner.h"

namespace ippl {

    template <unsigned Dim>
    int FieldLayout<Dim>::getMatchingIndex(int index) {
        // If we are touching another boundary component, that component must be parallel to us,
        // so any 2s are unchanged. All other digits must be swapped because otherwise we would
        // share a higher dimension boundary region with that other component and the digit
        // would be 2
        static const int digit_swap[3] = {1, 0, 2};
        int match                      = 0;
        for (unsigned d = 1; d < detail::countHypercubes(Dim); d *= 3) {
            match += digit_swap[index % 3] * d;
            index /= 3;
        }
        return match;
    }

    template <unsigned Dim>
    FieldLayout<Dim>::FieldLayout()
        : dLocalDomains_m("local domains (device)", 0)
        , hLocalDomains_m(Kokkos::create_mirror_view(dLocalDomains_m)) {
        for (unsigned int d = 0; d < Dim; ++d) {
            requestedLayout_m[d] = PARALLEL;
            minWidth_m[d]        = 0;
        }
    }

    template <unsigned Dim>
    FieldLayout<Dim>::FieldLayout(const NDIndex<Dim>& domain, e_dim_tag* p, bool isAllPeriodic)
        : FieldLayout() {
        initialize(domain, p, isAllPeriodic);
    }

    template <unsigned Dim>
    FieldLayout<Dim>::~FieldLayout() {}

    template <unsigned Dim>
    void FieldLayout<Dim>::updateLayout(const std::vector<NDIndex<Dim>>& domains) {
        if (domains.empty()) {
            return;
        }

        for (unsigned int i = 0; i < domains.size(); i++) {
            hLocalDomains_m(i) = domains[i];
        }

        findNeighbors();

        Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);

        calcWidths();
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain, e_dim_tag* userflags,
                                      bool isAllPeriodic) {
        int nRanks = Comm->size();

        gDomain_m = domain;

        isAllPeriodic_m = isAllPeriodic;

        // If the user did not specify parallel/serial flags then make all parallel.
        long totparelems = 1;
        isAllSerial_m = true;
        for (unsigned d = 0; d < Dim; ++d) {
            if (userflags == 0) {
                requestedLayout_m[d] = PARALLEL;
            } else {
                requestedLayout_m[d] = userflags[d];
            }

            if (requestedLayout_m[d] == PARALLEL) {
                totparelems *= domain[d].length();
                isAllSerial_m = false;
            }
        }

        if (nRanks < 2 || isAllSerial_m) {
            Kokkos::resize(dLocalDomains_m, nRanks);
            Kokkos::resize(hLocalDomains_m, nRanks);
            for (int r = 0; r < nRanks; r++) {
                hLocalDomains_m(r) = domain;
            }
            Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);
            return;
        }

        /* Check to see if we have too few elements to partition.  If so, reduce
         * the number of ranks (if necessary) to just the number of elements along
         * parallel dims.
         */
        if (totparelems < nRanks) {
            nRanks = totparelems;
        }

        Kokkos::resize(dLocalDomains_m, nRanks);
        Kokkos::resize(hLocalDomains_m, nRanks);

        detail::Partitioner<Dim> partition;

        partition.split(domain, hLocalDomains_m, requestedLayout_m, nRanks);

        findNeighbors();

        Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);

        calcWidths();
    }

    template <unsigned Dim>
    bool FieldLayout<Dim>::isAllSerial() const {
        return isAllSerial_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::NDIndex_t& FieldLayout<Dim>::getLocalNDIndex(int rank) const {
        return hLocalDomains_m(rank);
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::host_mirror_type FieldLayout<Dim>::getHostLocalDomains()
        const {
        return hLocalDomains_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::view_type FieldLayout<Dim>::getDeviceLocalDomains() const {
        return dLocalDomains_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::neighbor_list& FieldLayout<Dim>::getNeighbors() const {
        return neighbors_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::neighbor_range_list& FieldLayout<Dim>::getNeighborsSendRange()
        const {
        return neighborsSendRange_m;
    }

    template <unsigned Dim>
    const typename FieldLayout<Dim>::neighbor_range_list& FieldLayout<Dim>::getNeighborsRecvRange()
        const {
        return neighborsRecvRange_m;
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::write(std::ostream& out) const {
        if (Comm->rank() > 0) {
            return;
        }

        out << "Domain = " << gDomain_m << "\n"
            << "Total number of boxes = " << hLocalDomains_m.size() << "\n";

        using size_type = typename host_mirror_type::size_type;
        for (size_type i = 0; i < hLocalDomains_m.size(); ++i) {
            out << "    Box " << i << " " << hLocalDomains_m(i) << "\n";
        }
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::calcWidths() {
        // initialize widths first
        for (unsigned int d = 0; d < Dim; ++d) {
            minWidth_m[d] = gDomain_m[d].length();
        }

        using size_type = typename host_mirror_type::size_type;
        for (size_type i = 0; i < hLocalDomains_m.size(); ++i) {
            const NDIndex_t& dom = hLocalDomains_m(i);
            for (unsigned int d = 0; d < Dim; ++d) {
                if ((unsigned int)dom[d].length() < minWidth_m[d]) {
                    minWidth_m[d] = dom[d].length();
                }
            }
        }
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::findPeriodicNeighbors(const int nghost, const NDIndex<Dim>& localDomain,
                                                 NDIndex<Dim>& grown, NDIndex<Dim>& neighborDomain,
                                                 const int rank,
                                                 std::map<unsigned int, int>& offsets, unsigned d0,
                                                 unsigned codim) {
        for (unsigned int d = d0; d < Dim; ++d) {
            // 0 - check upper boundary
            // 1 - check lower boundary
            for (int k = 0; k < 2; ++k) {
                auto offset = offsets[d] = getPeriodicOffset(localDomain, d, k);
                if (offset == 0) {
                    continue;
                }

                grown[d] += offset;

                if (grown.touches(neighborDomain)) {
                    auto intersect = grown.intersect(neighborDomain);
                    for (auto& [d, offset] : offsets) {
                        neighborDomain[d] -= offset;
                    }
                    addNeighbors(grown, localDomain, neighborDomain, intersect, nghost, rank);
                    for (auto& [d, offset] : offsets) {
                        neighborDomain[d] += offset;
                    }
                }
                if (codim + 1 < Dim) {
                    findPeriodicNeighbors(nghost, localDomain, grown, neighborDomain, rank, offsets,
                                          d + 1, codim + 1);
                }

                grown[d] -= offset;
                offsets.erase(d);
            }
        }
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::findNeighbors(int nghost) {
        /* We need to reset the neighbor list
         * and its ranges because of the repartitioner.
         */
        for (size_t i = 0; i < detail::countHypercubes(Dim) - 1; i++) {
            neighbors_m[i].clear();
            neighborsSendRange_m[i].clear();
            neighborsRecvRange_m[i].clear();
        }

        int myRank = Comm->rank();

        // get my local box
        auto& nd = hLocalDomains_m[myRank];

        // grow the box by nghost cells in each dimension
        auto gnd = nd.grow(nghost);

        static IpplTimings::TimerRef findInternalNeighborsTimer =
            IpplTimings::getTimer("findInternal");
        static IpplTimings::TimerRef findPeriodicNeighborsTimer =
            IpplTimings::getTimer("findPeriodic");
        for (int rank = 0; rank < Comm->size(); ++rank) {
            if (rank == myRank) {
                // do not compare with my domain
                continue;
            }

            auto& ndNeighbor = hLocalDomains_m[rank];
            IpplTimings::startTimer(findInternalNeighborsTimer);
            // For inter-processor neighbors
            if (gnd.touches(ndNeighbor)) {
                auto intersect = gnd.intersect(ndNeighbor);
                addNeighbors(gnd, nd, ndNeighbor, intersect, nghost, rank);
            }
            IpplTimings::stopTimer(findInternalNeighborsTimer);

            IpplTimings::startTimer(findPeriodicNeighborsTimer);
            if (isAllPeriodic_m) {
                std::map<unsigned int, int> offsets;
                findPeriodicNeighbors(nghost, nd, gnd, ndNeighbor, rank, offsets);
            }
            IpplTimings::stopTimer(findPeriodicNeighborsTimer);
        }
    }

    template <unsigned Dim>
    void FieldLayout<Dim>::addNeighbors(const NDIndex_t& gnd, const NDIndex_t& nd,
                                        const NDIndex_t& ndNeighbor, const NDIndex_t& intersect,
                                        int nghost, int rank) {
        bound_type rangeSend, rangeRecv;
        rangeSend = getBounds(nd, ndNeighbor, nd, nghost);

        rangeRecv = getBounds(ndNeighbor, nd, nd, nghost);

        int index = 0;
        for (unsigned d = 0, digit = 1; d < Dim; d++, digit *= 3) {
            // For each dimension, check what kind of intersection we have and construct
            // an index for the component based on its base-3 representation with the
            // following properties for each digit:
            // 0 - touching the lower axis value
            // 1 - touching the upper axis value
            // 2 - parallel to the axis
            if (intersect[d].length() == 1) {
                if (gnd[d].first() != intersect[d].first()) {
                    index += digit;
                }
            } else {
                index += 2 * digit;
            }
        }
        neighbors_m[index].push_back(rank);
        neighborsSendRange_m[index].push_back(rangeSend);
        neighborsRecvRange_m[index].push_back(rangeRecv);
    }

    template <unsigned Dim>
    typename FieldLayout<Dim>::bound_type FieldLayout<Dim>::getBounds(const NDIndex_t& nd1,
                                                                      const NDIndex_t& nd2,
                                                                      const NDIndex_t& offset,
                                                                      int nghost) {
        NDIndex<Dim> gnd = nd2.grow(nghost);

        NDIndex<Dim> overlap = gnd.intersect(nd1);

        bound_type intersect;

        /* Obtain the intersection bounds with local ranges of the view.
         * Add "+1" to the upper bound since Kokkos loops always to "< extent".
         */
        for (size_t i = 0; i < Dim; ++i) {
            intersect.lo[i] = overlap[i].first() - offset[i].first() /*offset*/ + nghost;
            intersect.hi[i] = overlap[i].last() - offset[i].first() /*offset*/ + nghost + 1;
        }

        return intersect;
    }

    template <unsigned Dim>
    int FieldLayout<Dim>::getPeriodicOffset(const NDIndex_t& nd, const unsigned int d,
                                            const int k) {
        switch (k) {
            case 0:
                if (nd[d].max() == gDomain_m[d].max()) {
                    return -gDomain_m[d].length();
                }
                break;
            case 1:
                if (nd[d].min() == gDomain_m[d].min()) {
                    return gDomain_m[d].length();
                }
                break;
            default:
                throw IpplException("FieldLayout:getPeriodicOffset", "k  has to be either 0 or 1");
        }
        return 0;
    }

}  // namespace ippl
