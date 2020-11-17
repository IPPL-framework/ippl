//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks (vnodes). It performs the initial
//   partitioning, and stores a list of local and remote vnodes. The user may
//   request that a particular dimension not be partitioned by flagging that
//   axis as 'SERIAL' (instead of 'PARALLEL').
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
#include "FieldLayout/FieldLayout.h"
#include "Message/Communicate.h"
#include "Message/Message.h"
#include "Ippl.h"
#include "Utility/IpplStats.h"
#include "Utility/PAssert.h"

#include "Partition/Partitioner.h"


#include <cstdlib>
#include <limits>

namespace ippl {

    template<unsigned Dim>
    FieldLayout<Dim>::FieldLayout()
    : dLocalDomains_m("local domains (device)", 0)
    , hLocalDomains_m(Kokkos::create_mirror_view(dLocalDomains_m))
    {
        for (unsigned int d = 0; d < Dim; ++d)
            requestedLayout_m[d] = PARALLEL;
    }


    template<unsigned Dim>
    FieldLayout<Dim>::~FieldLayout() { }


    template<unsigned Dim>
    void
    FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
                                 e_dim_tag* userflags)
    {
        int nRanks = Ippl::Comm->size();
//         int rank = Ippl::Comm->rank();

        gDomain_m = domain;

        if (nRanks < 2) {
            Kokkos::resize(dLocalDomains_m, nRanks);
            Kokkos::resize(hLocalDomains_m, nRanks);
            hLocalDomains_m(0) = domain;
            Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);
            return;
        }


        // If the user did not specify parallel/serial flags then make all parallel.
        long totparelems = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            if (userflags == 0)
                requestedLayout_m[d] = PARALLEL;
            else
                requestedLayout_m[d] = userflags[d];

            if (requestedLayout_m[d] == PARALLEL) {
                totparelems *= domain[d].length();
            }
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

        Kokkos::deep_copy(dLocalDomains_m, hLocalDomains_m);

        // Calculate the widths of all the vnodes
//         calcWidths();
    }

//----------------------------------------------------------------------
//
// Completely repartition this FieldLayout and all of the Fields
// defined on it.
//
template<unsigned Dim>
void
FieldLayout<Dim>::Repartition(const NDIndex<Dim>* /*idxBegin*/,
			      const NDIndex<Dim>* /*idxEnd*/)
{
/*


    // Build a temporary FieldLayout to declare the temporary arrays.
    FieldLayout<Dim> tempLayout(Domain,idxBegin,idxEnd);

    // Repartition each field.
    iterator_if p, endp=end_if();
    for (p=begin_if(); p!=endp; ++p) {
        FieldLayoutUser *user = (FieldLayoutUser *)((*p).second);
        user->Repartition(&tempLayout);
    }

    // Copy back the layout information.
    Local_ac = tempLayout.Local_ac;
    Remotes_ac = tempLayout.Remotes_ac;

    // Calculate the widths of all the vnodes
    calcWidths();

    //INCIPPLSTAT(incRepartitions);*/
}


//----------------------------------------------------------------------
//
// calculate the minimum vnode sizes in each dimension
template<unsigned Dim>
void FieldLayout<Dim>::calcWidths() {
//     unsigned int d;
//
//     // initialize widths first
//     for (d=0; d < Dim; ++d)
//         MinWidth[d] = getDomain()[d].length();
//
//     // look for minimum width in local vnodes
//     for  (const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i) {
//         const NDIndex<Dim> &dom = (*v_i).second->getDomain();
//         for (d=0; d < Dim; ++d) {
//             if ((unsigned int) dom[d].length() < MinWidth[d])
//                 MinWidth[d] = dom[d].length();
//         }
//     }
//
//     // look for minimum width in remove vnodes
//     ac_domain_vnodes *v_ac = Remotes_ac[ gc0() ].get();
//     for (iterator_dv dv_i = v_ac->begin(); dv_i != v_ac->end(); ++ dv_i) {
//         const NDIndex<Dim> &dom = (*dv_i).first;
//         for (d=0; d < Dim; ++d) {
//             if ((unsigned int) dom[d].length() < MinWidth[d])
//                 MinWidth[d] = dom[d].length();
//         }
//     }
}


template<unsigned Dim>
const typename FieldLayout<Dim>::NDIndex_t&
FieldLayout<Dim>::getLocalNDIndex(int rank) const
{
    return hLocalDomains_m(rank);
}


template<unsigned Dim>
void FieldLayout<Dim>::write(std::ostream& out) const
{
    Kokkos::fence();
    detail::write<NDIndex_t>(dLocalDomains_m, out);
/*
    int icount;

    the whole domain, and the number of users
    out << "Domain = " << Domain << "\n";
//     out << "FieldLayoutUsers = " << size_if() << "\n";

    iterate over the local vnodes and print them out.
//     out << "Total number of vnodes = " << numVnodes() << std::endl;
//     out << "Local Vnodes = " << Local_ac.size() << "\n";
    icount = 0;
    tjw can now do operator<<() with whole Vnode objects
    tjw  for(const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i)
    tjw    out << " vnode " << icount++ <<" : "<< (*v_i).second->getDomain() << "\n";
    for(const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i)
        out << " vnode " << icount++ << ": " << *((*v_i).second) << std::endl;

    iterate over the remote vnodes and print them out.
    ac_domain_vnodes *v_ac = Remotes_ac[ gc0() ].get();
    out << "Remote Vnodes = " << v_ac->size() << "\n";
    icount = 0;
    for (iterator_dv dv_i = v_ac->begin(); dv_i != v_ac->end(); ++ dv_i)
        out << " vnode " << icount++ << " : " << *((*dv_i).second) << std::endl;
}*/

}
}