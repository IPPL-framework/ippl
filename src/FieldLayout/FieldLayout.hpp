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
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"
#include "Utility/PAssert.h"


#include <cstdlib>
#include <limits>

//////////////////////////////////////////////////////////////////////
// Default constructor, which should only be used if you are going to
// call 'initialize' soon after (before using in any context)
template<unsigned Dim>
FieldLayout<Dim>::FieldLayout() {
    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // for this kind of construction, we just take it that the user is
    // requesting all parallel axes
    for (unsigned int dl=0; dl < Dim; ++dl) RequestedLayout[dl] = PARALLEL;
}


//////////////////////////////////////////////////////////////////////
// Destructor: Everything deletes itself automatically ... the base
// class destructors inform all the FieldLayoutUser's we're going away.
template<unsigned Dim>
FieldLayout<Dim>::~FieldLayout() { }


// Initialization functions, only to be called by the user of FieldLayout
// objects when the FieldLayout was created using the default constructor;
// otherwise these are only called internally by the various non-default
// FieldLayout constructors:

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
			     e_dim_tag *p) {


    setup(domain, p);
}


//////////////////////////////////////////////////////////////////////

//
// Now, the meat of the construction.
//

//-----------------------------------------------------------------------------
//This setup() specifies only a total number of vnodes, taking complete control
//on how to do the vnode partitioning of the index space:

template<unsigned Dim>
void
FieldLayout<Dim>::setup(const NDIndex<Dim>& domain,
			e_dim_tag *userflags)
{


    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // Find the number processors.
    int nprocs = Ippl::getNodes();
    int myproc = Ippl::myNode();

    // If the user didn't specify the number of vnodes, make it equal nprocs
    int vnodes = nprocs;

    Inform dbgmsg("FieldLayout::setup", INFORM_ALL_NODES);

    // If the user did not specify parallel/serial flags then make all parallel.
    int parallel_count = 0;
    unsigned int flagdim = 0;
    long totparelems = 1;
    for (flagdim=0; flagdim < Dim; ++flagdim) {
        if (userflags == 0)
            RequestedLayout[flagdim] = PARALLEL;
        else
            RequestedLayout[flagdim] = userflags[flagdim];
        if (RequestedLayout[flagdim] == PARALLEL) {
            parallel_count++;
            totparelems *= domain[flagdim].length();
        }
    }
    // Check to see if we have too few elements to partition.  If so, reduced
    // the number of vnodes (if necessary) to just the number of elements along
    // parallel dims.
    if (totparelems < vnodes) {
        vnodes = totparelems;
    }

    e_dim_tag *flags = RequestedLayout;

    // Recursively split the domain until we have generated all the domains.
    Domain = domain;
    NDIndex<Dim> *domains_c = new NDIndex<Dim>[vnodes];
    NDIndex<Dim> *copy_c    = new NDIndex<Dim>[vnodes];
    NDIndex<Dim> leftDomain ;

    // Start with the whole domain.
    domains_c[0] = domain;
    int v;
    unsigned int d=0;

    int v1,v2,rm,vtot,vl,vr;
    double a,lmax,len;
    for (v=vnodes,rm=0;v>1;v/=2) { rm += (v % 2); }
    if (rm == 0) {

        // vnodes is a power of 2

        for (v=1; v<vnodes; v*=2) {
            // Go to the next parallel dimension.
            while(flags[d] != PARALLEL) if(++d == Dim) d = 0;

            // Split all the current vnodes.
            int i,j;
            for (i=0, j=0; i<v; ++i, j+=2)
                // Split to the left and to the right, saving both.
                domains_c[i].split( copy_c[j] , copy_c[j+1] , d );
            // Copy back.
            std::copy(copy_c,copy_c+v*2, domains_c);

            // On to the next dimension.
            if (++d == Dim) d = 0;
        }

    } else {

        vtot = 1; // count the number of vnodes to make sure that it worked
        // vnodes is not a power of 2 so we need to do some fancy splitting
        // sorry... this would be much cleaner with recursion
        /*
          The way this works is to recursively split on the longest dimension.
          Suppose you request 11 vnodes.  It will split the longest dimension
          in the ratio 5:6 and put the new domains in node 0 and node 5.  Then
          it splits the longest dimension of the 0 domain and puts the results
          in node 0 and node 2 and then splits the longest dimension of node 5
          and puts the results in node 5 and node 8. etc.
          The logic is kind of bizarre, but it works.
        */
        for (v=1; v<2*vnodes; ++v) {
            // kind of reverse the bits of v
            for (v2=v,v1=1;v2>1;v2/=2) { v1 = 2*v1+(v2%2); }
            vl = 0; vr = vnodes;
            while (v1>1) {
                if ((v1%2)==1) {
                    vl=vl+(vr-vl)/2;
                } else {
                    vr=vl+(vr-vl)/2;
                }
                v1/=2;
            }
            v2=vl+(vr-vl)/2;

            if (v2>vl) {
                a = v2-vl;
                a /= vr-vl;
                vr=v2;
                leftDomain=domains_c[vl];
                lmax=0;
                d=std::numeric_limits<unsigned int>::max();
                for (unsigned int dd=0;dd<Dim;++dd) {
                    if ( flags[dd] == PARALLEL ) {
                        if ((len = leftDomain[dd].length()) > lmax) {
                            lmax = len;
                            d = dd;
                        }
                    }
                }
                domains_c[vl].split( domains_c[vl] , domains_c[vr] , d , a);
                ++vtot;
            }
        }
        v=vtot;
    }
    // Make sure we had a power of two number of vnodes.
    PAssert_EQ( v, vnodes );

    // Now make the vnodes, using the domains just generated.
    // Some of them we store in the local list, others in the remote.
    ac_domain_vnodes *remote_ac = new ac_domain_vnodes( domain );
    typedef typename ac_gc_domain_vnodes::value_type vtype;
    Remotes_ac.insert( vtype(gc0(),remote_ac) );
    for (v=0; v<vnodes; ++v)
        {
            int p = (v*nprocs)/vnodes;
            bool nosplit = (domains_c[v].size() < 2);

            // Add v arg to Vnode constructor 3/19/98 --tjw:
            Vnode<Dim> *vnode = new Vnode<Dim>(domains_c[v], p, v);
            typedef typename ac_id_vnodes::value_type v1;
            typedef typename ac_domain_vnodes::value_type v2;
            if ( p==myproc )
                Local_ac.insert(v1(Unique::get(),vnode));
            else
                remote_ac->insert(v2(domains_c[v], vnode), nosplit);
        }

    // Delete the memory we allocated.
    delete [] domains_c;
    delete [] copy_c;

    // Calculate the widths of all the vnodes
    calcWidths();
}

//----------------------------------------------------------------------
//
// Completely repartition this FieldLayout and all of the Fields
// defined on it.
//
template<unsigned Dim>
void
FieldLayout<Dim>::Repartition(const NDIndex<Dim>* idxBegin,
			      const NDIndex<Dim>* idxEnd)
{



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

    //INCIPPLSTAT(incRepartitions);
}


//----------------------------------------------------------------------
//
// Completely repartition this FieldLayout and all of the Fields
// defined on it.
// This differs from the previous ctor in that it allows preservation of
// global Vnode integer ID numbers associated with the input Vnodes. --tjw
//
template<unsigned Dim>
void
FieldLayout<Dim>::Repartition(const Vnode<Dim>* idxBegin,
			      const Vnode<Dim>* idxEnd)
{



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
}


//----------------------------------------------------------------------
//
// calculate the minimum vnode sizes in each dimension
template<unsigned Dim>
void FieldLayout<Dim>::calcWidths() {
    unsigned int d;

    // initialize widths first
    for (d=0; d < Dim; ++d)
        MinWidth[d] = getDomain()[d].length();

    // look for minimum width in local vnodes
    for  (const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i) {
        const NDIndex<Dim> &dom = (*v_i).second->getDomain();
        for (d=0; d < Dim; ++d) {
            if ((unsigned int) dom[d].length() < MinWidth[d])
                MinWidth[d] = dom[d].length();
        }
    }

    // look for minimum width in remove vnodes
    ac_domain_vnodes *v_ac = Remotes_ac[ gc0() ].get();
    for (iterator_dv dv_i = v_ac->begin(); dv_i != v_ac->end(); ++ dv_i) {
        const NDIndex<Dim> &dom = (*dv_i).first;
        for (d=0; d < Dim; ++d) {
            if ((unsigned int) dom[d].length() < MinWidth[d])
                MinWidth[d] = dom[d].length();
        }
    }
}


template<unsigned Dim>
NDIndex<Dim> FieldLayout<Dim>::getLocalNDIndex()
{

    NDIndex<Dim> theId;
    for (iterator_iv localv = begin_iv(); localv != end_iv(); ++localv) {
        Vnode<Dim> *vn = (*localv).second.get();
        if(vn->getNode() == Ippl::myNode())
            theId = vn->getDomain();
    }
    return theId;
}


//---------------------------------------------------------------------
// output
//---------------------------------------------------------------------
template<unsigned Dim>
void FieldLayout<Dim>::write(std::ostream& out) const
{


    int icount;

    // the whole domain, and the number of users
    out << "Domain = " << Domain << "\n";
    out << "FieldLayoutUsers = " << size_if() << "\n";

    // iterate over the local vnodes and print them out.
    out << "Total number of vnodes = " << numVnodes() << std::endl;
    out << "Local Vnodes = " << Local_ac.size() << "\n";
    icount = 0;
    //tjw can now do operator<<() with whole Vnode objects
    //tjw  for(const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i)
    //tjw    out << " vnode " << icount++ <<" : "<< (*v_i).second->getDomain() << "\n";
    for(const_iterator_iv v_i = begin_iv() ; v_i != end_iv(); ++v_i)
        out << " vnode " << icount++ << ": " << *((*v_i).second) << std::endl;

    // iterate over the remote vnodes and print them out.
    ac_domain_vnodes *v_ac = Remotes_ac[ gc0() ].get();
    out << "Remote Vnodes = " << v_ac->size() << "\n";
    icount = 0;
    for (iterator_dv dv_i = v_ac->begin(); dv_i != v_ac->end(); ++ dv_i)
        out << " vnode " << icount++ << " : " << *((*dv_i).second) << std::endl;
}