// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "FieldLayout/FieldLayout.h"
#include "FieldLayout/VRB.h"
#include "Message/Communicate.h"
#include "Message/Message.h"
#include "Utility/DiscMeta.h"
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

    // just initialize basic things
    vnodesPerDirection_m = 0;

    // for this kind of construction, we just take it that the user is
    // requesting all parallel axes
    for (unsigned int dl=0; dl < Dim; ++dl) RequestedLayout[dl] = PARALLEL;
}


//////////////////////////////////////////////////////////////////////
// Constructor which reads in FieldLayout data from a file.  If the
// file contains data for an equal number of nodes as we are running on,
// then that vnode -> pnode mapping will be used.  If the file does not
// contain info for the same number of pnodes, the vnodes will be
// distributed in some other manner.
template<unsigned Dim>
FieldLayout<Dim>::FieldLayout(const char *filename) {



    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // try to initialize ourselves, by reading the info from the file.
    vnodesPerDirection_m = 0;

    // for this kind of construction, we just take it that the user is
    // requesting all parallel axes
    for (unsigned int dl=0; dl < Dim; ++dl) RequestedLayout[dl] = PARALLEL;

    // read in data from file
    read(filename);
}


//////////////////////////////////////////////////////////////////////
// Destructor: Everything deletes itself automatically ... the base
// class destructors inform all the FieldLayoutUser's we're going away.
template<unsigned Dim>
FieldLayout<Dim>::~FieldLayout() {
    if (vnodesPerDirection_m != 0)
        delete [] vnodesPerDirection_m;
}


//////////////////////////////////////////////////////////////////////

// Initialization functions, only to be called by the user of FieldLayout
// objects when the FieldLayout was created using the default constructor;
// otherwise these are only called internally by the various non-default
// FieldLayout constructors:

//-----------------------------------------------------------------------------
// These specify only a total number of vnodes, allowing the constructor
// complete control on how to do the vnode partitioning of the index space:

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
			     e_dim_tag *p, int vnodes) {


    setup(domain, p, vnodes);
}


template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, e_dim_tag p1, int vnodes) {



    PInsist(Dim==1,
            "Number of arguments does not match dimension of FieldLayout!!");
    NDIndex<Dim> ndi(i1);
    setup(ndi,&p1,vnodes);
}

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2,
			     e_dim_tag p1, e_dim_tag p2, int vnodes) {



    PInsist(Dim==2,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    setup(ndi,par,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     int vnodes) {

    PInsist(Dim==3,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    setup(ndi,par,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4,
			     int vnodes) {


    PInsist(Dim==4,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    setup(ndi,par,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4, const Index& i5,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4, e_dim_tag p5,
			     int vnodes) {

    PInsist(Dim==5,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    par[4] = p5;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    ndi[4] = i5;
    setup(ndi,par,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4, const Index& i5, const Index& i6,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
			     int vnodes) {

    PInsist(Dim==6,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    par[4] = p5;
    par[5] = p6;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    ndi[4] = i5;
    ndi[5] = i6;
    setup(ndi,par,vnodes);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously
// this restricts the number of vnodes to be a product of the numbers along
// each dimension (the constructor implementation checks this):

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const NDIndex<Dim>& domain,
			     e_dim_tag *p, unsigned* vnodesPerDirection,
			     bool recurse, int vnodes) {


    // Default to correct total vnodes:
    unsigned vnodesProduct = 1;
    for (unsigned int d=0; d<Dim; d++) vnodesProduct *= vnodesPerDirection[d];
    if (vnodes == -1) vnodes = vnodesProduct;
    // Verify than total vnodes is product of per-dimension vnode counts:
    if ((unsigned int) vnodes != vnodesProduct) {
        ERRORMSG("FieldLayout constructor: "
                 << "(vnodes != vnodesPerDirection[0]*vnodesPerDirection[1]*"
                 << "...*vnodesPerDirection[" << Dim-1 << "])"
                 << " ; vnodesPerDirection[0]*vnodesPerDirection[1]*"
                 << "...*vnodesPerDirection[" << Dim-1 << "] = "
                 << vnodesProduct << " ; vnodes = " << vnodes << endl);
    }
    setup(domain, p, vnodesPerDirection,recurse,vnodes);
}

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, e_dim_tag p1,
			     unsigned vnodes1, bool recurse, int vnodes) {



    PInsist(Dim==1,
            "Number of arguments does not match dimension of FieldLayout!!");
    NDIndex<Dim> ndi(i1);
    setup(ndi,&p1,&vnodes1,recurse,vnodes);
}

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2,
			     e_dim_tag p1, e_dim_tag p2,
			     unsigned vnodes1, unsigned vnodes2,
			     bool recurse, int vnodes) {

    PInsist(Dim==2,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    unsigned vnodesPerDirection[Dim];
    vnodesPerDirection[0] = vnodes1;
    vnodesPerDirection[1] = vnodes2;
    setup(ndi,par,vnodesPerDirection,recurse,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     unsigned vnodes1, unsigned vnodes2,
			     unsigned vnodes3,
			     bool recurse, int vnodes) {

    PInsist(Dim==3,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    unsigned vnodesPerDirection[Dim];
    vnodesPerDirection[0] = vnodes1;
    vnodesPerDirection[1] = vnodes2;
    vnodesPerDirection[2] = vnodes3;
    setup(ndi,par,vnodesPerDirection,recurse,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4,
			     unsigned vnodes1, unsigned vnodes2,
			     unsigned vnodes3, unsigned vnodes4,
			     bool recurse, int vnodes) {

    PInsist(Dim==4,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    unsigned vnodesPerDirection[Dim];
    vnodesPerDirection[0] = vnodes1;
    vnodesPerDirection[1] = vnodes2;
    vnodesPerDirection[2] = vnodes3;
    vnodesPerDirection[3] = vnodes4;
    setup(ndi,par,vnodesPerDirection,recurse,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4, const Index& i5,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4, e_dim_tag p5,
			     unsigned vnodes1, unsigned vnodes2,
			     unsigned vnodes3, unsigned vnodes4,
			     unsigned vnodes5,
			     bool recurse, int vnodes) {

    PInsist(Dim==5,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    par[4] = p5;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    ndi[4] = i5;
    unsigned vnodesPerDirection[Dim];
    vnodesPerDirection[0] = vnodes1;
    vnodesPerDirection[1] = vnodes2;
    vnodesPerDirection[2] = vnodes3;
    vnodesPerDirection[3] = vnodes4;
    vnodesPerDirection[4] = vnodes5;
    setup(ndi,par,vnodesPerDirection,recurse,vnodes);
}
template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const Index& i1, const Index& i2, const Index& i3,
			     const Index& i4, const Index& i5, const Index& i6,
			     e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			     e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
			     unsigned vnodes1, unsigned vnodes2,
			     unsigned vnodes3, unsigned vnodes4,
			     unsigned vnodes5, unsigned vnodes6,
			     bool recurse, int vnodes) {

    PInsist(Dim==6,
            "Number of arguments does not match dimension of FieldLayout!!");
    e_dim_tag par[Dim];
    par[0] = p1;
    par[1] = p2;
    par[2] = p3;
    par[3] = p4;
    par[4] = p5;
    par[5] = p6;
    NDIndex<Dim> ndi;
    ndi[0] = i1;
    ndi[1] = i2;
    ndi[2] = i3;
    ndi[3] = i4;
    ndi[4] = i5;
    ndi[5] = i6;
    unsigned vnodesPerDirection[Dim];
    vnodesPerDirection[0] = vnodes1;
    vnodesPerDirection[1] = vnodes2;
    vnodesPerDirection[2] = vnodes3;
    vnodesPerDirection[3] = vnodes4;
    vnodesPerDirection[4] = vnodes5;
    vnodesPerDirection[5] = vnodes6;
    setup(ndi,par,vnodesPerDirection,recurse,vnodes);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// A version of initialize that takes the total domain, and iterators
// over the subdomains the user wants along with iterators over the
// node assignments.  No communication is done
// so these lists must match on all nodes.  A bit of error checking
// is done for overlapping blocks and illegal nodes, but not exhaustive
// error checking.

template<unsigned Dim>
void
FieldLayout<Dim>::initialize(const NDIndex<Dim> &domain,
			     const NDIndex<Dim> *dombegin,
			     const NDIndex<Dim> *domend,
			     const int *nbegin, const int *nend)
{

    // Loop variables
    int i, j;
    unsigned int d;

    // Save the total domain.
    Domain = domain;

    // Find the number of vnodes requested
    int vnodes = (nend - nbegin);
    PInsist(vnodes > 0,
            "A user-specified FieldLayout must have at least one vnode.");
    PInsist(vnodes == (domend - dombegin),
            "A user-specified FieldLayout must have equal length node and domain lists");

    // Since we don't know any differently, indicate the requested
    // layout is all parallel
    for (d = 0; d < Dim; ++d)
        RequestedLayout[d] = PARALLEL;

    // This is not a grid-like layout, so set the pointer for this to 0
    vnodesPerDirection_m = 0;

    // Create the empty remote vnode list
    ac_domain_vnodes *remote_ac = new ac_domain_vnodes(domain);
    typedef typename ac_gc_domain_vnodes::value_type vntype;
    Remotes_ac.insert( vntype(gc0(), remote_ac) );

    // Loop through the vnodes, and add them to our local or remote lists.
    // Do a sanity check on the vnodes, making sure each one does not
    // intersect any other.  Also, add up the size of each vnode, if it
    // does not equal the size of the total domain, there are holes
    // and it is an error.
    size_t coverage = 0;
    for (i = 0; i < vnodes; ++i) {
        // Compare to other vnodes
        for (j = (i+1); j < vnodes; ++j) {
            PInsist(! (dombegin[i].touches(dombegin[j])),
                    "A user-specified FieldLayout cannot have overlapping domains.");
        }

        // Make sure the processor ID is OK
        PInsist(nbegin[i] >= 0 && nbegin[i] < Ippl::getNodes(),
                "A user-specified FieldLayout must have legal node assignments.");

        // Add in the volume of this domain
        coverage += dombegin[i].size();

        // Create a Vnode for this domain
        Vnode<Dim> *vnode = new Vnode<Dim>(dombegin[i], nbegin[i], i);
        typedef typename ac_id_vnodes::value_type v1;
        typedef typename ac_domain_vnodes::value_type v2;
        bool nosplit = (dombegin[i].size() < 2);

        // Based on the assigned node, add to our local or remote lists
        if (nbegin[i] == Ippl::myNode())
            Local_ac.insert(v1(Unique::get(), vnode));
        else
            remote_ac->insert(v2(dombegin[i], vnode), nosplit);
    }

    // Check the coverage, make sure it is complete
    PInsist(coverage == domain.size(),
            "A user-specified FieldLayout must completely cover the domain.");

    // At the end, calculate the widthds of all the vnodes
    calcWidths();
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
			e_dim_tag *userflags, int vnodes)
{


    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // Find the number processors.
    int nprocs = Ippl::getNodes();
    int myproc = Ippl::myNode();

    // If the user didn't specify the number of vnodes, make it equal nprocs
    if (vnodes <= 0) vnodes = nprocs;

    Inform dbgmsg("FieldLayout::setup", INFORM_ALL_NODES);
    // dbgmsg << "*** Domain=" << domain << ", nprocs=" << nprocs;
    // dbgmsg << ", myproc=" << myproc << ", vnodes=" << vnodes << endl;

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
    // Make sure at least one of the parallel/serial flags is parallel
    PInsist(parallel_count>0,"At least one dimension of a FieldLayout must be PARALLEL!");

    // Check to see if we have too few elements to partition.  If so, reduced
    // the number of vnodes (if necessary) to just the number of elements along
    // parallel dims.
    if (totparelems < vnodes) {
        //dbgmsg << "Total parallel lengths = " << totparelems << "; reducing ";
        //dbgmsg << "vnodes from " << vnodes << " to " << totparelems << endl;
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
    vnodesPerDirection_m = 0;
    for (v=vnodes,rm=0;v>1;v/=2) { rm += (v % 2); }
    if (rm == 0) {

        // vnodes is a power of 2

        // For power-of-two vnodes, allocate storage for vnodesPerDirection_m:
        //don't--tjw    vnodesPerDirection_m = new unsigned[Dim];
        // Fill in 1 for starters; remains 1 for serial directions:
        //don't--tjw    for (int d2=0; d2<Dim; d2++) vnodesPerDirection_m[d2] = 1;

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

            //don't--tjw      vnodesPerDirection_m[d] *= 2; // Update to reflect split

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

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// This setup() specifies both the total number of vnodes and the numbers of
// vnodes along each dimension for the partitioning of the index space, where
// the user wants this extra control over the partitioning of the index
// space. Obviously this restricts the number of vnodes to be a product of the
// numbers along each dimension (the constructor implementation checks this).
//
// The last argument is a bool for the algorithm to use for assigning
// vnodes to processors.
// If it is false, hand the vnodes to the processors in a very simple
// but probably inefficient manner.
// If it is true, use a binary recursive algorithm. This will usually be
// more efficient because it will generate less communication, but
// it will sometimes fail, particularly near the case of one vnode per
// processor.
//
//-----------------------------------------------------------------------------

template<unsigned Dim>
void
FieldLayout<Dim>::setup(const NDIndex<Dim>& domain,
			e_dim_tag *userflags, unsigned* vnodesPerDirection,
			bool recurse, int vnodes)
{

    // Find the number processors.
    int nprocs = Ippl::getNodes();
    int myproc = Ippl::myNode();

    // The number of vnodes must have been specified or computed by now:
    if (vnodes <= 0) ERRORMSG("FieldLayout::setup(): vnodes <= 0 "
                              << "for a vnodes-per-direction product "
                              << "specification; not allowed." << endl);

    // Loop indices:
    int v;
    unsigned int d, d2, vl;

    int parallel_count = 0;
    unsigned int flagdim = 0;
    for (flagdim=0; flagdim < Dim; ++flagdim) {
        if (userflags == 0)
            RequestedLayout[flagdim] = PARALLEL;
        else
            RequestedLayout[flagdim] = userflags[flagdim];

        // keep track of the number of parallel dimensions; we need at least one
        parallel_count += (RequestedLayout[flagdim] == PARALLEL);

        // make sure any SERIAL dimensions request only one vnode along them:
        if (RequestedLayout[flagdim] == SERIAL) {
            bool chk = vnodesPerDirection[flagdim] == 1;
            PInsist(chk,"SERIAL layout specified, yet vnodesPerDirection is not 1!");
        }
    }

    // Make sure at least one of the parallel/serial flags is parallel
    PInsist(parallel_count>0,"At least one dimension of a FieldLayout must be PARALLEL!");

    // Allocate and store vnodesPerDirection_m data-member array:
    vnodesPerDirection_m = new unsigned[Dim];
    for (d=0; d<Dim; d++) vnodesPerDirection_m[d] = vnodesPerDirection[d];

    // The domain of this FieldLayout object under construction:
    Domain = domain;

    // Set up a container of NDIndex's to store the index ranges for all vnodes:
    NDIndex<Dim> *domains_c = new NDIndex<Dim>[vnodes];

    // Divide the numbers of elements by the numbers of vnodes (along each
    // dimension). All vnodes will have at least this many elements. Some will
    // have one extra element:
    unsigned elementsPerVnode[Dim];
    unsigned nLargerVnodes[Dim];
    for (d=0; d<Dim; d++) {
        elementsPerVnode[d] = domain[d].length()/vnodesPerDirection[d];
        nLargerVnodes[d] = domain[d].length() % vnodesPerDirection[d];
    }

    // Set up the base, bound, and stride for the index range of all
    // vnodes. Organize by "vnode level." For this kind of partitioning we have a
    // the equivalent of a Dim-dimensional array of vnodes, with the number of
    // elements in dimension d being vnodesPerDirection[d]. By "vnode level" we
    // mean the index along a dimension in the "vnode array" of a given vnode. We
    // can store the vnode-index bases and bounds 2D arrays of arrays; we only
    // need a 1D array to store the vnode strides because they are the same
    // everywhere (and the same as the global strides for each dimension):
    int stride[Dim];
    for (d=0; d<Dim; d++) stride[d] = domain[d].stride();
    int* base[Dim];
    int* bound[Dim];
    for (d=0; d<Dim; d++) {
        base[d] = new int[vnodesPerDirection[d]];
        bound[d] = new int[vnodesPerDirection[d]];
    }
    int length;
    for (d=0; d<Dim; d++) {
        // Start things off for the zeroth vnode level using the global index base:
        length = elementsPerVnode[d];
        if (nLargerVnodes[d] > 0) length += 1;
        base[d][0] = domain[d].first();
        bound[d][0] = base[d][0] + (length-1)*stride[d];
        // Now go through all the other vnode levels
        for (vl=1; vl < vnodesPerDirection[d]; vl++) {
            base[d][vl] = bound[d][vl-1] + stride[d];
            length = elementsPerVnode[d];
            if (vl < nLargerVnodes[d]) length += 1;
            bound[d][vl] = base[d][vl] +(length-1)*stride[d];
        }
    }

    // Now actually initialize the values in the container of NDIndex's which
    // represent each vnode's subdomain:
    for (v=0; v<vnodes; v++) {
        for (d=0; d<Dim; d++) {
            // Compute this vnode's level in this direction:
            unsigned denom = 1;
            for (d2=0; d2<d; d2++) denom *= vnodesPerDirection[d2];
            int vnodeLevel = (v/denom) % vnodesPerDirection[d];
            // Now use the precomputed base, bound values to set the Index values:
            domains_c[v][d] = Index(base[d][vnodeLevel],bound[d][vnodeLevel],
                                    stride[d]);
            denom = 1; // for debugging
        }
    }

    // Now find what processor each vnode will end up on.
    // This is done with a recursive bisection algorithm.
    // This produces fairly squarish blocks -- and therefore
    // less communication at the expense of each processor getting
    // a less balanced load.
    int *vnodeProcs = new int[vnodes];
    int *sizes = new int[Dim];
    for (v=0; (unsigned int) v<Dim; ++v)
        sizes[v] = vnodesPerDirection[v];

    // If we have been instructed to use recursive bisection, do that.
    // if not, deal them out in a simple manner.
    if ( recurse )
        vnodeRecursiveBisection(Dim,sizes,nprocs,vnodeProcs);
    else
        for ( v=0; v<vnodes; ++v )
            vnodeProcs[v] = (v*nprocs)/vnodes;


    // Now make the vnodes, using the domains just generated.
    // Some of them we store in the local list, others in the remote.
    ac_domain_vnodes *remote_ac = new ac_domain_vnodes( domain );
    typedef typename ac_gc_domain_vnodes::value_type v1;
    Remotes_ac.insert(v1(gc0(),remote_ac) );
    for (v=0; v<vnodes; ++v) {
        int p = vnodeProcs[v];
        // Add v arg to Vnode constructor 3/19/98 --tjw:
        Vnode<Dim> *vnode = new Vnode<Dim>(domains_c[v], p, v);
        typedef typename ac_id_vnodes::value_type v2;
        if ( p==myproc )
            Local_ac.insert(v2(Unique::get(),vnode) );
        else
            // For domains of extent 1 in all directions, must call
            // DomainMap::insert() with the noSplit flag set to true, to avoid an
            // assertion failure in Index::split. Also do this for domains of
            // extent ZERO. These are not supposed to happen in IPPL, but when the
            // do when something like BinaryRepartition creates them, accomodating
            // them here will prevent subsequent code hangs or other errors and allow
            // recovery such as rejection of the result of BinaryRepartition (discard
            // the FieldLayout with zero-size domains and go on with the original
            // layout unchanged). (tjw)
            //      if (domains_c[v].size() == 1) {
            if (domains_c[v].size() <= 1) {
                typedef typename ac_domain_vnodes::value_type v1;
                remote_ac->insert(v1(domains_c[v], vnode), true);
            } else {
                typedef typename ac_domain_vnodes::value_type v1;
                remote_ac->insert(v1(domains_c[v], vnode), false);
            }
    }

    // Delete the memory we allocated.
    delete [] domains_c;
    delete [] vnodeProcs;
    delete [] sizes;
    for (d=0; d<Dim; d++) {
        delete [] base[d];
        delete [] bound[d];
    }

    // Calculate the widths of all the vnodes
    calcWidths();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Return number of vnodes along a direction.
template<unsigned Dim>
unsigned FieldLayout<Dim>::
getVnodesPerDirection(unsigned dir) {

    // If not set, then not appropriate to be calling this function. Example:
    // now-power-of-two-constructed FieldLayout which did *not* specify numbers
    // of vnodes along each direction:
    PAssert(vnodesPerDirection_m);

    return(vnodesPerDirection_m[dir]);
}
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------

template<unsigned Dim>
void FieldLayout<Dim>::new_gc_layout(const GuardCellSizes<Dim>& gc)
{



    // Build the guarded domain.
    NDIndex<Dim> guarded_domain( AddGuardCells(Domain,gc) );
    // Build a container for vnodes in that domain.
    ac_domain_vnodes *gr = new ac_domain_vnodes(guarded_domain);
    // Record pointer to that container using gc as the key.
    typedef typename ac_gc_domain_vnodes::value_type v1;
    Remotes_ac.insert(v1(gc,gr) );
    // Get the container of vnodes stored w/o guard cells.
    ac_domain_vnodes &v0 = *Remotes_ac[ gc0() ];
    // Loop over all the remote vnodes.
    for (iterator_dv v_i = v0.begin(); v_i != v0.end(); ++v_i)
        {
            // Build the domain for this vnode with gc guard cells.
            NDIndex<Dim> domain(AddGuardCells((*v_i).first,gc));
            // Record pointer to this vnode with these guard cells.
            typedef typename ac_domain_vnodes::value_type v2;
            gr->insert(v2(domain,(*v_i).second) );
        }
}

//----------------------------------------------------------------------
template<unsigned Dim>
FieldLayout<Dim>::FieldLayout(const NDIndex<Dim>& domain,
			      const NDIndex<Dim>* idx_begin,
			      const NDIndex<Dim>* idx_end)
    : Domain(domain)
{

    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // for this kind of construction, we just take it that the user is
    // requesting all parallel axes
    for (unsigned int dl=0; dl < Dim; ++dl) RequestedLayout[dl] = PARALLEL;

    // Build Vnodes for each of the local domains.
    vnodesPerDirection_m = 0;
    int mynode = Ippl::Comm->myNode();
    for (const NDIndex<Dim>* p=idx_begin; p!=idx_end; ++p)
        {
            typedef typename ac_id_vnodes::value_type v1;
            Local_ac.insert(v1(Unique::get(), new Vnode<Dim>(*p,mynode)));
        }

    // Everybody broadcasts their new local domains to everybody.
    // Build a message with the local domains and the id's for them.
    Message* bcast_mess = new Message;
    int count = idx_end - idx_begin;
    bcast_mess->put(count);
    // ::putMessage(*bcast_mess,idx_begin,idx_end);
    const NDIndex<Dim>* indxmsg = idx_begin;
    while (indxmsg != idx_end)
        (indxmsg++)->putMessage(*bcast_mess);
    // Send it to everybody except yourself, and record the number sent.
    int tag = Ippl::Comm->next_tag( F_REPARTITION_BCAST_TAG , F_TAG_CYCLE );
    int node_count = Ippl::Comm->broadcast_others(bcast_mess,tag);
    // Create the container for the remote vnodes.
    ac_domain_vnodes *remote_ac = new ac_domain_vnodes( Domain );
    typedef typename ac_gc_domain_vnodes::value_type v1;
    Remotes_ac.insert(v1(gc0(),remote_ac) );

    // Loop until we receive a message from each other node.
    while ((--node_count)>=0)
        {
            // Receive a broadcast message from any node.
            int other_node = COMM_ANY_NODE;
            Message *recv_mess = Ippl::Comm->receive_block(other_node,tag);
            PAssert(recv_mess);
            // Extract the number of vnodes coming in.
            int count = 0;
            recv_mess->get(count);
            // Now get the domains for the vnodes from the message.
            NDIndex<Dim> p;
            while (count-- > 0) {
                p.getMessage(*recv_mess);
                Vnode<Dim> *vnode = new Vnode<Dim>(p, other_node);
                // For domains of extent 1 in all directions, must call
                // DomainMap::insert() with the noSplit flag set to true, to avoid an
                // assertion failure in Index::split. Also do this for domains of
                // extent ZERO. These are not supposed to happen in IPPL, but when the
                // do when something like BinaryRepartition creates them, accomodating
                // them here will prevent subsequent code hangs or other errors and
                // allow recovery such as rejection of the result of BinaryRepartition
                // (discard the FieldLayout with zero-size domains and go on with the
                // original layout unchanged). (tjw)
                //	if (p.size() == 1) {
                if (p.size() <= 1) {
                    typedef typename ac_domain_vnodes::value_type v1;
                    remote_ac->insert(v1(p, vnode), true);
                } else {
                    typedef typename ac_domain_vnodes::value_type v1;
                    remote_ac->insert(v1(p, vnode), false);
                }
            }
            delete recv_mess;
        }

    // Calculate the widths of all the vnodes
    calcWidths();
}


//----------------------------------------------------------------------
// This differs from the previous ctor in that it allows preservation of
// global Vnode integer ID numbers associated with the input Vnodes. --tjw
template<unsigned Dim>
FieldLayout<Dim>::FieldLayout(const NDIndex<Dim>& domain,
			      const Vnode<Dim>* idx_begin,
			      const Vnode<Dim>* idx_end)
    : Domain(domain)
{

    // we have one more FieldLayout, indicate this
    //INCIPPLSTAT(incFieldLayouts);

    // for this kind of construction, we just take it that the user is
    // requesting all parallel axes
    for (unsigned int dl=0; dl < Dim; ++dl) RequestedLayout[dl] = PARALLEL;

    // Build Vnodes for each of the local domains.
    vnodesPerDirection_m = 0;
    int mynode = Ippl::Comm->myNode();
    for (const Vnode<Dim>* p=idx_begin; p!=idx_end; ++p) {
        typedef typename ac_id_vnodes::value_type v1;
        Local_ac.insert(v1(Unique::get(),
                           new Vnode<Dim>((*p).getDomain(),mynode,(*p).getVnode())));
    }

    // Everybody broadcasts their new local domains to everybody.
    // Build a message with the local domains and the id's for them.
    Message* bcast_mess = new Message;
    int count = idx_end - idx_begin;
    bcast_mess->put(count);
    // ::putMessage(*bcast_mess,idx_begin,idx_end);
    const Vnode<Dim>* indxmsg = idx_begin;
    while (indxmsg != idx_end)
        (indxmsg++)->putMessage(*bcast_mess);
    // Send it to everybody except yourself, and record the number sent.
    int tag = Ippl::Comm->next_tag( F_REPARTITION_BCAST_TAG , F_TAG_CYCLE );
    int node_count = Ippl::Comm->broadcast_others(bcast_mess,tag);
    // Create the container for the remote vnodes.
    ac_domain_vnodes *remote_ac = new ac_domain_vnodes( Domain );
    typedef typename ac_gc_domain_vnodes::value_type v1;
    Remotes_ac.insert(v1(gc0(),remote_ac) );

    // Loop until we receive a message from each other node.
    while ((--node_count)>=0)
        {
            // Receive a broadcast message from any node.
            int other_node = COMM_ANY_NODE;
            Message *recv_mess = Ippl::Comm->receive_block(other_node,tag);
            PAssert(recv_mess);
            // Extract the number of vnodes coming in.
            int count;
            recv_mess->get(count);
            // Now get the domains for the vnodes from the message.
            Vnode<Dim> p;
            while (count-- > 0) {
                p.getMessage(*recv_mess);
                Vnode<Dim> *vnode =
                    new Vnode<Dim>(p.getDomain(), other_node, p.getVnode());
                // For domains of extent 1 in all directions, must call
                // DomainMap::insert() with the noSplit flag set to true, to avoid an
                // assertion failure in Index::split. Also do this for domains of
                // extent ZERO. These are not supposed to happen in IPPL, but when the
                // do when something like BinaryRepartition creates them, accomodating
                // them here will prevent subsequent code hangs or other errors and
                // allow recovery such as rejection of the result of BinaryRepartition
                // (discard the FieldLayout with zero-size domains and go on with the
                // original layout unchanged). (tjw)
                //	if (p.getDomain().size() == 1) {
                if (p.getDomain().size() <= 1) {
                    typedef typename ac_domain_vnodes::value_type v1;
                    remote_ac->insert(v1(p.getDomain(), vnode), true);
                } else {
                    typedef typename ac_domain_vnodes::value_type v1;
                    remote_ac->insert(v1(p.getDomain(), vnode), false);
                }
            }
        }

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

    // (TJW) Copy the vnodesPerDirection information:
    if (vnodesPerDirection_m != 0) {
        tempLayout.vnodesPerDirection_m = new unsigned[Dim];
        for (unsigned int d=0; d<Dim; d++) {
            tempLayout.vnodesPerDirection_m[d] = vnodesPerDirection_m[d];
        }
    }

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

    // (TJW) Copy the vnodesPerDirection information:
    if (vnodesPerDirection_m != 0) {
        tempLayout.vnodesPerDirection_m = new unsigned[Dim];
        for (unsigned int d=0; d<Dim; d++) {
            tempLayout.vnodesPerDirection_m[d] = vnodesPerDirection_m[d];
        }
    }

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
// Write out a FieldLayout data to a file.  The file is in ascii format,
// with keyword=value.  Return success.
template<unsigned Dim>
bool FieldLayout<Dim>::write(const char *filename) {



    unsigned int d;

    // only do the read on node 0
    if (Ippl::myNode() == 0) {
        // create the file, make sure the creation is OK
        FILE *f = fopen(filename, "w");
        if (f == 0) {
            ERRORMSG("Could not create FieldLayout data file '" << filename);
            ERRORMSG("'." << endl);
            return false;
        }

        // write out FieldLayout information
        fprintf(f, "dim    = %d\n", Dim);
        fprintf(f, "vnodes = %d\n", numVnodes());
        fprintf(f, "pnodes = %d\n", Ippl::getNodes());

        fprintf(f, "domain =");
        for (d=0; d < Dim; ++d)
            fprintf(f, "  %d %d %d",
                    Domain[d].first(), Domain[d].last(), Domain[d].stride());
        fprintf(f, "\n");

        if (vnodesPerDirection_m != 0) {
            fprintf(f, "vnodesperdir =");
            for (d=0; d < Dim; ++d)
                fprintf(f, " %d", vnodesPerDirection_m[d]);
            fprintf(f, "\n");
        }

        for (iterator_iv localv = begin_iv(); localv != end_iv(); ++localv) {
            Vnode<Dim> *vn = (*localv).second.get();
            fprintf(f, "block  = %d %d", vn->getVnode(), vn->getNode());
            for (d=0; d < Dim; ++d)
                fprintf(f, "  %d %d %d", vn->getDomain()[d].first(),
                        vn->getDomain()[d].last(), vn->getDomain()[d].stride());
            fprintf(f, "\n");
        }

        for (iterator_dv remotev = begin_rdv(); remotev != end_rdv(); ++remotev) {
            Vnode<Dim> *vn = (*remotev).second;
            fprintf(f, "block  = %d %d", vn->getVnode(), vn->getNode());
            for (d=0; d < Dim; ++d)
                fprintf(f, "  %d %d %d", vn->getDomain()[d].first(),
                        vn->getDomain()[d].last(), vn->getDomain()[d].stride());
            fprintf(f, "\n");
        }

        fclose(f);
    }

    return true;
}


//----------------------------------------------------------------------
//
// Read in FieldLayout data from a file.  If the
// file contains data for an equal number of nodes as we are running on,
// then that vnode -> pnode mapping will be used.  If the file does not
// contain info for the same number of pnodes, the vnodes will be
// distributed in some other manner.
// Note that if this FieldLayout is initially empty (e.g., it has no
// local or remote vnodes), the domain will be changed to that in the
// file.  But if we already have some vnodes, we can only repartition
// if the domain matches that in the file, or if we do not have any
// users.  If an error occurs, return
// false and leave our own layout unchanged.
template<unsigned Dim>
bool FieldLayout<Dim>::read(const char *filename) {



    // generate a tag to use for communication
    int tag = Ippl::Comm->next_tag(F_LAYOUT_IO_TAG, F_TAG_CYCLE);

    // storage for data read from file
    NDIndex<Dim> fdomain;
    Vnode<Dim> *vnlist = 0;
    unsigned *vnodesPerDir = 0;
    int fdim = 0;
    int fvnodes = 0;
    int fpnodes = 0;
    int vnodesread = 0;
    int ok = 1;

    // only do the read on node 0
    if (Ippl::myNode() == 0) {
        // read the file, make sure the read is OK
        DiscMeta f(filename);

        // make sure it is OK
        DiscMeta::iterator metaline = f.begin();
        for ( ; ok == 1 && metaline != f.end(); ++metaline) {
            // get number of tokens and list of tokens in the line
            int numtokens  = (*metaline).second.first;
            std::string *tokens = (*metaline).second.second;

            // check first word
            if (tokens[0] == "dim") {
                if (fdim != 0) {
                    ERRORMSG("Repeated 'dim' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                fdim = atoi(tokens[1].c_str());
                if (fdim != Dim) {
                    ERRORMSG("Mismatched dimension in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
            } else if (tokens[0] == "vnodes") {
                if (fvnodes != 0 || vnlist != 0) {
                    ERRORMSG("Repeated 'vnodes' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                fvnodes = atoi(tokens[1].c_str());
                if (fvnodes < 1) {
                    ERRORMSG("Incorrect 'vnodes' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                } else {
                    vnlist = new Vnode<Dim>[fvnodes];
                }
            } else if (tokens[0] == "pnodes") {
                if (fpnodes != 0) {
                    ERRORMSG("Repeated 'pnodes' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                fpnodes = atoi(tokens[1].c_str());
                if (fpnodes < 1) {
                    ERRORMSG("Incorrect 'pnodes' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
            } else if (tokens[0] == "vnodesperdir") {
                if (vnodesPerDir != 0) {
                    ERRORMSG("Repeated 'vnodesperdir' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                if (numtokens - 1 != Dim) {
                    ERRORMSG("Incorrect 'vnodesperdir' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                vnodesPerDir = new unsigned[Dim];
                for (unsigned int d=0; d < Dim; ++d) {
                    vnodesPerDir[d] = atoi(tokens[d+1].c_str());
                    if (vnodesPerDir[d] < 1) {
                        ERRORMSG("Illegal vnode per direction value for dim=" << d);
                        ERRORMSG(" in FieldLayout data file '" << filename << "'."<<endl);
                        ok = 0;
                    }
                }
            } else if (tokens[0] == "domain") {
                // make sure we have (first,last,stride) for all dims
                if ((numtokens-1) % 3 != 0 || (numtokens-1) / 3 != Dim) {
                    ERRORMSG("Incorrect 'domain' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                }
                for (unsigned int d=0, dindx=1; d < Dim; ++d) {
                    fdomain[d] = Index(atoi(tokens[dindx].c_str()),
                                       atoi(tokens[dindx + 1].c_str()),
                                       atoi(tokens[dindx + 2].c_str()));
                    dindx += 3;
                }
            } else if (tokens[0] == "block") {
                // this is vnode information; it should be of the form
                // vnode# pnode# (domain, with 3 numbers/dimension)
                // So, the number of tokens should be a multipple of three
                if (numtokens % 3 != 0 || numtokens / 3 != (Dim+1)) {
                    ERRORMSG("Incorrect 'block' line in FieldLayout data file '");
                    ERRORMSG(filename << "'." << endl);
                    ok = 0;
                } else if (vnlist == 0) {
                    ERRORMSG("In FieldLayout data file '" << filename << ": You must ");
                    ERRORMSG("give the number vnodes before any block lines." << endl);
                    ok = 0;
                } else {
                    int vnum = atoi(tokens[1].c_str());
                    int pnum = atoi(tokens[2].c_str());
                    if (pnum < 0 || pnum >= fpnodes) {
                        ERRORMSG("Illegal pnode number in 'block' line ");
                        ERRORMSG("of file '" << filename << "'." << endl);
                        ok = 0;
                    }
                    NDIndex<Dim> vdom;
                    for (unsigned int d=0, dindx=3; d < Dim; ++d) {
                        vdom[d] = Index(atoi(tokens[dindx].c_str()),
                                        atoi(tokens[dindx + 1].c_str()),
                                        atoi(tokens[dindx + 2].c_str()));
                        dindx += 3;
                    }

                    // construct a new vnode from this info, and store it at the
                    // index given by the vnum value
                    vnlist[vnodesread++] = Vnode<Dim>(vdom, pnum, vnum);
                }
            } else {
                ERRORMSG("Unrecognized '" << tokens[0] << "' line in FieldLayout ");
                ERRORMSG(" data file '" << filename << "'." << endl);
                ok = 0;
            }
        }

        // do some final sanity checks
        if (ok != 0 && (fvnodes < 1 || fdim != Dim || vnodesread != fvnodes)) {
            ERRORMSG("Inconsistent FieldLayout data in file '" << filename);
            ERRORMSG("'." << endl);
            ok = 0;
        }

        // if we have users, we cannot change the global domain any
        if ( ok != 0 && getNumUsers() > 0 && !(fdomain == Domain) ) {
            ERRORMSG("You cannot change the global domain of a FieldLayout which");
            ERRORMSG(" has users." << endl);
            ok = 0;
        }

        // Send out new layout info to other nodes
        if (Ippl::getNodes() > 1) {
            Message *msg = new Message;
            msg->put(ok);
            if (ok != 0) {
                msg->put(fvnodes);
                msg->put(fpnodes);
                msg->put(fdomain);
                int vnpdok = (vnodesPerDir != 0 ? 1 : 0);
                msg->put(vnpdok);
                if (vnpdok != 0)
                    msg->put(vnodesPerDir, vnodesPerDir + Dim);
                for (int v=0; v < fvnodes; ++v) {
                    msg->put(vnlist[v]);
                }
            }

            Ippl::Comm->broadcast_others(msg, tag);
        }
    } else {
        // on the client nodes, get the info from node 0 and store it
        // first receive the message
        int node = 0;
        Message *msg = Ippl::Comm->receive_block(node, tag);
        PAssert(msg);

        // then get data out of the message
        msg->get(ok);
        if (ok != 0) {
            msg->get(fvnodes);
            msg->get(fpnodes);
            msg->get(fdomain);
            int vnpdok;
            msg->get(vnpdok);
            if (vnpdok != 0) {
                vnodesPerDir = new unsigned[Dim];
                msg->get_iter(vnodesPerDir);
            }
            if ((vnodesread = fvnodes) > 0) {
                vnlist = new Vnode<Dim>[fvnodes];
                for (int v=0; v < fvnodes; ++v) {
                    Vnode<Dim> vn;
                    msg->get(vn);
                    vnlist[v] = vn;
                }
            }
        }

        // done with the message, we can delete it
        delete msg;
    }

    // now, on all nodes, figure out which vnodes are ours, and which are
    // local.  But if an error occurred during reading, we just exit
    // without changing anything.
    if (ok != 1 || fvnodes < 1) {
        if (vnodesPerDir != 0)
            delete [] vnodesPerDir;
        if (vnlist != 0)
            delete [] vnlist;
        return false;
    }

    // save the new domain
    Domain = fdomain;

    // for each vnode, figure out which physical node it belongs on, and
    // sort those vnodes to the top of the list.
    int localvnodes = 0;
    for (int v=0; v < fvnodes; ++v) {
        int node = vnlist[v].getNode() % Ippl::getNodes();
        if (node == Ippl::myNode()) {
            if (v > localvnodes) {
                // move this vnode to the top
                Vnode<Dim> tempv(vnlist[localvnodes]);
                vnlist[localvnodes] = vnlist[v];
                vnlist[v] = tempv;
            }
            localvnodes++;
        }
    }

    // save the info on how many vnodes we have per direction
    if (vnodesPerDirection_m != 0)
        delete [] vnodesPerDirection_m;
    vnodesPerDirection_m = vnodesPerDir;

    // Repartition the system using our list of local vnodes
    Repartition(vnlist, vnlist + localvnodes);

    // success!
    delete [] vnlist;
    return true;
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

    //Inform dbgmsg("calcWidths", INFORM_ALL_NODES);
    //dbgmsg << "Calculated minimum widths for layout " << *this << endl;
    //for (d=0; d < Dim; ++d) {
    //  dbgmsg << " ==> MinWidth[" << d << "] = " << MinWidth[d];
    //  dbgmsg << " ... " << (getDistribution(d)==SERIAL?"serial":"parallel");
    //  dbgmsg << endl;
    //}
}


//----------------------------------------------------------------------
//
// Tell the FieldLayout that a FieldLayoutUser is using it
template<unsigned Dim>
void
FieldLayout<Dim>::checkin(FieldLayoutUser& f,
			  const GuardCellSizes<Dim>& gc)
{



    checkinUser(f);
    iterator_gdv guarded = Remotes_ac.find(gc);
    if ( guarded == Remotes_ac.end() )
        new_gc_layout(gc);
}

//----------------------------------------------------------------------
//
// Tell the FieldLayout that a Field is no longer using it.
template<unsigned Dim>
void
FieldLayout<Dim>::checkout(FieldLayoutUser& f)
{


    checkoutUser(f);
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

    // If applicable, vnodes per direction (tjw):
    if (vnodesPerDirection_m != 0) {
        out << "vnodesPerDirection_m[] =";
        for (unsigned int d=0; d<Dim; d++)
            out << " " << vnodesPerDirection_m[d];
        out << std::endl;
    }

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


/***************************************************************************
 * $RCSfile: FieldLayout.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: FieldLayout.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/