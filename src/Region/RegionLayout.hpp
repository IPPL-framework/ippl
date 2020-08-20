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
#include "Region/RegionLayout.h"
#include "Region/Rnode.h"
#include "Index/NDIndex.h"
#include "Field/GuardCellSizes.h"
#include "FieldLayout/FieldLayout.h"
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"


// static RegionLayout members
template < class T, unsigned Dim, class MeshType >
typename RegionLayout<T,Dim,MeshType>::RnodePool
  RegionLayout<T,Dim,MeshType>::StaticRnodePool;


//////////////////////////////////////////////////////////////////////
// default constructor
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout() {
  
  

  for (unsigned d=0; d < Dim; ++d) {
    IndexOffset[d] = 0;
    CenterOffset[d] = 0;
  }

  Remote_ac = 0;
  //  store_mesh(0, true);
  //  store_flayout(0, true);
  // Initialize the data members directly,
  // to avoid problems with non-zero default values of pointers
  theMesh = 0;
  WeOwnMesh = true;
  FLayout = 0;
  WeOwnFieldLayout = true;
}

//////////////////////////////////////////////////////////////////////
// copy constructor
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const RegionLayout<T,Dim,MeshType>& r) {
  
  

  for (unsigned d=0; d < Dim; ++d) {
    IndexOffset[d] = r.IndexOffset[d];
    CenterOffset[d] = r.CenterOffset[d];
  }

  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  
  if (r.WeOwnMesh) {
    // make local copy of mesh
    store_mesh(new MeshType(*(r.theMesh)), true);
  } else {
    // just copy the pointer
    store_mesh(r.theMesh, false);
  }

  if (r.WeOwnFieldLayout)
    changeDomain(r.Domain, r.size_iv() + r.size_rdv());
  else
    changeDomain(*((FieldLayout<Dim> *)(r.FLayout)));
}


//////////////////////////////////////////////////////////////////////
// destructor
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::~RegionLayout() {
  
  
 
  // delete all our existing rnodes
  delete_rnodes();

  // delete the FieldLayout
  delete_flayout();

  // delete the mesh
  delete_mesh();
}


//////////////////////////////////////////////////////////////////////
// constructor for an N-Dimensional PRegion
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const NDRegion<T,Dim>& domain,
					   MeshType& mesh, int vnodes) {
  
  

  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  changeDomain(domain, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 1-D PRegion
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const PRegion<T>& i1,
					   MeshType& mesh, int vnodes) {
  
  

  PInsist(Dim==1,
    "Number of PRegion arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDRegion<T,Dim> dom;
  dom[0] = i1;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 2-D PRegion
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const PRegion<T>& i1,
					   const PRegion<T>& i2,
					   MeshType& mesh,
					   int vnodes) {
  

  PInsist(Dim==2,
    "Number of PRegion arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDRegion<T,Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 3-D PRegion
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const PRegion<T>& i1,
					   const PRegion<T>& i2,
					   const PRegion<T>& i3,
					   MeshType& mesh,
					   int vnodes) {

  PInsist(Dim==3,
    "Number of PRegion arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDRegion<T,Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  dom[2] = i3;
  changeDomain(dom, vnodes);
}



//////////////////////////////////////////////////////////////////////
// constructor for an N-Dimensional PRegion, given an NDIndex
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const NDIndex<Dim>& domain,
					   int vnodes) {
  
  

  // build mesh on this domain, with each axis extended by one
  NDIndex<Dim> extended;
  for (unsigned int i=0; i<Dim; i++)
    extended[i] = Index(domain[i].first(), domain[i].last()+1,
                        domain[i].stride());
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(new MeshType(extended), true);
  changeDomain(domain, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 1-D PRegion, given an Index
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1, int vnodes) {
  
  
 
  PInsist(Dim==1,
    "Number of Index arguments does not match RegionLayout dimension!!");
  // build mesh on this Index, extended by one
  Index extended(i1.first(), i1.last()+1, i1.stride());
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(new MeshType(extended), true);
  NDIndex<Dim> dom;
  dom[0] = i1;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 2-D PRegion, given two Index objecs
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1, const Index& i2,
			   	       int vnodes) {
   
  

  PInsist(Dim==2,
    "Number of Index arguments does not match RegionLayout dimension!!");
  // build mesh on domain extended by one on each axis
  Index ex1(i1.first(), i1.last()+1, i1.stride());
  Index ex2(i2.first(), i2.last()+1, i2.stride());
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(new MeshType(ex1, ex2), true);
  NDIndex<Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 3-D PRegion, given three Index objecs
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1, const Index& i2,
				       const Index& i3, int vnodes) {

  PInsist(Dim==3,
    "Number of Index arguments does not match RegionLayout dimension!!");
  // build mesh on domain extended by one on each axis
  Index ex1(i1.first(), i1.last()+1, i1.stride());
  Index ex2(i2.first(), i2.last()+1, i2.stride());
  Index ex3(i3.first(), i3.last()+1, i3.stride());
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(new MeshType(ex1, ex2, ex3), true);
  NDIndex<Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  dom[2] = i3;
  changeDomain(dom, vnodes);
}



//////////////////////////////////////////////////////////////////////
// constructor for an N-Dimensional PRegion, given an NDIndex and Mesh
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const NDIndex<Dim>& domain,
					   MeshType& mesh,
					   int vnodes) {
  
  

  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  changeDomain(domain, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 1-D PRegion, given an Index and Mesh
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1,
					   MeshType& mesh,
					   int vnodes) {
  
  

  PInsist(Dim==1,
    "Number of Index arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDIndex<Dim> dom;
  dom[0] = i1;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 2-D PRegion, given two Index objecs and a Mesh
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1,
					   const Index& i2,
					   MeshType& mesh,
					   int vnodes) {

  PInsist(Dim==2,
    "Number of Index arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDIndex<Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  changeDomain(dom, vnodes);
}

//////////////////////////////////////////////////////////////////////
// constructor for just a 3-D PRegion, given three Index objecs and a Mesh
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(const Index& i1,
					   const Index& i2,
					   const Index& i3,
					   MeshType& mesh,
					   int vnodes) {

  PInsist(Dim==3,
    "Number of Index arguments does not match RegionLayout dimension!!");
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  NDIndex<Dim> dom;
  dom[0] = i1;
  dom[1] = i2;
  dom[2] = i3;
  changeDomain(dom, vnodes);
}


//////////////////////////////////////////////////////////////////////
// Constructor which takes a FieldLayout, and uses it for our layout
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(FieldLayout<Dim>& fl) {
  
  

  // build mesh on domain extended by one along each axis
  NDIndex<Dim> domain = fl.getDomain();
  NDIndex<Dim> extended;
  for (unsigned int i=0; i<Dim; i++)
    extended[i] = Index(domain[i].first(), domain[i].last()+1,
                        domain[i].stride());
  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(new MeshType(extended), true);
  changeDomain(fl);
}


//////////////////////////////////////////////////////////////////////
// Constructor which takes a FieldLayout and a Mesh
template < class T, unsigned Dim, class MeshType >
RegionLayout<T,Dim,MeshType>::RegionLayout(FieldLayout<Dim>& fl,
					   MeshType& mesh) {
  
  

  FLayout = 0;
  theMesh = 0;
  Remote_ac = 0;
  store_mesh(&mesh, false);
  changeDomain(fl);
}


//////////////////////////////////////////////////////////////////////
// Reconstruct this RegionLayout from the given domain.  This clears
// out the existing data, and generates a new partitioning.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::changeDomain(const NDIndex<Dim>& domain,
						int vnodes) {
  
  

  unsigned int d;			// loop variable

  // delete our existing FieldLayout
  delete_flayout();

  // create a mesh, if necessary
  if (theMesh == 0) {
    NDIndex<Dim> ex;
    for (d=0; d < Dim; ++d)
      ex[d] = Index(domain[d].first(),domain[d].last()+1,domain[d].stride());
    store_mesh(new MeshType(ex), true);
  }

  // set our index space offset
  for (d=0; d < Dim; ++d) {
    IndexOffset[d] = domain[d].first();
    CenterOffset[d] = (MeshVertices[d].length() > domain[d].length());
  }

  // build the FieldLayout from this NDIndex ... note that
  // we're making our own FieldLayout here.  Creating this FieldLayout
  // results in a distribution of vnodes, which we will use for our rnodes.
  store_flayout(new FieldLayout<Dim>(domain, 0, vnodes), true);

  // save the region, and note that our domain matches the FieldLayout's,
  // so that there is no offset
  Domain = convert_index(FLayout->getDomain());

  // from the FieldLayout and Domain, build our internal representation
  make_rnodes(Domain, *FLayout);
  return;
}


//////////////////////////////////////////////////////////////////////
// Reconstruct this RegionLayout from the given domain.  This clears
// out the existing data, and generates a new partitioning.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::changeDomain(const NDRegion<T,Dim>& domain,
						int vnodes) {
  
  

  unsigned int d;			// loop variable

  // delete our existing FieldLayout
  delete_flayout();

  // create a mesh, if necessary
  // right now, we assume that the current mesh is adequate, but 
  // we might have to pass a new one in or create one.
  PAssert(theMesh);

  // set our index space and centering offsets
  for (d=0; d < Dim; ++d) {
    IndexOffset[d] = 0;
    CenterOffset[d] = true;  // make the internal FieldLayout cell-centered
  }

  // make an NDIndex based on the NDRegion & mesh
  NDIndex<Dim> area = convert_region(domain);

  // build the FieldLayout from this NDIndex ... note that
  // we're making our own FieldLayout here.  Creating this FieldLayout
  // results in a distribution of vnodes, which we will use for our rnodes.
  store_flayout(new FieldLayout<Dim>(area, 0, vnodes), true);

  // save the region, and note that our domain is offset from the FieldLayout's
  Domain = domain;

  // from the FieldLayout, build our internal representation
  make_rnodes(Domain, *FLayout);
  return;
}


//////////////////////////////////////////////////////////////////////
// Reconstruct this RegionLayout from the given FieldLayout.  This
// we just use the FieldLayout as-is, instead of making a new one.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::changeDomain(FieldLayout<Dim>& fl) {
  
  

  unsigned int d;			// loop variable

  // delete our existing FieldLayout
  delete_flayout();

  // create a mesh, if necessary
  if (theMesh == 0) {
    NDIndex<Dim> ex;
    const NDIndex<Dim>& domain(fl.getDomain());
    for (d=0; d < Dim; ++d)
      ex[d] = Index(domain[d].first(),domain[d].last()+1,domain[d].stride());
    store_mesh(new MeshType(ex), true);
  }

  // set our index space offset
  for (d=0; d < Dim; ++d) {
    IndexOffset[d] = fl.getDomain()[d].first();
    CenterOffset[d] = (theMesh->gridSizes[d] > fl.getDomain()[d].length());
  }

  // save this new layout, and set up our data structures
  Domain = convert_index(fl.getDomain());
  store_flayout(&fl, false);
  make_rnodes(Domain, *FLayout);
  return;
}


//////////////////////////////////////////////////////////////////////
// convert a given NDIndex into an NDRegion ... if this object was
// constructed from a FieldLayout, this does nothing, but if we are maintaining
// our own internal FieldLayout, we must convert from the [0,N-1] index
// space to our own continuous NDRegion space.
// NOTE: THIS ASSUMES THAT REGION'S HAVE first() < last() !!
template < class T, unsigned Dim, class MeshType >
NDRegion<T,Dim>
RegionLayout<T,Dim,MeshType>::convert_index(const NDIndex<Dim>& ni) const {

  NDRegion<T,Dim> new_pregion; // Needed in TAU_TYPE_STRING

  unsigned int d;

  // find first and last points in NDIndex and get coordinates from mesh
  NDIndex<Dim> FirstPoint, LastPoint;
  for (d=0; d<Dim; d++) {
    int first = ni[d].first() - IndexOffset[d];
    int last  = ni[d].last()  - IndexOffset[d] + CenterOffset[d];
    FirstPoint[d] = Index(first, first);
    LastPoint[d] = Index(last, last);
  }

  // convert to mesh space
  Vektor<T,Dim> FirstCoord = theMesh->getVertexPosition(FirstPoint);
  Vektor<T,Dim> LastCoord = theMesh->getVertexPosition(LastPoint);
  for (d=0; d<Dim; d++) {
    if (!CenterOffset[d]) { // vertex centering, so offset region endpoints
      if ( !(FirstPoint[d] == Index(0,0)) ) {
        FirstPoint[d] = FirstPoint[d] - 1;
        FirstCoord = theMesh->getVertexPosition(FirstPoint);
        FirstCoord(d) = FirstCoord(d) +
                        0.5 * theMesh->getDeltaVertex(FirstPoint)(d);
      }
      int final = theMesh->gridSizes[d]-1; 
      if ( !(LastPoint[d] == Index(final,final)) )
        LastCoord(d) = LastCoord(d) +
                       0.5 * theMesh->getDeltaVertex(LastPoint)(d);
    }

    new_pregion[d] = PRegion<T>(FirstCoord(d), LastCoord(d));
  }

  return new_pregion;
}


//////////////////////////////////////////////////////////////////////
// perform the inverse of convert_index: convert a given NDRegion (with
// coordinates in the 'region' space) into an NDIndex (with values in
// the [0,N-1] 'index' space).  This will truncate values when converting
// from continuous to integer data.
template < class T, unsigned Dim, class MeshType >
NDIndex<Dim>
RegionLayout<T,Dim,MeshType>::convert_region(const NDRegion<T,Dim>& nr) const {
  NDIndex<Dim> index;
  
  

  unsigned d;

  // find mesh points corresponding to endpoints of region
  Vektor<T,Dim> FirstCoord, LastCoord;
  for (d=0; d<Dim; d++) {
    FirstCoord(d) = nr[d].first();
    LastCoord(d) = nr[d].last();
  }
  NDIndex<Dim> FirstPoint = theMesh->getNearestVertex(FirstCoord);
  NDIndex<Dim> LastPoint = theMesh->getNearestVertex(LastCoord);
  for (d=0; d<Dim; d++) {
    if (!CenterOffset[d]) { // vertex centering
      if (theMesh->getVertexPosition(FirstPoint)(d) < FirstCoord(d))
        FirstPoint[d] = FirstPoint[d] + 1;
      if (theMesh->getVertexPosition(LastPoint)(d) > LastCoord(d))
        LastPoint[d] = LastPoint[d] - 1;
    }
    index[d] = Index(FirstPoint[d].first() + IndexOffset[d],
  	             LastPoint[d].first() + IndexOffset[d] - CenterOffset[d]);
  }

  return index;
}

//////////////////////////////////////////////////////////////////////
// Scan through the internal FieldLayout and construct Rnodes based on
// the current FieldLayout and PRegion.  Put them into out local
// and remote Rnode containers.
template < class T, unsigned Dim, class MeshType >
void
RegionLayout<T,Dim,MeshType>::make_rnodes(const NDRegion<T,Dim>& dom,
					  FieldLayout<Dim>& FL) {
  
  

  //Inform dbgmsg("RegionLayout::make_rnodes");
  //dbgmsg << "Creating new rnodes based on FieldLayout = " << FL << endl;

  // delete the existing rnodes
  delete_rnodes();

  // for each local vnode in the FieldLayout, make a corresponding NDRegion
  // and put it in a local Rnode
  typedef typename ac_id_vnodes::value_type lrnode_t;
  typename FieldLayout<Dim>::iterator_iv lociter = FL.begin_iv();
  typename FieldLayout<Dim>::iterator_iv endloc  = FL.end_iv();
  for ( ; lociter != endloc; ++lociter) {
    Rnode<T,Dim> *rnode =
      StaticRnodePool.create_rnode(
		   convert_index((*((*lociter).second)).getDomain()),
		   (*((*lociter).second)).getNode());

    //dbgmsg << "  Created local rnode = " << rnode->getDomain();
    //dbgmsg << " from vnode = " << (*((*lociter).second)).getDomain() <<endl;
    
    Local_ac.insert( lrnode_t(Unique::get(), rnode) );
  }

  // similarly, for each remote vnode in the FieldLayout, make an Rnode
  Remote_ac = new ac_domain_vnodes(dom);
  typedef typename ac_domain_vnodes::value_type rrnode_t;
  typename FieldLayout<Dim>::iterator_dv remiter = FL.begin_rdv();
  typename FieldLayout<Dim>::iterator_dv endrem  = FL.end_rdv();
  for ( ; remiter != endrem; ++remiter) {
    Rnode<T,Dim> *rnode =
      StaticRnodePool.create_rnode(
		   convert_index((*((*remiter).second)).getDomain()),
		   (*((*remiter).second)).getNode());

    //dbgmsg << "  Created remote rnode = " << rnode->getDomain();
    //dbgmsg << " from vnode = " << (*((*remiter).second)).getDomain() <<endl;
    
    Remote_ac->insert( rrnode_t(rnode->getDomain(), rnode) );
  }

  // Since the rnodes changed, repartition each object using this
  // RegionLayout.  We have made sure only FieldLayoutUser's can
  // check in with us, so we know that all user's can be cast to
  // FieldLayoutUser.
  for (iterator_user p = begin_user(); p != end_user(); ++p) {
    FieldLayoutUser *user = (FieldLayoutUser *) (*p).second;
    user->Repartition(this);
  }
}


//////////////////////////////////////////////////////////////////////
// Delete the Rnodes in the given local and remote lists ... actually,
// just returns them back to the static pool.  Note that this DOES NOT
// remove the elements from the lists.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::delete_rnodes() {
  
  

  // delete local rnodes
  typename ac_id_vnodes::iterator lociter = Local_ac.begin();
  typename ac_id_vnodes::iterator endloc  = Local_ac.end();
  for ( ; lociter != endloc; ++lociter)
    StaticRnodePool.push_back((*lociter).second);
  Local_ac.erase(Local_ac.begin(), Local_ac.end());

  // delete remote rnodes
  if (Remote_ac != 0) {
    typename ac_domain_vnodes::iterator remiter = Remote_ac->begin();
    typename ac_domain_vnodes::iterator endrem  = Remote_ac->end();
    for ( ; remiter != endrem; ++remiter)
      StaticRnodePool.push_back((*remiter).second);
    delete Remote_ac;
    Remote_ac = 0;
  }
}


//////////////////////////////////////////////////////////////////////
// Store a FieldLayout pointer, and note if we own it or not.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::store_flayout(FieldLayout<Dim>* f,
						 bool WeOwn) {
  
  

  // get rid of the existing layout, if necessary
  delete_flayout();

  // save the pointer, and whether we own it.  Also, check in to the
  // layout
  FLayout = f;
  WeOwnFieldLayout = WeOwn;
  if (FLayout != 0)
    FLayout->checkin(*this);
}


//////////////////////////////////////////////////////////////////////
// Delete our current FLayout, and set it to NULL; we may have to
// check out from the layout
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::delete_flayout() {
  
  if (FLayout != 0) {
    FLayout->checkout(*this);
    if (WeOwnFieldLayout)
      delete FLayout;
    FLayout = 0;
  }
}


//////////////////////////////////////////////////////////////////////
// Store a Mesh pointer, and note if we own it or not.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::store_mesh(MeshType* m, bool WeOwn) {
   
  

  // get rid of the existing mesh, if necessary
  delete_mesh();

  // save the pointer, and whether we own it.  Also, check in to the
  // layout
  theMesh = m;
  WeOwnMesh = WeOwn;
  if (theMesh != 0) {
    theMesh->checkin(*this);
    for (unsigned int d=0; d < Dim; ++d)
      MeshVertices[d] = Index(theMesh->gridSizes[d]);
  }
}


//////////////////////////////////////////////////////////////////////
// Delete our current MeshType, and set it to NULL; we may have to
// check out from the mesh
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::delete_mesh() {
  
  
  if (theMesh != 0) {
    theMesh->checkout(*this);
    if (WeOwnMesh)
      delete theMesh;
    theMesh = 0;
  }
}


//////////////////////////////////////////////////////////////////////
// calculate the boundary region of the given mesh object
template < class T, unsigned Dim, class MeshType >
NDRegion<T,Dim> RegionLayout<T,Dim,MeshType>::getMeshDomain(MeshType *m) {
  NDRegion<T,Dim> retDomain;

  
  

  typename MeshType::MeshVektor_t morigin  = m->get_origin();
  NDIndex<Dim> meshCells;
  unsigned int d;
  for (d=0; d < Dim; ++d)
    meshCells[d] = Index(MeshVertices[d].first(),MeshVertices[d].last()-1);
  typename MeshType::MeshVektor_t msize    = m->getDeltaVertex(meshCells);
  for (d=0; d < Dim; ++d)
    retDomain[d] = PRegion<T>(morigin[d], morigin[d] + msize[d]);
  return retDomain;
}


//////////////////////////////////////////////////////////////////////
// calculate the number of vertices in the given mesh
template < class T, unsigned Dim, class MeshType >
NDIndex<Dim> RegionLayout<T,Dim,MeshType>::getMeshVertices(MeshType *m) {
  NDIndex<Dim> mvertices;

  
  

  for (unsigned int d=0; d < Dim; ++d)
    mvertices[d] = Index(m->gridSizes[d]);
  return mvertices;
}


//////////////////////////////////////////////////////////////////////
// output
template < class T, unsigned Dim, class MeshType >
std::ostream& operator<<(std::ostream& out, const RegionLayout<T,Dim,MeshType>& f) {
  
  
 
  int icount;

  // the whole domain
  out << "Total Domain = " << f.getDomain() << "\n";

  // iterate over the local vnodes and print them out.
  out << "Local Rnodes = " << f.size_iv() << "\n";
  typename RegionLayout<T,Dim,MeshType>::const_iterator_iv v_i = f.begin_iv();
  typename RegionLayout<T,Dim,MeshType>::const_iterator_iv v_e = f.end_iv();
  for(icount=0 ; v_i != v_e; ++v_i, ++icount)
    out << " rnode " << icount << " : " << (*v_i).second->getDomain() << "\n";

  // iterate over the remote vnodes and print them out.
  out << "Remote Rnodes = " << f.size_rdv() << "\n";
  if (f.size_rdv() > 0) {
    typename RegionLayout<T,Dim,MeshType>::const_iterator_dv dv_i = f.begin_rdv();
    typename RegionLayout<T,Dim,MeshType>::const_iterator_dv dv_e = f.end_rdv();
    for (icount=0 ; dv_i != dv_e; ++dv_i, ++icount)
      out << " rnode " << icount << " : " << (*dv_i).first << "\n";
  }

  // print out our internal FieldLayout
  out << "Internal FieldLayout = " << f.getFieldLayout();

  return out;
}


//////////////////////////////////////////////////////////////////////
// Repartition the region, from a list of NDIndex objects which
// represent our local domain.  This assumes two things:
//   1. We are repartitioning the same global domain, just in a different
//      way.  Thus, the encompassing NDRegion 'Domain' does not change.
//   2. The NDIndex objects cover a domain which corresponds to our
//      internal FieldLayout domain.  This may or may not directly
//      overlap with the NDRegion domain.  The basic point is that with
//      these NDIndex objects, we can replace the FieldLayout directly,
//      and then regenerate our RegionLayout (Rnode) data.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::RepartitionLayout(NDIndex<Dim>* ni,
						     NDIndex<Dim>* nf) {
  
  

  // repartition the FieldLayout, using the given list of local vnodes.  This
  // call will result in a distribution of the data to all the nodes.  Also,
  // as part of the Repartition process the FieldLayout will tell us to
  // recreate out rnodes by calling RegionLayout::Repartition.
  if (FLayout != 0)
    FLayout->Repartition(ni, nf);
}


//////////////////////////////////////////////////////////////////////
// Repartition onto a new layout, if the layout changes ... this is a
// virtual function called by a UserList, as opposed to the RepartitionLayout
// function used by the particle load balancing mechanism.
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::Repartition(UserList* userlist) {
  
  

  if (theMesh != 0 && userlist->getUserListID() == theMesh->get_Id()) {
    // perform actions to restructure our data due to a change in the
    // mesh

    //Inform dbgmsg("RegionLayout::Repartition");
    //dbgmsg << "Repartitioning RegionLayout due to mesh change ..." << endl;

    // have the number of mesh points changed?
    NDIndex<Dim> mvertices(getMeshVertices(theMesh));
    if (!(mvertices == MeshVertices)) {
      // yes they have ... we must reconstruct everything, including
      // our FieldLayout.
      MeshVertices = mvertices;
      if (WeOwnFieldLayout)
	changeDomain(getMeshDomain(theMesh), size_iv() + size_rdv());
      else
	changeDomain(*FLayout);
    } else {
      // since we have the same number of vertices, the layout has not
      // changed, only the size of the rnodes and the total domain.
      Domain = getMeshDomain(theMesh);
      make_rnodes(Domain, *FLayout);
    }

  } else {

    //Inform dbgmsg("RegionLayout::Repartition");
    //dbgmsg << "Repartitioning RegionLayout due to layout change ..." << endl;

    // Must be a FieldLayout change instead, so,
    // create new rnodes based on the vnodes in this layout.  Since this
    // is called by the FieldLayout we are using, there is no need to
    // change our FLayout pointer.
    FieldLayout<Dim>* newLayout = (FieldLayout<Dim>*)( userlist );
    make_rnodes(Domain, *newLayout);
  }
}


//////////////////////////////////////////////////////////////////////
// Tell the subclass that the FieldLayoutBase is being deleted, so
// don't use it anymore
template < class T, unsigned Dim, class MeshType >
void RegionLayout<T,Dim,MeshType>::notifyUserOfDelete(UserList* userlist) {
  
  

  if (FLayout != 0 && FLayout->get_Id() == userlist->getUserListID()) {
    // set our FieldLayout pointer to null, if it matches the given userlist.
    // This may render this RegionLayout instance useless,
    // so this should be followed only by a call to the destructor of
    // RegionLayout.
    FLayout = 0;

  } else if (theMesh != 0 && userlist->getUserListID() == theMesh->get_Id()) {
    // set our mesh pointer to null, if it matches the given userlist.
    // This may render this RegionLayout instance useless,
    // so this should be followed only by a call to the destructor of
    // RegionLayout.
    theMesh = 0;

  } else {
    // for now, print a warning ... but in general, this is OK and this
    // warning should be removed
    WARNMSG("RegionLayout: notified of unknown UserList being deleted.");
  }
}


/***************************************************************************
 * $RCSfile: RegionLayout.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:32 $
 * IPPL_VERSION_ID: $Id: RegionLayout.cpp,v 1.1.1.1 2003/01/23 07:40:32 adelmann Exp $ 
 ***************************************************************************/
