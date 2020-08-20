// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef REGION_LAYOUT_H
#define REGION_LAYOUT_H

/***************************************************************************
 * RegionLayout stores a partitioned set of NDRegion objects, to represent
 * the parallel layout of an encompassing NDRegion.  It also contains
 * functions to find the subsets of the NDRegion partitions which intersect
 * or touch a given NDRegion.  It is similar to FieldLayout, with the
 * following changes:
 * 1. It uses NDRegion instead of NDIndex, so it is templated on the position
 *    data type (although it can be constructed with an NDIndex and a Mesh
 *    as well);
 * 2. It does not contain any consideration for guard cells;
 * 3. It can store not only the partitioned domain, but periodic copies of
 *    the partitioned domain for use by particle periodic boundary conditions
 * 4. It also keeps a list of FieldLayoutUser's, so that it can notify them
 *    when the internal FieldLayout here is reparitioned or otherwise changed.
 *
 * If this is constructed with a FieldLayout, it stores a pointer to it
 * so that if we must repartition the copy of the FieldLayout that
 * is stored here, we will end up repartitioning all the registered Fields.
 ***************************************************************************/

// include files
#include "Region/Rnode.h"
#include "Index/NDIndex.h"
#include "DomainMap/DomainMap.h"
#include "FieldLayout/FieldLayoutUser.h"
#include "Utility/Unique.h"
#include "Utility/UserList.h"
#include "Utility/vmap.h"

#include <iostream>

// forward declarations
template <unsigned Dim> class FieldLayout;
template <unsigned Dim, class T> class UniformCartesian;
template <class T, unsigned Dim, class MeshType> class RegionLayout;
template <class T, unsigned Dim, class MeshType>
std::ostream& operator<<(std::ostream&, const RegionLayout<T,Dim,MeshType>&);


// the RegionLayout class definition
template < class T, unsigned Dim, class MeshType=UniformCartesian<Dim,T> >
class RegionLayout : public FieldLayoutUser, private UserList {

public:
  // Typedefs for containers.
  typedef vmap<Unique::type, Rnode<T,Dim> *> ac_id_vnodes;
  typedef DomainMap<NDRegion<T,Dim>, Rnode<T,Dim> *,
                    TouchesRegion<T,Dim>,
                    ContainsRegion<T,Dim>,
                    SplitRegion<T,Dim> > ac_domain_vnodes;

  // Typedefs for iterators.
  typedef typename ac_id_vnodes::iterator                    iterator_iv;
  typedef typename ac_id_vnodes::const_iterator              const_iterator_iv;
  typedef typename ac_domain_vnodes::iterator                iterator_dv;
  typedef typename ac_domain_vnodes::const_iterator          const_iterator_dv;
  typedef typename ac_domain_vnodes::touch_iterator          touch_iterator_dv;
  typedef std::pair<touch_iterator_dv,touch_iterator_dv> touch_range_dv;

public:
  // Default constructor.  To make this class actually work, the user
  // will have to later call 'changeDomain' to set the proper Domain
  // and get a new partitioning.
  RegionLayout();

  // Copy constructor.
  RegionLayout(const RegionLayout<T,Dim,MeshType>&);

  // Constructors which partition the given ND, 1D, 2D, or 3D PRegions
  RegionLayout(const NDRegion<T,Dim>& domain, MeshType& mesh, int vnodes=-1);
  RegionLayout(const PRegion<T>& i1, MeshType& mesh, int vnodes=-1);
  RegionLayout(const PRegion<T>& i1, const PRegion<T>& i2, MeshType& mesh,
	       int vnodes=-1);
  RegionLayout(const PRegion<T>& i1, const PRegion<T>& i2,
	       const PRegion<T>& i3, MeshType& mesh, int vnodes=-1);

  // Constructor which takes a FieldLayout, and stores a ref to it
  // This will assume a MeshType with unit spacing and that the domain of
  // the MeshType is one larger in each dimension than the domain of the 
  // FieldLayout (i.e., the FieldLayout index space refers to cell-centered
  // Field quantities).
  RegionLayout(FieldLayout<Dim>&);

  // Constructor which takes a FieldLayout and a MeshType
  // This one compares the domain of the FieldLayout and the domain of
  // the MeshType to determine the centering of the index space.
  RegionLayout(FieldLayout<Dim>&, MeshType&);

  // Constructor which takes an NDIndex and converts it to a RegionLayout.
  // These assume a MeshType with unit spacing and that the domain of
  // the MeshType is one larger in each dimension than the given domain 
  // (i.e., the index space refers to cell-centered Field quantities).
  RegionLayout(const NDIndex<Dim>& domain, int vnodes=-1);
  RegionLayout(const Index& i1,int vnodes=-1);
  RegionLayout(const Index& i1,const Index& i2,int vnodes=-1);
  RegionLayout(const Index& i1,const Index& i2,const Index& i3,int vnodes=-1);

  // Constructors which take NDIndex and MeshType and convert to RegionLayout.
  // These compare the given domain and the domain of
  // the MeshType to determine the centering of the index space.
  RegionLayout(const NDIndex<Dim>& domain, MeshType& mesh, int vnodes=-1);
  RegionLayout(const Index& i1, MeshType& mesh, int vnodes=-1);
  RegionLayout(const Index& i1, const Index& i2,MeshType& mesh,int vnodes=-1);
  RegionLayout(const Index& i1, const Index& i2, const Index& i3,
	       MeshType& mesh, int vnodes=-1);

  // Destructor.
  virtual ~RegionLayout();

  //
  // accessor member functions
  //

  // Has the domain been initialized and partitioned yet?
  bool initialized() const { return (FLayout != 0); }

  // Return the encompassing domain.
  const NDRegion<T,Dim>& getDomain() const { return Domain; }

  // Return the underlying FieldLayout.  This is not const, since we may
  // want to make Field's out of it (and have those Field's temporarily
  // register themselves with this FieldLayout).  We do supply a const
  // version, however.
  FieldLayout<Dim>& getFieldLayout() { return *FLayout; }
  const FieldLayout<Dim>& getFieldLayout() const { return *FLayout; }

  // get the mesh
  MeshType& getMesh() { return *theMesh; }
  const MeshType& getMesh() const { return *theMesh; }

  // Accessors for the locals domains by Id.
  typename ac_id_vnodes::size_type size_iv() const { return Local_ac.size(); }
  iterator_iv begin_iv()             { return Local_ac.begin(); }
  iterator_iv end_iv()               { return Local_ac.end(); }
  const_iterator_iv begin_iv() const { return Local_ac.begin(); }
  const_iterator_iv end_iv() const   { return Local_ac.end(); }

  // Accessors for the remote vnodes themselves.
  typename ac_domain_vnodes::size_type size_rdv() const {
    return  (Remote_ac != 0 ? Remote_ac->size() : 0);
  }
  iterator_dv begin_rdv()                { return Remote_ac->begin(); }
  iterator_dv end_rdv()                  { return Remote_ac->end(); }
  const_iterator_dv begin_rdv() const    { return Remote_ac->begin(); }
  const_iterator_dv end_rdv() const      { return Remote_ac->end(); }
  touch_range_dv touch_range_rdv(const NDRegion<T,Dim>& domain) {
    return Remote_ac->touch_range(domain);
  }

  //
  // operator/action methods
  //

  // Repartition the region, using the given layout as the new global
  // domain.  This is essentially the same as what occurs during construction,
  // but may be done any time after construction.
  void changeDomain(FieldLayout<Dim>&);
  void changeDomain(const NDIndex<Dim>&, int vnodes=-1);
  void changeDomain(const NDRegion<T,Dim>&, int vnodes=-1);

  // Repartition the region, from a list of NDIndex objects which
  // represent our local domain.  This assumes two things:
  //   1. We are repartitioning the same global domain, just in a different
  //      way.  Thus, the encompassing NDRegion 'Domain' does not change.
  //   2. The NDIndex objects cover a domain which corresponds to our
  //      internal FieldLayout domain.  This may or may not directly
  //      overlap with the NDRegion domain.  The basic point is that with
  //      these NDIndex objects, we can replace the FieldLayout directly,
  //      and then regenerate our RegionLayout (Rnode) data.
  void RepartitionLayout(NDIndex<Dim>*, NDIndex<Dim>*);
  void RepartitionLayout(NDIndex<Dim>& domain) {
    RepartitionLayout(&domain, (&domain) + 1);
  }

  // convert a given NDIndex into an NDRegion ... if this object was
  // constructed from a FieldLayout, this does nothing, but if we have
  // our own internal FieldLayout, we must convert from the [0,N-1] index
  // space to our own continuous NDRegion space.
  NDRegion<T,Dim> convert_index(const NDIndex<Dim>&) const;

  // perform the inverse of convert_index: convert a given NDRegion (with
  // coordinates in the 'region' space) into an NDIndex (with values in
  // the [0,N-1] 'index' space).  This will truncate values when converting
  // from continuous to integer data.
  NDIndex<Dim> convert_region(const NDRegion<T,Dim>&) const;

  // Compare RegionLayouts to see if they represent the same domain.
  bool operator==(const RegionLayout<T,Dim,MeshType>& x) {
    return Domain == x.Domain;
  }

  //
  // virtual functions for FieldLayoutUser's (and other UserList users)
  //

  // Repartition onto a new layout
  virtual void Repartition(UserList *);

  // Tell this object that an object is being deleted
  virtual void notifyUserOfDelete(UserList *);

  //
  // UserList operations, so that users of this RegionLayout can be
  // kept and notified of changes to the RegionLayout.  We use
  // FieldLayoutUser as the class which must check in here for
  // convenience, since FieldLayoutUser defines a virtual function
  // 'Repartition' which we'll need to call for the users when this
  // RegionLayout changes via repartitioning of itself or of it's
  // internal FieldLayout.
  //

  // Return our ID, as generated by UserList.
  UserList::ID_t get_Id() const { return getUserListID(); }

  // Tell this object that a FieldLayoutUser is using it.
  // This is different than the checkinUser from UserList, since we need
  // a FieldLayoutUser to be stored that has a Repartition virtual function.
  void checkin(FieldLayoutUser& f) { checkinUser(f); }

  // Tell this object that a FieldLayoutUser is no longer using it.
  // This is different than the checkoutUser from UserList,
  // for symmetry with checkin.  Both of these make sure this object
  // only allows FieldLayoutUser's to check in.
  void checkout(FieldLayoutUser& f) { checkoutUser(f); }

private:
  // The local and the remote subdomains which comprise the total domain
  ac_id_vnodes     Local_ac;
  ac_domain_vnodes *Remote_ac;

  // The total domain, i.e. the bounding box for the spatial region
  NDRegion<T,Dim>  Domain;

  // A FieldLayout which is used to represent a grid over our spatial area
  FieldLayout<Dim> *FLayout;
  bool             WeOwnFieldLayout;

  // A MeshType which is used to represent a grid over our spatial area
  MeshType *theMesh;
  bool WeOwnMesh;

  // The number of vertices in the mesh
  NDIndex<Dim> MeshVertices;

  // Offset from 'normal' Index space to 'Mesh' Index space
  int IndexOffset[Dim];

  // Offset needed between centering of Index space and Mesh points
  bool CenterOffset[Dim];

  // The routine which actually sets things up.
  void setup(const NDRegion<T,Dim>&, int);

  // Scan through the given FieldLayout and construct Rnodes based on
  // the FieldLayout and NDRegion.  Put them into out local
  // and remote Rnode containers.
  void make_rnodes(const NDRegion<T,Dim>&, FieldLayout<Dim>&);

  // Delete the Rnodes in out local and remote lists ... actually,
  // just returns them back to the static pool.
  void delete_rnodes();

  // Store a layout object pointer, and note if we own it or not.
  // Delete our current FLayout, and set it to NULL; we may have to
  // check out from the layout
  void store_flayout(FieldLayout<Dim>*, bool WeOwn);
  void delete_flayout();

  // Store a mesh object pointer, and note if we own it or not.
  // Delete our current MeshType, and set it to NULL; we may have to
  // check out from the mesh
  void store_mesh(MeshType*, bool WeOwn);
  void delete_mesh();

  // Calculate the boundary region of the given mesh, and the number of
  // vertices
  // Is this really a C++ error here, or just something mwerksCW4 doesn't grok?
  NDRegion<T,Dim> getMeshDomain(MeshType *);
  NDIndex<Dim>    getMeshVertices(MeshType *);

  //
  // Rnode pool - a static pool of unused Rnode's, used to cut down on new's
  //

  class RnodePool : public std::vector<Rnode<T,Dim> *> {
  public:
    // the constructor and destructor ... the destructor will clean up
    // the storage for allocated Rnode's when the program exits
    RnodePool() { }
    ~RnodePool() {
      while (this->size() > 0) {
	delete this->back();
	this->pop_back();
      }
    }

    // create a new Rnode, or get one from storage.
    Rnode<T,Dim>* create_rnode(const NDRegion<T,Dim>& nr, int node) {
      if (this->empty()) {
	return new Rnode<T,Dim>(nr, node);
      }
      else {
	Rnode<T,Dim>* rnode = this->back();
	this->pop_back();
	*rnode = Rnode<T,Dim>(nr, node);
	return rnode;
      }
    }

    // one more version of create, also specifying an offset
    Rnode<T,Dim>* create_rnode(const NDRegion<T,Dim>& nr,
			       const Vektor<T,Dim>& v,
			       int node) {
      if (this->empty()) {
	return new Rnode<T,Dim>(nr, v, node);
      }
      else {
	Rnode<T,Dim>* rnode = this->back();
	this->pop_back();
	*rnode = Rnode<T,Dim>(nr, v, node);
	return rnode;
      }
    }
  };

  // a static pool of Rnodes
  static RnodePool StaticRnodePool;

};

#include "Region/RegionLayout.hpp"

#endif // REGION_LAYOUT_H

/***************************************************************************
 * $RCSfile: RegionLayout.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:32 $
 * IPPL_VERSION_ID: $Id: RegionLayout.h,v 1.1.1.1 2003/01/23 07:40:32 adelmann Exp $ 
 ***************************************************************************/
