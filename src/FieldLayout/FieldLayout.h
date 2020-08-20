// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_LAYOUT_H
#define FIELD_LAYOUT_H

// FieldLayout describes how a given index space (represented by an NDIndex
// object) is partitioned into vnodes.  It performs the initial partitioning,
// and stores a list of local and remote vnodes. The user may request that a
// particular dimension not be partitioned by flagging that axis as 'SERIAL'
// (instead of 'PARALLEL').

// For a power-of-two number of vnodes, using the basic constructor, the
// partitioning is done using a recursive bisection method which makes cuts
// along whole hyperplanes in the global index space. For non-power-of-two
// vnodes with the basic constructor, the cuts do not necessarily go all the
// way across all the elements in all directions. There is also a constructor
// with extra arguments specifying the numbers of vnodes along each direction;
// this one makes the specified number of cuts to divide up the given
// directions. This last constructor obviously only works for numbers of vnodes
// expressible as products of N numbers (in ND), though 1 is an allowable
// number so it really allows any number of vnodes.

// include files
#include "FieldLayout/FieldLayoutUser.h"
#include "FieldLayout/Vnode.h"
#include "DomainMap/DomainMap.h"
#include "Index/NDIndex.h"
#include "Field/GuardCellSizes.h"
#include "Utility/IpplInfo.h"
#include "Utility/UserList.h"
#include "Utility/vmap.h"
#include "Utility/Unique.h"
#include "Utility/my_auto_ptr.h"
#include "Utility/RefCounted.h"

// #include "source/grid/brick.h"


#include <iostream>

// forward declarations
template <unsigned Dim> class FieldLayout;
template <unsigned Dim>
std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

// enumeration used to select serial or parallel axes
enum e_dim_tag { SERIAL=0, PARALLEL=1 } ;


// class definition ... inheritance is private, so that we hide the
// UserList checkin routines and instead replace them with our own
template<unsigned Dim>
class FieldLayout : private UserList
{

public:
  // Typedefs for containers.
  typedef vmap<Unique::type,my_auto_ptr<Vnode<Dim> > > ac_id_vnodes;
  typedef DomainMap<NDIndex<Dim>,RefCountedP< Vnode<Dim> >,
                    Touches<Dim>,Contains<Dim>,Split<Dim> > ac_domain_vnodes;
  typedef vmap<GuardCellSizes<Dim>,my_auto_ptr<ac_domain_vnodes> >
          ac_gc_domain_vnodes;

  // Typedefs for iterators.
  typedef typename ac_id_vnodes::iterator           iterator_iv;
  typedef typename ac_id_vnodes::const_iterator     const_iterator_iv;
  typedef typename ac_domain_vnodes::iterator       iterator_dv;
  typedef typename ac_domain_vnodes::touch_iterator touch_iterator_dv;
  typedef std::pair<touch_iterator_dv,touch_iterator_dv> touch_range_dv;
  typedef typename ac_gc_domain_vnodes::iterator    iterator_gdv;
  typedef iterator_user                             iterator_if;
  typedef size_type_user                            size_type_if;

private:
  // utility to return a zero-guard cell structure.
  static GuardCellSizes<Dim> gc0() { return GuardCellSizes<Dim>(0U); }

public:
  //
  // constructors and destructors
  //

  // Default constructor, which should only be used if you are going to
  // call 'initialize' soon after (before using in any context)
  FieldLayout();

  // Constructor which reads in FieldLayout data from a file.  If the
  // file contains data for an equal number of nodes as we are running on,
  // then that vnode -> pnode mapping will be used.  If the file does not
  // contain info for the same number of pnodes, the vnodes will be
  // distributed in some other manner.
  FieldLayout(const char *filename);

  // Constructors for 1 ... 6 dimensions
  // These specify only a total number of vnodes, allowing the constructor
  // complete control on how to do the vnode partitioning of the index space:
  FieldLayout(const Index& i1,
	      e_dim_tag p1=PARALLEL, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2,
	      e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
	      e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
	      e_dim_tag p3=PARALLEL, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4,
	      e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
	      e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
              int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4, const Index& i5,
	      e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
	      e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
	      e_dim_tag p5=PARALLEL,
              int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4, const Index& i5, const Index& i6,
	      e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
	      e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
	      e_dim_tag p5=PARALLEL, e_dim_tag p6=PARALLEL,
              int vnodes=-1);

  // These specify both the total number of vnodes and the numbers of vnodes
  // along each dimension for the partitioning of the index space. Obviously
  // this restricts the number of vnodes to be a product of the numbers along
  // each dimension (the constructor implementation checks this):
  FieldLayout(const Index& i1,
	      e_dim_tag p1,
	      unsigned vnodes1,
	      bool recurse=false,
              int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2,
	      e_dim_tag p1, e_dim_tag p2,
	      unsigned vnodes1, unsigned vnodes2,
	      bool recurse=false,int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
	      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
	      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
	      bool recurse=false, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4,
	      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
	      e_dim_tag p4,
	      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
	      unsigned vnodes4,
	      bool recurse=false, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4, const Index& i5,
	      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
	      e_dim_tag p4, e_dim_tag p5,
	      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
	      unsigned vnodes4, unsigned vnodes5,
              bool recurse=false, int vnodes=-1);
  FieldLayout(const Index& i1, const Index& i2, const Index& i3,
              const Index& i4, const Index& i5, const Index& i6,
	      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
	      e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
	      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
	      unsigned vnodes4, unsigned vnodes5, unsigned vnodes6,
              bool recurse=false, int vnodes=-1);

  // Next we have one for arbitrary dimension.
  // This one specifies only a total number of vnodes, allowing the constructor
  // complete control on how to do the vnode partitioning of the index space:
  FieldLayout(const NDIndex<Dim>& domain, e_dim_tag *p=0, int vnodes=-1) {
    initialize(domain,p,vnodes);
  }

  // This one specifies both the total number of vnodes and the numbers of
  // vnodes along each dimension for the partitioning of the index
  // space. Obviously this restricts the number of vnodes to be a product of
  // the numbers along each dimension (the constructor implementation checks
  // this):
  //
  // The last argument is a bool for the algorithm to use for assigning vnodes
  // to processors.  If it is false, hand the vnodes to the processors in a
  // very simple but probably inefficient manner.  If it is true, use a binary
  // recursive algorithm. This will usually be more efficient because it will
  // generate less communication, but it will sometimes fail, particularly
  // near the case of one vnode per processor. Because this can fail, it is
  // not the default. This algorithm should only be used when you have 4 or
  // more vnodes per processor.

  FieldLayout(const NDIndex<Dim>& domain, e_dim_tag *p, 
	      unsigned* vnodesPerDirection, 
	      bool recurse=false, int vnodes=-1 ) {
    initialize(domain,p,vnodesPerDirection,recurse,vnodes);
  }

  // Build a FieldLayout given the whole domain and
  // begin and end iterators for the set of domains for the local Vnodes.
  // It does a collective computation to find the remote Vnodes.
  FieldLayout(const NDIndex<Dim>& Domain,
	      const NDIndex<Dim>* begin, const NDIndex<Dim>* end);

  // Build a FieldLayout given the whole domain and
  // begin and end iterators for the set of Vnodes for the local Vnodes.
  // It does a collective computation to find the remote Vnodes.
  // This differs from the previous ctor in that it allows preservation of
  // global Vnode integer ID numbers associated with the input Vnodes. --tjw
  FieldLayout(const NDIndex<Dim>& Domain,
	      const Vnode<Dim>* begin, const Vnode<Dim>* end);

  // Constructor that takes a whole domain, and a pair of iterators over
  // a list of NDIndex's and nodes so that the user specifies the entire
  // decomposition.  No communication is done
  // so these lists must match on all nodes.  A bit of error checking
  // is done for overlapping blocks and illegal nodes, but not exhaustive
  // error checking.
  FieldLayout(const NDIndex<Dim>& Domain,
	      const NDIndex<Dim>* dombegin, const NDIndex<Dim>* domend,
	      const int *nbegin, const int *nend);

  // Destructor: Everything deletes itself automatically ... the base
  // class destructors inform all the FieldLayoutUser's we're going away.
  virtual ~FieldLayout();

  // Initialization functions, only to be called by the user of FieldLayout
  // objects when the FieldLayout was created using the default constructor;
  // otherwise these are only called internally by the various non-default
  // FieldLayout constructors:

  // These specify only a total number of vnodes, allowing the constructor
  // complete control on how to do the vnode partitioning of the index space:
  void initialize(const Index& i1,
		  e_dim_tag p1=PARALLEL, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2,
		  e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
		  e_dim_tag p3=PARALLEL, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4,
		  e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
		  e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
		  int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4, const Index& i5,
		  e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
		  e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
		  e_dim_tag p5=PARALLEL,
		  int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4, const Index& i5, const Index& i6,
		  e_dim_tag p1=PARALLEL, e_dim_tag p2=PARALLEL,
		  e_dim_tag p3=PARALLEL, e_dim_tag p4=PARALLEL,
		  e_dim_tag p5=PARALLEL, e_dim_tag p6=PARALLEL,
		  int vnodes=-1);
  void initialize(const NDIndex<Dim>& domain, e_dim_tag *p=0, int vnodes=-1);

  // These specify both the total number of vnodes and the numbers of vnodes
  // along each dimension for the partitioning of the index space. Obviously
  // this restricts the number of vnodes to be a product of the numbers along
  // each dimension (the constructor implementation checks this):
  void initialize(const Index& i1,
		  e_dim_tag p1, 
		  unsigned vnodes1, 
		  bool recurse=false, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2,
		  e_dim_tag p1, e_dim_tag p2, 
		  unsigned vnodes1, unsigned vnodes2,
		  bool recurse=false, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		  bool recurse=false, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4,
		  e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		  e_dim_tag p4,
		  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		  unsigned vnodes4,
		  bool recurse=false, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4, const Index& i5,
		  e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		  e_dim_tag p4, e_dim_tag p5,
		  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		  unsigned vnodes4, unsigned vnodes5,
		  bool recurse=false, int vnodes=-1);
  void initialize(const Index& i1, const Index& i2, const Index& i3,
		  const Index& i4, const Index& i5, const Index& i6,
		  e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		  e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
		  unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		  unsigned vnodes4, unsigned vnodes5, unsigned vnodes6,
		  bool recurse=false, int vnodes=-1);
  void initialize(const NDIndex<Dim>& domain, e_dim_tag *p, 
		  unsigned* vnodesPerDirection, 
		  bool recurse=false, int vnodes=-1);

  // Initialize that takes a whole domain, and a pair of iterators over
  // a list of NDIndex's and nodes so that the user specifies the entire
  // decomposition.  No communication is done
  // so these lists must match on all nodes.  A bit of error checking
  // is done for overlapping blocks and illegal nodes, but not exhaustive
  // error checking.
  void initialize(const NDIndex<Dim>& Domain,
		  const NDIndex<Dim>* dombegin, const NDIndex<Dim>* domend,
		  const int *nbegin, const int *nend);

  //
  // FieldLayout operations and information
  //

  // Let the user set the local vnodes.
  // this does everything necessary to realign all the fields
  // associated with this FieldLayout!
  // It inputs begin and end iterators for the local vnodes.
  void Repartition(const NDIndex<Dim>*, const NDIndex<Dim>*);
  void Repartition(const NDIndex<Dim>& domain) { Repartition(&domain,(&domain)+1); }

  // This differs from the previous prototype in that it allows preservation of
  // global Vnode integer ID numbers associated with the input Vnodes. --tjw
  void Repartition(const Vnode<Dim>*, const Vnode<Dim>*);

  // Return the domain.
  const NDIndex<Dim>& getDomain() const { return Domain; }

//tjw   // Compare FieldLayouts to see if they represent the same domain.
//tjw   bool operator==(const FieldLayout<Dim>& x) const {
//tjw     return Domain == x.Domain;
//tjw   }

  // Compare FieldLayouts to see if they represent the same domain; if
  // dimensionalities are different, the NDIndex operator==() will return
  // false:
  template <unsigned Dim2>
  bool operator==(const FieldLayout<Dim2>& x) const {
    return Domain == x.getDomain();
  }

  // Read information from the given file on how to repartition the data.
  // This works just like it does when constructing a FieldLayout from a
  // file, in fact this routine is called by the FieldLayout(const char *)
  // constructor.  Only node 0 will actually read the file.  Return success.
  bool read(const char *filename);

  // Write out info about this layout to the given file.  Only node 0 will
  // actually write a file.  Return success.
  bool write(const char *filename);

  //
  // local vnode, remote vnode, touch range, and FieldLayoutUser iterators
  //
  int numVnodes(void) const {
    return (size_iv() + size_rdv());
  }

  // Accessors for the locals by Id.
  typename ac_id_vnodes::size_type size_iv() const;
  iterator_iv                      begin_iv();
  iterator_iv                      end_iv();
  const_iterator_iv                begin_iv() const;
  const_iterator_iv                end_iv() const;

  // Accessors for the remote vnode containers.
  typename ac_gc_domain_vnodes::size_type size_rgdv() const;
  iterator_gdv                            begin_rgdv();
  iterator_gdv                            end_rgdv();

  // Accessors for the remote vnodes themselves.
  typename ac_domain_vnodes::size_type 
    size_rdv(const GuardCellSizes<Dim>& gc = gc0()) const;
  iterator_dv begin_rdv(const GuardCellSizes<Dim>& gc = gc0());
  iterator_dv end_rdv(const GuardCellSizes<Dim>& gc = gc0());
  touch_range_dv touch_range_rdv(const NDIndex<Dim>& domain,
				 const GuardCellSizes<Dim>& gc = gc0()) const; 

  // Accessors for the users accessing this FieldLayout
  size_type_if      size_if() const { return getNumUsers(); }
  iterator_if       begin_if() { return begin_user(); }
  iterator_if       end_if() { return end_user(); }

  //
  // Query for information about the vnode sizes
  //

  // check if the vnode sizes are OK, to match the given GuardCellSizes
  bool fitsGuardCells(const GuardCellSizes<Dim>& gc) const {
    for (unsigned int d=0; d < Dim; ++d)
      if (MinWidth[d] < gc.left(d) || MinWidth[d] < gc.right(d))
        return false;
    return true;
  }

  // for the requested dimension, report if the distribution is
  // SERIAL or PARALLEL
  e_dim_tag getDistribution(unsigned int d) const {
    e_dim_tag retval = PARALLEL;
    if (MinWidth[d] == (unsigned int) Domain[d].length())
      retval = SERIAL;
    return retval;
  }

  // for the requested dimension, report if the distribution was requested to
  // be SERIAL or PARALLEL
  e_dim_tag getRequestedDistribution(unsigned int d) const {
    return RequestedLayout[d];
  }

  // When stored, return number of vnodes along a direction:
  unsigned getVnodesPerDirection(unsigned dir);

  //
  // UserList operations
  //

  // Return our ID, as generated by UserList.
  UserList::ID_t get_Id() const { return getUserListID(); }

  // Tell the FieldLayout that a FieldLayoutUser has been declared on it.
  // This is different than the checkinUser from UserList,
  // since we have the GuardCellSizes argument.
  void checkin(FieldLayoutUser& f, const GuardCellSizes<Dim>& gc = gc0());

  // Tell the FieldLayout that a FieldLayoutUser is no longer using it.
  // This is different than the checkoutUser from UserList,
  // for symmetry with checkin
  void checkout(FieldLayoutUser& f);

  NDIndex<Dim> getLocalNDIndex();

  //
  // I/O
  //

  // Print it out.
  void write(std::ostream&) const;

private:
  // Container definitions.
  ac_id_vnodes        Local_ac;
  ac_gc_domain_vnodes Remotes_ac;

  // Record the domain.
  NDIndex<Dim> Domain;

  // The minimum width of vnodes in each dimension, and the type of
  // layout that the user requested (this might not be the case anymore).
  unsigned int MinWidth[Dim];
  e_dim_tag RequestedLayout[Dim];

  // Store the numbers of vnodes along each direction, when appropriate
  // constructors were called; otherwise leave pointer unset for assertion
  // checks.
  unsigned* vnodesPerDirection_m;

  // calculate the minimum vnode sizes in each dimension
  void calcWidths();

  // utility to return a zero-guard cell structure.
  void new_gc_layout(const GuardCellSizes<Dim>&);

  // The routine which actually sets things up.
  void setup(const NDIndex<Dim>&, e_dim_tag *, int);
  void setup(const NDIndex<Dim>&, e_dim_tag *, unsigned*, bool, int);
};


//////////////////////////////////////////////////////////////////////

// Definitions for the specialized constructors.
// Just turn it into a call to the general ctor.

//-----------------------------------------------------------------------------
// These specify only a total number of vnodes, allowing the constructor
// complete control on how to do the vnode partitioning of the index space:

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, e_dim_tag p1, int vnodes)
{
  initialize(i1, p1, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1,const Index& i2,
			      e_dim_tag p1, e_dim_tag p2, int vnodes)
{
  initialize(i1, i2, p1, p2, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      int vnodes)
{
  initialize(i1, i2, i3, p1, p2, p3, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4,
			      int vnodes)
{
  initialize(i1, i2, i3, i4, p1, p2, p3, p4, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4, 
			      const Index& i5,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4, e_dim_tag p5,
			      int vnodes)
{
  initialize(i1, i2, i3, i4, i5, p1, p2, p3, p4, p5, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4, 
			      const Index& i5, const Index& i6,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
			      int vnodes)
{
  initialize(i1, i2, i3, i4, i5, i6, p1, p2, p3, p4, p5, p6, vnodes);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// These specify both the total number of vnodes and the numbers of vnodes
// along each dimension for the partitioning of the index space. Obviously
// this restricts the number of vnodes to be a product of the numbers along
// each dimension (the constructor implementation checks this):

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, 
			      e_dim_tag p1, 
			      unsigned vnodes1,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes1 != vnodes)"
	    << " ; vnodes1 = " << vnodes1 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, p1, vnodes1, recurse,vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1,const Index& i2,
			      e_dim_tag p1, e_dim_tag p2, 
			      unsigned vnodes1, unsigned vnodes2,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1*vnodes2;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1*vnodes2) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes != vnodes1*vnodes2)"
	    << " ; vnodes1 = " << vnodes1 << " ; vnodes2 = " << vnodes2 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, i2, p1, p2, vnodes1, vnodes2, recurse, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      unsigned vnodes1, unsigned vnodes2, 
			      unsigned vnodes3,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1*vnodes2*vnodes3;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1*vnodes2*vnodes3) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes != vnodes1*vnodes2*vnodes3)"
	    << " ; vnodes1 = " << vnodes1 << " ; vnodes2 = " << vnodes2 
	    << " ; vnodes3 = " << vnodes3 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, i2, i3, p1, p2, p3, vnodes1, vnodes2, vnodes3, recurse, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4,
			      unsigned vnodes1, unsigned vnodes2, 
			      unsigned vnodes3, unsigned vnodes4,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1*vnodes2*vnodes3*vnodes4;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1*vnodes2*vnodes3*vnodes4) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes != vnodes1*vnodes2*vnodes3*vnodes4)"
	    << " ; vnodes1 = " << vnodes1 << " ; vnodes2 = " << vnodes2 
	    << " ; vnodes3 = " << vnodes3 << " ; vnodes4 = " << vnodes4 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, i2, i3, i4, p1, p2, p3, p4, 
	     vnodes1, vnodes2, vnodes3, recurse, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4, 
			      const Index& i5,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4, e_dim_tag p5,
			      unsigned vnodes1, unsigned vnodes2, 
			      unsigned vnodes3, unsigned vnodes4,
			      unsigned vnodes5,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1*vnodes2*vnodes3*vnodes4*vnodes5;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1*vnodes2*vnodes3*vnodes4*vnodes5) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes != vnodes1*vnodes2*vnodes3*vnodes4*vnodes5)"
	    << " ; vnodes1 = " << vnodes1 << " ; vnodes2 = " << vnodes2 
	    << " ; vnodes3 = " << vnodes3 << " ; vnodes4 = " << vnodes4 
	    << " ; vnodes5 = " << vnodes5 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, i2, i3, i4, i5, p1, p2, p3, p4, p5, 
	     vnodes1, vnodes2, vnodes3, vnodes4, vnodes5, recurse, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const Index& i1, const Index& i2, 
			      const Index& i3, const Index& i4, 
			      const Index& i5, const Index& i6,
			      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
			      e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
			      unsigned vnodes1, unsigned vnodes2, 
			      unsigned vnodes3, unsigned vnodes4,
			      unsigned vnodes5, unsigned vnodes6,
			      bool recurse, int vnodes)
{
  // Default to correct total vnodes:
  if (vnodes == -1) vnodes = vnodes1*vnodes2*vnodes3*vnodes4*vnodes5*vnodes6;
  // Verify than total vnodes is product of per-dimension vnode counts:
  if ((unsigned int) vnodes != vnodes1*vnodes2*vnodes3*vnodes4*vnodes5*vnodes6) {
    ERRORMSG("FieldLayout constructor: "
	    << "(vnodes != vnodes1*vnodes2*vnodes3*vnodes4*vnodes5*vnodes6)"
	    << " ; vnodes1 = " << vnodes1 << " ; vnodes2 = " << vnodes2 
	    << " ; vnodes3 = " << vnodes3 << " ; vnodes4 = " << vnodes4 
	    << " ; vnodes5 = " << vnodes5 << " ; vnodes6 = " << vnodes6 
	    << " ; vnodes = " << vnodes << endl);
  }
  initialize(i1, i2, i3, i4, i5, i6, p1, p2, p3, p4, p5, p6, 
	     vnodes1, vnodes2, vnodes3, vnodes4, vnodes5, vnodes6, recurse, vnodes);
}

template<unsigned Dim>
inline
FieldLayout<Dim>::FieldLayout(const NDIndex<Dim> &Domain,
			      const NDIndex<Dim> *dombegin,
			      const NDIndex<Dim> *domend,
			      const int *nbegin,
			      const int *nend)
{
  initialize(Domain, dombegin, domend, nbegin, nend);
}

//-----------------------------------------------------------------------------


//////////////////////////////////////////////////////////////////////

// Accessor definitions.

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_id_vnodes::size_type 
FieldLayout<Dim>::size_iv() const
{
  return Local_ac.size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_iv
FieldLayout<Dim>::begin_iv()
{
  return Local_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_iv
FieldLayout<Dim>::end_iv()
{
  return Local_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::const_iterator_iv
FieldLayout<Dim>::begin_iv() const
{
  return Local_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::const_iterator_iv
FieldLayout<Dim>::end_iv() const
{
  return Local_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_gc_domain_vnodes::size_type
FieldLayout<Dim>::size_rgdv() const
{
  return Remotes_ac.size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_gdv
FieldLayout<Dim>::begin_rgdv()
{
  return Remotes_ac.begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_gdv
FieldLayout<Dim>::end_rgdv()
{
  return Remotes_ac.end();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::ac_domain_vnodes::size_type
FieldLayout<Dim>::size_rdv(const GuardCellSizes<Dim>& gc) const
{
  return Remotes_ac[gc]->size();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_dv
FieldLayout<Dim>::begin_rdv(const GuardCellSizes<Dim>& gc)
{
  return Remotes_ac[gc]->begin();
}

template<unsigned Dim>
inline typename FieldLayout<Dim>::iterator_dv
FieldLayout<Dim>::end_rdv(const GuardCellSizes<Dim>& gc)
{
  return Remotes_ac[gc]->end();
}

template<unsigned Dim>
inline 
typename FieldLayout<Dim>::touch_range_dv
FieldLayout<Dim>::touch_range_rdv(const NDIndex<Dim>& domain,
				  const GuardCellSizes<Dim>& gc) const
{
  return Remotes_ac[gc]->touch_range(domain);
}


// I/O

template<unsigned Dim>
inline
std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
  
  

  f.write(out);
  return out;
}


#include "FieldLayout/FieldLayout.hpp"

#endif // FIELD_LAYOUT_H

/***************************************************************************
 * $RCSfile: FieldLayout.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: FieldLayout.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
