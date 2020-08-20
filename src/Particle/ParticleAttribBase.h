// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_ATTRIB_BASE_H
#define PARTICLE_ATTRIB_BASE_H

/*
 * ParticleAttribBase - base class for all particle attribute classes.
 *   This class is used as the generic base class for all (templated) classes
 *   which represent a single attribute of a Particle.  An attribute class
 *   contains an array of data for N particles, and methods to operate with
 *   this data.
 *
 *   This base class provides virtual methods used to create and destroy
 *   elements of the attribute array.
 */

// include files
#include <cstddef>

#include <vector>
#include <utility>

// forward declarations
class Inform;
class Message;


// ParticleAttribBase class definition
class ParticleAttribBase {

public:
  //
  // Public typedefs
  //

  // Type of storage used for "sort lists", that store indices of
  // where particles should be located in a sort operation.  Used with
  // virtual functions "calcSortList" and "sort".
  typedef long                    SortListIndex_t;
  typedef std::vector<SortListIndex_t> SortList_t;

  //
  // Constructors and Destructors
  //

  // constructor: just store the size of an element, in bytes.
  ParticleAttribBase(unsigned int size, const std::string &typestr)
    : ElementSize(size), TypeString(typestr), Temporary(false) { }

  // copy constructor
  ParticleAttribBase(const ParticleAttribBase &pa)
    : ElementSize(pa.ElementSize), TypeString(pa.TypeString), 
      Temporary(pa.Temporary) { }

  // destructor: does nothing, this is basically an interface class.
  virtual ~ParticleAttribBase() { }

  //
  // accessor methods
  //

  // return the size of a single element item, in bytes
  unsigned int elementSize() const { return ElementSize; }

  // return a simple string with a description of the data type.  This is
  // determined using the DiscType class
  const std::string &typeString() const { return TypeString; }

  // make this a temporary, or a non-temporary.  Temporary attribs just
  // make sure they keep the proper length when create or destroy are called,
  // they do not do any communication
  void setTemporary(bool t) { Temporary = t; }

  // query if this is a temporary attribute
  bool isTemporary() const { return Temporary; }

  //
  // virtual methods used to manipulate the normal attrib data
  //

  // Create storage for M particle attributes.  The storage is uninitialized.
  // New items are appended to the end of the array.
  virtual void create(size_t M) = 0;

  // Delete the attribute storage for M particle attributes, starting at
  // the position I.  Boolean flag indicates whether to use optimized method.
  virtual void destroy(size_t M, size_t I, bool optDestroy) = 0;
  // Delete the attribute storage for a list of particle destroy events
  // Boolean flag indicates whether to use optimized destroy method.
  virtual void destroy(const std::vector< std::pair<size_t,size_t> >& dlist,
                       bool optDestroy) = 0;

  // puts M particle's data starting from index I into a Message.
  // Return the number of particles put into the message.
  virtual size_t putMessage(Message&, size_t, size_t) = 0;
  // puts data for a list of particles into a Message.
  // Return the number of particles put into the message.
  virtual size_t putMessage(Message&, const std::vector<size_t>&) = 0;

  // Get data out of a Message containing N particle's attribute data,
  // and store it here.  Data is appended to the end of the list.  Return
  // the number of particles retrieved.
  virtual size_t getMessage(Message&, size_t) = 0;

  //
  // virtual methods used to manipulate the ghost particle data
  //

  // Delete the ghost attrib storage for M particles, starting at pos I.
  // Items from the end of the list are moved up to fill in the space.
  // Return the number of items actually destroyed.
  virtual size_t ghostDestroy(size_t M, size_t I) = 0;
  
  virtual void ghostCreate(size_t M) = 0;

  // puts M particle's data starting from index I into a Message.
  // Return the number of particles put into the message.  This is for
  // when particles are being swapped to build ghost particle interaction
  // lists.
  virtual size_t ghostPutMessage(Message&, size_t, size_t) = 0;
  // puts data for a list of particles into a Message, for interaction lists.
  // Return the number of particles put into the message.
  virtual size_t ghostPutMessage(Message&, const std::vector<size_t>&) = 0;

  // Get ghost particle data from a message.
  virtual size_t ghostGetMessage(Message&, size_t) = 0;

  //
  // virtual methods used to sort data
  //

  // Calculate a "sort list", which is an array of data of the same
  // length as this attribute, with each element indicating the
  // (local) index wherethe ith particle shoulkd go.  For example, 
  // if there are four particles, and the sort-list is {3,1,0,2}, that
  // means the particle currently with index=0 should be moved to the third
  // position, the one with index=1 should stay where it is, etc.
  // The optional second argument indicates if the sort should be ascending
  // (true, the default) or descending (false).
  virtual void calcSortList(SortList_t &slist, bool ascending = true) =  0;

  // Process a sort-list, as described for "calcSortList", to reorder
  // the elements in this attribute.  All indices in the sort list are
  // considered "local", so they should be in the range 0 ... localnum-1.
  // The sort-list does not have to have been calcualted by calcSortList,
  // it could be calculated by some other means, but it does have to
  // be in the same format.  Note that the routine may need to modify
  // the sort-list temporarily, but it will return it in the same state.
  virtual void sort(SortList_t &slist) = 0;

  // 
  //
  // other virtual functions
  //

  // Print out information for debugging purposes.
  virtual void printDebug(Inform&) = 0;

private:
  // the size, in bytes, of a single element
  unsigned int ElementSize;

  // the data type as a simple string
  std::string TypeString;

  // is this a temporary attribute (one that does not need to do any
  // communication, just keep the right length)?
  bool Temporary;
};

#endif // PARTICLE_ATTRIB_BASE_H


/***************************************************************************
 * $RCSfile: ParticleAttribBase.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleAttribBase.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $ 
 ***************************************************************************/
