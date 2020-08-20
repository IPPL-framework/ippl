// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_INTERACT_ATTRIB_H
#define PARTICLE_INTERACT_ATTRIB_H

/*
 * ParticleInteractAttrib - Templated class for all particle attribute classes.
 *
 * This templated class is used to represent a single particle attribute.
 * An attribute is one data element within a particle object, and is
 * stored as an array.  This class stores the type information for the
 * attribute, and provides methods to create and destroy new items, and
 * to perform operations involving this attribute with others.  It also
 * provides iterators to allow the user to operate on single particles
 * instead of the entire array.
 *
 * ParticleInteractAttrib is the primary element involved in expressions for
 * particles (just as Field is the primary element there).  This file
 * defines the necessary templated classes and functions to make
 * ParticleInteractAttrib a capable expression-template participant.
 *
 * For some types such as Vektor, Tenzor, etc. which have multipple items,
 * we want to involve just the Nth item from each data element in an 
 * expression.  The () operator here returns an object of type
 * ParticleInteractAttribElem, which will use the () operator on each
 * individual elem to access an item over which one can iterate and get the
 * Nth item from each data element.  For example, if we have an attribute
 * like this:
 *             ParticleInteractAttrib< Vektor<float, 4> > Data
 * we can involve just the 2nd item of each Vektor in an expression by
 * referring to this as
 *             Data(1)
 * which returns an object of type ParticleAttribElem that knows to return
 * just the 2nd item from each Vektor.  ParticleAttribElem is also expression-
 * template-aware; in fact, it is intended primarily for use in expressions
 * and not in many other situations.  The ParticleAttribElem will use the
 * () operator to get the Nth item out of each data element, so this requires
 * the user to define operator () for the particle attribute type being
 * used (for Vektor, Tenzor, etc., this has already been done).  This same
 * thing has been done for operator () involving either one or two indices,
 * which is needed to get the i,j element of a Tenzor, for example.
 *
 * Each ParticleInteractAttrib contains a set of ghost particle attributes as
 * well.  These are stored in a separate vector container, but to users
 * this object looks like a single array with (local + ghost) attributes total.
 */

// include files
#include "Particle/ParticleAttrib.h"
#include "PETE/IpplExpressions.h"

#include <vector>

// forward declarations
class Inform;
class Message;

// ParticleInteractAttrib class definition
template <class T>
class ParticleInteractAttrib : public ParticleAttrib<T> {

public:
  typedef typename ParticleAttrib<T>::ParticleList_t ParticleList_t;

  // default constructor
  ParticleInteractAttrib() { }

  // copy constructor
  ParticleInteractAttrib(const ParticleInteractAttrib<T>& pa)
    : ParticleAttrib<T>(pa), GhostList(pa.GhostList) { }
  ParticleInteractAttrib(const ParticleAttrib<T>& pa)
    : ParticleAttrib<T>(pa) { }

  // destructor: delete the storage of the attribute array
  ~ParticleInteractAttrib() { }

  //
  // Assignment operators
  //

  // assign a general expression
  template<class T1>
  const ParticleInteractAttrib<T>& operator=(const PETE_Expr<T1>& rhs) {
    assign(*this,rhs);
    return *this;
  }

  // assignment of a ParticleInteractAttrib
  const ParticleInteractAttrib<T>& operator=(const
					     ParticleInteractAttrib<T>& rhs) {
    if (this->size() != rhs.size()) {
      ERRORMSG("Attempting to copy particle attributes with unequal sizes.");
      ERRORMSG("\n" << this->size() << " != " << rhs.size() << endl);
    }
    assign(*this,rhs);
    return *this;
  }

  // assignment of a ParticleAttrib
  const ParticleInteractAttrib<T>& operator=(const ParticleAttrib<T>& rhs) {
    if (this->size() != rhs.size()) {
      ERRORMSG("Attempting to copy particle attributes with unequal sizes.");
      ERRORMSG("\n" << this->size() << " != " << rhs.size() << endl);
    }
    assign(*this,rhs);
    return *this;
  }

  // assignment of a scalar
  const ParticleInteractAttrib<T>& operator=(T rhs) {
    assign(*this,rhs);
    return *this;
  }

  //
  // methods specific to ghost particle data
  //

  // provide a new version of the [] operator; if the argument is greater
  // than the size of our normal storage, then it refers to one of the
  // 'ghost' particles
  typename ParticleList_t::reference operator[](size_t n) {
    return (n < this->size() ? this->ParticleList[n] : GhostList[n - this->size()]);
  }

  typename ParticleList_t::const_reference operator[](size_t n) const {
    return (n < this->size() ? this->ParticleList[n] : GhostList[n - this->size()]);
  }

  // return the attrib data for the Nth ghost particle
  typename ParticleList_t::reference ghostAttrib(size_t n) 
    { return GhostList[n]; }

  typename ParticleList_t::const_reference ghostAttrib(size_t n) const 
    { return GhostList[n]; }

  // puts M particle's data starting from index I into a Message.
  // Return the number of particles put into the message.
  virtual size_t putMessage(Message&, size_t, size_t);

  // Another version of putMessage, which takes list of indices
  // Return the number of particles put into the message.
  virtual size_t putMessage(Message& m, const std::vector<size_t>& v) {
    return ParticleAttrib<T>::putMessage(m, v);
  }

  //
  // methods used to manipulate the ghost particle data
  //

  // Delete the ghost attrib storage for M particles, starting at pos I.
  // Items from the end of the list are moved up to fill in the space.
  // Return the number of items actually destroyed.
  virtual size_t ghostDestroy(size_t M, size_t I);

  // puts M particle's data starting from index I into a Message.
  // Return the number of particles put into the message.  This is for
  // when particles are being swapped to build ghost particle interaction
  // lists.
  virtual size_t ghostPutMessage(Message &msg, size_t M, size_t I) {
    return putMessage(msg, M, I);
  }
  // puts data for a list of particles into a Message, for interaction lists.
  // Return the number of particles put into the message.
  virtual size_t ghostPutMessage(Message &msg, const std::vector<size_t> &v) {
    return putMessage(msg, v);
  }

  // Get ghost particle data from a message.
  virtual size_t ghostGetMessage(Message&, size_t);

  //
  // other functions
  //

  // Print out information for debugging purposes.
  virtual void printDebug(Inform&);

private:
  // storage for 'ghost' particles, those stored on the local node which
  // actually belong to other nodes
  ParticleList_t GhostList;
};

#include "Particle/ParticleInteractAttrib.hpp"

#endif  // PARTICLE_INTERACT_ATTRIB_H

/***************************************************************************
 * $RCSfile: ParticleInteractAttrib.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:29 $
 * IPPL_VERSION_ID: $Id: ParticleInteractAttrib.h,v 1.1.1.1 2003/01/23 07:40:29 adelmann Exp $ 
 ***************************************************************************/
