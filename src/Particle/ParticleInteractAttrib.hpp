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
#include "Particle/ParticleInteractAttrib.h"
#include "Message/Message.h"



/////////////////////////////////////////////////////////////////////
// put the data for M particles into a message, starting from index I.
// This will either put in local or ghost particle data, but not both.
template<class T>
size_t
ParticleInteractAttrib<T>::putMessage(Message& msg,
				      size_t M, size_t I) {

  if (I >= this->size()) {
    // put in ghost particles
    typename ParticleList_t::iterator currp = GhostList.begin() + (I - this->size());
    typename ParticleList_t::iterator endp = currp + M;
    ::putMessage(msg, currp, endp);
  }
  else {
    typename ParticleList_t::iterator currp = this->ParticleList.begin() + I;
    typename ParticleList_t::iterator endp = currp + M;
    ::putMessage(msg, currp, endp);
  }
  return M;
}


/////////////////////////////////////////////////////////////////////
// Delete the ghost attrib storage for M particles, starting at pos I.
// Items from the end of the list are moved up to fill in the space.
// Return the number of items actually destroyed.
template<class T>
size_t
ParticleInteractAttrib<T>::ghostDestroy(size_t M, size_t I) {

  if (M > 0) {
    // get iterators for where the data to be deleted begins, and where
    // the data we copy from the end begins
    typename ParticleList_t::iterator putloc = GhostList.begin() + I;
    typename ParticleList_t::iterator getloc = GhostList.end() - M;
    typename ParticleList_t::iterator endloc = GhostList.end();

    // make sure we do not copy too much
    if ((I + M) > (GhostList.size() - M))
      getloc = putloc + M;

    // copy over the data
    while (getloc != endloc)
      *putloc++ = *getloc++;

    // delete the last M items
    GhostList.erase(GhostList.end() - M, GhostList.end());
  }

  return M;
}


/////////////////////////////////////////////////////////////////////
// Get ghost data out of a Message containing M particle's attribute data,
// and store it here.  Data is appended to the end of the list.  Return
// the number of particles retrieved.
template<class T>
size_t
ParticleInteractAttrib<T>::ghostGetMessage(Message& msg, size_t M) {

  size_t currsize = GhostList.size();
  GhostList.insert(GhostList.end(), M, T());
  ::getMessage_iter(msg, GhostList.begin() + currsize);
  return M;
}


/////////////////////////////////////////////////////////////////////
// Print out information for debugging purposes.  This version just
// prints out static information, so it is static
template<class T>
void ParticleInteractAttrib<T>::printDebug(Inform& o) {

  o << "PAttr: size = " << this->ParticleList.size()
    << ", capacity = " << this->ParticleList.capacity()
    << ", ghosts = " << GhostList.size();
}


/***************************************************************************
 * $RCSfile: ParticleInteractAttrib.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:29 $
 * IPPL_VERSION_ID: $Id: ParticleInteractAttrib.cpp,v 1.1.1.1 2003/01/23 07:40:29 adelmann Exp $ 
 ***************************************************************************/
