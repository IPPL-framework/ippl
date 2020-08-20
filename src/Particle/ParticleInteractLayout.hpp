//
// Class ParticleInteractLayout
//   Please note: for the time being this class is *not* used! But since it
//   might be used in future projects, we keep this file.
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//

#include "Particle/ParticleInteractLayout.h"
#include "Particle/ParticleBConds.h"
#include "Particle/IpplParticleBase.h"
#include "Region/RegionLayout.h"
#include "FieldLayout/FieldLayout.h"
#include "Utility/IpplInfo.h"
#include "Message/Communicate.h"
#include "Message/Message.h"

#include <algorithm>

/////////////////////////////////////////////////////////////////////
// constructor, from a FieldLayout
template < class T, unsigned Dim, class Mesh >
ParticleInteractLayout<T,Dim,Mesh>::ParticleInteractLayout(FieldLayout<Dim>&
							   fl)
  : ParticleSpatialLayout<T,Dim,Mesh>(fl) {
  setup();			// perform necessary setup
}


/////////////////////////////////////////////////////////////////////
// constructor, from a FieldLayout
template < class T, unsigned Dim, class Mesh >
ParticleInteractLayout<T,Dim,Mesh>::ParticleInteractLayout(FieldLayout<Dim>&
							   fl, Mesh& mesh)
  : ParticleSpatialLayout<T,Dim,Mesh>(fl,mesh) {
  setup();			// perform necessary setup
}


/////////////////////////////////////////////////////////////////////
// constructor, from a RegionLayout
template < class T, unsigned Dim, class Mesh >
ParticleInteractLayout<T,Dim,Mesh>::ParticleInteractLayout(const
  RegionLayout<T,Dim,Mesh>& rl) : ParticleSpatialLayout<T,Dim,Mesh>(rl) {
  setup();			// perform necessary setup
}


/////////////////////////////////////////////////////////////////////
// default constructor ... this does not initialize the RegionLayout,
// it will be instead initialized during the first update.
template < class T, unsigned Dim, class Mesh >
ParticleInteractLayout<T,Dim,Mesh>::ParticleInteractLayout()
  : ParticleSpatialLayout<T,Dim,Mesh>() {
  setup();			// perform necessary setup
}


/////////////////////////////////////////////////////////////////////
// perform common constructor tasks
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::setup() {

  // create storage for message pointers used in swapping particles
  unsigned N = Ippl::getNodes();
  InterNodeList = new bool[N];
  SentToNodeList = new bool[N];
  InteractionNodes = 0;

  // initialize interaction radius information
  InterRadius = 0;
  InterRadiusArray = 0;
  MaxGlobalInterRadius = 0;
}


/////////////////////////////////////////////////////////////////////
// destructor
template < class T, unsigned Dim, class Mesh >
ParticleInteractLayout<T,Dim,Mesh>::~ParticleInteractLayout() {

  delete [] InterNodeList;
  delete [] SentToNodeList;

  for (int i=(PairList.size() - 1); i >= 0; --i)
    delete (PairList[i]);
}


/////////////////////////////////////////////////////////////////////
// Return the maximum interaction radius of the local particles.
template < class T, unsigned Dim, class Mesh >
T ParticleInteractLayout<T,Dim,Mesh>::getMaxLocalInteractionRadius() {

  if (InterRadiusArray != 0) {
    if (InterRadiusArray->size() > 0)
      return *(max_element(InterRadiusArray->begin(),
			   InterRadiusArray->end()));
    else
      return 0.0;
  } else {
    return InterRadius;
  }
}


/////////////////////////////////////////////////////////////////////
// Retrieve a Forward-style iterator for the beginning and end of the
// Nth (local) particle's nearest-neighbor pairlist.
// If this is the first call of this
// method after update(), this must make sure up-to-date info on particles
// from neighboring nodes is available.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::getPairlist(unsigned n,
  pair_iterator& bpi, pair_iterator& epi,
  IpplParticleBase< ParticleInteractLayout<T,Dim,Mesh> >& PData) {

  // check if we have any particle boundary conditions
  if (getUpdateFlag(ParticleLayout<T,Dim>::BCONDS)) {
    // check which boundaries, if any, are periodic
    ParticleBConds<T,Dim>& pBConds = this->getBConds();
    bool periodicBC[2*Dim];
    unsigned numPeriodicBC = 0;
    typename ParticleBConds<T,Dim>::ParticleBCond periodicBCond = ParticlePeriodicBCond;
    for (unsigned d=0; d<2*Dim; ++d) {
      periodicBC[d] = (pBConds[d] == periodicBCond);
      if (periodicBC[d]) ++numPeriodicBC;
    }
    if (numPeriodicBC>0) {
      // we need to reflect domains across all periodic boundaries
      // call specialized function to update ghost particle data
      swap_ghost_particles(PData.getLocalNum(), PData, periodicBC);
    }
    else { // no periodic boundaries, call standard function
      swap_ghost_particles(PData.getLocalNum(), PData);
    }
  }
  else { // no boundary conditions, call standard function
    // update ghost particle data if necessary ... this will also build
    // the pairlists if needed
    swap_ghost_particles(PData.getLocalNum(), PData);
  }

  // get iterators for Nth particle's pairlist ... no check for array sub.
  bpi = PairList[n]->begin();
  epi = PairList[n]->end();

  return;
}


/////////////////////////////////////////////////////////////////////
// for each dimension, calculate where neighboring Vnodes and physical
// nodes are located, and which nodes are within interaction distance
// of our own Vnodes.  Save this info for use in sending ghost particles.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::rebuild_interaction_data() {

  unsigned int j, d;			// loop variables

  // initialize data about interaction nodes, and get the inter radius
  InteractionNodes = 0;
  T interRad = 2.0 * getMaxInteractionRadius();

  // initialize the message list and initial node count
  unsigned N = Ippl::getNodes();
  for (j=0; j < N; ++j)
    InterNodeList[j] = false;

  // if no interaction radius, we're done
  if (interRad <= 0.0)
    return;

  // get RegionLayout iterators
  typename RegionLayout<T,Dim,Mesh>::iterator_iv localVN, endLocalVN = this->RLayout.end_iv();
  // determine which physical nodes are in our interaction domain
  for (localVN = this->RLayout.begin_iv(); localVN != endLocalVN; ++localVN) {
    // for each local Vnode, the domain to check equals the local Vnode dom
    // plus twice the interaction radius
    NDRegion<T,Dim> chkDom((*localVN).second->getDomain());
    for (d=0; d < Dim; ++d) {
      chkDom[d] = PRegion<T>(chkDom[d].first() - interRad,
			     chkDom[d].last() + interRad);
    }

    // use the RegionLayout to find all remote Vnodes which touch the domain
    // being checked here
    typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
      this->RLayout.touch_range_rdv(chkDom);
    typename RegionLayout<T,Dim,Mesh>::touch_iterator_dv tVN = touchingVN.first;
    for ( ; tVN != touchingVN.second; ++tVN) {
      // note that we need to send a message to the node which contains
      // this remote Vnode
      unsigned vn = ((*tVN).second)->getNode();
      if ( ! InterNodeList[vn] ) {
	InterNodeList[vn] = true;
	InteractionNodes++;
      }
    }
  }

  // set the flag indicating the swap ghost particle routine should
  // be called the next time we try to access a pairlist or do anything
  // of utility with the ghost particles
  NeedGhostSwap = true;
  return;
}


/////////////////////////////////////////////////////////////////////
// for each dimension, calculate where neighboring Vnodes and physical
// nodes are located, and which nodes are within interaction distance
// of our own Vnodes.  Save this info for use in sending ghost particles.
// Special version to handle periodic boundary conditions
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::rebuild_interaction_data(
  const bool periodicBC[2*Dim])
{
  unsigned int j, d;			// loop variables
  unsigned pe = Ippl::myNode();

  // initialize data about interaction nodes, and get the inter radius
  InteractionNodes = 0;
  T interRad = 2.0 * getMaxInteractionRadius();

  // initialize the message list and initial node count
  unsigned N = Ippl::getNodes();
  for (j=0; j < N; ++j)
    InterNodeList[j] = false;

  // if no interaction radius, we're done
  if (interRad <= 0.0)
    return;

  // get domain info
  const NDRegion<T,Dim>& globalDom = this->RLayout.getDomain();

  // some stuff for computing reflected domains
  T offset[Dim];
  unsigned numRef;
  bool flipBit, activeBit[Dim], refBit[Dim];
  NDRegion<T,Dim> chkDom, refDom;

  // get RegionLayout iterators
  typename RegionLayout<T,Dim,Mesh>::iterator_iv localVN, endLocalVN = this->RLayout.end_iv();
  typename RegionLayout<T,Dim,Mesh>::iterator_iv localVN2;
  typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN;
  typename RegionLayout<T,Dim,Mesh>::touch_iterator_dv tVN;

  // determine which physical nodes are in our interaction domain
  for (localVN = this->RLayout.begin_iv(); localVN != endLocalVN; ++localVN) {
    // for each local Vnode, the domain to check equals the local Vnode dom
    // plus twice the interaction radius
    chkDom = (*localVN).second->getDomain();
    for (d=0; d<Dim; ++d) {
      chkDom[d] = PRegion<T>(chkDom[d].first() - interRad,
	  		     chkDom[d].last() + interRad);
    }

    // use the RegionLayout to find all remote Vnodes which touch
    // the domain being checked here
    touchingVN = this->RLayout.touch_range_rdv(chkDom);
    for (tVN = touchingVN.first; tVN != touchingVN.second; ++tVN) {
      // note that we need to send a message to the node which contains
      // this remote Vnode
      unsigned vn = ((*tVN).second)->getNode();
      if ( ! InterNodeList[vn] ) {
        InterNodeList[vn] = true;
	InteractionNodes++;
      }
    }

    // look for boundary crossings and check reflected domains
    numRef = 0;
    for (d=0; d<Dim; ++d) {
      if (periodicBC[2*d] && chkDom[d].first()<globalDom[d].first()) {
        // crossed lower boundary
        offset[d] = globalDom[d].length();
        activeBit[d] = true;
        numRef = 2 * numRef + 1;
      }
      else if (periodicBC[2*d+1] && chkDom[d].last()>globalDom[d].last()) {
        // crossed upper boundary
        offset[d] = -globalDom[d].length();
        activeBit[d] = true;
        numRef = 2 * numRef + 1;
      }
      else {
        offset[d] = 0.0;
        activeBit[d] = false;
      }
      refBit[d] = false;  // reset reflected domain bools
    }

    // compute and check each domain reflection
    for (j=0; j<numRef; ++j) {
      // set up reflected domain: first initialize to original domain
      refDom = chkDom;
      // find next combination of dimension offsets
      d = 0;
      flipBit = false;
      while (d<Dim && !flipBit) {
        // first check if this dim is active
        if (activeBit[d]) {
          // now flip bit for this dimension
          if (refBit[d]) {
            // flip this bit off and proceed to next dim
            refBit[d] = false;
          }
          else { // refBit[d] is off
            // flip this bit on and indicate we're done
            refBit[d] = true;
            flipBit = true;
          }
        }
        ++d;
      }
      PAssert(flipBit);  // check that we found next combination

      // now offset the reflected domain
      for (d=0; d<Dim; ++d) {
        if (refBit[d]) refDom[d] = refDom[d] + offset[d];
      }

      // use the RegionLayout to find all remote Vnodes which touch
      // the domain being checked here
      touchingVN = this->RLayout.touch_range_rdv(refDom);
      for (tVN = touchingVN.first; tVN != touchingVN.second; ++tVN) {
        // note that we need to send a message to the node which contains
        // this remote Vnode
        unsigned vn = ((*tVN).second)->getNode();
        if ( ! InterNodeList[vn] ) {
          InterNodeList[vn] = true;
	  InteractionNodes++;
        }
      }

      if (!InterNodeList[pe]) { // check if we interact with our own domains
        // for reflected domains, we also must check against local domains
        bool interact = false;
        localVN2 = this->RLayout.begin_iv();
        while (localVN2 != endLocalVN && !interact) {
          interact = refDom.touches((*localVN2).second->getDomain());
          ++localVN2;
        }
        if (interact) {
          InterNodeList[pe] = true;
          InteractionNodes++;
        }
      }
    }

  }

  // set the flag indicating the swap ghost particle routine should
  // be called the next time we try to access a pairlist or do anything
  // of utility with the ghost particles
  NeedGhostSwap = true;
  return;
}


/////////////////////////////////////////////////////////////////////
// Update the location and indices of all atoms in the given IpplParticleBase
// object.  This handles swapping particles among processors if
// needed, and handles create and destroy requests.  When complete,
// all nodes have correct layout information.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::update(
  IpplParticleBase< ParticleInteractLayout<T,Dim,Mesh> >& PData,
  const ParticleAttrib<char>* canSwap) {

  unsigned N = Ippl::getNodes();
  unsigned myN = Ippl::myNode();
  unsigned LocalNum   = PData.getLocalNum();
  unsigned DestroyNum = PData.getDestroyNum();
  unsigned TotalNum;
  T maxrad = getMaxLocalInteractionRadius();
  int node;

  // delete particles in destroy list, update local num
  PData.performDestroy();
  LocalNum -= DestroyNum;

  // set up our layout, if not already done ... we could also do this if
  // we needed to expand our spatial region.
  if ( ! this->RLayout.initialized())
    rebuild_layout(LocalNum,PData);

  // apply boundary conditions to the particle positions
  if (getUpdateFlag(ParticleLayout<T,Dim>::BCONDS))
    apply_bconds(LocalNum, PData.R, this->getBConds(), this->RLayout.getDomain());

  // Now we can swap particles that have moved outside the region of
  // local field space.  This is done in several passes, one for each
  // spatial dimension.  The NodeCount values are updated by this routine.
  if (N > 1 && getUpdateFlag(this->SWAP)) {
    if (canSwap==0)
      LocalNum = swap_particles(LocalNum, PData);
    else
      LocalNum = swap_particles(LocalNum, PData, *canSwap);
  }

  // flag we need to update our ghost particles
  NeedGhostSwap = true;

  // Save how many local particles we have.
  TotalNum = this->NodeCount[myN] = LocalNum;

  // there is extra work to do if there are multipple nodes, to distribute
  // the particle layout data to all nodes
  if (N > 1) {
    // At this point, we can send our particle count updates to node 0, and
    // receive back the particle layout.
    int tag1 = Ippl::Comm->next_tag(P_SPATIAL_LAYOUT_TAG, P_LAYOUT_CYCLE);
    int tag2 = Ippl::Comm->next_tag(P_SPATIAL_RETURN_TAG, P_LAYOUT_CYCLE);
    if (myN != 0) {
      Message *msg = new Message;

      // put local particle count in the message
      msg->put(LocalNum);

      // also put in our maximum interaction radius
      msg->put(maxrad);

      // send this info to node 0
      Ippl::Comm->send(msg, 0, tag1);

      // receive back the number of particles on each node, and the maximum
      // interaction radius
      node = 0;
      msg = Ippl::Comm->receive_block(node, tag2);
      msg->get(this->NodeCount);
      msg->get(maxrad);
      msg->get(TotalNum);
    } else {			// do update tasks particular to node 0
      // receive messages from other nodes describing what they have
      int notrecvd = N - 1;	// do not need to receive from node 0
      TotalNum = LocalNum;
      while (notrecvd > 0) {
	// receive a message from another node.  After recv, node == sender.
	node = Communicate::COMM_ANY_NODE;
	Message *msg = Ippl::Comm->receive_block(node, tag1);
	int remNodeCount = 0;
	T remMaxRad = 0.0;
	msg->get(remNodeCount);
	msg->get(remMaxRad);
	delete msg;
	notrecvd--;

	// update values based on data from remote node
	TotalNum += remNodeCount;
	this->NodeCount[node] = remNodeCount;
	if (remMaxRad > maxrad)
	  maxrad = remMaxRad;
      }

      // send info back to all the client nodes
      Message *msg = new Message;
      msg->put(this->NodeCount, this->NodeCount + N);
      msg->put(maxrad);
      msg->put(TotalNum);
      Ippl::Comm->broadcast_others(msg, tag2);
    }
  }

  // update our particle number counts
  PData.setTotalNum(TotalNum);	// set the total atom count
  PData.setLocalNum(LocalNum);	// set the number of local atoms

  // if the interaction radius changed, must recalculate some things
  if (maxrad != getMaxInteractionRadius()) {
    setMaxInteractionRadius(maxrad);
    // check if we have any particle boundary conditions
    if (getUpdateFlag(ParticleLayout<T,Dim>::BCONDS)) {
      // check which boundaries, if any, are periodic
      ParticleBConds<T,Dim>& pBConds = this->getBConds();
      bool periodicBC[2*Dim];
      unsigned numPeriodicBC = 0;
      typename ParticleBConds<T,Dim>::ParticleBCond periodicBCond=ParticlePeriodicBCond;
      for (unsigned d=0; d<2*Dim; ++d) {
        periodicBC[d] = (pBConds[d] == periodicBCond);
        if (periodicBC[d]) ++numPeriodicBC;
      }
      if (numPeriodicBC>0) {
        // we need to reflect domains across all periodic boundaries
        // call specialized function
        rebuild_interaction_data(periodicBC);
      }
      else { // no periodic boundaries, call standard function
        rebuild_interaction_data();
      }
    }
    else { // no boundary conditions, call standard function
      rebuild_interaction_data();
    }
  }
  return;
}


/////////////////////////////////////////////////////////////////////
// copy particles to other nodes for pairlist computation.  The arguments
// are the current number of local particles, and the ParticleBase object.
// Make sure not to send any particles to, or receive particles from,
// nodes which have no particles on them.  This also takes care of
// building the pairlists.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::swap_ghost_particles(unsigned
							      LocalNum,
   IpplParticleBase< ParticleInteractLayout<T,Dim,Mesh> >& PData) {

  unsigned int i;			// loop variables

  // if we've already swapped particles since the last update, we're done
  if ( ! NeedGhostSwap ) return;

  // clear flag indicating we need to do this ghost particle swap again
  NeedGhostSwap = false;

  // delete all our current ghost particles; even if we have no local
  // particles now, we may have pairlists left over from earlier when we did
  // have local particles
  PData.ghostDestroy(PData.getGhostNum(), 0);

  // find the number of nodes we need to communicate with
  unsigned N = Ippl::getNodes();
  unsigned sendnum = 0;
  for (i=0; i < N; i++)
    if (InterNodeList[i] && this->NodeCount[i] > 0)
      sendnum++;

  // if there are no interaction nodes, we can just compute local pairlists
  // and then return
  if (sendnum == 0 || LocalNum == 0) {
    find_pairs(LocalNum, 0, LocalNum, true, PData);
    return;
  }

  // get the maximum interaction radius for the particles
  // we actually check twice the radius
  T interRad = 2.0 * getMaxInteractionRadius();

  // an NDRegion object used to store the interaction region of a particle
  NDRegion<T,Dim> pLoc;

  // Ghost particles are swapped in one pass: an interaction region for a
  // particle is created, and intersected with all the vnodes, and if the
  // particle needs to go to that vnode, it is sent.

  // create new messages to send to our neighbors
  for (i=0; i < N; i++)
    if (InterNodeList[i] && this->NodeCount[i] > 0)
      this->SwapMsgList[i] = new Message;

  // Go through the particles, find those with interaction radius
  // which overlaps with a neighboring left node, and copy into a message.
  // The interaction radius used to check for whether to send the particle
  // is (max inter. radius of system)*2.


  //  for (i=0; i < LocalNum; ++i) {

  // initialize the flags which indicate which node the particle will be
  // sent to

  // ada    memset((void *)SentToNodeList, 0, N * sizeof(bool));

  // get the position of the ith particle, and form an NDRegion which
  // is a cube with sides of length twice the interaction radius
  // ada    for (j=0; j < Dim; ++j)
  // ada      pLoc[j] = PRegion<T>(PData.R[i][j] - interRad,
  //			   PData.R[i][j] + interRad);

  // see which Vnodes this postion is in; if none, it is local
  // ada    typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN = this->RLayout.touch_range_rdv(pLoc);
  // ada   typename RegionLayout<T,Dim,Mesh>::touch_iterator_dv tVNit = touchingVN.first;
  // ada   for ( ; tVNit != touchingVN.second; ++tVNit) {
  // ada   Rnode<T,Dim> *tVN = (*tVNit).second;
  // ada  unsigned node = tVN->getNode();

  // the node has been found - copy particle data into a message,
  // ada  if (this->NodeCount[node] > 0 && ! SentToNodeList[node]) {
  // ada	if (! InterNodeList[node]) {
  // ada	  ERRORMSG("ParticleInteractLayout: Cannot send ghost " << i);
  // ada  ERRORMSG(" to node " << node << " ... skipping." << endl);
  // ada	}
  // ada        else {
  // ada	  PData.ghostPutMessage(*(this->SwapMsgList[node]), 1, i);
  // ada	  SentToNodeList[node] = true;
  //	}
  //      }
  //    }
  //  }

  // send out messages with ghost particles

  /*
     ada: buggy     BUGGY node hangs in later receive_block

  int tag = Ippl::Comm->next_tag(P_SPATIAL_GHOST_TAG, P_LAYOUT_CYCLE);
  for (i=0; i < N; ++i) {
    if (InterNodeList[i] && this->NodeCount[i] > 0) {
      // add a final 'zero' number of particles to indicate the end
      PData.ghostPutMessage(*(this->SwapMsgList[i]), (unsigned)0, (unsigned)0);

      // send the message
      Ippl::Comm->send(this->SwapMsgList[i], i, tag);
      }
    }
  */

  // while we're waiting for messages to arrive, calculate our local pairs
  find_pairs(LocalNum, 0, LocalNum, true, PData);

  // receive ghost particles from other nodes, and add them to our list

  /*
  while (sendnum-- > 0) {
    int node = Communicate::COMM_ANY_NODE;
    unsigned oldGN = PData.getGhostNum();
    Message *recmsg = Ippl::Comm->receive_block(node, tag);

    while (PData.ghostGetMessage(*recmsg, node) > 0);
    delete recmsg;

    // find pairs with these ghost particles
    find_pairs(LocalNum, LocalNum + oldGN, LocalNum + PData.getGhostNum(),
    false, PData);
	       }
  */

}


/////////////////////////////////////////////////////////////////////
// copy particles to other nodes for pairlist computation.  The arguments
// are the current number of local particles, and the IpplParticleBase object.
// Make sure not to send any particles to, or receive particles from,
// nodes which have no particles on them.  This also takes care of
// building the pairlists.
// special version to take care of periodic boundaries
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::swap_ghost_particles(
   unsigned LocalNum,
   IpplParticleBase< ParticleInteractLayout<T,Dim,Mesh> >& PData,
   const bool periodicBC[2*Dim])
{
  unsigned int i, j;			// loop variables
  unsigned d;

  // if we've already swapped particles since the last update, we're done
  if ( ! NeedGhostSwap ) return;

  // clear flag indicating we need to do this ghost particle swap again
  NeedGhostSwap = false;

  // delete all our current ghost particles; even if we have no local
  // particles now, we may have pairlists left over from earlier when we did
  // have local particles
  PData.ghostDestroy(PData.getGhostNum(), 0);

  // find the number of nodes we need to communicate with
  unsigned N = Ippl::getNodes();
  unsigned pe = Ippl::myNode();
  unsigned sendnum = 0;
  for (i=0; i < N; i++)
    if (InterNodeList[i] && this->NodeCount[i] > 0)
      sendnum++;

  // if there are no interaction nodes, we can just compute local pairlists
  // and then return
  if (sendnum == 0 || LocalNum == 0) {
    find_pairs(LocalNum, 0, LocalNum, true, PData);
    return;
  }

  // get the maximum interaction radius for the particles
  // we actually check twice the radius
  T interRad = 2.0 * getMaxInteractionRadius();

  // get domain info
  const NDRegion<T,Dim>& globalDom = this->RLayout.getDomain();

  // some stuff for computing reflected domains
  T offset[Dim];
  unsigned numRef;
  bool flipBit, activeBit[Dim], refBit[Dim];
  NDRegion<T,Dim> pLoc, refLoc;

  // region layout iterators
  typename RegionLayout<T,Dim,Mesh>::iterator_iv localVN, endLocalVN = this->RLayout.end_iv();
  typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN;
  typename RegionLayout<T,Dim,Mesh>::touch_iterator_dv tVNit;
  SingleParticlePos_t savePos;  // save position of reflected ghosts

  // Ghost particles are swapped in one pass: an interaction region for a
  // particle is created, and intersected with all the vnodes, and if the
  // particle needs to go to that vnode, it is sent.

  // create new messages to send to our neighbors
  for (i=0; i < N; i++)
    if (InterNodeList[i] && this->NodeCount[i] > 0)
      this->SwapMsgList[i] = new Message;

  // Go through the particles, find those with interaction radius
  // which overlaps with a neighboring left node, and copy into a message.
  // The interaction radius used to check for whether to send the particle
  // is (max inter. radius of system)*2.
  for (i=0; i < LocalNum; ++i) {

    // initialize flags indicating which nodes the particle has been sent to
    memset((void *)SentToNodeList, 0, N * sizeof(bool));

    // get the position of the ith particle, and form an NDRegion which
    // is a cube with sides of length twice the interaction radius
    for (j=0; j < (unsigned int) Dim; ++j)
      pLoc[j] = PRegion<T>(PData.R[i][j] - interRad,
			   PData.R[i][j] + interRad);

    // see which Vnodes this postion is in; if none, it is local
    touchingVN = this->RLayout.touch_range_rdv(pLoc);
    for (tVNit = touchingVN.first; tVNit != touchingVN.second; ++tVNit) {
      Rnode<T,Dim> *tVN = (*tVNit).second;
      unsigned node = tVN->getNode();

      // the node has been found - copy particle data into a message,
      if (this->NodeCount[node] > 0 && ! SentToNodeList[node]) {
	if (! InterNodeList[node]) {
	  ERRORMSG("ParticleInteractLayout: Cannot send ghost " << i);
	  ERRORMSG(" to node " << node << " ... skipping." << endl);
	}
        else {
	  PData.ghostPutMessage(*(this->SwapMsgList[node]), 1, i);
	  SentToNodeList[node] = true;
	}
      }
    }

    // look for boundary crossings and check reflected domains
    numRef = 0;
    for (d=0; d<Dim; ++d) {
      if (periodicBC[2*d] && pLoc[d].first()<globalDom[d].first()) {
        // crossed lower boundary
        offset[d] = globalDom[d].length();
        activeBit[d] = true;
        numRef = 2 * numRef + 1;
      }
      else if (periodicBC[2*d+1] && pLoc[d].last()>globalDom[d].last()) {
        // crossed upper boundary
        offset[d] = -globalDom[d].length();
        activeBit[d] = true;
        numRef = 2 * numRef + 1;
      }
      else {
        offset[d] = 0.0;
        activeBit[d] = false;
      }
      refBit[d] = false;  // reset bools indicating reflecting dims
    }

    if (numRef>0) savePos = PData.R[i];  // copy current particle position

    // loop over reflected domains
    for (j=0; j<numRef; ++j) {
      // set up reflected neighborhood and position
      refLoc = pLoc;
      PData.R[i] = savePos;
      // find next combination of dimension offsets
      d = 0;
      flipBit = false;
      while (d<Dim && !flipBit) {
        // first check if this dim is active
        if (activeBit[d]) {
          // now flip bit for this dimension
          if (refBit[d]) {
            // flip this bit off and proceed to next dim
            refBit[d] = false;
          }
          else { // refBit[d] is off
            // flip this bit on and indicate we're done
            refBit[d] = true;
            flipBit = true;
          }
        }
        ++d;
      }
      PAssert(flipBit);  // check that we found next combination

      // now offset the reflected neighborhood and particle position
      for (d=0; d<Dim; ++d) {
        if (refBit[d]) {
          refLoc[d] = refLoc[d] + offset[d];
          PData.R[i][d] = PData.R[i][d] + offset[d];
        }
      }

      // initialize flags indicating which nodes the particle has been sent to
      memset((void *)SentToNodeList, 0, N * sizeof(bool));

      // see which Vnodes this postion is in; if none, it is local
      touchingVN = this->RLayout.touch_range_rdv(refLoc);
      for (tVNit = touchingVN.first; tVNit != touchingVN.second; ++tVNit) {
        Rnode<T,Dim> *tVN = (*tVNit).second;
        unsigned node = tVN->getNode();

        // the node has been found - copy particle data into a message,
        if (this->NodeCount[node] > 0 && ! SentToNodeList[node]) {
	  if (! InterNodeList[node]) {
	    ERRORMSG("ParticleInteractLayout: Cannot send ghost " << i);
	    ERRORMSG(" to node " << node << " ... skipping." << endl);
	  }
          else {
	    PData.ghostPutMessage(*(this->SwapMsgList[node]), 1, i);
	    SentToNodeList[node] = true;
	  }
        }
      }

      if (InterNodeList[pe]) { // we may interact with local domains
        // for reflected domains, we also must check against local domains
        bool interact = false;
        localVN = this->RLayout.begin_iv();
        while (localVN != endLocalVN && !interact) {
          interact = refLoc.touches((*localVN).second->getDomain());
          ++localVN;
        }
        if (interact) {
          PData.ghostPutMessage(*(this->SwapMsgList[pe]), 1, i);
        }
      }
    }
    if (numRef>0) PData.R[i] = savePos;  // restore particle position data

  }

  // send out messages with ghost particles
  int tag = Ippl::Comm->next_tag(P_SPATIAL_GHOST_TAG, P_LAYOUT_CYCLE);
  for (i=0; i < N; ++i) {
    if (InterNodeList[i] && this->NodeCount[i] > 0) {
      // add a final 'zero' number of particles to indicate the end
      PData.ghostPutMessage(*(this->SwapMsgList[i]), (unsigned)0, (unsigned)0);

      // send the message
      Ippl::Comm->send(this->SwapMsgList[i], i, tag);
    }
  }

  // while we're waiting for messages to arrive, calculate our local pairs
  find_pairs(LocalNum, 0, LocalNum, true, PData);

  // receive ghost particles from other nodes, and add them to our list
  while (sendnum-- > 0) {
    int node = Communicate::COMM_ANY_NODE;
    unsigned oldGN = PData.getGhostNum();
    Message *recmsg = Ippl::Comm->receive_block(node, tag);

    while (PData.ghostGetMessage(*recmsg, node) > 0);
    delete recmsg;

    // find pairs with these ghost particles
    find_pairs(LocalNum, LocalNum + oldGN, LocalNum + PData.getGhostNum(),
	       false, PData);
  }
}


/////////////////////////////////////////////////////////////////////
// find the pairs between our local particles and particles a1 ... (a2 - 1).
// if the last argument is true, initialize all the pairlists to be empty.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::find_pairs(const unsigned LocalNum,
       const unsigned a1, const unsigned a2, const bool initLists,
       IpplParticleBase< ParticleInteractLayout<T,Dim,Mesh> >& PData) {

  unsigned i, j;			// loop variables

  // initialize the pairlist storage if requested
  if (initLists) {
    unsigned vlen = PairList.size();
    if (vlen > LocalNum)
      vlen = LocalNum;
    for (i=0; i < vlen; ++i)
      PairList[i]->erase(PairList[i]->begin(), PairList[i]->end());

    // make sure there are enough single particle pairlists
    if (PairList.size() < LocalNum) {
      int newamt = LocalNum - PairList.size();
      PairList.reserve(newamt);
      for (int k=0; k < newamt; ++k)
	PairList.push_back(new std::vector<pair_t>);
    }
  }

  // make sure we have something to do
  if (a2 <= a1) return;

  // find pairs between local particles and particles a1 ... a2
  for (i=0; i < LocalNum; ++i) {
    // get interaction radius of this particle
    T intrad1 = getInteractionRadius(i);

    // find starting index of inner loop
    j = (a1 > i ? a1 : i + 1);

    // do inner loop for particles up to the last local one
    // (these pairs must be stored twice)
    for (; j < LocalNum; ++j) {
      // add interaction radius of this particle
      T intrad2 = intrad1 + getInteractionRadius(j);
      intrad2 *= intrad2;	// (intrad1 + intrad2)^2

      // find distance^2 between these two particles
      Vektor<T,Dim> rsep = PData.R[j];
      rsep -= PData.R[i];
      T sep2 = dot(rsep, rsep);

      // if the separation is less than their interaction distance, book it
      // we store the pair twice, since we know both i and j are < LocalNum
      if (sep2 < intrad2) {
	PairList[i]->push_back(pair_t(j, sep2));
	PairList[j]->push_back(pair_t(i, sep2));
      }
    }

    // now do rest of loop for just ghost particles (only store the
    // pair once in this case)
    for (; j < a2; ++j) {
      // get interaction radius of this particle
      T intrad2 = intrad1 + getInteractionRadius(j);
      intrad2 *= intrad2;	// (intrad1 + intrad2)^2

      // find distance^2 between these two particles
      Vektor<T,Dim> rsep = PData.R[j];
      rsep -= PData.R[i];
      T sep2 = dot(rsep, rsep);

      // if the separation is less than their interaction distance, book it
      // we only store the pair for the local atom i once, since the other
      // atom j is a ghost atom
      if (sep2 < intrad2) {
	PairList[i]->push_back(pair_t(j, sep2));
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////
// print it out
template < class T, unsigned Dim, class Mesh >
std::ostream& operator<<(std::ostream& out, const ParticleInteractLayout<T,Dim,Mesh>& L) {

  out << "ParticleInteractLayout, with particle distribution:\n    ";
  for (unsigned int i=0; i < (unsigned int) Ippl::getNodes(); ++i)
    out << L.getNodeCount(i) << "  ";
  out << "\nInteractLayout decomposition = " << L.getLayout();
  return out;
}


//////////////////////////////////////////////////////////////////////
// Repartition onto a new layout, if the layout changes ... this is a
// virtual function called by a UserList, as opposed to the RepartitionLayout
// function used by the particle load balancing mechanism.
template < class T, unsigned Dim, class Mesh >
void ParticleInteractLayout<T,Dim,Mesh>::Repartition(UserList* userlist) {

  // perform actions to restructure our data due to a change in the
  // RegionLayout
  if (userlist->getUserListID() == this->RLayout.get_Id()) {
    // clear out current interaction node storage; if the next update
    // indicates we have a non-zero interaction radius, this info will be
    // rebuilt (by calling rebuild_interaction_data)
    InteractionNodes = 0;
    setMaxInteractionRadius(0);
    NeedGhostSwap = true;
  }
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
