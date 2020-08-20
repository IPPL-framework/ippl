// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef PARTICLE_SPATIAL_LAYOUT_H
#define PARTICLE_SPATIAL_LAYOUT_H

/*
 * ParticleSpatialLayout - particle layout based on spatial decomposition.
 *
 * This is a specialized version of ParticleLayout, which places particles
 * on processors based on their spatial location relative to a fixed grid.
 * In particular, this can maintain particles on processors based on a
 * specified FieldLayout or RegionLayout, so that particles are always on
 * the same node as the node containing the Field region to which they are
 * local.  This may also be used if there is no associated Field at all,
 * in which case a grid is selected based on an even distribution of
 * particles among processors.
 */

// include files
#include "Particle/ParticleLayout.h"
#include "Particle/IpplParticleBase.h"
#include "Region/RegionLayout.h"
#include "Message/Message.h"
#include "FieldLayout/FieldLayoutUser.h"
#include "Utility/IpplException.h"

#include <cstddef>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "BoxParticleCachingPolicy.h"

#include "Message/Formatter.h"
#include <mpi.h>

// forward declarations
class UserList;
template <class T> class ParticleAttrib;
template <unsigned Dim, class T> class UniformCartesian;
template <class T, unsigned Dim, class Mesh, class CachingPolicy> class ParticleSpatialLayout;
template <class T, unsigned Dim, class Mesh, class CachingPolicy>
std::ostream& operator<<(std::ostream&, const ParticleSpatialLayout<T,Dim,Mesh,CachingPolicy>&);

// ParticleSpatialLayout class definition.  Template parameters are the type
// and dimension of the ParticlePos object used for the particles.  The
// dimension of the position must match the dimension of the FieldLayout
// object used in this particle layout, if any.
// Optional template parameter for the mesh type
template < class T, unsigned Dim, class Mesh=UniformCartesian<Dim,T>, class CachingPolicy=BoxParticleCachingPolicy<T,Dim,Mesh > >
class ParticleSpatialLayout : public ParticleLayout<T, Dim>,
        public FieldLayoutUser, public CachingPolicy
{
    /*
      Enable cashing of particles. The size is
      given in multiples of the mesh size in each dimension.
     */

public:
    // pair iterator definition ... this layout does not allow for pairlists
    typedef int pair_t;
    typedef pair_t* pair_iterator;
    typedef typename ParticleLayout<T, Dim>::SingleParticlePos_t
    SingleParticlePos_t;
    typedef typename ParticleLayout<T, Dim>::Index_t Index_t;

    // type of attributes this layout should use for position and ID
    typedef ParticleAttrib<SingleParticlePos_t> ParticlePos_t;
    typedef ParticleAttrib<Index_t>             ParticleIndex_t;
    typedef RegionLayout<T,Dim,Mesh>			  RegionLayout_t;

public:
    // constructor: The Field layout to which we match our particle's
    // locations.
    ParticleSpatialLayout(FieldLayout<Dim>&);

    // constructor: this one also takes a Mesh
    ParticleSpatialLayout(FieldLayout<Dim>&, Mesh&);

    // a similar constructor, but this one takes a RegionLayout.
    ParticleSpatialLayout(const RegionLayout<T,Dim,Mesh>&);

    // a default constructor ... in this case, no layout will
    // be assumed by this class.  A layout may be given later via the
    // 'setLayout' method, either as a FieldLayout or as a RegionLayout.
    ParticleSpatialLayout();

    // destructor
    ~ParticleSpatialLayout();

    //
    // spatial decomposition layout information
    //

    // retrieve a reference to the FieldLayout object in use.  This may be used,
    // e.g., to construct a Field with the same layout as the Particles.  Note
    // that if this object was constructed by providing a RegionLayout in the
    // constructor, then this generated FieldLayout will not necessarily match
    // up with the Region (it will be offset by some amount).  But, if this
    // object was either 1) created with a FieldLayout to begin with, or 2)
    // created with no layout, and one was generated internally, then the
    // returned FieldLayout will match and can be used to make new Fields or
    // Particles.
    FieldLayout<Dim>& getFieldLayout()
    {
        return RLayout.getFieldLayout();
    }

    // retrieve a reference to the RegionLayout object in use
    RegionLayout<T,Dim,Mesh>& getLayout()
    {
        return RLayout;
    }
    const RegionLayout<T,Dim,Mesh>& getLayout() const
    {
        return RLayout;
    }

    // get number of particles on a physical node
    int getNodeCount(unsigned i) const
    {
        PAssert_LT(i, (unsigned int) Ippl::getNodes());
        return NodeCount[i];
    }

    // get flag for empty node domain
    bool getEmptyNode(unsigned i) const
    {
        PAssert_LT(i, (unsigned int) Ippl::getNodes());
        return EmptyNode[i];
    }

    //
    // Particle swapping/update routines
    //

    // Update the location and indices of all atoms in the given IpplParticleBase
    // object.  This handles swapping particles among processors if
    // needed, and handles create and destroy requests.  When complete,
    // all nodes have correct layout information.
    void update(IpplParticleBase< ParticleSpatialLayout<T,Dim,Mesh,CachingPolicy> >& p,
                const ParticleAttrib<char>* canSwap=0);


    //
    // I/O
    //

    // Print out information for debugging purposes.
    void printDebug(Inform&);

    //
    // virtual functions for FieldLayoutUser's (and other UserList users)
    //

    // Repartition onto a new layout
    virtual void Repartition(UserList *);

    // Tell this object that an object is being deleted
    virtual void notifyUserOfDelete(UserList *);

    void enableCaching() { caching = true; }
    void disableCaching() { caching = false; }

protected:
    // The RegionLayout which determines where our particles go.
    RegionLayout<T,Dim,Mesh> RLayout;

    // The number of particles located on each physical node.
    size_t *NodeCount;

    // Flag for which nodes have no local domain
    bool* EmptyNode;

    // a list of Message pointers used in swapping particles, and flags
    // for which nodes expect messages in each dimension
    bool* SwapNodeList[Dim];
    Message** SwapMsgList;
    unsigned NeighborNodes[Dim];
    std::vector<size_t>* PutList;

	bool caching;

    // perform common constructor tasks
    void setup();

    /////////////////////////////////////////////////////////////////////
    // Rebuild the RegionLayout entirely, by recalculating our min and max
    // domains, adding a buffer region, and then giving this new Domain to
    // our internal RegionLayout.  When this is done, we must rebuild all
    // our other data structures as well.
    template < class PB >
    void rebuild_layout(size_t haveLocal, PB& PData)
    {
        size_t i;
        unsigned d;			// loop variables

        //~ Inform dbgmsg("SpatialLayout::rebuild_layout", INFORM_ALL_NODES);
        //~ dbgmsg << "rebuild..." << endl;
        SingleParticlePos_t minpos = 0;
        SingleParticlePos_t maxpos = 0;
        int tag  = Ippl::Comm->next_tag(P_SPATIAL_RANGE_TAG, P_LAYOUT_CYCLE);
        int btag = Ippl::Comm->next_tag(P_SPATIAL_RANGE_TAG, P_LAYOUT_CYCLE);

        // if we have local particles, then find the min and max positions
        if (haveLocal > 0)
        {
            minpos = PData.R[0];
            maxpos = PData.R[0];
            for (i=1; i < haveLocal; ++i)
            {
                for (d=0; d < Dim; ++d)
                {
                    if (PData.R[i][d] < minpos[d])
                        minpos[d] = PData.R[i][d];
                    if (PData.R[i][d] > maxpos[d])
                        maxpos[d] = PData.R[i][d];
                }
            }
        }

        // if we're not on node 0, send data to node 0
        if (Ippl::myNode() != 0)
        {
            Message *msg = new Message;
            msg->put(haveLocal);
            if (haveLocal > 0)
            {
                minpos.putMessage(*msg);
                maxpos.putMessage(*msg);
            }
            Ippl::Comm->send(msg, 0, tag);

            // now receive back min and max range as provided by the master node.
            // These will include some buffer region, and will be integral values,
            // so we can make a FieldLayout and use it to initialize the RegionLayout.
            int node = 0;
            msg = Ippl::Comm->receive_block(node, btag);
            minpos.getMessage(*msg);
            maxpos.getMessage(*msg);
            delete msg;

        }
        else  			// on node 0, collect data and compute region
        {
            SingleParticlePos_t tmpminpos;
            SingleParticlePos_t tmpmaxpos;
            size_t tmphaveLocal = 0;
            unsigned unreceived = Ippl::getNodes() - 1;

            // collect data from other nodes
            while (unreceived > 0)
            {
                int node = COMM_ANY_NODE;
                Message *msg = Ippl::Comm->receive_block(node, tag);
                msg->get(tmphaveLocal);
                if (tmphaveLocal > 0)
                {
                    tmpminpos.getMessage(*msg);
                    tmpmaxpos.getMessage(*msg);
                    for (i=0; i < Dim; ++i)
                    {
                        if (tmpminpos[i] < minpos[i])
                            minpos[i] = tmpminpos[i];
                        if (tmpmaxpos[i] > maxpos[i])
                            maxpos[i] = tmpmaxpos[i];
                    }
                }
                delete msg;
                unreceived--;
            }

            // adjust min and max to include a buffer region and fall on integral
            // values
            SingleParticlePos_t extrapos = (maxpos - minpos) * ((T)0.125);
            maxpos += extrapos;
            minpos -= extrapos;
            for (i=0; i < Dim; ++i)
            {
                if (minpos[i] >= 0.0)
                    minpos[i] = (int)(minpos[i]);
                else
                    minpos[i] = (int)(minpos[i] - 1);
                maxpos[i] = (int)(maxpos[i] + 1);
            }

            // send these values out to the other nodes
            if (Ippl::getNodes() > 1)
            {
                Message *bmsg = new Message;
                minpos.putMessage(*bmsg);
                maxpos.putMessage(*bmsg);
                Ippl::Comm->broadcast_others(bmsg, btag);
            }
        }

        // determine the size of the new domain, and the number of blocks into
        // which it should be broken
        NDIndex<Dim> range;
        for (i=0; i < Dim; ++i)
            range[i] = Index((int)(minpos[i]), (int)(maxpos[i]));
        int vn = -1;
        if (RLayout.initialized())
            vn = RLayout.size_iv() + RLayout.size_rdv();

        // ask the RegionLayout to change the paritioning to match this size
        // and block count.  This will eventually end up by calling Repartition
        // here, which will lead to rebuilding the neighbor data, etc., so we
        // are done.
        RLayout.changeDomain(range, vn);
    }

    // swap particles to neighboring nodes if they have moved too far
    // PB is the type of IpplParticleBase which should have it's layout rebuilt.
    //mwerks  template<class PB>
    //mwerks  unsigned swap_particles(unsigned, PB&);
    /////////////////////////////////////////////////////////////////////
    // go through all our local particles, and send particles which must
    // be swapped to another node to that node.
    template < class PB >
    size_t swap_particles(size_t LocalNum, PB& PData)
    {

//~ Inform dbgmsg("SpatialLayout::swap_particles", INFORM_ALL_NODES);
        //~ dbgmsg << "swap..." << endl;

        Inform msg("ParticleSpatialLayout ERROR ", INFORM_ALL_NODES);

        unsigned d, i, j;			// loop variables
        size_t ip;
        unsigned N = Ippl::getNodes();
        unsigned myN = Ippl::myNode();

        // iterators used to search local domains
        typename RegionLayout<T,Dim,Mesh>::iterator_iv localV, localEnd = RLayout.end_iv();

        // iterators used to search remote domains
        typename RegionLayout<T,Dim,Mesh>::iterator_dv remoteV; //  remoteEnd = RLayout.end_rdv();

        // JCC: This "nudge factor" stuff was added when we were experiencing
        // problems with particles getting lost in between PRegions on
        // neighboring nodes.  This problem has since been resolved by
        // fixing the way in which PRegion boundaries are computed, so I am
        // commenting this out for now.  We can bring it back later if the
        // need arises.

        /*

        // Calculate a 'nudge factor', an amount that can get added to a
        // particle position to determine where it should be located.  The nudge
        // factor equals 1/100th the smallest width of the rnodes in each dimension.
        // When we try to find where a particle is located, we check what vnode
        // contains this particle 'nudge region', a box around the particle's pos
        // of the size of the nudge factor.
        T pNudge[Dim];
        for (d=0; d < Dim; ++d) {
          // initialize to the first rnode's width
          T minval = (*(RLayout.begin_iv())).second->getDomain()[d].length();

          // check the local rnodes
          for (localV = RLayout.begin_iv(); localV != localEnd; ++localV) {
        T checkval = (*localV).second->getDomain()[d].length();
        if (checkval < minval)
          minval = checkval;
          }

          // check the remote rnodes
          for (remoteV = RLayout.begin_rdv(); remoteV != remoteEnd; ++remoteV) {
        T checkval = (*remoteV).second->getDomain()[d].length();
        if (checkval < minval)
          minval = checkval;
          }

          // now rescale the minval, and save it
          pNudge[d] = 0.00001 * minval;
        }

        */

        // An NDRegion object used to store a particle position.
        NDRegion<T,Dim> pLoc;

        // get new message tag for particle exchange with empty domains
        int etag = Ippl::Comm->next_tag(P_SPATIAL_RETURN_TAG,P_LAYOUT_CYCLE);

        if (!getEmptyNode(myN))
        {

            // Particles are swapped in multipple passes, one for each dimension.
            // The tasks completed here for each dimension are the following:
            //   1. For each local Vnode, find the remote Vnodes which exist along
            //      same axis as the current axis (i.e. all Vnodes along the x-axis).
            //   2. From this list, determine which nodes we send messages to.
            //   3. Go through all the particles, finding those which have moved to
            //      an off-processor vnode, and store index in an array for that node
            //   4. Send off the particles to the nodes (if no particles are
            //      going to a node, send them a message with 0 in it)
            //   5. Delete the send particles from our local list
            //   6. Receive particles sent to us by other nodes (some messages may
            //      say that we're receiving 0 particles from that node).

            // Initialize NDRegion with a position inside the first Vnode.
            // We can skip dim 0, since it will be filled below.
            for (d = 1; d < Dim; ++d)
            {
                T first = (*(RLayout.begin_iv())).second->getDomain()[d].first();
                T last  = (*(RLayout.begin_iv())).second->getDomain()[d].last();
                T mid   = first + 0.5 * (last - first);
                pLoc[d] = PRegion<T>(mid, mid);
            }

            for (d = 0; d < Dim; ++d)
            {

                // get new message tag for particle exchange along this dimension
                int tag = Ippl::Comm->next_tag(P_SPATIAL_TRANSFER_TAG,P_LAYOUT_CYCLE);

                // we only need to do the rest if there are other nodes in this dim
                if (NeighborNodes[d] > 0)
                {
                    // create new messages to send to our neighbors
                    for (i = 0; i < N; i++)
                        if (SwapNodeList[d][i])
                            SwapMsgList[i] = new Message;

                    // Go through the particles and find those moving in the current dir.
                    // When one is found, copy it into outgoing message and delete it.
                    for (ip=0; ip<LocalNum; ++ip)
                    {
                        // get the position of particle ip, and find the closest grid pnt
                        // for just the dimensions 0 ... d
                        for (j = 0; j <= d; j++)
                            pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

                        // first check local domains (in this dimension)
                        bool foundit = false;
                        // JCC:	  int nudged = 0;
                        while (!foundit)
                        {
                            for (localV = RLayout.begin_iv();
                                    localV != localEnd && !foundit; ++localV)
                            {
                                foundit= (((*localV).second)->getDomain())[d].touches(pLoc[d]);
                            }

                            // if not found, it might be remote
                            if (!foundit)
                            {
                                // see which Vnode this postion is in
                                typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
                                    RLayout.touch_range_rdv(pLoc);

                                // make sure we have a vnode to send it to
                                if (touchingVN.first == touchingVN.second)
                                {
                                    // JCC:		if (nudged >= Dim) {
                                    ERRORMSG("Local particle " << ip << " with ID=");
                                    ERRORMSG(PData.ID[ip] << " at ");
                                    ERRORMSG(PData.R[ip] << " is outside of global domain ");
                                    ERRORMSG(RLayout.getDomain() << endl);
                                    ERRORMSG("This occurred when searching for point " << pLoc);
                                    ERRORMSG(" in RegionLayout = " << RLayout << endl);
                                    Ippl::abort();
                                }
                                else
                                {

				    // the node has been found - add index to put list
				    unsigned node = (*(touchingVN.first)).second->getNode();
				    PAssert_EQ(SwapNodeList[d][node], true);
				    PutList[node].push_back(ip);

				    // .. and then add to DestroyList
				    PData.destroy(1, ip);

				    // indicate we found it to quit this check
				    foundit = true;
                                }
                            }
                        }
                    }

                    // send the particles to their destination nodes
                    for (i = 0; i < N; i++)
                    {
                        if (SwapNodeList[d][i])
                        {
                            // put data for particles on this put list into message
                            PData.putMessage( *(SwapMsgList[i]), PutList[i] );

                            // add a final 'zero' number of particles to indicate the end
                            PData.putMessage(*(SwapMsgList[i]), (size_t) 0, (size_t) 0);

                            // send the message
                            // Inform dbgmsg("SpatialLayout", INFORM_ALL_NODES);
                            //dbgmsg << "Swapping "<<PutList[i].size() << " particles to node ";
                            //dbgmsg << i<<" with tag " << tag << " (" << 'x' + d << ")" << endl;
                            //dbgmsg << "  ... msg = " << *(SwapMsgList[i]) << endl;
                            int node = i;
                            Ippl::Comm->send(SwapMsgList[i], node, tag);

                            // clear the list
                            PutList[i].erase(PutList[i].begin(), PutList[i].end());
                        }
                    }

                    LocalNum -= PData.getDestroyNum();  // update local num
                    ADDIPPLSTAT(incParticlesSwapped, PData.getDestroyNum());
                    PData.performDestroy();

                    // receive particles from neighbor nodes, and add them to our list
                    unsigned sendnum = NeighborNodes[d];
                    while (sendnum-- > 0)
                    {
                        int node = Communicate::COMM_ANY_NODE;
                        Message *recmsg = Ippl::Comm->receive_block(node, tag);
                        size_t recvd;
                        while ((recvd = PData.getMessage(*recmsg)) > 0)
                            LocalNum += recvd;
                        delete recmsg;
                    }
                }  // end if (NeighborNodes[d] > 0)

                if (d == 0)
                {
                    // receive messages from any empty nodes
                    for (i = 0; i < N; ++i)
                    {
                        if (getEmptyNode(i))
                        {
                            int node = i;
                            Message *recmsg = Ippl::Comm->receive_block(node, etag);
                            size_t recvd;
                            while ((recvd = PData.getMessage(*recmsg)) > 0)
                                LocalNum += recvd;
                            delete recmsg;
                        }
                    }
                }

            }  // end for (d=0; d<Dim; ++d)

        }
        else   // empty node sends, but does not receive
        {
            msg << "case getEmptyNode(myN) " << endl;
            // create new messages to send to our neighbors along dim 0
            for (i = 0; i < N; i++)
                if (SwapNodeList[0][i])
                    SwapMsgList[i] = new Message;

            // Go through the particles and find those moving to other nodes.
            // When one is found, copy it into outgoing message and delete it.
            for (ip=0; ip<LocalNum; ++ip)
            {
                // get the position of particle ip, and find the closest grid pnt
                for (j = 0; j < Dim; j++)
                    pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

                // see which remote Vnode this postion is in
                typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
                    RLayout.touch_range_rdv(pLoc);

                // make sure we have a vnode to send it to
                if (touchingVN.first == touchingVN.second)
                {
                    ERRORMSG("Local particle " << ip << " with ID=");
                    ERRORMSG(PData.ID[ip] << " at ");
                    ERRORMSG(PData.R[ip] << " is outside of global domain ");
                    ERRORMSG(RLayout.getDomain() << endl);
                    ERRORMSG("This occurred when searching for point " << pLoc);
                    ERRORMSG(" in RegionLayout = " << RLayout << endl);
                    Ippl::abort();
                }
                else
                {
		    // the node has been found - add index to put list
		    unsigned node = (*(touchingVN.first)).second->getNode();
		    PAssert_EQ(SwapNodeList[0][node], true);
		    PutList[node].push_back(ip);

		    // .. and then add to DestroyList
		    PData.destroy(1, ip);
                }
            }

            // send the particles to their destination nodes
            for (i = 0; i < N; i++)
            {
                if (SwapNodeList[0][i])
                {
                    // put data for particles on this put list into message
                    PData.putMessage( *(SwapMsgList[i]), PutList[i] );

                    // add a final 'zero' number of particles to indicate the end
                    PData.putMessage(*(SwapMsgList[i]), (size_t) 0, (size_t) 0);

                    // send the message
                    int node = i;
                    Ippl::Comm->send(SwapMsgList[i], node, etag);

                    // clear the list
                    PutList[i].erase(PutList[i].begin(), PutList[i].end());
                }
            }

            LocalNum -= PData.getDestroyNum();  // update local num
            ADDIPPLSTAT(incParticlesSwapped, PData.getDestroyNum());
            PData.performDestroy();

        }

        // return how many particles we have now
        return LocalNum;
    }


/*
 * Simplified version for testing purposes.
 */

   template < class PB >
    size_t short_swap_particles(size_t LocalNum, PB& PData)
    {
    	static int sent = 0, old_sent=0;


        unsigned d, i, j;			// loop variables
        size_t ip;
        unsigned N = Ippl::getNodes();

        // iterators used to search local domains
        typename RegionLayout<T,Dim,Mesh>::iterator_iv localV, localEnd = RLayout.end_iv();

        // iterators used to search remote domains
        typename RegionLayout<T,Dim,Mesh>::iterator_dv remoteV; // remoteEnd = RLayout.end_rdv();


        // An NDRegion object used to store a particle position.
        NDRegion<T,Dim> pLoc;

            // Initialize NDRegion with a position inside the first Vnode.
            // We can skip dim 0, since it will be filled below.
            for (d = 1; d < Dim; ++d)
            {
                T first = (*(RLayout.begin_iv())).second->getDomain()[d].first();
                T last  = (*(RLayout.begin_iv())).second->getDomain()[d].last();
                T mid   = first + 0.5 * (last - first);
                pLoc[d] = PRegion<T>(mid, mid);
            }

            for (d = 0; d < Dim; ++d)
            {

                // get new message tag for particle exchange along this dimension
                int tag = Ippl::Comm->next_tag(P_SPATIAL_TRANSFER_TAG,P_LAYOUT_CYCLE);

                // we only need to do the rest if there are other nodes in this dim
                if (NeighborNodes[d] > 0)
                {
                    // create new messages to send to our neighbors
                    for (i = 0; i < N; i++)
                        if (SwapNodeList[d][i])
                            SwapMsgList[i] = new Message;

                    // Go through the particles and find those moving in the current dir.
                    // When one is found, copy it into outgoing message and delete it.
                    for (ip=0; ip<LocalNum; ++ip)
                    {
                        // get the position of particle ip, and find the closest grid pnt
                        // for just the dimensions 0 ... d
                        for (j = 0; j <= d; j++)
                            pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

                        // first check local domains (in this dimension)
                        bool foundit = false;

                            for (localV = RLayout.begin_iv();
                                    localV != localEnd && !foundit; ++localV)
                            {
                                foundit= (((*localV).second)->getDomain())[d].touches(pLoc[d]);
                            }

                            // if not found, it might be remote
                            if (!foundit)
                            {
                                // see which Vnode this postion is in
                                typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
                                    RLayout.touch_range_rdv(pLoc);


                                    // the node has been found - add index to put list
                                    unsigned node = (*(touchingVN.first)).second->getNode();
                                    PAssert_EQ(SwapNodeList[d][node], true);
                                    PutList[node].push_back(ip);

                                    // .. and then add to DestroyList
                                    PData.destroy(1, ip);

                                    // indicate we found it to quit this check
                                    foundit = true;
                                    sent++;
                            }
                    }

					std::vector<MPI_Request> requests;
					std::vector<MsgBuffer*> buffers;

	                // send the particles to their destination nodes
                    for (i = 0; i < N; i++)
                    {
                        if (SwapNodeList[d][i])
                        {

                            // put data for particles on this put list into message
                            PData.putMessage( *(SwapMsgList[i]), PutList[i] );

                            // add a final 'zero' number of particles to indicate the end
                            PData.putMessage(*(SwapMsgList[i]), (size_t) 0, (size_t) 0);

                            int node = i;
                            Ippl::Comm->send(SwapMsgList[i], node, tag);

                            // clear the list
                            PutList[i].erase(PutList[i].begin(), PutList[i].end());



                        }
                    }

                    LocalNum -= PData.getDestroyNum();  // update local num
                    ADDIPPLSTAT(incParticlesSwapped, PData.getDestroyNum());
                    PData.performDestroy();

                    // receive particles from neighbor nodes, and add them to our list
                    unsigned sendnum = NeighborNodes[d];
                    while (sendnum-- > 0)
                    {
                        int node = Communicate::COMM_ANY_NODE;
                        Message *recmsg = Ippl::Comm->receive_block(node, tag);
                        size_t recvd;
                        while ((recvd = PData.getMessage(*recmsg)) > 0)
                            LocalNum += recvd;
                        delete recmsg;
                    }

                }  // end if (NeighborNodes[d] > 0)

            }  // end for (d=0; d<Dim; ++d)

		//std::cout << "node " << Ippl::myNode() << " sent particles " << sent - old_sent << std::endl;
		old_sent = sent;

        // return how many particles we have now
        return LocalNum;
    }





    // PB is the type of IpplParticleBase which should have it's layout rebuilt.
    //mwerks  template<class PB>
    //mwerks  unsigned swap_particles(unsigned, PB&, const ParticleAttrib<char>&);
    /////////////////////////////////////////////////////////////////////
    // go through all our local particles, and send particles which must
    // be swapped to another node to that node.
    template < class PB >
    size_t swap_particles(size_t LocalNum, PB& PData,
                          const ParticleAttrib<char>& canSwap)
    {

        unsigned d, i, j;			// loop variables
        size_t ip;
        unsigned N = Ippl::getNodes();
        unsigned myN = Ippl::myNode();

        // iterators used to search local domains
        typename RegionLayout<T,Dim,Mesh>::iterator_iv localV, localEnd = RLayout.end_iv();

        // iterators used to search remote domains
        typename RegionLayout<T,Dim,Mesh>::iterator_dv remoteV; // remoteEnd = RLayout.end_rdv();

        // JCC: This "nudge factor" stuff was added when we were experiencing
        // problems with particles getting lost in between PRegions on
        // neighboring nodes.  This problem has since been resolved by
        // fixing the way in which PRegion boundaries are computed, so I am
        // commenting this out for now.  We can bring it back later if the
        // need arises.

        /*

        // Calculate a 'nudge factor', an amount that can get added to a
        // particle position to determine where it should be located.  The nudge
        // factor equals 1/100th the smallest width of the rnodes in each dimension.
        // When we try to find where a particle is located, we check what vnode
        // contains this particle 'nudge region', a box around the particle's pos
        // of the size of the nudge factor.
        T pNudge[Dim];
        for (d=0; d < Dim; ++d) {
          // initialize to the first rnode's width
          T minval = (*(RLayout.begin_iv())).second->getDomain()[d].length();

          // check the local rnodes
          for (localV = RLayout.begin_iv(); localV != localEnd; ++localV) {
        T checkval = (*localV).second->getDomain()[d].length();
        if (checkval < minval)
          minval = checkval;
          }

          // check the remote rnodes
          for (remoteV = RLayout.begin_rdv(); remoteV != remoteEnd; ++remoteV) {
        T checkval = (*remoteV).second->getDomain()[d].length();
        if (checkval < minval)
          minval = checkval;
          }

          // now rescale the minval, and save it
          pNudge[d] = 0.00001 * minval;
        }

        */

        // An NDRegion object used to store a particle position.
        NDRegion<T,Dim> pLoc;

        // get new message tag for particle exchange with empty domains
        int etag = Ippl::Comm->next_tag(P_SPATIAL_RETURN_TAG,P_LAYOUT_CYCLE);

        if (!getEmptyNode(myN))
        {

            // Particles are swapped in multipple passes, one for each dimension.
            // The tasks completed here for each dimension are the following:
            //   1. For each local Vnode, find the remote Vnodes which exist along
            //      same axis as the current axis (i.e. all Vnodes along the x-axis).
            //   2. From this list, determine which nodes we send messages to.
            //   3. Go through all the particles, finding those which have moved to
            //      an off-processor vnode, and store index in an array for that node
            //   4. Send off the particles to the nodes (if no particles are
            //      going to a node, send them a message with 0 in it)
            //   5. Delete the send particles from our local list
            //   6. Receive particles sent to us by other nodes (some messages may
            //      say that we're receiving 0 particles from that node).

            // Initialize NDRegion with a position inside the first Vnode.
            // We can skip dim 0, since it will be filled below.
            for (d = 1; d < Dim; ++d)
            {
                T first = (*(RLayout.begin_iv())).second->getDomain()[d].first();
                T last  = (*(RLayout.begin_iv())).second->getDomain()[d].last();
                T mid   = first + 0.5 * (last - first);
                pLoc[d] = PRegion<T>(mid, mid);
            }

            for (d = 0; d < Dim; ++d)
            {

                // get new message tag for particle exchange along this dimension
                int tag = Ippl::Comm->next_tag(P_SPATIAL_TRANSFER_TAG,P_LAYOUT_CYCLE);

                // we only need to do the rest if there are other nodes in this dim
                if (NeighborNodes[d] > 0)
                {
                    // create new messages to send to our neighbors
                    for (i = 0; i < N; i++)
                        if (SwapNodeList[d][i])
                            SwapMsgList[i] = new Message;

                    // Go through the particles and find those moving in the current dir.
                    // When one is found, copy it into outgoing message and delete it.
                    for (ip=0; ip<LocalNum; ++ip)
                    {
                        if (!bool(canSwap[ip])) continue;  // skip if can't swap
                        // get the position of particle ip, and find the closest grid pnt
                        // for just the dimensions 0 ... d
                        for (j = 0; j <= d; j++)
                            pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

                        // first check local domains (in this dimension)
                        bool foundit = false;
                        // JCC:	  int nudged = 0;
                        while (!foundit)
                        {
                            for (localV = RLayout.begin_iv();
                                    localV != localEnd && !foundit; ++localV)
                            {
                                foundit= (((*localV).second)->getDomain())[d].touches(pLoc[d]);
                            }

                            // if not found, it might be remote
                            if (!foundit)
                            {
                                // see which Vnode this postion is in
                                typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
                                    RLayout.touch_range_rdv(pLoc);

                                // make sure we have a vnode to send it to
                                if (touchingVN.first == touchingVN.second)
                                {
                                    // JCC:		if (nudged >= Dim) {
                                    ERRORMSG("Local particle " << ip << " with ID=");
                                    ERRORMSG(PData.ID[ip] << " at ");
                                    ERRORMSG(PData.R[ip] << " is outside of global domain ");
                                    ERRORMSG(RLayout.getDomain() << endl);
                                    ERRORMSG("This occurred when searching for point " << pLoc);
                                    ERRORMSG(" in RegionLayout = " << RLayout << endl);
                                    Ippl::abort();
                                }
                                else
                                {
                                    // the node has been found - add index to put list
                                    unsigned node = (*(touchingVN.first)).second->getNode();
                                    PAssert_EQ(SwapNodeList[d][node], true);
                                    PutList[node].push_back(ip);

                                    // .. and then add to DestroyList
                                    PData.destroy(1, ip);

                                    // indicate we found it to quit this check
                                    foundit = true;
                                }
                            }
                        }
                    }

                    // send the particles to their destination nodes
                    for (i = 0; i < N; i++)
                    {
                        if (SwapNodeList[d][i])
                        {
                            // put data for particles on this put list into message
                            PData.putMessage( *(SwapMsgList[i]), PutList[i] );

                            // add a final 'zero' number of particles to indicate the end
                            PData.putMessage(*(SwapMsgList[i]), (size_t) 0, (size_t) 0);

                            // send the message
                            //Inform dbgmsg("SpatialLayout", INFORM_ALL_NODES);
                            //dbgmsg << "Swapping "<<PutList[i].size() << " particles to node ";
                            //dbgmsg << i<<" with tag " << tag << " (" << 'x' + d << ")" << endl;
                            //dbgmsg << "  ... msg = " << *(SwapMsgList[i]) << endl;
                            int node = i;
                            Ippl::Comm->send(SwapMsgList[i], node, tag);

                            // clear the list
                            PutList[i].erase(PutList[i].begin(), PutList[i].end());
                        }
                    }

                    LocalNum -= PData.getDestroyNum();  // update local num
                    ADDIPPLSTAT(incParticlesSwapped, PData.getDestroyNum());
                    PData.performDestroy();

                    // receive particles from neighbor nodes, and add them to our list
                    unsigned sendnum = NeighborNodes[d];
                    while (sendnum-- > 0)
                    {
                        int node = Communicate::COMM_ANY_NODE;
                        Message *recmsg = Ippl::Comm->receive_block(node, tag);
                        size_t recvd;
                        while ((recvd = PData.getMessage(*recmsg)) > 0)
                            LocalNum += recvd;
                        delete recmsg;
                    }
                }  // end if (NeighborNodes[d] > 0)

                if (d == 0)
                {
                    // receive messages from any empty nodes
                    for (i = 0; i < N; ++i)
                    {
                        if (getEmptyNode(i))
                        {
                            int node = i;
                            Message *recmsg = Ippl::Comm->receive_block(node, etag);
                            size_t recvd;
                            while ((recvd = PData.getMessage(*recmsg)) > 0)
                                LocalNum += recvd;
                            delete recmsg;
                        }
                    }
                }

            }  // end for (d=0; d<Dim; ++d)

        }
        else   // empty node sends, but does not receive
        {
            // create new messages to send to our neighbors along dim 0
            for (i = 0; i < N; i++)
                if (SwapNodeList[0][i])
                    SwapMsgList[i] = new Message;

            // Go through the particles and find those moving to other nodes.
            // When one is found, copy it into outgoing message and delete it.
            for (ip=0; ip<LocalNum; ++ip)
            {
                if (!bool(canSwap[ip])) continue;  // skip if can't swap
                // get the position of particle ip, and find the closest grid pnt
                for (j = 0; j < Dim; j++)
                    pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

                // see which remote Vnode this postion is in
                typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN =
                    RLayout.touch_range_rdv(pLoc);

                // make sure we have a vnode to send it to
                if (touchingVN.first == touchingVN.second)
                {
                    ERRORMSG("Local particle " << ip << " with ID=");
                    ERRORMSG(PData.ID[ip] << " at ");
                    ERRORMSG(PData.R[ip] << " is outside of global domain ");
                    ERRORMSG(RLayout.getDomain() << endl);
                    ERRORMSG("This occurred when searching for point " << pLoc);
                    ERRORMSG(" in RegionLayout = " << RLayout << endl);
                    Ippl::abort();
                }
                else
                {
                    // the node has been found - add index to put list
                    unsigned node = (*(touchingVN.first)).second->getNode();
                    PAssert_EQ(SwapNodeList[0][node], true);
                    PutList[node].push_back(ip);

                    // .. and then add to DestroyList
                    PData.destroy(1, ip);
                }
            }

            // send the particles to their destination nodes
            for (i = 0; i < N; i++)
            {
                if (SwapNodeList[0][i])
                {
                    // put data for particles on this put list into message
                    PData.putMessage( *(SwapMsgList[i]), PutList[i] );

                    // add a final 'zero' number of particles to indicate the end
                    PData.putMessage(*(SwapMsgList[i]), (size_t) 0, (size_t) 0);

                    // send the message
                    int node = i;
                    Ippl::Comm->send(SwapMsgList[i], node, etag);

                    // clear the list
                    PutList[i].erase(PutList[i].begin(), PutList[i].end());
                }
            }

            LocalNum -= PData.getDestroyNum();  // update local num
            ADDIPPLSTAT(incParticlesSwapped, PData.getDestroyNum());
            PData.performDestroy();

        }

        // return how many particles we have now
        return LocalNum;
    }




/*
 * Newer (cleaner) version of swap particles that uses less bandwidth
 * and drastically lowers message counts for real cases.
 */
    template < class PB >
    size_t new_swap_particles(size_t LocalNum, PB& PData)
    {
        Ippl::Comm->barrier();
        static int sent = 0;

        unsigned N = Ippl::getNodes();
        unsigned myN = Ippl::myNode();

        typename RegionLayout<T,Dim,Mesh>::iterator_iv localV, localEnd = RLayout.end_iv();
        typename RegionLayout<T,Dim,Mesh>::iterator_dv remoteV;

        std::vector<int> msgsend(N, 0);
        std::vector<int> msgrecv(N, 0);

        NDRegion<T,Dim> pLoc;

        std::multimap<unsigned, unsigned> p2n; //<node ID, particle ID>

        int minParticlesPerNode = PData.getMinimumNumberOfParticlesPerCore();
        int particlesLeft = LocalNum;
        bool responsibleNodeNotFound = false;
        for (unsigned int ip=0; ip<LocalNum; ++ip)
        {
            for (unsigned int j = 0; j < Dim; j++)
                pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

            unsigned destination = myN;
            bool found = false;
            for (localV = RLayout.begin_iv(); localV != localEnd && !found; ++localV)
            {
                if ((((*localV).second)->getDomain()).touches(pLoc))
                    found = true; // particle is local and doesn't need to be sent anywhere
            }

            if (found)
                continue;

            if (particlesLeft <= minParticlesPerNode)
	        break; //leave atleast minimum number of particles per core

            typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN = RLayout.touch_range_rdv(pLoc);

            //external location
            if (touchingVN.first == touchingVN.second) {
                responsibleNodeNotFound = true;
                break;
            }
            destination = (*(touchingVN.first)).second->getNode();

            msgsend[destination] = 1;

            p2n.insert(std::pair<unsigned, unsigned>(destination, ip));
            sent++;
            particlesLeft--;
        }

        allreduce(&responsibleNodeNotFound,
                  1,
                  std::logical_or<bool>());

        if (responsibleNodeNotFound) {
            throw IpplException("ParticleSpatialLayout::new_swap_particles",
                                "could not find node responsible for particle");
        }

        //reduce message count so every node knows how many messages to receive
        allreduce(msgsend.data(), msgrecv.data(), N, std::plus<int>());

        int tag = Ippl::Comm->next_tag(P_SPATIAL_TRANSFER_TAG,P_LAYOUT_CYCLE);

        typename std::multimap<unsigned, unsigned>::iterator i = p2n.begin();

        std::unique_ptr<Format> format(PData.getFormat());


        std::vector<MPI_Request> requests;
        std::vector<std::shared_ptr<MsgBuffer> > buffers;

        while (i!=p2n.end())
        {
            unsigned cur_destination = i->first;

            std::shared_ptr<MsgBuffer> msgbuf(new MsgBuffer(format.get(), p2n.count(i->first)));

            for (; i!=p2n.end() && i->first == cur_destination; ++i)
            {
                Message msg;
                PData.putMessage(msg, i->second);
                PData.destroy(1, i->second);
                msgbuf->add(&msg);
            }

            MPI_Request request = Ippl::Comm->raw_isend( msgbuf->getBuffer(), msgbuf->getSize(), cur_destination, tag);

            //remember request and buffer so we can delete them later
            requests.push_back(request);
            buffers.push_back(msgbuf);
        }

        LocalNum -= PData.getDestroyNum();  // update local num
        PData.performDestroy();

        //receive new particles
        for (int k = 0; k<msgrecv[myN]; ++k)
        {
            int node = Communicate::COMM_ANY_NODE;
            char *buffer = 0;
            int bufsize = Ippl::Comm->raw_probe_receive(buffer, node, tag);
            MsgBuffer recvbuf(format.get(), buffer, bufsize);

            Message *msg = recvbuf.get();
            while (msg != 0)
            {
                LocalNum += PData.getSingleMessage(*msg);
                delete msg;
                msg = recvbuf.get();
            }


        }

        //wait for communication to finish and clean up buffers
        MPI_Waitall(requests.size(), &(requests[0]), MPI_STATUSES_IGNORE);

        return LocalNum;
    }

   template < class PB >
    size_t new_swap_particles(size_t LocalNum, PB& PData,
                              const ParticleAttrib<char>& canSwap)
    {
        Ippl::Comm->barrier();
        static int sent = 0;

        unsigned N = Ippl::getNodes();
        unsigned myN = Ippl::myNode();

        typename RegionLayout<T,Dim,Mesh>::iterator_iv localV, localEnd = RLayout.end_iv();
        typename RegionLayout<T,Dim,Mesh>::iterator_dv remoteV;

        std::vector<int> msgsend(N, 0);
        std::vector<int> msgrecv(N, 0);

        NDRegion<T,Dim> pLoc;

        std::multimap<unsigned, unsigned> p2n; //<node ID, particle ID>

        int minParticlesPerNode = PData.getMinimumNumberOfParticlesPerCore();
        int particlesLeft = LocalNum;
        bool responsibleNodeNotFound = false;
        for (unsigned int ip=0; ip<LocalNum; ++ip)
        {
            if (!bool(canSwap[ip]))//skip if it can't be swapped
                continue;

            for (unsigned int j = 0; j < Dim; j++)
                pLoc[j] = PRegion<T>(PData.R[ip][j], PData.R[ip][j]);

            unsigned destination = myN;
            bool found = false;
            for (localV = RLayout.begin_iv(); localV != localEnd && !found; ++localV)
            {
                if ((((*localV).second)->getDomain()).touches(pLoc))
                    found = true; // particle is local and doesn't need to be sent anywhere
            }

            if (found)
                continue;

            if (particlesLeft <= minParticlesPerNode)
                continue; //leave atleast minimum number of particles per core

            typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchingVN = RLayout.touch_range_rdv(pLoc);

            //external location
            if (touchingVN.first == touchingVN.second) {
                responsibleNodeNotFound = true;
                break;
            }
            destination = (*(touchingVN.first)).second->getNode();

            msgsend[destination] = 1;

            p2n.insert(std::pair<unsigned, unsigned>(destination, ip));
            sent++;
            particlesLeft--;
        }

        allreduce(&responsibleNodeNotFound,
                  1,
                  std::logical_or<bool>());

        if (responsibleNodeNotFound) {
            throw IpplException("ParticleSpatialLayout::new_swap_particles",
                                "could not find node responsible for particle");
        }

        //reduce message count so every node knows how many messages to receive
        allreduce(msgsend.data(), msgrecv.data(), N, std::plus<int>());

        int tag = Ippl::Comm->next_tag(P_SPATIAL_TRANSFER_TAG,P_LAYOUT_CYCLE);

        typename std::multimap<unsigned, unsigned>::iterator i = p2n.begin();

        std::unique_ptr<Format> format(PData.getFormat());

        std::vector<MPI_Request> requests;
        std::vector<std::shared_ptr<MsgBuffer> > buffers;

        while (i!=p2n.end())
        {
            unsigned cur_destination = i->first;

            std::shared_ptr<MsgBuffer> msgbuf(new MsgBuffer(format.get(), p2n.count(i->first)));

            for (; i!=p2n.end() && i->first == cur_destination; ++i)
            {
                Message msg;
                PData.putMessage(msg, i->second);
                PData.destroy(1, i->second);
                msgbuf->add(&msg);
            }

            MPI_Request request = Ippl::Comm->raw_isend( msgbuf->getBuffer(), msgbuf->getSize(), cur_destination, tag);

            //remember request and buffer so we can delete them later
            requests.push_back(request);
            buffers.push_back(msgbuf);
        }

        LocalNum -= PData.getDestroyNum();  // update local num
        PData.performDestroy();

        //receive new particles
        for (int k = 0; k<msgrecv[myN]; ++k)
        {
            int node = Communicate::COMM_ANY_NODE;
            char *buffer = 0;
            int bufsize = Ippl::Comm->raw_probe_receive(buffer, node, tag);
            MsgBuffer recvbuf(format.get(), buffer, bufsize);

            Message *msg = recvbuf.get();
            while (msg != 0)
            {
                LocalNum += PData.getSingleMessage(*msg);
                delete msg;
                msg = recvbuf.get();
            }
        }

        //wait for communication to finish and clean up buffers
        MPI_Waitall(requests.size(), &(requests[0]), 0);

        return LocalNum;
    }

};

#include "Particle/ParticleSpatialLayout.hpp"

#endif // PARTICLE_SPATIAL_LAYOUT_H
