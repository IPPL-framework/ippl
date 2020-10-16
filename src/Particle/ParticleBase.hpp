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

// #include "Particle/Kokkos_ParticleBase.h"
// #include "Particle/ParticleLayout.h"
// #include "Particle/ParticleAttrib.h"
// #include "Message/Message.h"
// #include "Message/Communicate.h"
// #include "Utility/Inform.h"
// #include "Utility/PAssert.h"
// #include "Utility/IpplInfo.h"
// #include "Utility/IpplStats.h"
// #include "Utility/IpplException.h"
// #include <algorithm>

namespace ippl {

    template<class PLayout>
    ParticleBase<PLayout>::ParticleBase()
    : ParticleBase(nullptr)
    { }

    template<class PLayout>
    ParticleBase<PLayout>::ParticleBase(std::shared_ptr<PLayout>& layout)
    : ParticleBase()
    {
        initialize(layout);
    }

    template<class PLayout>
    ParticleBase<PLayout>::ParticleBase(std::shared_ptr<PLayout>&& layout)
    : layout_m(std::move(layout))
    , totalNum_m(0)
    , localNum_m(0)
    , nextID_m(Ippl::Comm->myNode())
    , numNodes_m(Ippl::Comm->getNodes())
    {
        addAttribute(R);
        addAttribute(ID);
    }


    template<class PLayout>
    void ParticleBase<PLayout>::addAttribute(ParticleAttribBase& pa)
    {
        attributes_m.push_back(&pa);
    }

    template<class PLayout>
    void ParticleBase<PLayout>::initialize(std::shared_ptr<PLayout>& layout) {
        PAssert(layout_m == nullptr);

        std::cout << "initialize ";
        // save the layout, and perform setup tasks
        layout_m = std::move(layout);

        std::cout << (layout_m != nullptr) << std::endl;
    }
//
//
//
//
//     /////////////////////////////////////////////////////////////////////
//     // Return a boolean value indicating if we are on a processor which can
//     // be used for single-node particle creation and initialization
//
//     template<class PLayout>
//     bool ParticleBase<PLayout>::singleInitNode() const {
//         return (Ippl::Comm->myNode() == 0);
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // Reset the particle ID's to be globally consecutive, 0 thru TotalNum.
//     template <class PLayout>
//     void ParticleBase<PLayout>::resetID(void) {
//
//
//
//     unsigned int nodes = Ippl::getNodes();
//     unsigned int myNode = Ippl::myNode();
//     size_t localNum = this->getLocalNum();
//     size_t ip;
//     int master = 0;
//     if (myNode == (unsigned int) master) {
//         // Node 0 can immediately set its ID's to 0 thru LocalNum-1
//         for (ip=0; ip<localNum; ++ip)
//         this->ID[ip] = ip;
//         // if there is only one processor, we are done
//         if (nodes == 1) return;
//         // parallel case: must find out how many particles each processor has
//         // Node 0 gathers this information into an array
//         size_t *lp;
//         lp = new size_t[nodes];
//         lp[0] = localNum;  // enter our own number of particles
//         // get next message tag and receive messages
//         int tag1 = Ippl::Comm->next_tag(P_RESET_ID_TAG,P_LAYOUT_CYCLE);
//         Message* msg1;
//         for (ip=1; ip<nodes; ++ip) {
//         int rnode = COMM_ANY_NODE;
//         msg1 = Ippl::Comm->receive_block(rnode,tag1);
//         PAssert(msg1);
//         msg1->get(lp[rnode]);
//         delete msg1;
//         }
//         // now we should have all the localnum values.
//         // figure out starting ID for each processor and send back
//         size_t current, sum = 0;
//         for (ip=0; ip<nodes; ++ip) {
//         current = lp[ip];
//         lp[ip] = sum;
//         sum += current;
//         }
//         // send initial ID values back out
//         int tag2 = Ippl::Comm->next_tag(P_RESET_ID_TAG,P_LAYOUT_CYCLE);
//         for (ip=1; ip<nodes; ++ip) {
//         Message* msg2 = new Message;
//         msg2->put(lp[ip]);
//         bool success = Ippl::Comm->send(msg2,ip,tag2);
//         if (success == false) {
//                 throw IpplException (
//                         "ParticleBase<PLayout>::resetID()",
//                         "sending initial ID values failed.");
//         }
//         }
//         // we are done
//         return;
//     }
//     else {
//         // first send number of local particles to Node 0
//         int tag1 = Ippl::Comm->next_tag(P_RESET_ID_TAG,P_LAYOUT_CYCLE);
//         Message* msg1 = new Message;
//         msg1->put(localNum);
//         bool success = Ippl::Comm->send(msg1,master,tag1);
//         if (success == false) {
//                 throw IpplException (
//                         "ParticleBase<PLayout>::resetID()",
//                         "sending initial ID values failed.");
//         }
//         // now receive back our initial ID number
//         size_t initialID = 0;
//         int tag2 = Ippl::Comm->next_tag(P_RESET_ID_TAG,P_LAYOUT_CYCLE);
//         Message* msg2 = Ippl::Comm->receive_block(master,tag2);
//         PAssert(msg2);
//         msg2->get(initialID);
//         delete msg2;
//         // now reset our particle ID's using this initial value
//         for (ip=0; ip<localNum; ++ip)
//         this->ID[ip] = ip + initialID;
//         // we are done
//         return;
//     }
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // put the data for M particles starting from local index I in a Message
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::putMessage(Message& msg, size_t M, size_t I) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // put into message the number of items in the message
//     msg.put(M);
//
//     // go through all the attributes and put their data in the message
//     if (M > 0) {
//         // this routine should only be called for local particles; call
//         // ghostPutMessage to put in particles which might be ghost particles
//         PAssert_LT(I, R.size());
//         PAssert_LE(I + M, R.size());
//
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->putMessage(msg, M, I);
//     }
//
//     return M;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // put the data for a list of particles in a Message
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::putMessage(Message& msg,
//                                     const std::vector<size_t>& putList)
//     {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     std::vector<size_t>::size_type M = putList.size();
//     msg.put(M);
//
//     // go through all the attributes and put their data in the message
//     if (M > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; ++abeg )
//         (*abeg)->putMessage(msg, putList);
//     }
//
//     return M;
//     }
//
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::putMessage(Message& msg, size_t I) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // put into message the number of items in the message
//
//     // go through all the attributes and put their data in the message
//
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->putMessage(msg, 1, I);
//
//     return 1;
//     }
//
//     template<class PLayout>
//     Format* ParticleBase<PLayout>::getFormat()
//     {
//             //create dummy particle so we can obtain the format
//         bool wasempty = false;
//         if(this->getLocalNum()==0)
//         {
//                     this->create(1);
//                     wasempty = true;
//             }
//
//             //obtain the format
//         Message *msg = new Message;
//         this->putMessage(*msg, (size_t) 0);
//         Format *format = new Format(msg);
//         delete msg;
//
//             //remove the dummy particle again
//         if (wasempty)
//             this->destroy(1, 0, true);
//
//         return format;
//     }
//
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::writeMsgBuffer(MsgBuffer *&msgbuf, const std::vector<size_t> &list)
//     {
//         msgbuf = new MsgBuffer(this->getFormat(), list.size());
//
//         for (unsigned int i = 0;i<list.size();++i)
//         {
//             Message msg;
//             this->putMessage(msg, list[i]);
//             msgbuf->add(&msg);
//         }
//         return list.size();
//     }
//
//     template<class PLayout>
//     template<class O>
//     size_t
//     ParticleBase<PLayout>::writeMsgBufferWithOffsets(MsgBuffer *&msgbuf, const std::vector<size_t> &list, const std::vector<O> &offset)
//     {
//         msgbuf = new MsgBuffer(this->getFormat(), list.size());
//         typename PLayout::SingleParticlePos_t oldpos;
//         for (unsigned int i = 0;i<list.size();++i)
//             {
//                 oldpos = R[list[i]];
//                 for(int d = 0;d<Dim;++d)
//                     {
//                         R[list[i]][d] += offset[i][d];
//                     }
//
//                 Message msg;
//                 this->putMessage(msg, list[i]);
//                 msgbuf->add(&msg);
//
//                 R[list[i]] = oldpos;
//             }
//         return list.size();
//     }
//
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::readMsgBuffer(MsgBuffer *msgbuf)
//     {
//             size_t added = 0;
//             Message *msg = msgbuf->get();
//         while (msg != 0)
//         {
//                     added += this->getSingleMessage(*msg);
//             delete msg;
//             msg = msgbuf->get();
//         }
//         return added;
//     }
//
//
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::readGhostMsgBuffer(MsgBuffer *msgbuf, int node)
//     {
//             size_t added = 0;
//             Message *msg = msgbuf->get();
//         while (msg != 0)
//         {
//                     added += this->ghostGetSingleMessage(*msg, node);
//             delete msg;
//             msg = msgbuf->get();
//         }
//         return added;
//     }
//
//     /////////////////////////////////////////////////////////////////////
//     // retrieve particles from the given message and store them
//     template<class PLayout>
//     size_t ParticleBase<PLayout>::getMessage(Message& msg) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // get the number of items in the message
//     size_t numitems = 0;
//     msg.get(numitems);
//
//     // go through all the attributes and get their data from the message
//     if (numitems > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->getMessage(msg, numitems);
//     }
//
//     return numitems;
//     }
//
//     /////////////////////////////////////////////////////////////////////
//     // retrieve particles from the given message and store them
//     template<class PLayout>
//     size_t ParticleBase<PLayout>::getSingleMessage(Message& msg) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // get the number of items in the message
//     size_t numitems=1;
//
//     // go through all the attributes and get their data from the message
//     if (numitems > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->getMessage(msg, numitems);
//     }
//
//     return numitems;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // retrieve particles from the given message and store them, also
//     // signaling we are creating the given number of particles.  Return the
//     // number of particles created.
//     template<class PLayout>
//     size_t ParticleBase<PLayout>::getMessageAndCreate(Message& msg) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // call the regular create, and add in the particles to our LocalNum
//     size_t numcreate = getMessage(msg);
//     LocalNum += numcreate;
//     ADDIPPLSTAT(incParticlesCreated,numcreate);
//     return numcreate;
//     }
//
//


    template<class PLayout>
    void ParticleBase<PLayout>::create(size_t nLocal)
    {
//     // make sure we've been initialized
//     PAssert(Layout != 0);

        for (attribute_iterator it = attributes_m.begin();
             it != attributes_m.end(); ++it) {
            (*it)->create(nLocal);
        }

        // set the unique ID value for these new particles
        Kokkos::parallel_for("ParticleBase<PLayout>::create(size_t)",
                             Kokkos::RangePolicy(localNum_m, nLocal),
                             KOKKOS_CLASS_LAMBDA(const size_t i) {
                                 ID(i) = this->nextID_m + this->numNodes_m * i;
                             });
        nextID_m += numNodes_m * (nLocal - localNum_m);

        // remember that we're creating these new particles
        localNum_m += nLocal;
    }

    template<class PLayout>
    void ParticleBase<PLayout>::createWithID(index_type id) {
//         // make sure we've been initialized
//         PAssert(Layout != 0);

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m = id;
        numNodes_m = 0;

        create(1);

        nextID_m = tmpNextID;
        numNodes_m = Ippl::Comm->getNodes();
    }

    template<class PLayout>
    void ParticleBase<PLayout>::globalCreate(size_t nTotal) {
//         // make sure we've been initialized
//         PAssert(Layout != 0);

        // Compute the number of particles local to each processor
        size_t nLocal = nTotal / numNodes_m;

        const size_t rank = Ippl::Comm->myNode();

        size_t rest = nTotal - nLocal * rank;
        if (rank < rest)
            ++nLocal;

        create(nLocal);
    }


//
//

//
//
//     /////////////////////////////////////////////////////////////////////
//     // delete M particles, starting with the Ith particle.  If the last argument
//     // is true, the destroy will be done immediately, otherwise the request
//     // will be cached.
//     template<class PLayout>
//     void ParticleBase<PLayout>::destroy(size_t M, size_t I, bool doNow) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     if (M > 0) {
//         if (doNow) {
//         // find out if we are using optimized destroy method
//         bool optDestroy = getUpdateFlag(PLayout::OPTDESTROY);
//         // loop over attributes and carry out the destroy request
//         attrib_container_t::iterator abeg, aend = AttribList.end();
//         for (abeg = AttribList.begin(); abeg != aend; ++abeg)
//             (*abeg)->destroy(M,I,optDestroy);
//         LocalNum -= M;
//         }
//         else {
//         // add this group of particle indices to our list of items to destroy
//         std::pair<size_t,size_t> destroyEvent(I,M);
//         DestroyList.push_back(destroyEvent);
//         DestroyNum += M;
//         }
//
//         // remember we have this many more items to destroy (or have destroyed)
//         ADDIPPLSTAT(incParticlesDestroyed,M);
//     }
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // Update the particle object after a timestep.  This routine will change
//     // our local, total, create particle counts properly.
//     template<class PLayout>
//     void ParticleBase<PLayout>::update() {
//
//
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // ask the layout manager to update our atoms, etc.
//     Layout->update(*this);
//     INCIPPLSTAT(incParticleUpdates);
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // Update the particle object after a timestep.  This routine will change
//     // our local, total, create particle counts properly.
//     template<class PLayout>
//     void ParticleBase<PLayout>::update(const ParticleAttrib<char>& canSwap) {
//
//
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // ask the layout manager to update our atoms, etc.
//     Layout->update(*this, &canSwap);
//     INCIPPLSTAT(incParticleUpdates);
//     }
//
//
//     // Actually perform the delete atoms action for all the attributes; the
//     // calls to destroy() only stored a list of what to do.  This actually
//     // does it.  This should in most cases only be called by the layout manager.
//     template<class PLayout>
//     void ParticleBase<PLayout>::performDestroy(bool updateLocalNum) {
//
//
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // nothing to do if destroy list is empty
//     if (DestroyList.empty()) return;
//
//     // before processing the list, we should make sure it is sorted
//     bool isSorted = true;
//     typedef std::vector< std::pair<size_t,size_t> > dlist_t;
//     dlist_t::const_iterator curr = DestroyList.begin();
//     const dlist_t::const_iterator last = DestroyList.end();
//     dlist_t::const_iterator next = curr + 1;
//     while (next != last && isSorted) {
//         if (*next++ < *curr++) isSorted = false;
//     }
//     if (!isSorted)
//         std::sort(DestroyList.begin(),DestroyList.end());
//
//     // find out if we are using optimized destroy method
//     bool optDestroy = getUpdateFlag(PLayout::OPTDESTROY);
//
//     // loop over attributes and process destroy list
//     attrib_container_t::iterator abeg, aend = AttribList.end();
//     for (abeg = AttribList.begin(); abeg != aend; ++abeg)
//         (*abeg)->destroy(DestroyList,optDestroy);
//
//     if (updateLocalNum) {
//         for (curr = DestroyList.begin(); curr != last; ++ curr) {
//             LocalNum -= curr->second;
//         }
//     }
//
//     // clear destroy list and update destroy num counter
//     DestroyList.erase(DestroyList.begin(),DestroyList.end());
//     DestroyNum = 0;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // delete M ghost particles, starting with the Ith particle.
//     // This is done immediately.
//     template<class PLayout>
//     void ParticleBase<PLayout>::ghostDestroy(size_t M, size_t I) {
//
//
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     if (M > 0) {
//         // delete the data from the attribute containers
//         size_t dnum = 0;
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; ++abeg )
//         dnum = (*abeg)->ghostDestroy(M, I);
//         GhostNum -= dnum;
//     }
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // Put the data for M particles starting from local index I in a Message.
//     // Return the number of particles put in the Message.  This is for building
//     // ghost particle interaction lists.
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::ghostPutMessage(Message &msg, size_t M, size_t I) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // put into message the number of items in the message
//     if (I >= R.size()) {
//         // we're putting in ghost particles ...
//         if ((I + M) > (R.size() + GhostNum))
//         M = (R.size() + GhostNum) - I;
//     } else {
//         // we're putting in local particles ...
//         if ((I + M) > R.size())
//         M = R.size() - I;
//     }
//     msg.put(M);
//
//     // go through all the attributes and put their data in the message
//     if (M > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->ghostPutMessage(msg, M, I);
//     }
//
//     return M;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // put the data for particles on a list into a Message, given list of indices
//     // Return the number of particles put in the Message.  This is for building
//     // ghost particle interaction lists.
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::ghostPutMessage(Message &msg,
//                                         const std::vector<size_t>& pl) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     std::vector<size_t>::size_type M = pl.size();
//     msg.put(M);
//
//     // go through all the attributes and put their data in the message
//     if (M > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; ++abeg )
//         (*abeg)->ghostPutMessage(msg, pl);
//     }
//
//     return M;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // retrieve particles from the given message and sending node and store them
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::ghostGetMessage(Message& msg, int /*node*/) {
//
//
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // get the number of items in the message
//     size_t numitems;
//     msg.get(numitems);
//     GhostNum += numitems;
//
//     // go through all the attributes and get their data from the message
//     if (numitems > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->ghostGetMessage(msg, numitems);
//     }
//
//     return numitems;
//     }
//
//     template<class PLayout>
//     size_t
//     ParticleBase<PLayout>::ghostGetSingleMessage(Message& msg, int /*node*/) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     // get the number of items in the message
//     size_t numitems=1;
//     GhostNum += numitems;
//
//     // go through all the attributes and get their data from the message
//     if (numitems > 0) {
//         attrib_container_t::iterator abeg = AttribList.begin();
//         attrib_container_t::iterator aend = AttribList.end();
//         for ( ; abeg != aend; abeg++ )
//         (*abeg)->ghostGetMessage(msg, numitems);
//     }
//
//     return numitems;
//     }
//
//     /////////////////////////////////////////////////////////////////////
//     // Apply the given sort-list to all the attributes.  The sort-list
//     // may be temporarily modified, thus it must be passed by non-const ref.
//     template<class PLayout>
//     void ParticleBase<PLayout>::sort(SortList_t &sortlist) {
//     attrib_container_t::iterator abeg = AttribList.begin();
//     attrib_container_t::iterator aend = AttribList.end();
//     for ( ; abeg != aend; ++abeg )
//         (*abeg)->sort(sortlist);
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // print it out
//     template<class PLayout>
//     std::ostream& operator<<(std::ostream& out, const ParticleBase<PLayout>& P) {
//
//
//     out << "Particle object contents:";
//     out << "\n  Total particles: " << P.getTotalNum();
//     out << "\n  Local particles: " << P.getLocalNum();
//     out << "\n  Attributes (including R and ID): " << P.numAttributes();
//     out << "\n  Layout = " << P.getLayout();
//     return out;
//     }
//
//
//     /////////////////////////////////////////////////////////////////////
//     // print out debugging information
//     template<class PLayout>
//     void ParticleBase<PLayout>::printDebug(Inform& o) {
//
//     o << "PBase: total = " << getTotalNum() << ", local = " << getLocalNum();
//     o << ", attributes = " << AttribList.size() << endl;
//     for (attrib_container_t::size_type i=0; i < AttribList.size(); ++i) {
//         o << "    ";
//         AttribList[i]->printDebug(o);
//         o << endl;
//     }
//     o << "    ";
//     Layout->printDebug(o);
//     }
}
