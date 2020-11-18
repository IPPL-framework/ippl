//
// Class ParticleSpatialLayout
//   Particle layout based on spatial decomposition.
//
//   This is a specialized version of ParticleLayout, which places particles
//   on processors based on their spatial location relative to a fixed grid.
//   In particular, this can maintain particles on processors based on a
//   specified FieldLayout or RegionLayout, so that particles are always on
//   the same node as the node containing the Field region to which they are
//   local.  This may also be used if there is no associated Field at all,
//   in which case a grid is selected based on an even distribution of
//   particles among processors.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
// #include "Particle/ParticleBConds.h"
// #include "Index/NDIndex.h"
// #include "Region/RegionLayout.h"
// #include "Message/Communicate.h"
// #include "Message/Message.h"
// #include "Utility/IpplInfo.h"
// #include "Utility/IpplStats.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh>
    ParticleSpatialLayout<T, Dim, Mesh>::ParticleSpatialLayout(
        FieldLayout<Dim>& fl,
        Mesh& mesh)
    : rlayout_m(fl, mesh)
    {
        setup();
    }


    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::setup()
    {
/*
        unsigned i;			// loop variable

        // check ourselves in as a user of the RegionLayout
        RLayout.checkin(*this);

        // create storage for message pointers used in swapping particles
        unsigned N = Ippl::Comm->getNodes();
        SwapMsgList = new Message*[N];
        for (i = 0; i < Dim; ++i)
            SwapNodeList[i] = new bool[N];
        PutList = new std::vector<size_t>[N];

        // create storage for the number of particles on each node
        // and flag for empty node domain
        NodeCount = new size_t[N];
        EmptyNode = new bool[N];
        for (i = 0; i < N; ++i)
        {
            NodeCount[i] = 0;
            EmptyNode[i] = false;
        }*/
    }


    template <typename T, unsigned Dim, class Mesh>
    ParticleSpatialLayout<T, Dim, Mesh>::~ParticleSpatialLayout()
    {
//         delete [] NodeCount;
//         delete [] EmptyNode;
//         delete [] SwapMsgList;
//         for (unsigned int i=0; i < Dim; i++)
//             delete [] (SwapNodeList[i]);
//         delete [] PutList;
//
//         // check ourselves out as a user of the RegionLayout
//         RLayout.checkout(*this);
    }


    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::update(
        ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& pdata)
    {
        this->applyBC(pdata.R, rlayout_m.getDomain());

        if (Ippl::Comm->size() < 2) {
            // delete invalidated particles
            pdata.destroy();
            return;
        }


        /* particle MPI exchange:
         *   1. figure out which particles need to go where
         *   2. fill send buffer
         *   3. send / receive particles
         *   4. delete invalidated particles
         */

        size_t localnum = pdata.getLocalNum();

        // 1st step
        locate_type ranks("MPI ranks", localnum);

        locateParticles(pdata, ranks);

        /*
         * 2nd step
         */

        // send
        for (int rank = 0; rank < Ippl::Comm->size(); ++rank) {
            std::cout << "rank = " << rank << std::endl;
            if (rank == Ippl::Comm->rank()) {
                // we do not need to send to ourself
                continue;
            }

            size_t nSends = numberOfSends(rank, ranks);

            std::cout << "nSends = " << nSends << std::endl;

            if (nSends > 0) {
                hash_type hash("hash", localnum);
                fillHash(rank, ranks, hash);

                ParticleBase<ParticleSpatialLayout<T, Dim, Mesh> > buffer(pdata.getLayout());
                buffer.create(nSends);
//                 pdata.pack(buffer, hash);
            }
        }

        // 3rd step


        // 4th step
        pdata.destroy();


//         // At this point, we can send our particle count updates to node 0, and
//         // receive back the particle layout.
//         int tag1 = Ippl::Comm->next_tag(P_SPATIAL_LAYOUT_TAG, P_LAYOUT_CYCLE);
//         int tag2 = Ippl::Comm->next_tag(P_SPATIAL_RETURN_TAG, P_LAYOUT_CYCLE);
//         if (myN != 0)
//         {
//             Message *msg = new Message;
//
//             // put local particle count in the message
//             msg->put(LocalNum);
//             // send this info to node 0
//             Ippl::Comm->send(msg, 0, tag1);
//
//             // receive back the number of particles on each node
//             node = 0;
//             Message* recmsg = Ippl::Comm->receive_block(node, tag2);
//             recmsg->get(NodeCount);
//             recmsg->get(TotalNum);
//             delete recmsg;
//         }
//         else  			// do update tasks particular to node 0
//         {
//             // receive messages from other nodes describing what they have
//             int notrecvd = N - 1;	// do not need to receive from node 0
//             TotalNum = LocalNum;
//             while (notrecvd > 0)
//             {
//                 // receive a message from another node.  After recv, node == sender.
//                 node = Communicate::COMM_ANY_NODE;
//                 Message *recmsg = Ippl::Comm->receive_block(node, tag1);
//                 size_t remNodeCount = 0;
//                 recmsg->get(remNodeCount);
//                 delete recmsg;
//                 notrecvd--;
//
//                 // update values based on data from remote node
//                 TotalNum += remNodeCount;
//                 NodeCount[node] = remNodeCount;
//             }
//
//             // send info back to all the client nodes
//             Message *msg = new Message;
//             msg->put(NodeCount, NodeCount + N);
//             msg->put(TotalNum);
//             Ippl::Comm->broadcast_others(msg, tag2);
//         }
    }


    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::locateParticles(
        const ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& /*pdata*/,
        locate_type& ranks) const
    {
//         auto& positions = pdata.R.getView();
        Kokkos::parallel_for(
            "ParticleSpatialLayout::locateParticles()",
            ranks.size(),
            KOKKOS_CLASS_LAMBDA(const size_t /*i*/) {
//             ranks(i)
        });
    }


    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::fillHash(int rank,
                                                       const locate_type& ranks,
                                                       hash_type& hash)
    {
        /* Compute the prefix sum and fill the hash
         */
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()",
            ranks.size(),
            KOKKOS_LAMBDA(const int i, int& idx, const bool final) {
                if (final) {
                    hash(i) = idx;
                }

                if (rank == ranks(i)) {
                    idx += 1;
                }
            });
    }


    template <typename T, unsigned Dim, class Mesh>
    size_t ParticleSpatialLayout<T, Dim, Mesh>::numberOfSends(
        int rank,
        const locate_type& ranks)
    {
        size_t nSends = 0;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()",
            ranks.size(),
            KOKKOS_CLASS_LAMBDA(const size_t i,
                                size_t& num)
            {
                num += size_t(rank == ranks(i));
            }, nSends);
        return nSends;
    }
}