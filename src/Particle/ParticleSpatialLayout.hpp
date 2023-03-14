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
//   After each 'time step' in a calculation, which is defined as a period
//   in which the particle positions may change enough to affect the global
//   layout, the user must call the 'update' routine, which will move
//   particles between processors, etc.  After the Nth call to update, a
//   load balancing routine will be called instead.  The user may set the
//   frequency of load balancing (N), or may supply a function to
//   determine if load balancing should be done or not.
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
#include <memory>
#include <numeric>
#include <vector>
#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh>
    ParticleSpatialLayout<T, Dim, Mesh>::ParticleSpatialLayout(FieldLayout<Dim>& fl, Mesh& mesh)
        : rlayout_m(fl, mesh) {}

    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::updateLayout(FieldLayout<Dim>& fl, Mesh& mesh) {
        rlayout_m.changeDomain(fl, mesh);
    }

    template <typename T, unsigned Dim, class Mesh>
    template <class BufferType>
    void ParticleSpatialLayout<T, Dim, Mesh>::update(BufferType& pdata, BufferType& buffer) {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(pdata.R, rlayout_m.getDomain());
        IpplTimings::stopTimer(ParticleBCTimer);

        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);
        int nRanks = Ippl::Comm->size();

        if (nRanks < 2) {
            return;
        }

        /* particle MPI exchange:
         *   1. figure out which particles need to go where
         *   2. fill send buffer and send particles
         *   3. delete invalidated particles
         *   4. receive particles
         */

        static IpplTimings::TimerRef locateTimer = IpplTimings::getTimer("locateParticles");
        IpplTimings::startTimer(locateTimer);
        size_type localnum = pdata.getLocalNum();

        // 1st step

        /* the values specify the rank where
         * the particle with that index should go
         */
        locate_type ranks("MPI ranks", localnum);

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        locateParticles(pdata, ranks, invalid);
        IpplTimings::stopTimer(locateTimer);

        // 2nd step

        // figure out how many receives
        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);
        MPI_Win win;
        std::vector<size_type> nRecvs(nRanks, 0);
        MPI_Win_create(nRecvs.data(), nRanks * sizeof(size_type), sizeof(size_type), MPI_INFO_NULL,
                       Ippl::getComm(), &win);

        std::vector<size_type> nSends(nRanks, 0);

        MPI_Win_fence(0, win);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == Ippl::Comm->rank()) {
                // we do not need to send to ourselves
                continue;
            }
            nSends[rank] = numberOfSends(rank, ranks);
            MPI_Put(nSends.data() + rank, 1, MPI_LONG_LONG_INT, rank, Ippl::Comm->rank(), 1,
                    MPI_LONG_LONG_INT, win);
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
        IpplTimings::stopTimer(preprocTimer);

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);
        // send
        std::vector<MPI_Request> requests(0);

        using buffer_type = Communicate::buffer_type;

        int tag = Ippl::Comm->next_tag(P_SPATIAL_LAYOUT_TAG, P_LAYOUT_CYCLE);

        int sends = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nSends[rank] > 0) {
                hash_type hash("hash", nSends[rank]);
                fillHash(rank, ranks, hash);

                requests.resize(requests.size() + 1);

                pdata.pack(buffer, hash);
                size_type bufSize = pdata.packedSize(nSends[rank]);

                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARTICLE_SEND + sends, bufSize);

                Ippl::Comm->isend(rank, tag, buffer, *buf, requests.back(), nSends[rank]);
                buf->resetWritePos();

                ++sends;
            }
        }
        IpplTimings::stopTimer(sendTimer);

        // 3rd step
        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        size_type invalidCount = 0;
        auto pIDs              = pdata.ID.getView();
        Kokkos::parallel_reduce(
            "set/count invalid", localnum,
            KOKKOS_LAMBDA(const size_t i, size_type& nInvalid) {
                if (invalid(i)) {
                    pIDs(i) = -1;
                    nInvalid += 1;
                }
            },
            invalidCount);
        Kokkos::fence();

        pdata.destroy(invalid, invalidCount);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);
        // 4th step
        int recvs = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs[rank] > 0) {
                size_type bufSize = pdata.packedSize(nRecvs[rank]);
                buffer_type buf   = Ippl::Comm->getBuffer(IPPL_PARTICLE_RECV + recvs, bufSize);

                Ippl::Comm->recv(rank, tag, buffer, *buf, bufSize, nRecvs[rank]);
                buf->resetReadPos();

                pdata.unpack(buffer, nRecvs[rank]);

                ++recvs;
            }
        }
        IpplTimings::stopTimer(recvTimer);

        IpplTimings::startTimer(sendTimer);

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        IpplTimings::stopTimer(sendTimer);

        IpplTimings::stopTimer(ParticleUpdateTimer);
    }

    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::locateParticles(
        const ParticleBase<ParticleSpatialLayout<T, Dim, Mesh>>& pdata, locate_type& ranks,
        bool_type& invalid) const {
        auto& positions                            = pdata.R.getView();
        typename RegionLayout_t::view_type Regions = rlayout_m.getdLocalRegions();
        using view_size_t                          = typename RegionLayout_t::view_type::size_type;
        using mdrange_type                         = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        int myRank                                 = Ippl::Comm->rank();
        Kokkos::parallel_for(
            "ParticleSpatialLayout::locateParticles()",
            mdrange_type({0, 0}, {ranks.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const view_size_t j) {
                bool xyz_bool = false;
                xyz_bool      = ((positions(i)[0] >= Regions(j)[0].min())
                            && (positions(i)[0] <= Regions(j)[0].max())
                            && (positions(i)[1] >= Regions(j)[1].min())
                            && (positions(i)[1] <= Regions(j)[1].max())
                            && (positions(i)[2] >= Regions(j)[2].min())
                            && (positions(i)[2] <= Regions(j)[2].max()));
                if (xyz_bool) {
                    ranks(i)   = j;
                    invalid(i) = (myRank != ranks(i));
                }
            });
        Kokkos::fence();
    }

    template <typename T, unsigned Dim, class Mesh>
    void ParticleSpatialLayout<T, Dim, Mesh>::fillHash(int rank, const locate_type& ranks,
                                                       hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        Kokkos::parallel_scan(
            "ParticleSpatialLayout::fillHash()", ranks.extent(0),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final) {
                    if (rank == ranks(i)) {
                        hash(idx) = i;
                    }
                }

                if (rank == ranks(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();
    }

    template <typename T, unsigned Dim, class Mesh>
    size_t ParticleSpatialLayout<T, Dim, Mesh>::numberOfSends(int rank, const locate_type& ranks) {
        size_t nSends = 0;
        Kokkos::parallel_reduce(
            "ParticleSpatialLayout::numberOfSends()", ranks.extent(0),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }
}  // namespace ippl
