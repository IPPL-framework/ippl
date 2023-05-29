//
// Class ParticleBase
//   Base class for all user-defined particle classes.
//
//   ParticleBase is a container and manager for a set of particles.
//   The user must define a class derived from ParticleBase which describes
//   what specific data attributes the particle has (e.g., mass or charge).
//   Each attribute is an instance of a ParticleAttribute<T> class; ParticleBase
//   keeps a list of pointers to these attributes, and performs particle creation
//   and destruction.
//
//   ParticleBase is templated on the ParticleLayout mechanism for the particles.
//   This template parameter should be a class derived from ParticleLayout.
//   ParticleLayout-derived classes maintain the info on which particles are
//   located on which processor, and performs the specific communication
//   required between processors for the particles.  The ParticleLayout is
//   templated on the type and dimension of the atom position attribute, and
//   ParticleBase uses the same types for these items as the given
//   ParticleLayout.
//
//   ParticleBase and all derived classes have the following common
//   characteristics:
//       - The spatial positions of the N particles are stored in the
//         particle_position_type variable R
//       - The global index of the N particles are stored in the
//         particle_index_type variable ID
//       - A pointer to an allocated layout class.  When you construct a
//         ParticleBase, you must provide a layout instance, and ParticleBase
//         will delete this instance when it (the ParticleBase) is deleted.
//
//   To use this class, the user defines a derived class with the same
//   structure as in this example:
//
//     class UserParticles :
//              public ParticleBase< ParticleSpatialLayout<double,3> > {
//     public:
//       // attributes for this class
//       ParticleAttribute<double> rad;  // radius
//       particle_position_type    vel;  // velocity, same storage type as R
//
//       // constructor: add attributes to base class
//       UserParticles(ParticleSpatialLayout<double,2>* L) : ParticleBase(L) {
//         addAttribute(rad);
//         addAttribute(vel);
//       }
//     };
//
//   This example defines a user class with 3D position and two extra
//   attributes: a radius rad (double), and a velocity vel (a 3D Vector).
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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

namespace ippl {

    template <class PLayout, typename... IP>
    ParticleBase<PLayout, IP...>::ParticleBase()
        : layout_m(nullptr)
        , localNum_m(0)
        , nextID_m(Ippl::Comm->myNode())
        , numNodes_m(Ippl::Comm->getNodes()) {
        addAttribute(ID);  // needs to be added first due to destroy function
        addAttribute(R);
    }

    template <class PLayout, typename... IP>
    ParticleBase<PLayout, IP...>::ParticleBase(PLayout& layout)
        : ParticleBase() {
        initialize(layout);
    }

    template <class PLayout, typename... IP>
    template <typename... Properties>
    void ParticleBase<PLayout, IP...>::addAttribute(detail::ParticleAttribBase<Properties...>& pa) {
        using memory_space   = typename detail::ParticleAttribBase<Properties...>::memory_space;
        using attribute_type = detail::ParticleAttribBase<memory_space>;
        attributes_m.template get<memory_space>().push_back((attribute_type*)&pa);
        pa.setParticleCount(localNum_m);
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::initialize(PLayout& layout) {
        //         PAssert(layout_m == nullptr);

        // save the layout, and perform setup tasks
        layout_m = &layout;
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::create(size_type nLocal) {
        PAssert(layout_m != nullptr);

        attributes_m.forAll([&]<typename Attributes>(Attributes& att) {
            for (auto& attribute : att) {
                attribute->create(nLocal);
            }
        });

        // set the unique ID value for these new particles
        auto pIDs     = ID.getView();
        auto nextID   = this->nextID_m;
        auto numNodes = this->numNodes_m;
        Kokkos::parallel_for(
            "ParticleBase<...>::create(size_t)", Kokkos::RangePolicy<size_type>(localNum_m, nLocal),
            KOKKOS_LAMBDA(const std::int64_t i) { pIDs(i) = nextID + numNodes * i; });
        // nextID_m += numNodes_m * (nLocal - localNum_m);
        nextID_m += numNodes_m * nLocal;

        // remember that we're creating these new particles
        localNum_m += nLocal;
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::createWithID(index_type id) {
        PAssert(layout_m != nullptr);

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m             = id;
        numNodes_m           = 0;

        create(1);

        nextID_m   = tmpNextID;
        numNodes_m = Ippl::Comm->getNodes();
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::globalCreate(size_type nTotal) {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_type nLocal = nTotal / numNodes_m;

        const size_t rank = Ippl::Comm->myNode();

        size_type rest = nTotal - nLocal * rank;
        if (rank < rest) {
            ++nLocal;
        }

        create(nLocal);
    }

    template <class PLayout, typename... IP>
    template <typename... Properties>
    void ParticleBase<PLayout, IP...>::destroy(const Kokkos::View<bool*, Properties...>& invalid,
                                               const size_type destroyNum) {
        PAssert(destroyNum <= localNum_m);

        // If there aren't any particles to delete, do nothing
        if (destroyNum == 0) {
            return;
        }

        // If we're deleting all the particles, there's no point in doing
        // anything because the valid region will be empty; we only need to
        // update the particle count
        if (destroyNum == localNum_m) {
            localNum_m = 0;
            return;
        }

        using memory_space   = typename Kokkos::View<bool*, Properties...>::memory_space;
        auto& locDeleteIndex = deleteIndex_m.get<memory_space>();
        auto& locKeepIndex   = keepIndex_m.get<memory_space>();

        // Resize buffers, if necessary
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            if (attributes_m.template get<memory_space>().size() > 0) {
                int overalloc = Ippl::Comm->getDefaultOverallocation();
                auto& del     = deleteIndex_m.get<memory_space>();
                auto& keep    = keepIndex_m.get<memory_space>();
                if (del.size() < destroyNum) {
                    Kokkos::realloc(del, destroyNum * overalloc);
                    Kokkos::realloc(keep, destroyNum * overalloc);
                }
            }
        });

        // Reset index buffer
        Kokkos::deep_copy(locDeleteIndex, -1);

        // Find the indices of the invalid particles in the valid region
        Kokkos::parallel_scan(
            "Scan in ParticleBase::destroy()", localNum_m - destroyNum,
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && invalid(i)) {
                    locDeleteIndex(idx) = i;
                }
                if (invalid(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();

        // Determine the total number of invalid particles in the valid region
        size_type maxDeleteIndex = 0;
        Kokkos::parallel_reduce(
            "Reduce in ParticleBase::destroy()", destroyNum,
            KOKKOS_LAMBDA(const size_t i, size_t& maxIdx) {
                if (locDeleteIndex(i) >= 0 && i > maxIdx) {
                    maxIdx = i;
                }
            },
            Kokkos::Max<size_type>(maxDeleteIndex));

        // Find the indices of the valid particles in the invalid region
        Kokkos::parallel_scan(
            "Second scan in ParticleBase::destroy()",
            Kokkos::RangePolicy<size_type>(localNum_m - destroyNum, localNum_m),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && !invalid(i)) {
                    locKeepIndex(idx) = i;
                }
                if (!invalid(i)) {
                    idx += 1;
                }
            });

        Kokkos::fence();

        localNum_m -= destroyNum;

        auto filter = [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        };
        deleteIndex_m.copyToOtherSpaces<memory_space>(filter);
        keepIndex_m.copyToOtherSpaces<memory_space>(filter);

        // Partition the attributes into valid and invalid regions
        attributes_m.forAll([&]<typename Attributes>(Attributes& att) {
            using att_memory_space =
                typename std::remove_pointer_t<typename Attributes::value_type>::memory_space;
            auto& del  = deleteIndex_m.get<att_memory_space>();
            auto& keep = keepIndex_m.get<att_memory_space>();
            for (auto& attribute : att) {
                attribute->destroy(del, keep, maxDeleteIndex + 1);
            }
        });
    }

    template <class PLayout, typename... IP>
    template <typename HashType, typename BufferType>
    void ParticleBase<PLayout, IP...>::sendToRank(int rank, int tag, int& sendNum,
                                                  std::vector<MPI_Request>& requests,
                                                  const HashType& hash, BufferType& buffer) {
        size_type nSends = hash.size();
        requests.resize(requests.size() + 1);

        auto hashes = hash_container_type(hash, [&]<typename MemSpace>() {
            return attributes_m.template get<MemSpace>().size() > 0;
        });
        pack(buffer, hashes);
        detail::runForAllSpaces([&]<typename MemSpace>() {
            size_type bufSize = packedSize<MemSpace>(nSends);
            if (bufSize == 0) {
                return;
            }

            auto buf = Ippl::Comm->getBuffer<MemSpace>(IPPL_PARTICLE_SEND + sendNum, bufSize);

            Ippl::Comm->isend(rank, tag, buffer, *buf, requests.back(), nSends);
            buf->resetWritePos();

            ++sendNum;
        });
    }

    template <class PLayout, typename... IP>
    template <typename BufferType>
    void ParticleBase<PLayout, IP...>::recvFromRank(int rank, int tag, int& recvNum,
                                                    size_type nRecvs, BufferType& buffer) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            auto buf = Ippl::Comm->getBuffer<MemorySpace>(IPPL_PARTICLE_RECV + recvNum, bufSize);

            Ippl::Comm->recv(rank, tag, buffer, *buf, bufSize, nRecvs);
            buf->resetReadPos();

            ++recvNum;
        });
        unpack(buffer, nRecvs);
    }

    template <class PLayout, typename... IP>
    template <typename Archive>
    void ParticleBase<PLayout, IP...>::serialize(Archive& ar, size_type nsends) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->serialize(ar, nsends);
        });
    }

    template <class PLayout, typename... IP>
    template <typename Archive>
    void ParticleBase<PLayout, IP...>::deserialize(Archive& ar, size_type nrecvs) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->deserialize(ar, nrecvs);
        });
    }

    template <class PLayout, typename... IP>
    template <typename MemorySpace>
    detail::size_type ParticleBase<PLayout, IP...>::packedSize(const size_type count) const {
        size_type total = 0;
        forAllAttributes<MemorySpace>([&]<typename Attribute>(const Attribute& att) {
            total += att->packedSize(count);
        });
        return total;
    }

    template <class PLayout, typename... IP>
    template <class Buffer>
    void ParticleBase<PLayout, IP...>::pack(Buffer& buffer, const hash_container_type& hash) {
        forAllAttributes([&]<typename Attributes>(Attributes& att) {
            using memory_space =
                typename std::remove_pointer_t<typename Attributes::value_type>::memory_space;
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->pack(buffer.template getAttribute<memory_space>(j),
                             hash.template get<memory_space>());
            }
        });
    }

    template <class PLayout, typename... IP>
    template <class Buffer>
    void ParticleBase<PLayout, IP...>::unpack(Buffer& buffer, size_type nrecvs) {
        forAllAttributes([&]<typename Attributes>(Attributes& att) {
            using memory_space =
                typename std::remove_pointer_t<typename Attributes::value_type>::memory_space;
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->unpack(buffer.template getAttribute<memory_space>(j), nrecvs);
            }
            localNum_m += nrecvs;
        });
    }
}  // namespace ippl
