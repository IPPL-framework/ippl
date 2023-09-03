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

namespace ippl {

    template <typename T, unsigned Dim, typename... IP>
    ParticleBase<T, Dim, IP...>::ParticleBase()
        : layout_m(nullptr)
        , localNum_m(0)
        , nextID_m(Comm->rank())
        , numNodes_m(Comm->size()) {
        if constexpr (EnableIDs) {
            addAttribute(ID);
        }
        addAttribute(R);
    }

    template <typename T, unsigned Dim, typename... IP>
    ParticleBase<T, Dim, IP...>::ParticleBase(Layout_t& layout)
        : ParticleBase() {
        initialize(layout);
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename MemorySpace>
    void ParticleBase<T, Dim, IP...>::addAttribute(detail::ParticleAttribBase<MemorySpace>& pa) {
        attributes_m.template get<MemorySpace>().push_back(&pa);
        pa.setParticleCount(localNum_m);
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::initialize(Layout_t& layout) {
        //         PAssert(layout_m == nullptr);

        // save the layout, and perform setup tasks
        layout_m = &layout;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::create(size_type nLocal) {
        PAssert(layout_m != nullptr);

        forAllAttributes([&]<typename Attribute>(Attribute& attribute) {
            attribute->create(nLocal);
        });

        if constexpr (EnableIDs) {
            // set the unique ID value for these new particles
            using policy_type =
                Kokkos::RangePolicy<size_type, typename particle_index_type::execution_space>;
            auto pIDs     = ID.getView();
            auto nextID   = this->nextID_m;
            auto numNodes = this->numNodes_m;
            Kokkos::parallel_for(
                "ParticleBase<...>::create(size_t)", policy_type(localNum_m, nLocal),
                KOKKOS_LAMBDA(const std::int64_t i) { pIDs(i) = nextID + numNodes * i; });
            // nextID_m += numNodes_m * (nLocal - localNum_m);
            nextID_m += numNodes_m * nLocal;
        }

        // remember that we're creating these new particles
        localNum_m += nLocal;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::createWithID(index_type id) {
        PAssert(layout_m != nullptr);

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m             = id;
        numNodes_m           = 0;

        create(1);

        nextID_m   = tmpNextID;
        numNodes_m = Comm->getNodes();
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::globalCreate(size_type nTotal) {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_type nLocal = nTotal / numNodes_m;

        const size_t rank = Comm->myNode();

        size_type rest = nTotal - nLocal * rank;
        if (rank < rest) {
            ++nLocal;
        }

        create(nLocal);
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename... Properties>
    void ParticleBase<T, Dim, IP...>::destroy(const Kokkos::View<bool*, Properties...>& invalid,
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

        using view_type       = Kokkos::View<bool*, Properties...>;
        using memory_space    = typename view_type::memory_space;
        using execution_space = typename view_type::execution_space;
        using policy_type     = Kokkos::RangePolicy<execution_space>;
        auto& locDeleteIndex  = deleteIndex_m.get<memory_space>();
        auto& locKeepIndex    = keepIndex_m.get<memory_space>();

        // Resize buffers, if necessary
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            if (attributes_m.template get<memory_space>().size() > 0) {
                int overalloc = Comm->getDefaultOverallocation();
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
            "Scan in ParticleBase::destroy()", policy_type(0, localNum_m - destroyNum),
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
            "Reduce in ParticleBase::destroy()", policy_type(0, destroyNum),
            KOKKOS_LAMBDA(const size_t i, size_t& maxIdx) {
                if (locDeleteIndex(i) >= 0 && i > maxIdx) {
                    maxIdx = i;
                }
            },
            Kokkos::Max<size_type>(maxDeleteIndex));

        // Find the indices of the valid particles in the invalid region
        Kokkos::parallel_scan(
            "Second scan in ParticleBase::destroy()",
            Kokkos::RangePolicy<size_type, execution_space>(localNum_m - destroyNum, localNum_m),
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
        // NOTE: The vector elements are pointers, but we want to extract
        // the memory space from the class type, so we explicitly
        // make the lambda argument a pointer to the template parameter
        forAllAttributes([&]<typename Attribute>(Attribute*& attribute) {
            using att_memory_space = typename Attribute::memory_space;
            auto& del              = deleteIndex_m.get<att_memory_space>();
            auto& keep             = keepIndex_m.get<att_memory_space>();
            attribute->destroy(del, keep, maxDeleteIndex + 1);
        });
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename HashType>
    void ParticleBase<T, Dim, IP...>::sendToRank(int rank, int tag, int sendNum,
                                                 std::vector<MPI_Request>& requests,
                                                 const HashType& hash) {
        size_type nSends = hash.size();
        requests.resize(requests.size() + 1);

        auto hashes = hash_container_type(hash, [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        });
        pack(hashes);
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nSends);
            if (bufSize == 0) {
                return;
            }

            auto buf = Comm->getBuffer<MemorySpace>(IPPL_PARTICLE_SEND + sendNum, bufSize);

            Comm->isend(rank, tag++, *this, *buf, requests.back(), nSends);
            buf->resetWritePos();
        });
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::recvFromRank(int rank, int tag, int recvNum,
                                                   size_type nRecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            if (bufSize == 0) {
                return;
            }

            auto buf = Comm->getBuffer<MemorySpace>(IPPL_PARTICLE_RECV + recvNum, bufSize);

            Comm->recv(rank, tag++, *this, *buf, bufSize, nRecvs);
            buf->resetReadPos();
        });
        unpack(nRecvs);
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename Archive>
    void ParticleBase<T, Dim, IP...>::serialize(Archive& ar, size_type nsends) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->serialize(ar, nsends);
        });
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename Archive>
    void ParticleBase<T, Dim, IP...>::deserialize(Archive& ar, size_type nrecvs) {
        using memory_space = typename Archive::buffer_type::memory_space;
        forAllAttributes<memory_space>([&]<typename Attribute>(Attribute& att) {
            att->deserialize(ar, nrecvs);
        });
    }

    template <typename T, unsigned Dim, typename... IP>
    template <typename MemorySpace>
    detail::size_type ParticleBase<T, Dim, IP...>::packedSize(const size_type count) const {
        size_type total = 0;
        forAllAttributes<MemorySpace>([&]<typename Attribute>(const Attribute& att) {
            total += att->packedSize(count);
        });
        return total;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::pack(const hash_container_type& hash) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->pack(hash.template get<MemorySpace>());
            }
        });
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::unpack(size_type nrecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->unpack(nrecvs);
            }
        });
        localNum_m += nrecvs;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::update() {
        static IpplTimings::TimerRef ParticleBCTimer = IpplTimings::getTimer("particleBC");
        IpplTimings::startTimer(ParticleBCTimer);
        this->applyBC(layout_m->getDomain());
        IpplTimings::stopTimer(ParticleBCTimer);

        static IpplTimings::TimerRef ParticleUpdateTimer = IpplTimings::getTimer("updateParticle");
        IpplTimings::startTimer(ParticleUpdateTimer);
        int nRanks = Comm->size();

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
        size_type localnum = this->getLocalNum();

        // 1st step

        /* the values specify the rank where
         * the particle with that index should go
         */
        locate_type ranks("MPI ranks", localnum);

        /* 0 --> particle valid
         * 1 --> particle invalid
         */
        bool_type invalid("invalid", localnum);

        size_type invalidCount = locateParticles(ranks, invalid);
        IpplTimings::stopTimer(locateTimer);

        // 2nd step

        // figure out how many receives
        static IpplTimings::TimerRef preprocTimer = IpplTimings::getTimer("sendPreprocess");
        IpplTimings::startTimer(preprocTimer);
        MPI_Win win;
        std::vector<size_type> nRecvs(nRanks, 0);
        MPI_Win_create(nRecvs.data(), nRanks * sizeof(size_type), sizeof(size_type), MPI_INFO_NULL,
                       Comm->getCommunicator(), &win);

        std::vector<size_type> nSends(nRanks, 0);

        MPI_Win_fence(0, win);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == Comm->rank()) {
                // we do not need to send to ourselves
                continue;
            }
            nSends[rank] = numberOfSends(rank, ranks);
            MPI_Put(nSends.data() + rank, 1, MPI_LONG_LONG_INT, rank, Comm->rank(), 1,
                    MPI_LONG_LONG_INT, win);
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
        IpplTimings::stopTimer(preprocTimer);

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        IpplTimings::startTimer(sendTimer);
        // send
        std::vector<MPI_Request> requests(0);

        int tag = Comm->next_tag(P_SPATIAL_LAYOUT_TAG, P_LAYOUT_CYCLE);

        int sends = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nSends[rank] > 0) {
                hash_type hash("hash", nSends[rank]);
                fillHash(rank, ranks, hash);

                this->sendToRank(rank, tag, sends++, requests, hash);
            }
        }
        IpplTimings::stopTimer(sendTimer);

        // 3rd step
        static IpplTimings::TimerRef destroyTimer = IpplTimings::getTimer("particleDestroy");
        IpplTimings::startTimer(destroyTimer);

        this->destroy(invalid, invalidCount);
        Kokkos::fence();

        IpplTimings::stopTimer(destroyTimer);
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        IpplTimings::startTimer(recvTimer);
        // 4th step
        int recvs = 0;
        for (int rank = 0; rank < nRanks; ++rank) {
            if (nRecvs[rank] > 0) {
                this->recvFromRank(rank, tag, recvs++, nRecvs[rank]);
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

    template <typename T, unsigned Dim, typename... IP>
    template <size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr bool ParticleBase<T, Dim, IP...>::positionInRegion(
        const std::index_sequence<Idx...>&, const vector_type& pos, const region_type& region) {
        return ((pos[Idx] >= region[Idx].min()) && ...) && ((pos[Idx] <= region[Idx].max()) && ...);
    };

    template <typename T, unsigned Dim, typename... IP>
    detail::size_type ParticleBase<T, Dim, IP...>::locateParticles(locate_type& ranks,
                                                                   bool_type& invalid) const {
        auto& positions                            = this->R.getView();
        typename RegionLayout_t::view_type Regions = layout_m->getdLocalRegions();

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, position_execution_space>;

        int myRank = Comm->rank();

        const auto is = std::make_index_sequence<Dim>{};

        size_type invalidCount = 0;
        Kokkos::parallel_reduce(
            "ParticleBase::locateParticles()",
            mdrange_type({0, 0}, {ranks.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j, size_type& count) {
                bool xyz_bool = positionInRegion(is, positions(i), Regions(j));
                if (xyz_bool) {
                    ranks(i)   = j;
                    invalid(i) = (myRank != ranks(i));
                    count += invalid(i);
                }
            },
            Kokkos::Sum<size_type>(invalidCount));
        Kokkos::fence();

        return invalidCount;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::fillHash(int rank, const locate_type& ranks,
                                               hash_type& hash) {
        /* Compute the prefix sum and fill the hash
         */
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_scan(
            "ParticleBase::fillHash()", policy_type(0, ranks.extent(0)),
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

    template <typename T, unsigned Dim, typename... IP>
    size_t ParticleBase<T, Dim, IP...>::numberOfSends(int rank, const locate_type& ranks) {
        size_t nSends     = 0;
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_reduce(
            "ParticleBase::numberOfSends()", policy_type(0, ranks.extent(0)),
            KOKKOS_LAMBDA(const size_t i, size_t& num) { num += size_t(rank == ranks(i)); },
            nSends);
        Kokkos::fence();
        return nSends;
    }

    template <typename T, unsigned Dim, typename... IP>
    void ParticleBase<T, Dim, IP...>::applyBC(const NDRegion<T, Dim>& nr) {
        /* loop over all faces
         * 0: lower x-face
         * 1: upper x-face
         * 2: lower y-face
         * 3: upper y-face
         * etc...
         */
        Kokkos::RangePolicy<position_execution_space> policy{0,
                                                             (unsigned)this->R.getParticleCount()};
        for (unsigned face = 0; face < 2 * Dim; ++face) {
            // unsigned face = i % Dim;
            unsigned d   = face / 2;
            bool isUpper = face & 1;
            switch (bcs_m[face]) {
                case BC::PERIODIC:
                    // Periodic faces come in pairs and the application of
                    // BCs checks both sides, so there is no reason to
                    // apply periodic conditions twice
                    if (isUpper) {
                        break;
                    }

                    Kokkos::parallel_for("Periodic BC", policy,
                                         detail::PeriodicBC(this->R.getView(), nr, d, isUpper));
                    break;
                case BC::REFLECTIVE:
                    Kokkos::parallel_for("Reflective BC", policy,
                                         detail::ReflectiveBC(this->R.getView(), nr, d, isUpper));
                    break;
                case BC::SINK:
                    Kokkos::parallel_for("Sink BC", policy,
                                         detail::SinkBC(this->R.getView(), nr, d, isUpper));
                    break;
                case BC::NO:
                default:
                    break;
            }
        }
    }

}  // namespace ippl
