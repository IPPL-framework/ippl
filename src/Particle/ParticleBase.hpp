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

    template <class PLayout, typename... IP>
    ParticleBase<PLayout, IP...>::ParticleBase()
        : layout_m(nullptr)
        , localNum_m(0)
        , totalNum_m(0)
        , nextID_m(Comm->rank())
        , numNodes_m(Comm->size()) {
        if constexpr (EnableIDs) {
            addAttribute(ID);
        }
        addAttribute(R);
    }

    template <class PLayout, typename... IP>
    ParticleBase<PLayout, IP...>::ParticleBase(PLayout& layout)
        : ParticleBase() {
        initialize(layout);
    }

    template <class PLayout, typename... IP>
    template <typename MemorySpace>
    void ParticleBase<PLayout, IP...>::addAttribute(detail::ParticleAttribBase<MemorySpace>& pa) {
        attributes_m.template get<MemorySpace>().push_back(&pa);
        pa.setParticleCount(localNum_m);
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::initialize(PLayout& layout) {
        //         PAssert(layout_m == nullptr);

        // save the layout, and perform setup tasks
        layout_m = &layout;
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::create(size_type nLocal, bool non_destructive) {
        PAssert(layout_m != nullptr);

        if (nLocal > 0) {
            forAllAttributes([&]<typename Attribute>(Attribute& attribute) {
                attribute->create(nLocal, non_destructive);
            });

            if constexpr (EnableIDs) {
                // Set the unique ID value for these new particles. The new entries
                // live at indices [localNum_m, localNum_m + nLocal); the k-th new
                // particle on this rank gets ID nextID + numNodes * k.
                using policy_type =
                    Kokkos::RangePolicy<size_type, typename particle_index_type::execution_space>;
                auto pIDs     = ID.getView();
                auto nextID   = this->nextID_m;
                auto numNodes = this->numNodes_m;
                auto offset   = localNum_m;
                Kokkos::parallel_for(
                    "ParticleBase<...>::create(size_t,bool)",
                    policy_type(localNum_m, localNum_m + nLocal),
                    KOKKOS_LAMBDA(const std::int64_t i) {
                        pIDs(i) = nextID + numNodes * (i - offset);
                    });
                nextID_m += numNodes_m * nLocal;
            }

            // remember that we're creating these new particles
            localNum_m += nLocal;
        }

        Comm->allreduce(localNum_m, totalNum_m, 1, std::plus<size_type>());
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::alloc(size_type nLocal) {
        PAssert(layout_m != nullptr);
        forAllAttributes([&]<typename Attribute>(Attribute& attribute) {
            attribute->alloc(nLocal);
        });
        // Intentionally does NOT touch localNum_m, totalNum_m, nextID_m.
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::createWithID(index_type id) {
        PAssert(layout_m != nullptr);

        size_type n = (id > -1) ? 1 : 0;

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m             = id;
        numNodes_m           = 0;

        create(n);

        nextID_m   = tmpNextID;
        numNodes_m = Comm->size();
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::globalCreate(size_type nTotal) {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_type nLocal = nTotal / numNodes_m;

        const size_t rank = Comm->rank();

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
        this->internalDestroy(invalid, destroyNum);

        Comm->allreduce(localNum_m, totalNum_m, 1, std::plus<size_type>());
    }

    template <class PLayout, typename... IP>
    template <typename memory_space, typename execution_space, typename F, typename... Properties>
    void ParticleBase<PLayout, IP...>::internalDestroy(const F& invalid_functor,
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

        using policy_type    = Kokkos::RangePolicy<execution_space>;
        auto& locDeleteIndex = deleteIndex_m.get<memory_space>();
        auto& locKeepIndex   = keepIndex_m.get<memory_space>();

        // Resize buffers per-memory-space. The previous code mistakenly used
        // the outer `memory_space` template parameter inside the per-space
        // lambda, leaving every other space's del/keep buffers undersized.
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            if (attributes_m.template get<MemorySpace>().size() > 0) {
                int overalloc = Comm->getDefaultOverallocation();
                auto& del     = deleteIndex_m.template get<MemorySpace>();
                auto& keep    = keepIndex_m.template get<MemorySpace>();
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
                if (final && invalid_functor(i)) {
                    locDeleteIndex(idx) = i;
                }
                if (invalid_functor(i)) {
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
                if (final && !invalid_functor(i)) {
                    locKeepIndex(idx) = i;
                }
                if (!invalid_functor(i)) {
                    idx += 1;
                }
            });

        Kokkos::fence();

        localNum_m -= destroyNum;

        // We need to delete particles in all memory spaces. If there are any attributes not stored
        // in the memory space we've already been using, we need to copy the index views to the
        // other spaces.
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

    template <class PLayout, typename... IP>
    template <typename HashType>
    std::vector<MPI_Request> ParticleBase<PLayout, IP...>::sendToRank(int rank, int tag,
                                                                      const HashType& hash) {
        std::vector<MPI_Request> requests;
        sendToRank(rank, tag, requests, hash);
        return requests;
    }

    template <class PLayout, typename... IP>
    template <typename HashType>
    void ParticleBase<PLayout, IP...>::sendToRank(int rank, int tag,
                                                  std::vector<MPI_Request>& requests,
                                                  const HashType& hash) {
        const size_type nSends = hash.size();

        auto hashes = hash_container_type(hash, [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        });
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nSends);
            if (bufSize == 0) {
                return;
            }
            auto buf = Comm->getBuffer<MemorySpace>(bufSize);

            forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                att->serialize(*buf, hashes.template get<MemorySpace>(), nSends);
            });

            Comm->isend(rank, tag++, *buf, requests);
            buf->resetWritePos();
        });
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::recvFromRank(int rank, int tag, size_type nRecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            if (bufSize == 0) {
                return;
            }

            auto buf = Comm->getBuffer<MemorySpace>(bufSize);

            Comm->recv(rank, tag++, *buf, bufSize);
            forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                att->deserialize(*buf, localNum_m, nRecvs);
            });

            buf->resetReadPos();
        });
        localNum_m += nRecvs;
    }

    template <class PLayout, typename... IP>
    std::pair<std::vector<MPI_Request>, std::function<void(size_t)>>
    ParticleBase<PLayout, IP...>::postRecvFromRank(int rank, int tag, size_type nRecvs) {
        std::vector<MPI_Request> requests;

        // Collect (buf, nRecvs) per memory space for deferred deserialization
        auto deferred = std::make_shared<std::vector<std::function<void(size_type)>>>();

        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            if (bufSize == 0)
                return;

            auto buf = Comm->getBuffer<MemorySpace>(bufSize);
            Comm->irecv(rank, tag++, *buf, requests, bufSize);

            deferred->push_back([this, buf, nRecvs](size_type offset) {
                forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                    att->deserialize(*buf, offset, nRecvs);
                });
                buf->resetReadPos();
            });
        });

        // Finalize takes the offset (localNum_m after destruction) and advances it
        return {std::move(requests), [this, deferred, nRecvs](size_type offset) {
                    for (auto& fn : *deferred)
                        fn(offset);
                    localNum_m += nRecvs;
                }};
    }

    template <class PLayout, typename... IP>
    template <typename HashType>
    std::function<void()> ParticleBase<PLayout, IP...>::postSendToRank(
        int rank, int tag, std::vector<MPI_Request>& requests, const HashType& hash) {
        const size_type nSends = hash.size();
        auto releases          = std::make_shared<std::vector<std::function<void()>>>();

        auto hashes = hash_container_type(hash, [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        });
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nSends);
            if (bufSize == 0) {
                return;
            }
            auto buf = Comm->getBuffer<MemorySpace>(bufSize);

            forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                att->serialize(*buf, hashes.template get<MemorySpace>(), nSends);
            });

            Comm->isend(rank, tag++, *buf, requests);
            buf->resetWritePos();
            releases->push_back([buf]() {
                Comm->freeBuffer<MemorySpace>(buf);
            });
        });
        return [releases]() {
            for (auto& release : *releases)
                release();
        };
    }

    template <class PLayout, typename... IP>
    std::pair<std::vector<MPI_Request>,
              std::function<void(detail::size_type, const typename ParticleBase<
                                                        PLayout, IP...>::hash_container_type*)>>
    ParticleBase<PLayout, IP...>::postRecvFromRankIndexed(int rank, int tag, size_type nRecvs) {
        using finalizer_type = std::function<void(size_type, const hash_container_type*)>;
        std::vector<MPI_Request> requests;
        auto deferred = std::make_shared<std::vector<finalizer_type>>();

        detail::runForAllSpaces([&]<typename MemorySpace>() {
            size_type bufSize = packedSize<MemorySpace>(nRecvs);
            if (bufSize == 0)
                return;

            auto buf = Comm->getBuffer<MemorySpace>(bufSize);
            Comm->irecv(rank, tag++, *buf, requests, bufSize);

            deferred->push_back([this, buf, nRecvs](size_type offset,
                                                    const hash_container_type* targets) {
                forAllAttributes<MemorySpace>([&]<typename Attribute>(Attribute& att) {
                    if (targets) {
                        att->deserializeIndexed(*buf, targets->template get<MemorySpace>(), nRecvs);
                    } else {
                        att->deserialize(*buf, offset, nRecvs);
                    }
                });
                buf->resetReadPos();
                Comm->freeBuffer<MemorySpace>(buf);
            });
        });

        return {std::move(requests),
                [deferred](size_type offset, const hash_container_type* targets) {
                    for (auto& fn : *deferred)
                        fn(offset, targets);
                }};
    }

    template <class PLayout, typename... IP>
    template <typename HashType>
    void ParticleBase<PLayout, IP...>::exchangeInRounds(
        int tag, const std::vector<std::pair<int, HashType>>& sends,
        const std::vector<std::pair<int, size_type>>& recvs, bool sentParticlesLeave) {
        using index_memory_space = typename HashType::memory_space;
        using index_exec_space   = typename HashType::execution_space;
        using index_view_type    = detail::hash_type<index_memory_space>;
        using policy_type        = Kokkos::RangePolicy<index_exec_space>;

        static IpplTimings::TimerRef sendTimer = IpplTimings::getTimer("particleSend");
        static IpplTimings::TimerRef recvTimer = IpplTimings::getTimer("particleRecv");
        static IpplTimings::TimerRef waitTimer = IpplTimings::getTimer("particleWait");
        static IpplTimings::TimerRef deserializeTimer =
            IpplTimings::getTimer("particleDeserialize");
        static IpplTimings::TimerRef reserveTimer    = IpplTimings::getTimer("particleDeserResize");
        static IpplTimings::TimerRef freeBufferTimer = IpplTimings::getTimer("particleFreeBuffers");
        static IpplTimings::TimerRef destroyTimer    = IpplTimings::getTimer("particleDestroy");

        size_type bytesPerParticle = 0;
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            bytesPerParticle += packedSize<MemorySpace>(1);
        });
        if (bytesPerParticle == 0) {
            return;
        }
        const size_type chunk =
            std::max<size_type>(1, Comm->getMaxMessageBytes() / bytesPerParticle);
        // Particles a list of the given length sends or receives in the given round.
        const auto chunkIn = [chunk](size_type count, size_type round) -> size_type {
            const size_type begin = round * chunk;
            return begin < count ? std::min(chunk, count - begin) : 0;
        };

        size_type rounds     = 0;
        size_type totalSends = 0;
        size_type maxRecv    = 0;
        for (const auto& [rank, ids] : sends) {
            rounds = std::max(rounds, (ids.extent(0) + chunk - 1) / chunk);
            totalSends += ids.extent(0);
        }
        for (const auto& [rank, count] : recvs) {
            rounds  = std::max(rounds, (count + chunk - 1) / chunk);
            maxRecv = std::max(maxRecv, count);
        }

        // Rounds are posted in order and completed oldest first, at most `window` in flight.
        // A sent particle's slot is free once its round is packed, since the send buffer holds
        // a copy; received particles take free slots in completion order.
        const size_type window = Comm->getMessageRoundsInFlight();

        // Replay that schedule on counts only to find how many received particles are
        // appended, so storage grows once instead of once per chunk.
        size_type appended = 0;
        {
            size_type freeSlots       = 0;
            const auto completeCounts = [&](size_type round) {
                for (const auto& [rank, count] : recvs) {
                    const size_type n    = chunkIn(count, round);
                    const size_type fill = std::min(n, freeSlots);
                    freeSlots -= fill;
                    appended += n - fill;
                }
            };
            for (size_type round = 0; round < rounds; ++round) {
                if (sentParticlesLeave) {
                    for (const auto& [rank, ids] : sends) {
                        freeSlots += chunkIn(ids.extent(0), round);
                    }
                }
                if (round + 1 >= window) {
                    completeCounts(round + 1 - window);
                }
            }
            for (size_type round = rounds >= window ? rounds - window + 1 : 0; round < rounds;
                 ++round) {
                completeCounts(round);
            }
        }

        IpplTimings::startTimer(reserveTimer);
        if (appended > 0) {
            const size_type capacity = localNum_m + appended;
            forAllAttributes([&]<typename Attribute>(Attribute*& attribute) {
                attribute->reserve(capacity);
            });
        }
        IpplTimings::stopTimer(reserveTimer);

        // Slots of packed particles in posting order; [slotHead, slotEnd) are free.
        const auto scratch = [](index_view_type& view, size_type size) -> index_view_type {
            if (view.extent(0) < size) {
                view = index_view_type(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "exchangeInRounds scratch"),
                    size);
            }
            return view;
        };
        index_view_type slots = scratch(exchangeSlots_m.template get<index_memory_space>(),
                                        sentParticlesLeave ? totalSends : 0);
        index_view_type targets =
            scratch(exchangeTargets_m.template get<index_memory_space>(), std::min(chunk, maxRecv));
        size_type slotHead = 0;
        size_type slotEnd  = 0;

        auto attributeSpaces = [&]<typename MemorySpace>() {
            return attributes_m.template get<MemorySpace>().size() > 0;
        };

        struct Round {
            std::vector<MPI_Request> requests;
            std::vector<std::function<void()>> releaseSends;
            std::vector<
                std::pair<size_type, std::function<void(size_type, const hash_container_type*)>>>
                finalizers;
        };
        std::deque<Round> inFlight;

        const auto complete = [&](Round& round) {
            IpplTimings::startTimer(waitTimer);
            if (!round.requests.empty()) {
                MPI_Waitall(static_cast<int>(round.requests.size()), round.requests.data(),
                            MPI_STATUSES_IGNORE);
            }
            IpplTimings::stopTimer(waitTimer);

            IpplTimings::startTimer(freeBufferTimer);
            for (auto& release : round.releaseSends) {
                release();
            }
            IpplTimings::stopTimer(freeBufferTimer);

            IpplTimings::startTimer(deserializeTimer);
            for (auto& [n, finalize] : round.finalizers) {
                const size_type fill  = std::min(n, slotEnd - slotHead);
                const size_type first = localNum_m;
                if (fill == 0) {
                    // Nothing to refill: append contiguously, which needs no index list.
                    localNum_m += n;
                    finalize(first, nullptr);
                    continue;
                }
                const size_type head = slotHead;
                auto slotsView       = slots;
                auto targetView      = targets;
                Kokkos::parallel_for(
                    "exchangeInRounds targets", policy_type(0, n), KOKKOS_LAMBDA(const size_t i) {
                        targetView(i) =
                            i < fill ? slotsView(head + i) : static_cast<int>(first + (i - fill));
                    });
                Kokkos::fence();
                slotHead += fill;
                localNum_m += n - fill;
                // Deserializes and returns the receive buffer to the pool.
                const hash_container_type chunkTargets(
                    Kokkos::subview(targets, Kokkos::make_pair(size_type(0), n)), attributeSpaces);
                finalize(first, &chunkTargets);
            }
            IpplTimings::stopTimer(deserializeTimer);
        };

        for (size_type round = 0; round < rounds; ++round) {
            const size_type begin = round * chunk;
            Round& posted         = inFlight.emplace_back();

            IpplTimings::startTimer(sendTimer);
            for (const auto& [rank, ids] : sends) {
                if (const size_type n = chunkIn(ids.extent(0), round); n > 0) {
                    const auto chunkIds = Kokkos::subview(ids, Kokkos::make_pair(begin, begin + n));
                    posted.releaseSends.push_back(
                        postSendToRank(rank, tag, posted.requests, chunkIds));
                    if (sentParticlesLeave) {
                        Kokkos::deep_copy(
                            Kokkos::subview(slots, Kokkos::make_pair(slotEnd, slotEnd + n)),
                            chunkIds);
                        slotEnd += n;
                    }
                }
            }
            IpplTimings::stopTimer(sendTimer);

            IpplTimings::startTimer(recvTimer);
            for (const auto& [rank, count] : recvs) {
                if (const size_type n = chunkIn(count, round); n > 0) {
                    auto [chunkRequests, finalize] = postRecvFromRankIndexed(rank, tag, n);
                    posted.requests.insert(posted.requests.end(), chunkRequests.begin(),
                                           chunkRequests.end());
                    posted.finalizers.emplace_back(n, std::move(finalize));
                }
            }
            IpplTimings::stopTimer(recvTimer);

            if (inFlight.size() >= window) {
                complete(inFlight.front());
                inFlight.pop_front();
            }
        }
        while (!inFlight.empty()) {
            complete(inFlight.front());
            inFlight.pop_front();
        }

        // Slots that no received particle refilled are holes; move tail particles into them.
        if (const size_type holes = slotEnd - slotHead; holes > 0) {
            IpplTimings::startTimer(destroyTimer);
            Kokkos::View<bool*, index_memory_space> isHole("exchangeInRounds holes", localNum_m);
            auto slotsView = slots;
            Kokkos::parallel_for(
                "exchangeInRounds mark holes", policy_type(slotHead, slotEnd),
                KOKKOS_LAMBDA(const size_t i) { isHole(slotsView(i)) = true; });
            Kokkos::fence();
            internalDestroy<index_memory_space, index_exec_space>(
                KOKKOS_LAMBDA(size_t i) { return isHole(i); }, holes);
            Kokkos::fence();
            IpplTimings::stopTimer(destroyTimer);
        }
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
    void ParticleBase<PLayout, IP...>::pack(const hash_container_type& hash) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->pack(hash.template get<MemorySpace>());
            }
        });
    }

    template <class PLayout, typename... IP>
    void ParticleBase<PLayout, IP...>::unpack(size_type nrecvs) {
        detail::runForAllSpaces([&]<typename MemorySpace>() {
            auto& att = attributes_m.template get<MemorySpace>();
            for (unsigned j = 0; j < att.size(); j++) {
                att[j]->unpack(nrecvs);
            }
        });
        localNum_m += nrecvs;
    }
}  // namespace ippl
