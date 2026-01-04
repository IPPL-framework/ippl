namespace ippl{

    template <typename T>
    FEMVector<T>::FEMVector(size_t n, std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs) : 
                data_m("FEMVector::data", n), boundaryInfo_m(new BoundaryInfo(std::move(neighbors),
                    std::move(sendIdxs), std::move(recvIdxs))) {
        
    }


    template <typename T>
    FEMVector<T>::FEMVector(size_t n) : data_m("FEMVector::data", n), boundaryInfo_m(nullptr){

    }


    
    template <typename T>
    KOKKOS_FUNCTION
    FEMVector<T>::FEMVector(const FEMVector<T>& other) : data_m(other.data_m),
                                                    boundaryInfo_m(other.boundaryInfo_m) {

    }


    template <typename T>
    FEMVector<T>::BoundaryInfo::BoundaryInfo(std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs) : 
            neighbors_m(neighbors), sendIdxs_m(sendIdxs), recvIdxs_m(recvIdxs) {
            
    }


    template <typename T>
    void FEMVector<T>::fillHalo() {
        // check that we have halo information
        if (!boundaryInfo_m) {
            throw IpplException(
                "FEMVector::fillHalo()",
                "Cannot do halo operations, as no MPI communication information is provided. "
                "Did you use the correct constructor to construct the FEMVector?");
        }

        using memory_space = typename Kokkos::View<size_t*>::memory_space;
        // List of MPI requests which we send
        std::vector<MPI_Request> requests(boundaryInfo_m->neighbors_m.size());

        // Send loop.
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            size_t neighborRank = boundaryInfo_m->neighbors_m[i];
            size_t nsends = boundaryInfo_m->sendIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + ippl::Comm->rank();
            
            // pack data, i.e., copy values from data_m to commBuffer_m
            pack(boundaryInfo_m->sendIdxs_m[i]);


            // ippl MPI communication which sends data to neighbors
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nsends);
            ippl::Comm->isend(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive,
                requests[i], nsends);
            archive->resetWritePos();
        }

        // Recieve loop
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            size_t neighborRank = boundaryInfo_m->neighbors_m[i];
            size_t nrecvs = boundaryInfo_m->recvIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + boundaryInfo_m->neighbors_m[i];

            // ippl MPI communication which will recive data from neighbors,
            // will put data into commBuffer_m
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nrecvs);
            ippl::Comm->recv(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive,
                nrecvs * sizeof(T), nrecvs);
            archive->resetReadPos();

            // unpack recieved data, i.e., copy from commBuffer_m to data_m
            unpack<Assign>(boundaryInfo_m->recvIdxs_m[i]);
        }

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        ippl::Comm->freeAllBuffers();
    }


    template <typename T>
    void FEMVector<T>::accumulateHalo() {
        // check that we have halo information
        if (!boundaryInfo_m) {
            throw IpplException(
                "FEMVector::accumulateHalo()",
                "Cannot do halo operations, as no MPI communication information is provided. "
                "Did you use the correct constructor to construct the FEMVector?");
        }

        using memory_space = typename Kokkos::View<size_t*>::memory_space;
        // List of MPI requests which we send
        std::vector<MPI_Request> requests(boundaryInfo_m->neighbors_m.size());

        // Send loop.
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            size_t neighborRank = boundaryInfo_m->neighbors_m[i];
            size_t nsends = boundaryInfo_m->recvIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + ippl::Comm->rank();
            
            // pack data, i.e., copy values from data_m to commBuffer_m
            pack(boundaryInfo_m->recvIdxs_m[i]);


            // ippl MPI communication which sends data to neighbors
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nsends);
            ippl::Comm->isend(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive,
                requests[i], nsends);
            archive->resetWritePos();
        }

        // Receive loop
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            size_t neighborRank = boundaryInfo_m->neighbors_m[i];
            size_t nrecvs = boundaryInfo_m->sendIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + boundaryInfo_m->neighbors_m[i];

            // ippl MPI communication which will receive data from neighbors,
            // will put data into commBuffer_m
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nrecvs);
            ippl::Comm->recv(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive,
                nrecvs * sizeof(T), nrecvs);
            archive->resetReadPos();

            // unpack recieved data, i.e., copy from commBuffer_m to data_m
            unpack<AssignAdd>(boundaryInfo_m->sendIdxs_m[i]);
        }

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        ippl::Comm->freeAllBuffers();
    }


    template <typename T>
    void FEMVector<T>::setHalo(T setValue) {
        // check that we have halo information
        if (!boundaryInfo_m) {
            return;
            throw IpplException(
                "FEMVector::setHalo()",
                "Cannot do halo operations, as no MPI communication information is provided. "
                "Did you use the correct constructor to construct the FEMVector?");
        }

        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            auto& view = boundaryInfo_m->recvIdxs_m[i];
            Kokkos::parallel_for("FEMVector::setHalo()",view.extent(0),
                KOKKOS_CLASS_LAMBDA(const size_t& j){
                    data_m[view(j)] = setValue;
                }
            );
        }
    }


    template <typename T>
    FEMVector<T>& FEMVector<T>::operator= (T value) {
        Kokkos::parallel_for("FEMVector::operator=(T value)", data_m.extent(0),
            KOKKOS_CLASS_LAMBDA(const size_t& i){
                data_m[i] = value;
            }
        );
        return *this;
    }


    template <typename T>
    template <typename E, size_t N>
    FEMVector<T>& FEMVector<T>::operator= (const detail::Expression<E, N>& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
        Kokkos::parallel_for("FEMVector::operator=(Expression)", data_m.extent(0),
            KOKKOS_CLASS_LAMBDA(const size_t& i){
                data_m[i] = expr_(i);
            }
        );
        return *this;
    }


    template <typename T>
    FEMVector<T>& FEMVector<T>::operator= (const FEMVector<T>& v) {
        auto view = v.getView();
        Kokkos::parallel_for("FEMVector::operator=(FEMVector)", data_m.extent(0),
            KOKKOS_CLASS_LAMBDA(const size_t& i){
                data_m[i] = view(i);
            }
        );
        return *this;
    }


    template <typename T>
    KOKKOS_INLINE_FUNCTION T FEMVector<T>::operator[] (size_t i) const {
        return data_m(i);
    }


    template <typename T>
    KOKKOS_INLINE_FUNCTION T FEMVector<T>::operator() (size_t i) const {
        return this->operator[](i);
    }


    template <typename T>
    const Kokkos::View<T*>& FEMVector<T>::getView() const {
        return data_m;
    }

    
    template <typename T>
    size_t FEMVector<T>::size() const {
        return data_m.extent(0);
    }

    template <typename T>
    FEMVector<T> FEMVector<T>::deepCopy() const {
        // We have to check if we have boundary information or not
        if (boundaryInfo_m) {
            // The neighbor_m can be simply passed to the new vector, the sendIdxs_m
            // and recvIdxs_m need to be explicitly copied.
            std::vector< Kokkos::View<size_t*> > newSendIdxs;
            std::vector< Kokkos::View<size_t*> > newRecvIdxs;
            
            for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
                newSendIdxs.emplace_back(Kokkos::View<size_t*>(boundaryInfo_m->sendIdxs_m[i].label(), 
                                            boundaryInfo_m->sendIdxs_m[i].extent(0)));
                Kokkos::deep_copy(newSendIdxs[i], boundaryInfo_m->sendIdxs_m[i]);
                
                newRecvIdxs.emplace_back(Kokkos::View<size_t*>(boundaryInfo_m->recvIdxs_m[i].label(), 
                                            boundaryInfo_m->recvIdxs_m[i].extent(0)));
            
                Kokkos::deep_copy(newRecvIdxs[i], boundaryInfo_m->recvIdxs_m[i]);
            }

            // create the new FEMVector
            FEMVector<T> newVector(size(), boundaryInfo_m->neighbors_m, newSendIdxs, newRecvIdxs);
            // copy over the values 
            newVector = *this;

            return newVector;
        } else {
            FEMVector<T> newVector(size());
            // copy over the values 
            newVector = *this;

            return newVector;
        }
    }

    template <typename T>
    template <typename K>
    FEMVector<K> FEMVector<T>::skeletonCopy() const {
        // We have to check if we have boundary information or not
        if (boundaryInfo_m) {
            // The neighbor_m can be simply passed to the new vector, the sendIdxs_m
            // and recvIdxs_m need to be explicitly copied.
            std::vector< Kokkos::View<size_t*> > newSendIdxs;
            std::vector< Kokkos::View<size_t*> > newRecvIdxs;
            
            for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
                newSendIdxs.emplace_back(Kokkos::View<size_t*>(boundaryInfo_m->sendIdxs_m[i].label(), 
                                            boundaryInfo_m->sendIdxs_m[i].extent(0)));
                Kokkos::deep_copy(newSendIdxs[i], boundaryInfo_m->sendIdxs_m[i]);
                
                newRecvIdxs.emplace_back(Kokkos::View<size_t*>(boundaryInfo_m->recvIdxs_m[i].label(), 
                                            boundaryInfo_m->recvIdxs_m[i].extent(0)));
            
                Kokkos::deep_copy(newRecvIdxs[i], boundaryInfo_m->recvIdxs_m[i]);
            }

            // create the new FEMVector
            FEMVector<K> newVector(size(), boundaryInfo_m->neighbors_m, newSendIdxs, newRecvIdxs);

            return newVector;
        } else {
            FEMVector<K> newVector(size());
            
            return newVector;
        }
    }


    template <typename T>
    void FEMVector<T>::pack(const Kokkos::View<size_t*>& idxStore) {
        // check that we have halo information
        if (!boundaryInfo_m) {
            throw IpplException(
                "FEMVector::pack()",
                "Cannot do halo operations, as no MPI communication information is provided. "
                "Did you use the correct constructor to construct the FEMVector?");
        }

        size_t nIdxs = idxStore.extent(0);
        auto& bufferData = boundaryInfo_m->commBuffer_m.buffer;

        if (bufferData.size() < nIdxs) {
            int overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(bufferData, nIdxs * overalloc);
        }

        Kokkos::parallel_for("FEMVector::pack()", nIdxs,
            KOKKOS_CLASS_LAMBDA(const size_t& i) {
                bufferData(i) = data_m(idxStore(i));
            }
        );
        Kokkos::fence();
    }


    template <typename T>
    template <typename Op>
    void FEMVector<T>::unpack(const Kokkos::View<size_t*>& idxStore) {
        // check that we have halo information
        if (!boundaryInfo_m) {
            throw IpplException(
                "FEMVector::unpack()",
                "Cannot do halo operations, as no MPI communication information is provided. "
                "Did you use the correct constructor to construct the FEMVector?");
        }

        size_t nIdxs = idxStore.extent(0);
        auto& bufferData = boundaryInfo_m->commBuffer_m.buffer;        
        if (bufferData.size() < nIdxs) {
            int overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(bufferData, nIdxs * overalloc);
        }

        Op op;
        Kokkos::parallel_for("FEMVector::unpack()", nIdxs,
            KOKKOS_CLASS_LAMBDA(const size_t& i) {
                op(data_m(idxStore(i)), bufferData(i));
            }
        );
        Kokkos::fence();
    }



} // namespace ippl
