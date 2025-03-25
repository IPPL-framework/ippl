namespace ippl{

    template <typename T>
    FEMVector<T>::FEMVector(size_t n, std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs) : 
            data_m("FEMVector::data", n) {
            
        boundaryInfo_m = new BoundaryInfo(std::move(neighbors), std::move(sendIdxs),
                                            std::move(recvIdxs));
    }


    template <typename T>
    FEMVector<T>::BoundaryInfo::BoundaryInfo(std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs) : 
            neighbors_m(neighbors), sendIdxs_m(sendIdxs), recvIdxs_m(recvIdxs) {
            
    }


    template <typename T>
    void FEMVector<T>::fillHalo() {
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
            ippl::Comm->isend(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive, requests[i], nsends);
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
            ippl::Comm->recv(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive, nrecvs * sizeof(T), nrecvs);
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
            ippl::Comm->isend(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive, requests[i], nsends);
            archive->resetWritePos();
        }

        // Recieve loop
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            size_t neighborRank = boundaryInfo_m->neighbors_m[i];
            size_t nrecvs = boundaryInfo_m->sendIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + boundaryInfo_m->neighbors_m[i];

            // ippl MPI communication which will recive data from neighbors,
            // will put data into commBuffer_m
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nrecvs);
            ippl::Comm->recv(neighborRank, tag, boundaryInfo_m->commBuffer_m, *archive, nrecvs * sizeof(T), nrecvs);
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
    void FEMVector<T>::clearHalo(T clearValue) {
        for (size_t i = 0; i < boundaryInfo_m->neighbors_m.size(); ++i) {
            auto& view = boundaryInfo_m->recvIdxs_m[i];
            Kokkos::parallel_for("FEMVector::clearHalo()",view.extent(0),
                KOKKOS_CLASS_LAMBDA(const size_t& j){
                    data_m[view(j)] = clearValue;
                }
            );
        }
    }


    template <typename T>
    void FEMVector<T>::operator= (T value) {
        Kokkos::parallel_for("FEMVector::operator=(T value)", data_m.extent(0),
            KOKKOS_CLASS_LAMBDA(const size_t& i){
                data_m[i] = value;
            }
        );
    }


    template <typename T>
    template <typename E, size_t N>
    void FEMVector<T>::operator= (const detail::Expression<E, N>& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
        Kokkos::parallel_for("FEMVector::operator=(FEMVector)", data_m.extent(0),
            KOKKOS_CLASS_LAMBDA(const size_t& i){
                data_m[i] = expr_(i);
            }
        );
    }


    template <typename T>
    T FEMVector<T>::operator[] (size_t i) const {
        return data_m(i);
    }


    template <typename T>
    T FEMVector<T>::operator() (size_t i) const {
        return this->operator[](i);
    }


    template <typename T>
    const Kokkos::View<T*>& FEMVector<T>::getView() const {
        return data_m;
    }


    template <typename T>
    T FEMVector<T>::getVolumeAverage() const {
        return T();
    }


    template <typename T>
    void FEMVector<T>::pack(const Kokkos::View<size_t*>& idxStore) {
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
    void FEMVector<T>::unpack<Op>(const Kokkos::View<size_t*>& idxStore) {
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