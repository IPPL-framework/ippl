namespace ippl{

    template <typename T>
    FEMVector<T>::FEMVector(size_t n, std::vector<size_t> neighbors,
        std::vector< Kokkos::View<size_t*> > sendIdxs,
        std::vector< Kokkos::View<size_t*> > recvIdxs) : 
        data_m("FEMVector::data", n), neighbors_m(neighbors),
        sendIdxs_m(sendIdxs), recvIdxs_m(recvIdxs) {
    }


    template <typename T>
    void FEMVector<T>::fillHalo() {
        using memory_space = typename Kokkos::View<size_t*>::memory_space;
        // List of MPI requests which we send
        std::vector<MPI_Request> requests(neighbors_m.size());

        // Send loop.
        for (size_t i = 0; i < neighbors_m.size(); ++i) {
            size_t neighborRank = neighbors_m[i];
            size_t nsends = sendIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + ippl::Comm->rank();
            
            // pack data, i.e., copy values from data_m to commBuffer_m
            pack(sendIdxs_m[i]);


            // ippl MPI communication which sends data to neighbors
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nsends);
            ippl::Comm->isend(neighborRank, tag, commBuffer_m, *archive, requests[i], nsends);
            archive->resetWritePos();
        }

        // Recieve loop
        for (size_t i = 0; i < neighbors_m.size(); ++i) {
            size_t neighborRank = neighbors_m[i];
            size_t nrecvs = recvIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + neighbors_m[i];

            // ippl MPI communication which will recive data from neighbors,
            // will put data into commBuffer_m
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nrecvs);
            ippl::Comm->recv(neighborRank, tag, commBuffer_m, *archive, nrecvs * sizeof(T), nrecvs);
            archive->resetReadPos();

            // unpack recieved data, i.e., copy from commBuffer_m to data_m
            unpack<Assign>(recvIdxs_m[i]);
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
        std::vector<MPI_Request> requests(neighbors_m.size());

        // Send loop.
        for (size_t i = 0; i < neighbors_m.size(); ++i) {
            size_t neighborRank = neighbors_m[i];
            size_t nsends = recvIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + ippl::Comm->rank();
            
            // pack data, i.e., copy values from data_m to commBuffer_m
            pack(recvIdxs_m[i]);


            // ippl MPI communication which sends data to neighbors
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nsends);
            ippl::Comm->isend(neighborRank, tag, commBuffer_m, *archive, requests[i], nsends);
            archive->resetWritePos();
        }

        // Recieve loop
        for (size_t i = 0; i < neighbors_m.size(); ++i) {
            size_t neighborRank = neighbors_m[i];
            size_t nrecvs = sendIdxs_m[i].extent(0);
            size_t tag = mpi::tag::FEMVECTOR + neighbors_m[i];

            // ippl MPI communication which will recive data from neighbors,
            // will put data into commBuffer_m
            mpi::Communicator::buffer_type<memory_space> archive =
                ippl::Comm->getBuffer<memory_space, T>(nrecvs);
            ippl::Comm->recv(neighborRank, tag, commBuffer_m, *archive, nrecvs * sizeof(T), nrecvs);
            archive->resetReadPos();

            // unpack recieved data, i.e., copy from commBuffer_m to data_m
            unpack<AssignAdd>(sendIdxs_m[i]);
        }

        if (requests.size() > 0) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        ippl::Comm->freeAllBuffers();
    }


    template <typename T>
    void FEMVector<T>::clearHalo(T clearValue) {
        for (size_t i = 0; i < neighbors_m.size(); ++i) {
            auto& view = recvIdxs_m[i];
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
    const Kokkos::View<T*>& FEMVector<T>::getView() const {
        return data_m;
    }


    template <typename T>
    void FEMVector<T>::pack(const Kokkos::View<size_t*>& idxStore) {
        size_t nIdxs = idxStore.extent(0);
        auto& bufferData = commBuffer_m.buffer;

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
        auto& bufferData = commBuffer_m.buffer;
        Op op;
        Kokkos::parallel_for("FEMVector::unpack()", nIdxs,
            KOKKOS_CLASS_LAMBDA(const size_t& i) {
                op(data_m(idxStore(i)), bufferData(i));
            }
        );
        Kokkos::fence();
    }



} // namespace ippl