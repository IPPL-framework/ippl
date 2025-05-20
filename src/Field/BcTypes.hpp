//   This file contains the abstract base class for
//   field boundary conditions and other child classes
//   which represent specific BCs. At the moment the
//   following field BCs are supported
//
//   1. Periodic BC
//   2. Zero BC
//   3. Specifying a constant BC
//   4. No BC (default option)
//   5. Constant extrapolation BC
//   Only cell-centered field BCs are implemented
//   at the moment.
//

#include "Utility/IpplException.h"

#include "Field/HaloCells.h"

namespace ippl {
    namespace detail {

        template <typename Field>
        BCondBase<Field>::BCondBase(unsigned int face)
            : face_m(face)
            , changePhysical_m(false) {}

        template <typename Field>
        inline std::ostream& operator<<(std::ostream& os, const BCondBase<Field>& bc) {
            bc.write(os);
            return os;
        }

    }  // namespace detail

    template <typename Field>
    void ExtrapolateFace<Field>::apply(Field& field) {
        // We only support constant extrapolation for the moment, other
        // higher order extrapolation stuffs need to be added.

        unsigned int face = this->face_m;
        unsigned d        = face / 2;
        if (field.getCommunicator().size() > 1) {
            const Layout_t& layout = field.getLayout();
            const auto& lDomains   = layout.getHostLocalDomains();
            const auto& domain     = layout.getDomain();
            int myRank             = field.getCommunicator().rank();

            bool isBoundary = (lDomains[myRank][d].max() == domain[d].max())
                              || (lDomains[myRank][d].min() == domain[d].min());

            if (!isBoundary) {
                return;
            }
        }

        // If we are here then it is a processor with the face on the physical
        // boundary or it is the single core case. Then the following code is same
        // irrespective of either it is a single core or multi-core case as the
        // non-periodic BC is local to apply.
        typename Field::view_type& view = field.getView();
        const int nghost                = field.getNghost();
        int src, dest;

        // It is not clear what it exactly means to do extrapolate
        // BC for nghost >1
        if (nghost > 1) {
            throw IpplException("ExtrapolateFace::apply", "nghost > 1 not supported");
        }

        if (d >= Dim) {
            throw IpplException("ExtrapolateFace::apply", "face number wrong");
        }

        // If face & 1 is true, then it is an upper BC
        if (face & 1) {
            src  = view.extent(d) - 2;
            dest = src + 1;
        } else {
            src  = 1;
            dest = src - 1;
        }

        using exec_space = typename Field::execution_space;
        using index_type = typename RangePolicy<Dim, exec_space>::index_type;
        Kokkos::Array<index_type, Dim> begin, end;
        for (unsigned i = 0; i < Dim; i++) {
            begin[i] = nghost;
            end[i]   = view.extent(i) - nghost;
        }
        begin[d]               = src;
        end[d]                 = src + 1;
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        ippl::parallel_for(
            "Assign extrapolate BC", createRangePolicy<Dim, exec_space>(begin, end),
            KOKKOS_CLASS_LAMBDA(index_array_type & args) {
                // to avoid ambiguity with the member function
                using ippl::apply;

                T value = apply(view, args);

                args[d] = dest;

                apply(view, args) = slope_m * value + offset_m;
            });
    }

    template <typename Field>
    void ExtrapolateFace<Field>::write(std::ostream& out) const {
        out << "Constant Extrapolation Face"
            << ", Face = " << this->face_m;
    }

    template <typename Field>
    void NoBcFace<Field>::write(std::ostream& out) const {
        out << "NoBcFace"
            << ", Face = " << this->face_m;
    }

    template <typename Field>
    void ConstantFace<Field>::write(std::ostream& out) const {
        out << "ConstantFace"
            << ", Face = " << this->face_m << ", Constant = " << this->offset_m;
    }

    template <typename Field>
    void ZeroFace<Field>::write(std::ostream& out) const {
        out << "ZeroFace"
            << ", Face = " << this->face_m;
    }

    template <typename Field>
    void PeriodicFace<Field>::write(std::ostream& out) const {
        out << "PeriodicFace"
            << ", Face = " << this->face_m;
    }

    template <typename Field>
    void PeriodicFace<Field>::findBCNeighbors(Field& field) {
        auto& comm = field.getCommunicator();
        // For cell centering only face neighbors are needed
        unsigned int face      = this->face_m;
        unsigned int d         = face / 2;
        const int nghost       = field.getNghost();
        int myRank             = comm.rank();
        const Layout_t& layout = field.getLayout();
        const auto& lDomains   = layout.getHostLocalDomains();
        const auto& domain     = layout.getDomain();

        for (auto& neighbors : faceNeighbors_m) {
            neighbors.clear();
        }

        if (lDomains[myRank][d].length() < domain[d].length()) {
            // Only along this dimension we need communication.

            bool isBoundary = (lDomains[myRank][d].max() == domain[d].max())
                              || (lDomains[myRank][d].min() == domain[d].min());

            if (isBoundary) {
                // this face is  on mesh/physical boundary
                //  get my local box
                auto& nd = lDomains[myRank];

                // grow the box by nghost cells in dimension d of face
                auto gnd = nd.grow(nghost, d);

                int offset;
                if (face & 1) {
                    // upper face
                    offset = -domain[d].length();
                } else {
                    // lower face
                    offset = domain[d].length();
                }
                // shift by offset
                gnd[d] = gnd[d] + offset;

                // Now, we are ready to intersect
                for (int rank = 0; rank < comm.size(); ++rank) {
                    if (rank == myRank) {
                        continue;
                    }

                    if (gnd.touches(lDomains[rank])) {
                        faceNeighbors_m[face].push_back(rank);
                    }
                }
            }
        }
    }

    template <typename Field>
    void PeriodicFace<Field>::apply(Field& field) {
        auto& comm                      = field.getCommunicator();
        unsigned int face               = this->face_m;
        unsigned int d                  = face / 2;
        typename Field::view_type& view = field.getView();
        const Layout_t& layout          = field.getLayout();
        const int nghost                = field.getNghost();
        int myRank                      = comm.rank();
        const auto& lDomains            = layout.getHostLocalDomains();
        const auto& domain              = layout.getDomain();

        // We have to put tag here so that the matchtag inside
        // the if is proper.
        int tag = comm.next_tag(mpi::tag::BC_PARALLEL_PERIODIC, mpi::tag::BC_CYCLE);

        if (lDomains[myRank][d].length() < domain[d].length()) {
            // Only along this dimension we need communication.

            bool isBoundary = (lDomains[myRank][d].max() == domain[d].max())
                              || (lDomains[myRank][d].min() == domain[d].min());

            if (isBoundary) {
                // this face is  on mesh/physical boundary
                //  get my local box
                auto& nd = lDomains[myRank];

                int offset, offsetRecv, matchtag;
                if (face & 1) {
                    // upper face
                    offset     = -domain[d].length();
                    offsetRecv = nghost;
                    matchtag   = comm.preceding_tag(mpi::tag::BC_PARALLEL_PERIODIC);
                } else {
                    // lower face
                    offset     = domain[d].length();
                    offsetRecv = -nghost;
                    matchtag   = comm.following_tag(mpi::tag::BC_PARALLEL_PERIODIC);
                }

                auto& neighbors = faceNeighbors_m[face];

                using memory_space = typename Field::memory_space;
                using buffer_type  = mpi::Communicator::buffer_type<memory_space>;
                std::vector<MPI_Request> requests(neighbors.size());

                using HaloCells_t = typename Field::halo_type;
                using range_t     = typename HaloCells_t::bound_type;
                HaloCells_t& halo = field.getHalo();
                std::vector<range_t> rangeNeighbors;

                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int rank = neighbors[i];

                    auto ndNeighbor = lDomains[rank];
                    ndNeighbor[d]   = ndNeighbor[d] - offset;

                    NDIndex<Dim> gndNeighbor = ndNeighbor.grow(nghost, d);

                    NDIndex<Dim> overlap = gndNeighbor.intersect(nd);

                    range_t range;

                    for (size_t j = 0; j < Dim; ++j) {
                        range.lo[j] = overlap[j].first() - nd[j].first() + nghost;
                        range.hi[j] = overlap[j].last() - nd[j].first() + nghost + 1;
                    }

                    rangeNeighbors.push_back(range);

                    detail::size_type nSends;
                    halo.pack(range, view, haloData_m, nSends);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nSends);

                    comm.isend(rank, tag, haloData_m, *buf, requests[i], nSends);
                    buf->resetWritePos();
                }

                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int rank = neighbors[i];

                    range_t range = rangeNeighbors[i];

                    range.lo[d] = range.lo[d] + offsetRecv;
                    range.hi[d] = range.hi[d] + offsetRecv;

                    detail::size_type nRecvs = range.size();

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nRecvs);
                    comm.recv(rank, matchtag, haloData_m, *buf, nRecvs * sizeof(T), nRecvs);
                    buf->resetReadPos();

                    using assign_t = typename HaloCells_t::assign;
                    halo.template unpack<assign_t>(range, view, haloData_m);
                }
                if (!requests.empty()) {
                    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                }
                comm.freeAllBuffers();
            }
            // For all other processors do nothing
        } else {
            if (d >= Dim) {
                throw IpplException("PeriodicFace::apply", "face number wrong");
            }

            auto N = view.extent(d) - 1;

            using exec_space = typename Field::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;
            Kokkos::Array<index_type, Dim> begin, end;

            // For the axis along which BCs are being applied, iterate
            // through only the ghost cells. For all other axes, iterate
            // through all internal cells.
            for (size_t i = 0; i < Dim; ++i) {
                end[i]   = view.extent(i) - nghost;
                begin[i] = nghost;
            }
            begin[d] = 0;
            end[d]   = nghost;

            using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_for(
                "Assign periodic field BC", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_CLASS_LAMBDA(index_array_type & coords) {
                    // The ghosts are filled starting from the inside of
                    // the domain proceeding outwards for both lower and
                    // upper faces.

                    // to avoid ambiguity with the member function
                    using ippl::apply;

                    // x -> nghost + x
                    coords[d] += nghost;
                    auto&& left = apply(view, coords);

                    // nghost + x -> N - (nghost + x) = N - nghost - x
                    coords[d]    = N - coords[d];
                    auto&& right = apply(view, coords);

                    // N - nghost - x -> nghost - 1 - x
                    coords[d] += 2 * nghost - 1 - N;
                    apply(view, coords) = right;

                    // nghost - 1 - x -> N - (nghost - 1 - x)
                    //     = N - (nghost - 1) + x
                    coords[d]           = N - coords[d];
                    apply(view, coords) = left;
                });
        }
    }

    template <typename Field>
    void PeriodicFace<Field>::assignGhostToPhysical(Field& field) {
        unsigned int face               = this->face_m;
        unsigned int d                  = face / 2;
        typename Field::view_type& view = field.getView();
        const Layout_t& layout          = field.getLayout();
        const int nghost                = field.getNghost();
        const auto& ldom                = layout.getLocalNDIndex();
        const auto& domain              = layout.getDomain();

        if (d >= Dim) {
            throw IpplException("PeriodicFace::apply", "face number wrong");
        }

        bool upperFace = (face & 1);
        bool isBoundary = ((ldom[d].max() == domain[d].max()) && upperFace)
                           || ((ldom[d].min() == domain[d].min()) && !(upperFace));

        if (isBoundary) {

            auto N = view.extent(d) - 1;

            using exec_space = typename Field::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;
            Kokkos::Array<index_type, Dim> begin, end;

            // For the axis along which BCs are being applied, iterate
            // through only the ghost cells. For all other axes, iterate
            // through all internal cells.
            bool isCorner = (d != 0);
            for (size_t i = 0; i < Dim; ++i) {
                bool upperFace_i = (ldom[i].max() == domain[i].max());
                bool lowerFace_i = (ldom[i].min() == domain[i].min());
                end[i]   = view.extent(i) - nghost - (upperFace_i)*(isCorner);
                begin[i] = nghost + (lowerFace_i)*(isCorner);
            }
            begin[d] = ((0 + nghost - 1) * (1 - upperFace)) + (N * upperFace);
            end[d]   = begin[d] + 1;

            using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_for(
                "Assign periodic field BC", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_CLASS_LAMBDA(index_array_type & coords) {
                    // we add the ghost cell values to the appropriate
                    // neighbouring physical boundary cell

                    // to avoid ambiguity with the member function
                    using ippl::apply;

                    // get the value at ghost cells
                    auto&& right = apply(view, coords);

                    // apply to the last physical cells (boundary)
                    int shift = 1 - (2 * upperFace);
                    coords[d] += shift;

                    apply(view, coords) += right;
            });
        }
    }

    template <typename Field>
    void ExtrapolateFace<Field>::assignGhostToPhysical(Field& field) {
        unsigned int face               = this->face_m;
        unsigned int d                  = face / 2;
        typename Field::view_type& view = field.getView();
        const Layout_t& layout          = field.getLayout();
        const int nghost                = field.getNghost();
        const auto& ldom                = layout.getLocalNDIndex();
        const auto& domain              = layout.getDomain();

        if (d >= Dim) {
            throw IpplException("ExtrapolateFace::apply", "face number wrong");
        }

        bool upperFace = (face & 1);
        bool isBoundary = ((ldom[d].max() == domain[d].max()) && upperFace)
                           || ((ldom[d].min() == domain[d].min()) && !(upperFace));

        if (isBoundary) {
            auto N = view.extent(d) - 1;

            using exec_space = typename Field::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;
            Kokkos::Array<index_type, Dim> begin, end;

            // For the axis along which BCs are being applied, iterate
            // through only the ghost cells. For all other axes, iterate
            // through all internal cells.
            for (size_t i = 0; i < Dim; ++i) {
                end[i]   = view.extent(i) - nghost;
                begin[i] = nghost;
            }
            begin[d] = ((0 + nghost - 1) * (1 - upperFace)) + (N * upperFace);
            end[d]   = begin[d] + 1;

            using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_for(
                "Assign field BC", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_CLASS_LAMBDA(index_array_type & coords) {
                    // we assign the ghost cell values to the appropriate
                    // neighbouring physical boundary cell

                    // to avoid ambiguity with the member function
                    using ippl::apply;

                    // get the value at ghost cells
                    auto&& right = apply(view, coords);

                    // apply to the last physical cells (boundary)
                    int shift = 1 - (2 * upperFace);

                    coords[d] += shift;

                    apply(view, coords) = right;
            });
        }
    }
}  // namespace ippl
