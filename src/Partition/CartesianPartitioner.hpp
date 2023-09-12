//
// Class CartesianPartitioner
//   Partition a domain into subdomains.
//
#include <algorithm>

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        mpi::Communicator CartesianPartitioner<Dim>::partition(
            const mpi::Communicator& comm, const NDIndex<Dim>& /*domain*/,
            const std::array<bool, Dim>& isParallel) const {
            //             using NDIndex_t = NDIndex<Dim>;

            const int ndims = std::count(isParallel.cbegin(), isParallel.cend(), true);

            // ndims <= Dim
            int dims[Dim];

            MPI_Dims_create(comm.size(), ndims, dims);

            int periods[ndims];
            for (int i = 0; i < ndims; ++i) {
                periods[i] = 0;
            }

            MPI_Comm tmpcomm;
            MPI_Cart_create(comm, ndims, dims, periods, 0, &tmpcomm);
            mpi::Communicator cartcomm(tmpcomm);

            int coords[Dim];
            MPI_Cart_coords(cartcomm, cartcomm.rank(), ndims, coords);

            return cartcomm;
        }
    }  // namespace detail
}  // namespace ippl
