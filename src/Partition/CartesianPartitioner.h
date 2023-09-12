//
// Class CartesianPartitioner
//   Partition a domain into subdomains.
//
#ifndef IPPL_CARTESIAN_PARTITIONER_H
#define IPPL_CARTESIAN_PARTITIONER_H

#include <array>

#include "Communicate/Communicator.h"
#include "Index/NDIndex.h"

namespace ippl {

    namespace detail {

        template <unsigned Dim>
        class CartesianPartitioner {
        public:
            CartesianPartitioner()  = default;
            ~CartesianPartitioner() = default;

            mpi::Communicator partition(const mpi::Communicator& comm, const NDIndex<Dim>& domain,
                                        const std::array<bool, Dim>& decomp) const;
        };
    }  // namespace detail
}  // namespace ippl

#include "Partition/CartesianPartitioner.hpp"

#endif
