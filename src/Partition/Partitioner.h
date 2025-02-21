//
// Class Partitioner
//   Partition a domain into subdomains.
//
#ifndef IPPL_PARTITIONER_H
#define IPPL_PARTITIONER_H

#include "Index/NDIndex.h"

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        class Partitioner {
        public:
            Partitioner()  = default;
            ~Partitioner() = default;

            template <typename view_type>
            void split(const NDIndex<Dim>& domain, view_type& view,
                       const std::array<bool, Dim>& isParallel, int nSplits) const;
        };
    }  // namespace detail
}  // namespace ippl

#include "Partition/Partitioner.hpp"

#endif
