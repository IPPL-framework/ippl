//
// Class Partitioner
//   Partition a domain into subdomains.
//
#ifndef IPPL_PARTITIONER_H
#define IPPL_PARTITIONER_H

#include <utility>

#include "Communicate/Communicate.h"
#include "Index/NDIndex.h"

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        class Partitioner {
        private:
            using pair_type = std::pair<int, int>;

        public:
            Partitioner()  = default;
            ~Partitioner() = default;

            template <typename view_type>
            void split(Communicate* communicate, const NDIndex<Dim>& domain, view_type& view,
                       e_dim_tag* decomp, int nSplits) const;

        private:
            pair_type getLocalBounds(int nglobal, int coords, int dims) const;
        };
    }  // namespace detail
}  // namespace ippl

#include "Partition/Partitioner.hpp"

#endif
