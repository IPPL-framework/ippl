#ifndef IPPL_GRIDPATH_SEGMENTER_H
#define IPPL_GRIDPATH_SEGMENTER_H
#include <array>

namespace ippl {

template<unsigned Dim, typename T>
struct Segment {
  ippl::Vector<T,Dim> p0, p1;
};

struct DefaultCellCrossingRule {};

template<unsigned Dim, typename T, typename Rule = DefaultCellCrossingRule>
struct GridPathSegmenter {

  static KOKKOS_INLINE_FUNCTION
  std::array<Segment<Dim,T>, Dim+1>
  split(const ippl::Vector<T, Dim>& A,
        const ippl::Vector<T, Dim>& B,
        const ippl::Vector<T, Dim>& origin,
        const ippl::Vector<T, Dim>& h);
};

} // namespace ippl 

#include "GridPathSegmenter.hpp"

#endif
