#ifndef IPPL_GRIDPATH_SEGMENTER_H
#define IPPL_GRIDPATH_SEGMENTER_H
#include <array>

namespace ippl {

template<unsigned Dim, typename T>
struct Segment {
  std::array<T,Dim> p0, p1;
};

struct DefaultCellCrossingRule {};
struct ZigZagScheme {};

template<unsigned Dim, typename T, typename Rule = DefaultCellCrossingRule>
struct GridPathSegmenter {

  static KOKKOS_INLINE_FUNCTION
  std::array<Segment<Dim,T>, Dim+1>
  split(const std::array<T,Dim>& A,
        const std::array<T,Dim>& B,
        const std::array<T,Dim>& origin,
        const std::array<T,Dim>& h);
};

} // namespace ippl 

#include "GridPathSegmenter.hpp"

#endif
