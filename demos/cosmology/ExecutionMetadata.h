#ifndef IPPL_COSMOLOGY_EXECUTION_METADATA_H
#define IPPL_COSMOLOGY_EXECUTION_METADATA_H

#include <Kokkos_Core.hpp>
#include <ostream>

namespace cosmology {

// Host-only reporting: this does not launch kernels or change their execution.
// Keep the historical "threads" field unchanged: on CUDA it is device
// concurrency, not the number of host OpenMP threads or resident GPU threads.
inline void writeExecutionMetadata(std::ostream& output) {
    const int executionConcurrency = Kokkos::DefaultExecutionSpace().concurrency();
    output << "threads=" << executionConcurrency
           << "\nexecution_concurrency=" << executionConcurrency
           << "\nhost_threads=" << Kokkos::DefaultHostExecutionSpace().concurrency()
           << "\nexecution_space=" << Kokkos::DefaultExecutionSpace::name()
           << "\nmemory_space=" << Kokkos::DefaultExecutionSpace::memory_space::name() << '\n';
}

} // namespace cosmology

#endif
