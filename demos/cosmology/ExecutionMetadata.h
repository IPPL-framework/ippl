/**
 * @brief Scientific implementation and contracts for ExecutionMetadata.h.
 *
 * @file ExecutionMetadata.h
 * @ingroup cosmology_core
 * @see cosmology_model cosmology_numerics cosmology_contracts
 */
#ifndef IPPL_COSMOLOGY_EXECUTION_METADATA_H
#define IPPL_COSMOLOGY_EXECUTION_METADATA_H

#include <Kokkos_Core.hpp>
#include <ostream>

namespace cosmology {

// Host-only reporting: this does not launch kernels or change their execution.
// Keep the historical "threads" field unchanged: on CUDA it is device
// concurrency, not the number of host OpenMP threads or resident GPU threads.
/**
 * @brief Write host/device concurrency and memory-space identity without changing execution.
 *
 * Host-only; no kernels, copies or MPI collectives. Historical threads means execution-space concurrency, not host threads on CUDA.
 * @see cosmology_parallel
 *
 * @param output Host output stream receiving newline-separated name=value fields.
 */
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
