# AGENTS.md

## Repository purpose
IPPL is a C++ Particle and fields Framwork. Preserve physical correctness over stylistic cleanup.

## Build
- Configure with CMake using the project’s normal toolchain and dependency prefixes.
- Build in the local build or build_openmp directory.
- build instructions can be found online.
- Run the relevant test subset after changes.
- External dependencies are in the _deps directory fetched with cmake

## External dependencies
- heFFte and Kokkos are external dependencies.
- Prefer changing IPPL adapter/wrapper code over patching upstream dependencies.
- When a change touches parallel kernels, explain execution space, memory space, and data movement implications.

## Physics / numerics rules
- For algorithmic changes, state expected impact on conservation, stability, and reproducibility.
- Flag any change that may alter floating-point behavior or MPI/GPU execution order.
- Add or update at least one regression/sanity test for physics-facing changes.
- Keep am eye on parallel efficincy at least on the openmp level