# AGENTS.md

## Repository purpose
IPPL is a C++ Particle and fields Framework. Preserve physical correctness over stylistic cleanup.

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

# Naming Conventions

- Variables should use camel casing
- Compile time constants should use capital casing
- Member variables should be suffixed with `_m`

# Math

Mathematical constants should be obtained from `Kokkos::numbers` (moved out of `experimental` in Kokkos 4). Any instance of a mathematical function in host-only code should use symbols from the standard library, e.g. `std::sqrt`. Any instances occurring in device-code or code that might be run on a device (such as those marked with `KOKKOS_INLINE_FUNCTION`) should use the symbols from Kokkos, e.g. `Kokkos::sqrt`, to ensure performance and portability on GPUs.
