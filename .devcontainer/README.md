# IPPL Dev Container

This container is the browser-editable path for manual examples that need the real IPPL stack:
MPI, Kokkos, HeFFTe, and FFTW.

It is intentionally CPU-first. GPU containers should be separate CUDA/HIP variants because they
depend on host drivers, device runtime versions, and cluster-specific MPI behavior.

## What It Builds

- Ubuntu 24.04 development container with a non-root `vscode` user
- OpenMPI compiler wrappers and runtime
- FFTW development libraries, including MPI and OpenMP variants
- IPPL configured with `SERIAL;OPENMP`
- Kokkos fetched through IPPL CMake as `git.4.7.02`
- HeFFTe fetched through IPPL CMake as `git.v2.4.1`
- `IPPL_ENABLE_FFT=ON`, `IPPL_ENABLE_SOLVERS=ON`, examples, integration tests, and unit tests

## GitHub Codespaces

Open this repository in Codespaces from the branch containing the devcontainer. Codespaces will
build the image and run:

```bash
.devcontainer/scripts/configure.sh
```

The configured build tree is `build/devcontainer`.

## Local VS Code Dev Containers

Install Docker and the VS Code Dev Containers extension, then run **Dev Containers: Reopen in
Container** from VS Code.

## Useful Commands

Configure explicitly:

```bash
.devcontainer/scripts/configure.sh
```

Build everything in the devcontainer preset:

```bash
.devcontainer/scripts/build.sh
```

Build and run the smoke test that exercises IPPL FFT through HeFFTe:

```bash
.devcontainer/scripts/test-smoke.sh
```

Build one target:

```bash
.devcontainer/scripts/build.sh --target FFT
```

Run tests directly:

```bash
ctest --test-dir build/devcontainer --output-on-failure
```
