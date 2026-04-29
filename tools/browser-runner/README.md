# IPPL Browser Runner

This is the local companion service for browser-based manual examples. It gives a static manual page
a narrow local API for the edit -> compile -> run -> inspect-stdout loop.

The first runner is intentionally minimal:

- CPU-only IPPL build
- MPI + Kokkos
- no FFT, solvers, HeFFTe, or GPU stack
- one editable C++ source file
- JSON response with configure/build/run stdout and stderr

## Build locally

```bash
docker build -t ippl-runner:local -f tools/browser-runner/Dockerfile .
```

## Start the published image

Bind only to `127.0.0.1` on the host:

```bash
docker run --rm \
  -p 127.0.0.1:5050:5050 \
  ghcr.io/ippl-framework/ippl-runner:latest
```

## Start a locally built image

Bind only to `127.0.0.1` on the host:

```bash
docker run --rm \
  -p 127.0.0.1:5050:5050 \
  ippl-runner:local
```

The runner prints a token on startup:

```text
IPPL runner token: ...
```

Paste that token into the manual playground before running code.

## API

```text
GET  /health
GET  /example/hello
POST /compile-run
```

`POST /compile-run` accepts:

```json
{
  "token": "token printed by the runner",
  "source": "#include \"Ippl.h\" ...",
  "ranks": 2
}
```

The service creates a temporary CMake project, compiles it with `mpicxx`, runs it with `mpiexec`,
and returns JSON with stdout/stderr for each stage.

## Safety boundaries

- Run the container with `-p 127.0.0.1:5050:5050`.
- Do not bind the runner to a public interface.
- The token prevents arbitrary websites from silently triggering local compiles.
- Source size, MPI ranks, and command time are bounded by the runner.
