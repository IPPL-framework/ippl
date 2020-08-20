# Independent Parallel Particle Layer (IPPL)

IPPL represents the backend of OPAL implementing particles, meshes and operations on them to do PIC codes. This repository serves
as a temporary development environment decoupled from OPAL in order to enhance the backend with the performance portable capabilities
of [Kokkos](https://github.com/kokkos).


## Compile
In order to compile IPPL follow the subsequent steps

```bash
cd $BUILD_DIR
cmake $SRC_DIR
make
```
where `$BUILD_DIR` is the build directory and `$SRC_DIR` points to the
root directory of the IPPL repository. The compilation with `-DENABLE_IPPLTESTS`
is currently not working.

