# Independent Parallel Particle Layer (IPPL)

IPPL represents the backend of OPAL implementing particles, meshes and operations on them to do PIC codes. This branch is the 
version 1.0 of the library which is MPI parallelized and work with only CPUs. 


## Compile
In order to compile IPPL follow the subsequent steps. Load the same set of modules 
as in the version 2.0 of this library (please see install page).

```bash
cd $BUILD_DIR
CC=mpicc CXX=mpicxx cmake $SRC_DIR
make -j 4
```
where `$BUILD_DIR` is the build directory and `$SRC_DIR` points to the
root directory of the IPPL repository.
