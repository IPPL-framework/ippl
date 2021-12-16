# Independent Parallel Particle Layer (IPPL) v 1.0

IPPL represents the backend of OPAL implementing particles, meshes and operations on them to do PIC codes. This branch is the 
version 1.0 of the library which is MPI parallelized and work with only CPUs. 


## Compile
In order to compile IPPL follow the subsequent steps. Load the following
modules.

```bash
module use unstable
module use /afs/psi.ch/project/amas/modulefiles
module load OPAL/toolchain/2021.1_slurm

cd $BUILD_DIR
CC=mpicc CXX=mpicxx cmake -DENABLE_IPPLTESTS=ON $SRC_DIR
make -j 4
```
where `$BUILD_DIR` is the build directory and `$SRC_DIR` points to the
root directory of the IPPL repository.
