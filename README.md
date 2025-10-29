[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5940225.svg)](https://doi.org/10.5281/zenodo.8389192)
[![License](https://img.shields.io/github/license/IPPL-framework/ippl)](https://github.com/IPPL-framework/ippl/blob/master/LICENSE)

# Independent Parallel Particle Layer (IPPL)

## Table of Contents
- [Independent Parallel Particle Layer (IPPL)](#independent-parallel-particle-layer-ippl)
  - [Table of Contents](#table-of-contents)
- [CI/CD](#cicd)
- [Installing IPPL and its dependencies](#installing-ippl-and-its-dependencies)
  - [Requirements](#requirements)
    - [Optional requirements](#optional-requirements)
  - [Compilation](#compilation)
      - [None of the options have to be set explicitly, all have a default.](#none-of-the-options-have-to-be-set-explicitly-all-have-a-default)
    - [Examples](#examples)
      - [CMakeUserPresets](#cmakeuserpresets)
      - [Serial debug build with tests and newest Kokkos](#serial-debug-build-with-tests-and-newest-kokkos)
      - [OpenMP release build with alpine and FFTW](#openmp-release-build-with-alpine-and-fftw)
      - [Cuda alpine release build](#cuda-alpine-release-build)
      - [HIP release build (LUMI)](#hip-release-build-lumi)
      - [Enable Scripts](#enable-scripts)
- [Contributions](#contributions)
  - [Citing IPPL](#citing-ippl)
- [Job scripts for running on Merlin and Gwendolen (at PSI)](#job-scripts-for-running-on-merlin-and-gwendolen-at-psi)
  - [Merlin CPU (MPI + OpenMP)](#merlin-cpu-mpi--openmp)
  - [Gwendolen GPU](#gwendolen-gpu)
  - [LUMI GPU partition](#lumi-gpu-partition)
- [Profiling IPPL MPI calls](#profiling-ippl-mpi-calls)
- [Build Instructions](#build-instructions)
  - [MERLIN 7 (PSI)](#merlin-7-psi)
  - [ALPS (CSCS)](#alps-cscs)

Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs).

All IPPL releases (< 3.2.0) are available under the BSD 3-clause license. Since version 3.2.0, this repository includes a modified version of the `variant` header by GNU, created to support compilation under CUDA 12.2 with GCC 12.3.0. This header file is available under the same terms as the [GNU Standard Library](https://github.com/gcc-mirror/gcc); note the GNU runtime library exception. As long as this file is not removed, IPPL is available under GNU GPL version 3.

# CI/CD
Check out the latest [results](https://ippl-bc4558.pages.jsc.fz-juelich.de/)

# Installing IPPL and its dependencies

All the new developments of IPPL are merged into the `master` branch which can make it potentially unstable from time to time. So if you want a stable and more tested version
please checkout the tagged branch correspodning to the latest release (e.g. `git checkout tags/IPPL-x.x.x`). Otherwise if you want the latest developments go with the master with the above caveat in mind.

## Requirements

* [CMake](https://cmake.org/download/)
* A C++ compilation toolchain (GPU-capable for GPU builds, e.g. [nvcc](https://developer.nvidia.com/cuda-downloads), [clang]() or [rocmcc](https://rocm.docs.amd.com/en/docs-5.0.2/reference/rocmcc/rocmcc.html))
* MPI (GPU-aware if building for GPUs)

### Optional requirements
* FFTW
* CuFFT

## Compilation
IPPL is a CMake Project and can be configured by passing options in CMake syntax:
```
cmake <src_dir> -D<option>=<value>
```
#### None of the options have to be set explicitly, all have a default.

The relevant options of IPPL are
- IPPL_PLATFORMS, can be one of `SERIAL, OPENMP, CUDA, "OPENMP;CUDA"`, default `SERIAL`
- `Kokkos_VERSION`, default `4.5.00` 
- `Heffte_VERSION`, default `4.7.1`
  - If set to `MASTER`, an additional flag `Heffte_COMMIT_HASH` can be set, default `9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda`
  - Currently, this is the only compatible commit of Heffte
- `IPPL_DYL`, default `OFF`
- `IPPL_ENABLE_SOLVERS`, default `OFF`
- `IPPL_ENABLE_FFT`, default `OFF`
  - If `IPPL_ENABLE_FFT` is set, `Heffte_ENABLE_CUDA` will default to `ON` if `IPPL_PLATFORMS` contains `cuda`
  - Otherwise, `Heffte_ENABLE_AVX2` is enabled. FFTW has to be enabled explicitly.
- `Heffte_ENABLE_FFTW`, default `OFF`
- `IPPL_ENABLE_TESTS`, default `OFF`
- `IPPL_ENABLE_UNIT_TESTS`, default `OFF`
- `IPPL_ENABLE_ALPINE`, default `OFF`
- `IPPL_USE_ALTERNATIVE_VARIANT`, default `OFF`. Can be turned on for GPU builds where the use of the system-provided variant doesn't work.  
- `IPPL_ENABLE_SANITIZER`, default `OFF`
- `IPPL_ENABLE_SCRIPTS`, default `OFF`
  
`Kokkos` and `Heffte` by default will try to use version that are found on the sytem, if the system has `kokkos@4.6` and you set `Kokkos_VERSION=4.5` then cmake's find_package will consider the system version a match (newer than requested) and use it. The same applies for `Heffte`. You can override the variable to checkout any version by setting a git tag/sha/branch such as 
```
cmake -DKokkos_version=git.4.7.01 -DHeffte_VERSION=git.v2.4.1 ...  
# or for a very specific version 
cmake -DHeffte_VERSION=git.9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda ...  
```
Note that by default, Kokkos git tags use a format `x.x.xx (eg. 4.7.01)` and Heffte git tags (extra 'v') are of the form `vx.x.x (eg. v2.4.1)`

Furthermore, be aware of `CMAKE_BUILD_TYPE`, which can be either
- `Release` for optimized builds
- `RelWithDebInfo` for optimized builds with debug info (default)
- `Debug` for debug builds (with [**Sanitizers enabled**](https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Instrumentation-Options.html))

### Examples
Download and setup a build directory:
```
https://github.com/IPPL-framework/ippl
cd ippl
mkdir build
cd build
```
#### CMakeUserPresets
In the root IPPL source folder, there is a cmake user presets file which can be used to set some default cmake settings, they may be used in the following way
```
cmake --prefix=release-testing ...
```
This will set the following variables automatically (exact values may change over time)
```
        "IPPL_ENABLE_TESTS": "ON",
        "IPPL_ENABLE_UNIT_TESTS": "ON"
        "BUILD_SHARED_LIBS": "ON",
        "CMAKE_BUILD_TYPE": "Release",
        "Kokkos_VERSION_DEFAULT": "4.5.01",
        "Heffte_VERSION_DEFAULT": "2.4.0",
        "IPPL_PLATFORMS": "OPENMP;CUDA",
        "IPPL_ENABLE_FFT": "ON",
        "IPPL_ENABLE_ALPINE": "ON",
        "IPPL_ENABLE_COSMOLOGY": "ON",
        "IPPL_USE_STANDARD_FOLDERS": "OFF"
```
Users are encouraged to define additional sets of flags and create presets for them.

#### Serial debug build with tests and newest Kokkos
```
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_STANDARD=20 \
    -DIPPL_ENABLE_TESTS=True \
    -DKokkos_VERSION=4.2.00
```
#### OpenMP release build with alpine and FFTW
```
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DIPPL_ENABLE_FFT=ON \
    -DIPPL_ENABLE_SOLVERS=ON \
    -DIPPL_ENABLE_ALPINE=True \
    -DIPPL_ENABLE_TESTS=ON \
    -DIPPL_PLATFORMS=openmp \
    -DHeffte_ENABLE_FFTW=True
```
#### Cuda alpine release build
```
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ARCH_[architecture]=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DIPPL_ENABLE_FFT=ON \
    -DIPPL_ENABLE_TESTS=ON \
    -DIPPL_USE_ALTERNATIVE_VARIANT=ON \
    -DIPPL_ENABLE_SOLVERS=ON \
    -DIPPL_ENABLE_ALPINE=True \
    -DIPPL_PLATFORMS=cuda
```
#### HIP release build (LUMI)
```
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_FLAGS=--offload-arch=gfx90a \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
      -DKokkos_ENABLE_DEBUG=OFF \
      -DKokkos_ARCH_ZEN3=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DKokkos_ENABLE_HIP=ON \
      -DIPPL_PLATFORMS="HIP;OPENMP" \
      -DIPPL_ENABLE_TESTS=ON \
      -DIPPL_ENABLE_FFT=ON  \
      -DIPPL_ENABLE_SOLVERS=ON \
      -DIPPL_ENABLE_ALPINE=ON \
      -DHeffte_ENABLE_ROCM=ON \
      -DHeffte_ENABLE_GPU_AWARE_MPI=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/mpich/8.1.28/ofi/amd/5.0/lib -L/opt/cray/pe/mpich/8.1.28/gtl/lib -L/opt/cray/pe/libsci/24.03.0/AMD/5.0/x86_64/lib -L/opt/cray/pe/dsmml/0.3.0/dsmml
/lib -L/opt/cray/xpmem/2.8.2-1.0_5.1__g84a27a5.shasta/lib64 -lsci_amd_mpi -lsci_amd -ldl -lmpi_amd -lmpi_gtl_hsa -ldsmml -lxpmem -L/opt/rocm-6.0.3/lib/lib -L/opt/rocm-6.0.3/lib/lib64 -L/opt/roc
m-6.0.3/lib/llvm/lib"
```


`[architecture]` should be the target architecture, e.g.
- `PASCAL60`
- `PASCAL61`
- `VOLTA70`
- `VOLTA72`
- `TURING75`
- `AMPERE80` (PSI GWENDOLEN machine)
- `AMD_GFX90A` (LUMI machine)
- `HOPPER90` (Merlin7 GPUs)

### Enable Scripts
We add `IPPL_ENABLE_SCRIPTS=ON/OFF` and when enabled, cmake will use `configure_file` to copy the scripts to the build dir, and replace some strings in them with cmake generated ones with the correct paths/values in. This allows the user to
```
cmake -DIPPL_ENABLE_SCRIPTS=ON  ....
make LandauDamping
...
-- Scripts configured in /capstor/scratch/cscs/biddisco/build-santis/scripts
```
then
```
./scripts/landau/strong-scaling-alps/generate.sh
```
and the result will be something like
```
Generating job for node count 004 in /capstor/scratch/cscs/biddisco/build-santis/ippl/strongscaling_landau/nodes_004
Submitted batch job 396349
...
Generating job for node count 256 in /capstor/scratch/cscs/biddisco/build-santis/ippl/strongscaling_landau/nodes_256
Submitted batch job 396355
```
# Contributions
We are open and welcome contributions from others. Please open an issue and a corresponding pull request in the main repository if it is a bug fix or a minor change.

For larger projects we recommend to fork the main repository and then submit a pull request from it. More information regarding github workflow for forks can be found in this [page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks) and how to submit a pull request from a fork can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Please follow the coding guidelines as mentioned in this [page](https://github.com/IPPL-framework/ippl/blob/master/WORKFLOW.md).

You can add an upstream to be able to get all the latest changes from the master. For example, if you are working with a fork of the main repository, you can add the upstream by:
```bash
$ git remote add upstream git@github.com:IPPL-framework/ippl.git
```
You can then easily pull by typing
```bash
$ git pull upstream master
```
All the contributions (except for bug fixes) need to be accompanied with a unit test. For more information on unit tests in IPPL please
take a look at this [page](https://github.com/IPPL-framework/ippl/blob/master/UNIT_TESTS.md).

## Citing IPPL

```
@inproceedings{muralikrishnan2024scaling,
  title={Scaling and performance portability of the particle-in-cell scheme for plasma physics applications
         through mini-apps targeting exascale architectures},
  author={Muralikrishnan, Sriramkrishnan and Frey, Matthias and Vinciguerra, Alessandro and Ligotino, Michael
          and Cerfon, Antoine J and Stoyanov, Miroslav and Gayatri, Rahulkumar and Adelmann, Andreas},
  booktitle={Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing (PP)},
  pages={26--38},
  year={2024},
  organization={SIAM}
}
```

# Job scripts for running on Merlin and Gwendolen (at PSI)
You can use the following example job scripts to run on the local PSI computing cluster, which uses slurm.
More documentation on the local cluster can be found [here](https://lsm-hpce.gitpages.psi.ch/merlin6/introduction.html) (need to be in the PSI network to access).

## Merlin CPU (MPI + OpenMP)
For example, to run a job on 1 MPI node, with 44 OpenMP threads:
```
#!/bin/bash
#SBATCH --partition=hourly      # Using 'hourly' will grant higher priority
#SBATCH --nodes=1               # No. of nodes
#SBATCH --ntasks-per-node=1     # No. of MPI ranks per node. Merlin CPU nodes have 44 cores
#SBATCH --cpus-per-task=44      # No. of OMP threads
#SBATCH --time=00:05:00         # Define max time job will run (e.g. here 5 mins)
#SBATCH --hint=nomultithread    # Without hyperthreading
##SBATCH --exclusive            # The allocations will be exclusive if turned on (remove extra hashtag to turn on)

#SBATCH --output=<output_file_name>.out  # Name of output file
#SBATCH --error=<error_file_name>.err    # Name of error file

export OMP_NUM_THREADS=44
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# need to pass the --cpus-per-task option to srun otherwise will not use more than 1 core per task
# (see https://lsm-hpce.gitpages.psi.ch/merlin6/known-problems.html#sbatch-using-one-core-despite-setting--ccpus-per-task)

srun --cpus-per-task=44 ./<your_executable> <args>
```

## Gwendolen GPU
For example, to run a job on 4 GPUs (max on Gwendolen is 8 GPUs, which are all on a single node):
```
#!/bin/bash
#SBATCH --time=00:05:00         # Define max time job will run (e.g. here 5 mins)
#SBATCH --nodes=1               # No. of nodes (there is only 1 node on Gwendolen)
#SBATCH --ntasks=4              # No. of tasks (max. 8)
#SBATCH --clusters=gmerlin6     # Specify that we are running on the GPU cluster
#SBATCH --partition=gwendolen   # Running on the Gwendolen partition of the GPU cluster
#SBATCH --account=gwendolen
##SBATCH --exclusive            # The allocations will be exclusive if turned on (remove extra hashtag to turn on)
#SBATCH --gpus=4                # No. of GPUs (max. 8)

#SBATCH --output=<output_file_name>.out  # Name of output file
#SBATCH --error=<error_file_name>.err    # Name of error file

srun ./<your_executable> <args> --kokkos-map-device-id-by=mpi_rank
```
## LUMI GPU partition
For example, to run a job on 8 nodes with 8 GPUs each:
```
#!/bin/bash
#SBATCH --job-name=TestGaussian
#SBATCH --error=TestGaussian-%j.error
#SBATCH --output=TestGaussian-%j.out
#SBATCH --partition=dev-g  # partition name
#SBATCH --time=00:10:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 64 total (8x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank per node
#SBATCH --account=project_xxx
#SBATCH --hint=nomultithread
module load  LUMI/24.03 partition/G cpeAMD rocm buildtools/24.03
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -s unlimited
export EXE_DIR=/users/adelmann/sandbox/vico-paper/build/test/solver
cat << EOF > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu
srun ./select_gpu ${EXE_DIR}/TestGaussian 1024 1024 1024 pencils a2av no-reorder HOCKNEY --info 5
rm -rf ./select_gpu
```

# Profiling IPPL MPI calls

You can use the mpiP tool (https://github.com/LLNL/mpiP) to get statistics about the MPI calls in IPPL. 

To use it, download it from [Github](https://github.com/LLNL/mpiP) and follow the instructions to install it. You may run into some issues while installing, here is a list of common issues and the solution:
- On Cray systems "MPI_Init not defined": This I fixed by passing the correct Cray wrappers for the compilers to the configure: `./configure CC=cc FC=ftn F77=ftn`
- If you have an issue with it not recognizing a function symbol in Fortran 77, you need to substitute the line `echo "main(){ FF(); return 0; }" > flink.c` (line 706) in the file `configure.ac` by the following line `echo "extern void FF(); int main() { FF(); return 0; }" > flink.c`
- During the `make all`, if you run into an issue of some Testing file not recognizing mpi.h, then you need to add the following line `CXX = CC` in the file `Testing/Makefile`.

If the installation was successful, you should have the library `libmpip.so` in the mpiP directory. 

To instument your application with the mpiP library, add the following line to your jobscript (or run it in your command line if you are running locally/on an interactive node):
`export LD_PRELOAD=$[path to mpip directory]/libmpiP.so`
To pass any options to mpiP, you can export the variable MPIP with the options you want. For example, if you would like to get a histogram of the data sent by MPI calls (option `-y`), you would need to add the following line to your jobscript:
`export MPIP="-y"`

If you application has been correctly instrumented, you will see that mpiP has been found and its version is printed at the top of the standard output. At the end of the standard output, you will get the name of the file containing the MPI statistics:
`Storing mpiP output in ...`

To get a total amount of bytes moved around by your application, you can use the python script mpiP.py (found in the top level IPPL directory) in the following form:
`python3 mpiP.py [path/to/directory]`
where path/to/directory refers to the place where the .mpiP output can be found. This python script will then print out the total amount of Bytes moved by MPI in your application.

Happy profiling!

# Build Instructions
Here we compile links to recipies for easy build on various HPC systems. 

## MERLIN 7 (PSI)
[IPPL build for A100 and HG](https://hpce.pages.psi.ch/merlin7/ippl.html)

## ALPS (CSCS)
Start by loading a `uenv` that contains most of the tools we want. Note that in future an `official` uenv will be provided in the CSCS uenv repository, but until testing is complete, use the following ...

```bash
uenv start --view=develop \
/capstor/store/cscs/cscs/csstaff/biddisco/uenvs/opal-x-gh200-mpich-gcc-2025-09-28.squashfs
```
or, look for a newer one and pick the one with the latest date in the name using
```bash
ls -al /capstor/store/cscs/cscs/csstaff/biddisco/uenvs/opal-x-gh200-*.squashfs
``` 
At the time of writing, the uenv provides (as well as many other packages)
```yaml
cmake@4.1.1         ~doc+ncurses+ownlibs~qtgui 
cray-mpich@9.0.0    +cuda+cxi~rocm 
cuda@12.8.1         ~allow-unsupported-compilers~dev 
eigen@3.4.0         ~ipo~nightly~rocm 
fftw@3.3.10         +mpi~openmp~pfft_patches+shared 
gcc@13.4.0          ~binutils+bootstrap~graphite~mold~nvptx~piclibs+profiled+strip 
googletest@1.17.0   ~absl+gmock~ipo+pthreads+shared 
gsl@2.8             ~external-cblas+pic+shared 
h5hut@master        ~fortran+mpi 
hdf5@1.14.6         +cxx~fortran+hl~ipo~java~map+mpi+shared~subfiling+szip~threadsafe+tools api=default 
heffte@2.4.1        +cuda+fftw~fortran~ipo~magma~mkl~python~rocm+shared 
hpctoolkit@2025.0.1 +cuda~docs~level_zero~mpi~opencl+papi~python~rocm~strip+viewer 
hwloc@2.11.1        ~cairo~cuda~gl~level_zero~libudev~libxml2~nvml~opencl+pci~rocm 
kokkos@4.7.00       ~aggressive_vectorization~alloc_async~cmake_lang~compiler_warnings+complex_align+cuda~cuda_constexpr~cuda_lambda~cuda_ldg_intrinsic~cuda_relocatable_device_code~cuda_uvm~debug~debug_bounds_check~debug_dualview_modify_check~deprecated_code~examples~hip_relocatable_device_code~hpx~hpx_async_dispatch~hwloc~ipo~memkind~numactl+openmp~openmptarget~pic~rocm+serial+shared~sycl~tests~threads~tuning+wrapper build_system=cmake build_type=Release cuda_arch=90 cxxstd=20 generator=make intel_gpu_arch=none 
ninja@1.13.0        +re2c 
```
You can check what is inside the uenv by executing the command
```bash
# This will show all packages installed by spack (including any ones you might have installed yourself outside of a uenv)
spack find -flv

# use this to only show packages inside the uenv (ie. not any you have installed elsewhere)
spack -C /user-environment/config find -flv
```
It is important to use the `--view=develop` when loading the uenv as this sets-up the paths to packages in the spack environment ready for you to use them (without needing to manually `spack load xxx` packages individually) (In fact it will also add `/user-environment/env/develop/` to your `CMAKE_PREFIX_PATH`) which makes cmake-built packages 'just work'. 

To build, try the following which uses default CMake settings (`release-testing`) taken from `CMakeUserPresets.json` (in the root ippl source of the cmake-alps branch - it is not required to use this branch, but cmake support has been cleaned up)
```bash
ssh daint
uenv start --view=develop /capstor/scratch/cscs/biddisco/uenvs/gh200-opalxgccmpich-2025-07-23.squashfs

# clone IPPL
mkdir -p $HOME/src/ippl
cd $HOME/src
git clone https://github.com/IPPL-framework/ippl

# (optionally) checkout the cmake-alps branch since it is not yet merged to master
cd $HOME/src/ippl
git remote add biddisco https://github.com/biddisco/ippl.git
git fetch biddisco cmake-alps
git checkout cmake-alps

# create a build dir
mkdir -p $HOME/build/ippl
cd $HOME/build/ippl

# run cmake (note that ninja is available in the uenv and "cmake -G Ninja" can be used)
cmake --preset=release-testing -DCMAKE_INSTALL_PREFIX=$HOME/apps/ippl -DCMAKE_CUDA_ARCHITECTURES=90 $HOME/src/ippl/
```
