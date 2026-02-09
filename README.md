<p align="center">
  <img src="./assets/ippl-logo.png" alt="IPPL" width="30%" height="30%"><br>
</p>
<p align="right">
  <sub><sub><sup><sup><em> 
  IPPL-logo; design by S.A.T.Klapproth</em> </sup></sup></sup></sub>
</p>

<p align="center">
  <a href="https://ippl-bc4558.pages.jsc.fz-juelich.de/">
    <img alt="CI/CD" src="https://img.shields.io/badge/CI/CD-red.svg">
  </a>
  <a href="https://github.com/IPPL-framework/ippl/blob/master/ci/cscs/cscs-ci-cd.md">
    <img alt="CI/CD CSCS" src="ci/cscs/CSCS_logo_short.jpg" width=50>
  </a>
  <a href="https://en.wikipedia.org/wiki/C%2B%2B20">
    <img alt="C++ standard" src="https://img.shields.io/badge/c%2B%2B-20-blue.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.8389192">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.5940225.svg">
  </a>
  <a href="https://github.com/IPPL-framework/ippl/blob/master/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/IPPL-framework/ippl">
  </a>
</p>

The IPPL (Independent Parallel Particle Layer) library
provides performance portable and dimension independent
building blocks for scientific simulations requiring particle-mesh
methods, with Eulerian (mesh-based) and Lagrangian (particle-based) approaches.
IPPL makes use of [Kokkos](https://github.com/kokkos/kokkos), [HeFFTe](https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable,
massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs).

**[Installation](#installation)** |
**[Contributions](#contributions)** |
**[CI/CD](#cicd-and-pr-testing)** |
**[Citing IPPL](#citing-ippl)** |
**[SLURM Job scripts](#slurm-job-scripts)** |
**[Profiling](#profiling)**

# Installation
We compiled installation [instructions](./INSTALLATION.md) for many HPC systems. 

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

# CI/CD and PR testing
Please see [Julich CI results](https://ippl-bc4558.pages.jsc.fz-juelich.de/) and [CSCS PR testing](ci/cscs/cscs-ci-cd.md) for further information.

# Citing IPPL

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

# SLURM Job scripts

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


# Profiling

## MPI Calls 
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

## Profiling on LUMI

### rocprof

Analysis with: https://ui.perfetto.dev/

```
#!/bin/bash -l
#
#SBATCH --job-name=opalx1
#SBATCH --error=opalx-%j.error
#SBATCH --output=opalx-2-%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes 1
#SBATCH --ntasks-per-core=1
#SBATCH -c 56 --threads-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465001705 
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1
 
ulimit -s unlimited
export EXE_DIR=/users/adelmann/sandbox/opalx/build/src/
module load cray-python/3.11.7 
module use /appl/local/containers/test-modules
module load LUMI/24.03 partition/G cpeAMD rocm/6.1.3 buildtools/24.03

cat << EOF > select_gpu
#!/bin/bash
export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu
srun ./select_gpu rocprof --hip-trace ${EXE_DIR}/opalx input.in --info 5
rm -rf ./select_gpu

```


### omniperf (do not use omnitrace)

doc url: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.2.4/how-to/profile/mode.html

```
#!/bin/bash -l
#
#SBATCH --job-name=opalx1
#SBATCH --error=opalx-%j.error
#SBATCH --output=opalx-2-%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes 1
#SBATCH --ntasks-per-core=1
#SBATCH -c 56 --threads-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465001705 
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1
 
ulimit -s unlimited
export EXE_DIR=/users/adelmann/sandbox/opalx/build/src/
module load cray-python/3.11.7 
module use /appl/local/containers/test-modules
module load LUMI/24.03 partition/G cpeAMD rocm/6.1.3 buildtools/24.03
module load omniperf
cat << EOF > select_gpu
#!/bin/bash
#export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu
srun ./select_gpu omniperf profile --name opalx  --roof-only --kernel-names -- ${EXE_DIR}/opalx input.in --info 5
rm -rf ./select_gpu
```









Happy profiling!
