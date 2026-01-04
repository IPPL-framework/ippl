# Installing IPPL on EULER {#Installation}

This guide outlines the steps to install the IPPL library on the EULER cluster. Before beginning, ensure you are connected to the ETH-VPN to access the cluster.

## Connecting to the EULER Cluster

Use SSH to connect to EULER. Replace '\<username\>' with your actual username.

        ssh -Y <username>@euler.ethz.ch

The '-Y' flag enables trusted X11 forwarding, necessary for running graphical applications remotely.

## Preparing the Environment

_Load the New Software Stack:_ Transition to the new software stack to access the latest dependencies:

        env2lmod

_Clean the Environment:_ Ensure no previous modules are loaded to avoid conflicts:

        module purge

_Load Dependencies:_ Load the required modules for IPPL:

        module load gcc/11.4.0 cmake/3.26.3 cuda/12.1.1 openmpi/4.1.4

## Cloning the IPPL Library

Clone the IPPL library from its repository:

        git clone https://github.com/IPPL-framework/ippl.git

## Building IPPL

#### Setup a build directory:

        cd ippl
        mkdir build
        cd build

#### Create build files

Choose from the following options based on your needs. If necessary, you can build multiple versions in separate directories.

**Serial Version** (for single-node computing)

        cmake .. -DCMAKE_CXX_STANDARD=20 -DIPPL_PLATFORMS=SERIAL -DIPPL_ENABLE_SOLVERS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_ALPINE=ON

**OpenMP Version** (for multi-threaded computing):

        cmake .. -DCMAKE_CXX_STANDARD=20 -DIPPL_PLATFORMS=OPENMP -DIPPL_ENABLE_SOLVERS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_ALPINE=ON

**Cuda Version** (for GPU computing):

        cmake .. -DCMAKE_CXX_STANDARD=20 -DIPPL_PLATFORMS=CUDA -DIPPL_ENABLE_SOLVERS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_ALPINE=ON -DIPPL_USE_ALTERNATIVE_VARIANT=OFF -DKokkos_ARCH_[architecture]=ON

[architecture] should be the target architecture, e.g.

- PASCAL60
- VOLTA70
- TURING75
- AMPERE80


#### Compile

        make

## Testing Your Installation

Launch an interactive job on EULER to test your installation:

        srun -n 1 --time=1:00:00 --mem-per-cpu=32g --pty bash

This command allocates one computing node with 32GB of RAM for 60 minutes.

**Task** : Execute a miniapp in the '/alpine' folder to verify the installation (don't forget to compile with Make).


## Links

Repository for Independent Parallel Particle Layer (IPPL)

        https://github.com/IPPL-framework/ippl
