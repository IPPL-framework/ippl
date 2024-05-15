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

## Cloning the IPPL Library and Build Scripts 

Clone the IPPL library from its repository:

        git clone https://github.com/IPPL-framework/ippl.git

Clone the IPPL build scripts for a simplified installation process: 

        git clone https://github.com/IPPL-framework/ippl-build-scripts.git

## Building IPPL 

Choose from the following options based on your needs. If necessary, you can build multiple versions in separate directories.

**Serial Version** (for single-node computing)
        
        ./ippl-build-scripts/999-build-everything -t serial -k -f -i -u

**OpenMP Version** (for multi-threaded computing):
        
        ./ippl-build-scripts/999-build-everything -t openmp -k -f -i -u

**CUDA Version** (for GPU computing):
        
        ./ippl-build-scripts/999-build-everything -t cuda -k -f -i -u

## Testing Your Installation

Launch an interactive job on EULER to test your installation:

        srun -n 1 --time=1:00:00 --mem-per-cpu=32g --pty bash

This command allocates one computing node with 32GB of RAM for 60 minutes.

**Task** : Execute a miniapp in the '/alpine' folder to verify the installation (don't forget to compile with Make).


## Links

Repository for Independent Parallel Particle Layer (IPPL)
        
        https://github.com/IPPL-framework/ippl

Repository for IPPL Build Scripts

        https://github.com/IPPL-framework/ippl-build-scripts