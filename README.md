# Independent Parallel Particle Layer (IPPL)
Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs). 

## Repository organization
You can find the presentations based on IPPL in [IPPL-presentations](https://github.com/IPPL-framework/ippl-presentations). 

## Getting the source code
The main repository of IPPL is obtained with
```bash
$ git clone git@github.com:IPPL-framework/ippl.git
```
or
```bash
$ git clone https://github.com/IPPL-framework/ippl.git
```

### Working with a fork
For larger projects we recommend to fork the main repository. 

You can add an upstream to be able to get all the latest changes from the master. For example, if you are working with a fork of the main repository, you can add the upstream by:
```bash
$ git remote add upstream git@github.com:IPPL-framework/ippl.git
```
You can then easily pull by typing
```bash
$ git pull upstream master
````

