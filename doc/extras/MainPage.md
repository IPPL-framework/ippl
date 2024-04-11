# Independent Parallel Particle Layer (IPPL)
Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs). 

All IPPL releases (< 3.2.0) are available under the BSD 3-clause license. Since version 3.2.0, this repository includes a modified version of the `variant` header by GNU, created to support compilation under CUDA 12.2 with GCC 12.3.0. This header file is available under the same terms as the [GNU Standard Library](https://github.com/gcc-mirror/gcc); note the GNU runtime library exception. As long as this file is not removed, IPPL is available under GNU GPL version 3.


# Goals of Independent Parallel Particle Layer (IPPL)
- Open source modern C++ (requires at least C++ 20) library for grid and particle-based methods
- Performance Portability across heterogeneous parallel architectures (different CPUs, GPUs etc.)
- Development of reusable, cross-domain components to enable rapid application development
- Prototyping library for the development of novel numerical methods targeting exascale architectures
- Shorter time from problem inception to working parallel simulations
- Primary application is particle-in-cell methods (backend of production particle accelerator code OPAL) but applicable for other use cases too

