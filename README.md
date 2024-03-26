[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5940225.svg)](https://doi.org/10.5281/zenodo.8389192)
[![License](https://img.shields.io/github/license/IPPL-framework/ippl)](https://github.com/IPPL-framework/ippl/blob/master/LICENSE)

# Independent Parallel Particle Layer (IPPL)
Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs). 

All IPPL releases (< 3.2.0) are available under the BSD 3-clause license. Since version 3.2.0, this repository includes a modified version of the `variant` header by GNU, created to support compilation under CUDA 12.2 with GCC 12.3.0. This header file is available under the same terms as the [GNU Standard Library](https://github.com/gcc-mirror/gcc); note the GNU runtime library exception. As long as this file is not removed, IPPL is available under GNU GPL version 3.

## Installing IPPL and its dependencies

All the new developments of IPPL are merged in to the `master` branch which can make it potentially unstable from time to time. So if you want a stable and more tested version
please checkout the tagged branch correspodning to the last release (e.g. `git checkout tags/IPPL-x.x.x`). Otherwise if you want the latest developments go with the master with the above caveat in mind.

### Requirements
The following libraries are required:

* MPI (GPU-aware if building for GPUs)
* [Kokkos](https://github.com/kokkos) >= 4.1.00
* [HeFFTe](https://github.com/icl-utk-edu/heffte) >= 2.2.0; only required if IPPL is built with FFTs enabled (`ENABLE_FFT=ON`)

To build IPPL and its dependencies, we recommend using the [IPPL build scripts](https://github.com/IPPL-framework/ippl-build-scripts). See the [documentation](https://github.com/IPPL-framework/ippl-build-scripts#readme) for more info on how to use the IPPL build script.


## Contributions
We are open and welcome contributions from others. Please open an issue and a corresponding pull request in the main repository if it is a bug fix or a minor change.

For larger projects we recommend to fork the main repository and then submit a pull request from it. More information regarding github workflow for forks can be found in this [page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks) and how to submit a pull request from a fork can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Please follow the coding guidelines as mentioned in this [page](https://github.com/IPPL-framework/ippl/blob/master/WORKFLOW.md). 

You can add an upstream to be able to get all the latest changes from the master. For example, if you are working with a fork of the main repository, you can add the upstream by:
```bash
$ git remote add upstream git@github.com:IPPL-framework/ippl.git
```
You can then easily pull by typing
```bash
$ git pull upstream master
````
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
