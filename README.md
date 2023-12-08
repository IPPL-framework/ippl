[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5940225.svg)](https://doi.org/10.5281/zenodo.8389192)
[![License](https://img.shields.io/github/license/matt-frey/epic)](https://github.com/IPPL-framework/ippl/blob/master/LICENSE)

# Independent Parallel Particle Layer (IPPL)
Independent Parallel Particle Layer (IPPL) is a performance portable C++ library for Particle-Mesh methods. IPPL makes use of Kokkos (https://github.com/kokkos/kokkos), HeFFTe (https://github.com/icl-utk-edu/heffte), and MPI (Message Passing Interface) to deliver a portable, massively parallel toolkit for particle-mesh methods. IPPL supports simulations in one to six dimensions, mixed precision, and asynchronous execution in different execution spaces (e.g. CPUs and GPUs). 

## Installing IPPL and its dependencies

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
@article{muralikrishnan2022scaling,
  title={Scaling and performance portability of the particle-in-cell scheme for plasma physics
         applications through mini-apps targeting exascale architectures},
  author={Muralikrishnan, Sriramkrishnan and Frey, Matthias and Vinciguerra, Alessandro
          and Ligotino, Michael and Cerfon, Antoine J and Stoyanov, Miroslav and
          Gayatri, Rahulkumar and Adelmann, Andreas},
  journal={arXiv preprint arXiv:2205.11052},
  year={2022}
}
```
