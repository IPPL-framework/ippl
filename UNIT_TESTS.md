# Overview

IPPL allows the user to customize the behavior of its data types using template parameters. The unit tests verify that everything is working under all relevant and available configurations of the parameters. In particular, the tests verify behavior under
- mixed precision, i.e. components using single or double precision (`float`, `double`)
- varying dimensionality, i.e. components in 1D, 2D, 3D, etc. up to 6D (the maximum supported by Kokkos)
- mixed execution spaces, i.e. components using different accelerators (e.g. OpenMP, CUDA) in different memory spaces (host, device)
    - The unit tests are instantiated for all available execution spaces. These are determined based on the compilation settings for Kokkos, which exposes compiler macros to indicate which execution spaces are available.

As an example, we look at the unit tests for `BareField`. Consider the following Kokkos configuration:
```sh
cmake \
    -DCMAKE_INSTALL_PREFIX=${KOKKOS_PREFIX} \
    -DCMAKE_CXX_COMPILER="${compiler}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DKokkos_ENABLE_SERIAL=OFF \
    -DKokkos_ARCH_AMPERE80=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_UVM=OFF \
    kokkos
```

This compiles Kokkos to include support for OpenMP and CUDA as accelerators. All Kokkos kernels can thus use either of these execution spaces. In this case, each individual unit test is compiled 24 times: once for each possible combination of precision, rank, and execution space. This is visible in the first few lines of output:
```
[==========] Running 192 tests from 24 test suites.
[----------] Global test environment set-up.
[----------] 8 tests from BareFieldTest/0, where TypeParam = Parameters<double, Kokkos::OpenMP, Rank<1u> >
[ RUN      ] BareFieldTest/0.DeepCopy
[       OK ] BareFieldTest/0.DeepCopy (0 ms)
[ RUN      ] BareFieldTest/0.Sum
[       OK ] BareFieldTest/0.Sum (0 ms)
[ RUN      ] BareFieldTest/0.Min
[       OK ] BareFieldTest/0.Min (0 ms)
[ RUN      ] BareFieldTest/0.Max
[       OK ] BareFieldTest/0.Max (0 ms)
[ RUN      ] BareFieldTest/0.Prod
[       OK ] BareFieldTest/0.Prod (0 ms)
[ RUN      ] BareFieldTest/0.ScalarMultiplication
[       OK ] BareFieldTest/0.ScalarMultiplication (0 ms)
[ RUN      ] BareFieldTest/0.DotProduct
[       OK ] BareFieldTest/0.DotProduct (0 ms)
[ RUN      ] BareFieldTest/0.AllFuncs
[       OK ] BareFieldTest/0.AllFuncs (3 ms)
[----------] 8 tests from BareFieldTest/0 (6 ms total)
```

Here, we see that the first configuration for the unit tests is for a 1D field in double precision using OpenMP. In particular, the field data is stored in host memory. Among the other tests, we also have the following configurations:
```
[----------] 8 tests from BareFieldTest/9, where TypeParam = Parameters<float, Kokkos::OpenMP, Rank<3u> >
[----------] 8 tests from BareFieldTest/10, where TypeParam = Parameters<double, Kokkos::Cuda, Rank<3u> >
[----------] 8 tests from BareFieldTest/15, where TypeParam = Parameters<float, Kokkos::Cuda, Rank<4u> >
```

The configurations using `Kokkos::Cuda` as the execution space store the field data on GPUs, as one would expect.

Not all sets of unit tests have the same level of configuration. Data structures that do not involve parallel execution don't have an associated execution space; instantiations of these unit tests are thus independent of the number of Kokkos backends. On the other hand, some data structures can use multiple execution spaces. One example is particle bunches, which can store attributes in different execution spaces. Here, we have the following setup:
```cpp
template <typename T, typename IDSpace, typename PositionSpace, unsigned Dim>
class ParticleBaseTest<Parameters<T, IDSpace, PositionSpace, Rank<Dim>>> : public ::testing::Test
```

Unit tests are instantiated for a particle bunch with the following properties:
- Particles have an attribute of type `T` (which will be `double` or `float`)
- The particle positions have type `T` (controls the precision of the position data)
- The particle positions use the execution space `PositionSpace`
- The particles are identified by IDs, which use the execution space `IDSpace`
- The particle positions are `Dim`-vectors (controls the dimensionality of the particle layout)

By instantiating unit tests for all possible combinations of execution spaces of multiple components, we can verify not only the correctness of the computations, but also that data is being copied where needed and that data in all execution spaces is being handled correctly. These are some of the test configurations generated for particle bunches using the same Kokkos configuration as the previous example:
```
... TypeParam = Parameters<float, Kokkos::OpenMP, Kokkos::Cuda, Rank<3u> >
... TypeParam = Parameters<double, Kokkos::Cuda, Kokkos::Cuda, Rank<3u> >
... TypeParam = Parameters<float, Kokkos::Cuda, Kokkos::Cuda, Rank<3u> >
... TypeParam = Parameters<double, Kokkos::OpenMP, Kokkos::OpenMP, Rank<4u> >
... TypeParam = Parameters<float, Kokkos::OpenMP, Kokkos::OpenMP, Rank<4u> >
```

# Implementation

The generation of test configurations is implemented in `unit_tests/TestUtils.h`. It is controlled using the following struct:

```cpp
template <typename... Ts>
struct Parameters { ... };

struct TestParams {
    using Spaces     = ippl::detail::TypeForAllSpaces<std::tuple>::exec_spaces_type;
    using Precisions = std::tuple<double, float>;
    using Combos     = CreateCombinations<Precisions, Spaces>::type;

    template <unsigned... Dims>
    using Ranks = std::tuple<Rank<Dims>...>;

    template <unsigned... Dims>
    using CombosWithRanks = typename CreateCombinations<Precisions, Spaces, Ranks<Dims...>>::type;

    template <unsigned... Dims>
    using tests = typename TestForTypes<
        std::conditional_t<sizeof...(Dims) == 0, Combos, CombosWithRanks<Dims...>>>::type;
...
};
```

First, we generate a tuple containing all the available execution spacs (see `Utility/TypeUtils.h`). We then create tuples with the other parameters we want to test. The `CreateCombinations` struct recursively generates all combinations of the chosen parameters at compile time and instantiates the `Parameter` type to hold these combinations. We then use GoogleTest's `testing::Types<...>` to instantiate all the unit tests for each combination. Example:

```cpp
template <typename>
class FieldTest;

template <typename T, typename ExecSpace, unsigned Dim>
class FieldTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test { ... };

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_CASE(FieldTest, Tests);

TYPED_TEST(FieldTest, DeepCopy) { ... }
```

If we want to generate combinations beyond just ranks/precision/execution space like above, we can use the tuple generation with other types. For example, the particle bunch tests use two execution spaces:

```cpp
template <typename>
class ParticleBaseTest;

template <typename T, typename IDSpace, typename PositionSpace, unsigned Dim>
class ParticleBaseTest<Parameters<T, IDSpace, PositionSpace, Rank<Dim>>> : public ::testing::Test { ... };

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Ranks      = TestParams::Ranks<1, 2, 3, 4, 5, 6>;
using Combos     = CreateCombinations<Precisions, Spaces, Spaces, Ranks>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(ParticleBaseTest, Tests);

TYPED_TEST(ParticleBaseTest, CreateAndDestroy) { ... }
```

# Run all Unit tests

```bash
#!/bin/bash

for file in `find $1/unit_tests -type f`
do
    if [[ -x "$file" ]]
    then
        $file
    fi
done
```

The first argument needs to point to the directory were you build ippl.
