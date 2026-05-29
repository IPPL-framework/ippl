#include "Ippl.h"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <numeric>
#include <string>

#include "Utility/ViewUtils.h"

#include "Interpolation/Gather/Gather.h"
#include "Interpolation/Gather/GatherConfig.h"
#include "Interpolation/Kernels.h"
#include "Interpolation/Scatter/Scatter.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

//=============================================================================
// Test Particle Bunch with Additional Attributes
//=============================================================================

namespace ippl {
    namespace test {

        template <typename T, typename PLayout>
        class ComprehensiveTestBunch : public ParticleBase<PLayout> {
        public:
            using Base         = ParticleBase<PLayout>;
            using complex_type = Kokkos::complex<T>;

            // Particle attributes for various tests
            ParticleAttrib<T> weight;                // Scalar weights for scatter
            ParticleAttrib<T> gathered_scalar;       // Scalar gathered values
            ParticleAttrib<complex_type> Q_scatter;  // Complex for adjointness tests
            ParticleAttrib<complex_type> Q_gather;   // Complex for adjointness tests

            explicit ComprehensiveTestBunch(PLayout& layout)
                : Base(layout) {
                this->addAttribute(weight);
                this->addAttribute(gathered_scalar);
                this->addAttribute(Q_scatter);
                this->addAttribute(Q_gather);
            }
        };

        // Helper to compute complex inner product for particles
        template <typename Bunch, typename Attrib1, typename Attrib2>
        Kokkos::complex<typename Attrib1::value_type::value_type>
        computeParticleInnerProductComplex(Bunch& bunch, Attrib1& a1, Attrib2& a2) {
            using T            = typename Attrib1::value_type::value_type;
            using complex_type = Kokkos::complex<T>;
            using ExecSpace    = typename Attrib1::execution_space;

            auto view1 = a1.getView();
            auto view2 = a2.getView();
            size_t n   = bunch.getLocalNum();

            T localReal = 0.0;
            T localImag = 0.0;

            Kokkos::parallel_reduce(
                "particle_inner_product", Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(size_t i, T& sumReal, T& sumImag) {
                    complex_type prod = view1(i) * Kokkos::conj(view2(i));
                    sumReal += prod.real();
                    sumImag += prod.imag();
                },
                localReal, localImag);
            Kokkos::fence();

            T globalReal = 0.0, globalImag = 0.0;
            MPI_Allreduce(&localReal, &globalReal, 1,
                          std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                          ippl::Comm->getCommunicator());
            MPI_Allreduce(&localImag, &globalImag, 1,
                          std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM,
                          ippl::Comm->getCommunicator());

            return complex_type(globalReal, globalImag);
        }

        // Initialize field with random complex values
        template <typename Field>
        void initializeRandomFieldComplex(Field& field) {
            using T                = typename Field::value_type::value_type;
            using complex_type     = typename Field::value_type;
            using ExecSpace        = typename Field::execution_space;
            constexpr unsigned Dim = Field::dim;
            using index_array_type = RangePolicy<Dim>::index_array_type;
            const auto& view       = field.getView();

            using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
            RandPool randPool(123 + Comm->rank());
            ippl::parallel_for(
                "init_random_complex", field.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    typename RandPool::generator_type gen = randPool.get_state();
                    T re                                  = gen.drand(-1.0, 1.0);
                    T im                                  = gen.drand(-1.0, 1.0);
                    apply(view, args)                     = complex_type(re, im);
                    randPool.free_state(gen);
                });
            Kokkos::fence();
        }

    }  // namespace test
}  // namespace ippl

//=============================================================================
// Kernel Type Wrapper for Testing Multiple Kernels
//=============================================================================

template <typename T>
struct KernelInfo {
    std::string name;
    int width;
    int expectedOrder;  // Interpolation order for accuracy tests
};

// Kernel tag types for typed tests
struct NGPTag {};
struct LinearTag {};
struct QuadraticTag {};
struct CubicTag {};

template <typename T, typename Tag>
struct KernelTraits;

template <typename T>
struct KernelTraits<T, NGPTag> {
    using kernel_type                 = ippl::Interpolation::NGPKernel<T>;
    static constexpr int order        = 0;
    static constexpr const char* name = "NGP";
};

template <typename T>
struct KernelTraits<T, LinearTag> {
    using kernel_type                 = ippl::Interpolation::LinearKernel<T>;
    static constexpr int order        = 1;
    static constexpr const char* name = "Linear";
};

template <typename T>
struct KernelTraits<T, QuadraticTag> {
    using kernel_type                 = ippl::Interpolation::QuadraticKernel<T>;
    static constexpr int order        = 2;
    static constexpr const char* name = "Quadratic";
};

template <typename T>
struct KernelTraits<T, CubicTag> {
    using kernel_type                 = ippl::Interpolation::CubicKernel<T>;
    static constexpr int order        = 3;
    static constexpr const char* name = "Cubic";
};

//=============================================================================
// Test Parameters
//=============================================================================

template <typename T, typename ExecSpace, unsigned Dim, typename KernelTag>
struct TestParameters {
    using value_type              = T;
    using exec_space              = ExecSpace;
    static constexpr unsigned dim = Dim;
    using kernel_tag              = KernelTag;
};

// Generate test parameter combinations
using TestTypes = ::testing::Types<
    // 3D tests with different kernels
    TestParameters<double, Kokkos::DefaultExecutionSpace, 3, NGPTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 3, LinearTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 3, QuadraticTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 3, CubicTag>,
    // 2D tests with different kernels
    TestParameters<double, Kokkos::DefaultExecutionSpace, 2, NGPTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 2, LinearTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 2, QuadraticTag>,
    TestParameters<double, Kokkos::DefaultExecutionSpace, 2, CubicTag>>;

//=============================================================================
// Test Fixture
//=============================================================================

template <typename Params>
class ScatterGatherTest : public ::testing::Test {
public:
    using value_type              = Params::value_type;
    using exec_space              = Params::exec_space;
    using T                       = value_type;
    using ExecSpace               = exec_space;
    using complex_type            = Kokkos::complex<T>;
    static constexpr unsigned Dim = Params::dim;

    using kernel_tag    = Params::kernel_tag;
    using kernel_traits = KernelTraits<T, kernel_tag>;
    using kernel_type   = kernel_traits::kernel_type;

    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using centering_type = mesh_type::DefaultCentering;
    using field_type     = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>::uniform_type;
    using complex_field_type =
        ippl::Field<complex_type, Dim, mesh_type, centering_type, ExecSpace>::uniform_type;
    using layout_type = ippl::FieldLayout<Dim>;

    using playout_type = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using bunch_type   = ippl::test::ComprehensiveTestBunch<T, playout_type>;

    using scatter_config_type = ippl::Interpolation::ScatterConfig<Dim>;
    using gather_config_type  = ippl::Interpolation::GatherConfig<Dim>;

    std::shared_ptr<layout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout;
    std::shared_ptr<bunch_type> bunch;

    kernel_type kernel;

    ippl::Vector<T, Dim> origin;
    ippl::Vector<T, Dim> extent;
    ippl::Vector<T, Dim> hx;
    ippl::Vector<size_t, Dim> gridSize;

    int nRanks{};
    int myRank{};
    int nghost{};

    ScatterGatherTest() {
        const T pi = Kokkos::numbers::pi_v<T>;
        for (unsigned d = 0; d < Dim; ++d) {
            origin[d]   = 0.0;
            extent[d]   = 2.0 * pi;
            gridSize[d] = 32;
        }
        nghost = kernel.width() / 2 + 1;
    }

    void SetUp() override {
        nRanks = ippl::Comm->size();
        myRank = ippl::Comm->rank();

        std::array<ippl::Index, Dim> domains;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        for (unsigned d = 0; d < Dim; ++d) {
            domains[d] = ippl::Index(gridSize[d]);
            hx[d]      = extent[d] / gridSize[d];
        }

        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
        layout     = std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel, true, nghost);
        mesh       = std::make_shared<mesh_type>(owned, hx, origin);

        playout = std::make_shared<playout_type>(*layout, *mesh);
        bunch   = std::make_shared<bunch_type>(*playout);

    }

    void TearDown() override {
        bunch.reset();
        playout.reset();
        mesh.reset();
        layout.reset();
    }

    //=========================================================================
    // Helper Functions
    //=========================================================================

    void createUniformParticles(size_t nParticlesPerRank) {
        bunch->create(nParticlesPerRank);

        auto R_view      = bunch->R.getView();
        auto weight_view = bunch->weight.getView();

        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + myRank * 1000);

        auto origin_local = origin;
        auto extent_local = extent;

        Kokkos::parallel_for(
            "create_uniform_particles", Kokkos::RangePolicy<ExecSpace>(0, nParticlesPerRank),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();

                ippl::Vector<T, Dim> pos;
                for (unsigned d = 0; d < Dim; ++d) {
                    pos[d] = origin_local[d] + gen.drand() * extent_local[d];
                }
                R_view(i)      = pos;
                weight_view(i) = gen.drand();
                randPool.free_state(gen);
            });
        bunch->update();
        Kokkos::fence();
    }

    void createBoundaryParticles(size_t nParticlesPerBoundary) {
        size_t nBoundaries    = 2 * Dim;
        size_t totalParticles = nBoundaries * nParticlesPerBoundary;

        bunch->create(totalParticles);

        auto R_view      = bunch->R.getView();
        auto weight_view = bunch->weight.getView();

        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(123 + myRank * 1000);

        auto origin_local = origin;
        auto extent_local = extent;

        Kokkos::parallel_for(
            "create_boundary_particles", Kokkos::RangePolicy<ExecSpace>(0, totalParticles),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();

                size_t boundaryIdx   = i / nParticlesPerBoundary;
                unsigned boundaryDim = boundaryIdx / 2;
                bool isLower         = (boundaryIdx % 2 == 0);

                ippl::Vector<T, Dim> pos;
                for (unsigned d = 0; d < Dim; ++d) {
                    if (d == boundaryDim) {
                        T boundaryWidth = 0.1 * extent_local[d];
                        if (isLower) {
                            pos[d] = origin_local[d] + gen.drand() * boundaryWidth;
                        } else {
                            pos[d] =
                                origin_local[d] + extent_local[d] - gen.drand() * boundaryWidth;
                        }
                    } else {
                        pos[d] = origin_local[d] + gen.drand() * extent_local[d];
                    }
                }

                R_view(i)      = pos;
                weight_view(i) = 1.0;

                randPool.free_state(gen);
            });
        bunch->update();
        Kokkos::fence();
    }

    size_t countTotalParticles() {
        size_t localCount  = bunch->getLocalNum();
        size_t globalCount = 0;
        ippl::Comm->allreduce(localCount, globalCount, 1, std::plus<size_t>{});
        return globalCount;
    }

    // Check for NaN/Inf in field
    int countInvalidFieldValues(const field_type& field) {
        auto field_view        = field.getView();
        using index_array_type = ippl::RangePolicy<Dim>::index_array_type;
        int invalidCount       = 0;
        ippl::parallel_reduce(
            "init_random_complex", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args, int& count) {
                T val = apply(field, args);
                if (Kokkos::isnan(val) || Kokkos::isinf(val)) {
                    count++;
                }
            },
            invalidCount);
        ippl::fence();

        int globalInvalid = 0;
        ippl::Comm->allreduce(invalidCount, globalInvalid, 1, std::plus<int>{});
        return globalInvalid;
    }

    //=========================================================================
    // Test Methods
    //=========================================================================

    // Test 1: Conservation - partition of unity kernels should conserve total charge
    void runConservationTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field = T(0.0);

        size_t nParticles = 1000;
        createUniformParticles(nParticles);

        auto weight_view = bunch->weight.getView();
        Kokkos::parallel_for(
            "set_weights", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) { weight_view(i) = 1.0; });
        ippl::fence();
        bunch->update();
        ippl::fence();

        T totalWeight = bunch->weight.sum();
        ippl::fence();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);
        ippl::fence();

        T fieldSum = ippl::norm(field, 1);
        T relError = std::abs(fieldSum - totalWeight) / std::abs(totalWeight);

        if (myRank == 0) {
            EXPECT_LT(relError, 1e-10)
                << "Conservation failed for " << kernel_traits::name << " kernel: "
                << "field sum = " << fieldSum << ", weight sum = " << totalWeight
                << ", relative error = " << relError;
        }
    }

    // Test 2: Conservation with varying weights
    void runVaryingWeightConservationTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field = T(0.0);

        size_t nParticles = 1000;
        createUniformParticles(nParticles);

        auto weight_view = bunch->weight.getView();
        using RandPool   = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + myRank);

        Kokkos::parallel_for(
            "set_random_weights", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                weight_view(i)                        = gen.drand(0.1, 10.0);
                randPool.free_state(gen);
            });
        ippl::fence();

        bunch->update();

        T totalWeight = bunch->weight.sum();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);

        T fieldSum = ippl::norm(field, 1);
        T relError = std::abs(fieldSum - totalWeight) / std::abs(totalWeight);

        if (myRank == 0) {
            EXPECT_LT(relError, 1e-10)
                << "Varying weight conservation failed for " << kernel_traits::name;
        }
    }

    // Test 3: Periodic boundary handling
    void runPeriodicBoundaryTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field = T(0.0);

        size_t nParticlesPerBoundary = 100;
        createBoundaryParticles(nParticlesPerBoundary);

        auto weight_view = bunch->weight.getView();
        Kokkos::parallel_for(
            "set_weights", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) { weight_view(i) = 1.0; });
        ippl::fence();

        bunch->update();

        T totalWeight = bunch->weight.sum();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);

        T fieldSum = ippl::norm(field, 1);
        T relError = std::abs(fieldSum - totalWeight) / std::abs(totalWeight);

        if (myRank == 0) {
            EXPECT_LT(relError, 1e-10)
                << "Periodic boundary conservation failed for " << kernel_traits::name
                << ": field sum = " << fieldSum << ", weight sum = " << totalWeight;
        }

        int invalidCount = countInvalidFieldValues(field);
        if (myRank == 0) {
            EXPECT_EQ(invalidCount, 0) << "Found " << invalidCount << " NaN/Inf values in field";
        }
    }

    // Test 5: Gather from constant field should return constant
    void runGatherConstantFieldTest(const gather_config_type& config) {
        field_type field(*mesh, *layout, nghost);

        const T constantValue = 42.0;
        field                 = constantValue;

        size_t nParticles = 500;
        createUniformParticles(nParticles);
        bunch->update();

        auto gathered_view = bunch->gathered_scalar.getView();
        Kokkos::parallel_for(
            "zero_gathered", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) { gathered_view(i) = 0.0; });
        ippl::fence();

        auto gather = ippl::Gather<decltype(kernel), Dim>(kernel, config);
        gather(field, bunch->R, bunch->gathered_scalar);

        T localMaxError = 0.0;
        Kokkos::parallel_reduce(
            "check_constant", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i, T& maxErr) {
                T error = Kokkos::abs(gathered_view(i) - constantValue);
                if (error > maxErr)
                    maxErr = error;
            },
            Kokkos::Max<T>(localMaxError));
        Kokkos::fence();

        T globalMaxError = 0.0;
        ippl::Comm->allreduce(localMaxError, globalMaxError, 1, std::greater<T>{});

        if (myRank == 0) {
            EXPECT_LT(globalMaxError, 1e-10)
                << "Gather from constant field failed for " << kernel_traits::name
                << ": max error = " << globalMaxError;
        }
    }

    // Test 7: Adjointness with complex values
    void runAdjointnessTest(const scatter_config_type& scatterCfg,
                            const gather_config_type& gatherCfg, T testTolerance = 1e-10) {
        complex_field_type fieldScatter(*mesh, *layout, nghost);
        complex_field_type fieldGather(*mesh, *layout, nghost);

        size_t nParticles = 1000;
        createUniformParticles(nParticles);
        bunch->update();

        ippl::test::initializeRandomFieldComplex(fieldGather);

        auto QScatter_view = bunch->Q_scatter.getView();
        auto QGather_view  = bunch->Q_gather.getView();

        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + myRank);

        Kokkos::parallel_for(
            "init_charges", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                T re                                  = gen.drand(-1.0, 1.0);
                T im                                  = gen.drand(-1.0, 1.0);
                QScatter_view(i)                      = complex_type(re, im);
                QGather_view(i)                       = complex_type(0.0, 0.0);
                randPool.free_state(gen);
            });
        Kokkos::fence();

        // Scatter: S * q -> fieldScatter
        fieldScatter = complex_type(0.0, 0.0);

        auto scatter = ippl::Scatter(kernel, scatterCfg);
        scatter(fieldScatter, bunch->R, bunch->Q_scatter);

        // Gather: G * fieldGather -> Q_gather
        bunch->Q_gather = complex_type(0.0, 0.0);
        auto gather     = ippl::Gather<decltype(kernel), Dim>(kernel, gatherCfg);
        gather(fieldGather, bunch->R, bunch->Q_gather);

        // Left IP: <S*q, g>
        complex_type leftIP = ippl::innerProduct(fieldScatter, fieldGather);

        // Right IP: <q, G*g>
        complex_type rightIP = ippl::test::computeParticleInnerProductComplex(
            *bunch, bunch->Q_scatter, bunch->Q_gather);

        if (myRank == 0) {
            T diff     = Kokkos::abs(leftIP - rightIP);
            T relError = diff / Kokkos::abs(leftIP);

            bool isAdjoint = relError < testTolerance;

            if (!isAdjoint) {
                std::cout << "Adjointness test failed for " << kernel_traits::name << ":\n"
                          << "  Left IP:   " << leftIP << "\n"
                          << "  Right IP:  " << rightIP << "\n"
                          << "  Abs diff:  " << diff << "\n"
                          << "  Rel error: " << relError << std::endl;
            }

            EXPECT_TRUE(isAdjoint)
                << "Adjointness relative error for " << kernel_traits::name << ": " << relError;
        }
    }

    // Test 9: High particle density
    void runHighDensityTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field = T(0.0);

        size_t nParticles = 1000;
        bunch->create(nParticles);

        T centerX = origin[0] + 0.5 * extent[0];
        T centerY = origin[1] + 0.5 * extent[1];
        T spread  = 0.01 * extent[0];

        auto R_view      = bunch->R.getView();
        auto weight_view = bunch->weight.getView();

        using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + myRank);
        auto origin_local = origin;
        auto extent_local = extent;

        Kokkos::parallel_for(
            "create_dense_particles", Kokkos::RangePolicy<ExecSpace>(0, nParticles),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                (void)extent_local; // Fix nvcc problems
                (void)origin_local;

                ippl::Vector<T, Dim> pos;
                pos[0] = centerX + (gen.drand() - 0.5) * spread;
                pos[1] = centerY + (gen.drand() - 0.5) * spread;
                if constexpr (Dim == 3) {
                    T centerZ = origin_local[2] + 0.5 * extent_local[2];
                    pos[2]    = centerZ + (gen.drand() - 0.5) * spread;
                }

                R_view(i)      = pos;
                weight_view(i) = 1.0;

                randPool.free_state(gen);
            });
        Kokkos::fence();

        bunch->update();

        T totalWeight = bunch->weight.sum();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);

        T fieldSum = ippl::norm(field, 1);

        if (myRank == 0) {
            T relError = std::abs(fieldSum - totalWeight) / std::abs(totalWeight);
            EXPECT_LT(relError, 1e-10)
                << "High density test failed for " << kernel_traits::name
                << ": field sum = " << fieldSum << ", weight sum = " << totalWeight;
        }
    }

    // Test 10: Symmetry test
    void runSymmetryTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field             = T(0.0);
        size_t nParticles = myRank == 0 ? (Dim == 3) ? 8 : 4 : 0;

        bunch->create(nParticles);
        if (myRank == 0) {
            auto R_view      = bunch->R.getView();
            auto weight_view = bunch->weight.getView();

            T cx = origin[0] + (static_cast<T>(gridSize[0] / 2) + T(0.5)) * hx[0];
            T cy = origin[1] + (static_cast<T>(gridSize[1] / 2) + T(0.5)) * hx[1];
            T cz =
                (Dim == 3) ? origin[2] + (static_cast<T>(gridSize[2] / 2) + T(0.5)) * hx[2] : T(0);
            T offset = 0.1 * extent[0];

            auto R_host      = Kokkos::create_mirror_view(R_view);
            auto weight_host = Kokkos::create_mirror_view(weight_view);

            size_t idx = 0;
            for (int dx = -1; dx <= 1; dx += 2) {
                for (int dy = -1; dy <= 1; dy += 2) {
                    if constexpr (Dim == 3) {
                        for (int dz = -1; dz <= 1; dz += 2) {
                            R_host(idx) = {cx + dx * offset, cy + dy * offset, cz + dz * offset};
                            weight_host(idx) = 1.0;
                            idx++;
                        }
                    } else {
                        R_host(idx)      = {cx + dx * offset, cy + dy * offset};
                        weight_host(idx) = 1.0;
                        idx++;
                    }
                }
            }

            Kokkos::deep_copy(R_view, R_host);
            Kokkos::deep_copy(weight_view, weight_host);
        }

        bunch->update();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);

        auto field_view  = field.getView();
        const auto& lDom = layout->getLocalNDIndex();
        int ng           = nghost;

        T maxAsymmetry = 0.0;

        int centerI = gridSize[0] / 2;
        int centerJ = gridSize[1] / 2;

        if (centerI >= lDom[0].first() && centerI <= lDom[0].last() && centerJ >= lDom[1].first()
            && centerJ <= lDom[1].last()) {
            auto field_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field_view);

            for (int di = -3; di <= 3; ++di) {
                for (int dj = -3; dj <= 3; ++dj) {
                    int i1 = centerI + di - lDom[0].first() + ng;
                    int j1 = centerJ + dj - lDom[1].first() + ng;
                    int i2 = centerI - di - lDom[0].first() + ng;
                    int j2 = centerJ - dj - lDom[1].first() + ng;

                    if (i1 >= 0 && i1 < (int)(lDom[0].length() + 2 * ng) && j1 >= 0
                        && j1 < (int)(lDom[1].length() + 2 * ng) && i2 >= 0
                        && i2 < (int)(lDom[0].length() + 2 * ng) && j2 >= 0
                        && j2 < (int)(lDom[1].length() + 2 * ng)) {
                        T val1, val2;
                        if constexpr (Dim == 3) {
                            int centerK = gridSize[2] / 2;
                            int k       = centerK - lDom[2].first() + ng;
                            val1        = field_host(i1, j1, k);
                            val2        = field_host(i2, j2, k);
                        } else {
                            val1 = field_host(i1, j1);
                            val2 = field_host(i2, j2);
                        }

                        T asymmetry = std::abs(val1 - val2);
                        if (asymmetry > maxAsymmetry) {
                            maxAsymmetry = asymmetry;
                        }
                    }
                }
            }
        }

        T globalMaxAsymmetry = 0.0;
        MPI_Allreduce(&maxAsymmetry, &globalMaxAsymmetry, 1,
                      std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, MPI_MAX,
                      ippl::Comm->getCommunicator());

        if (myRank == 0) {
            EXPECT_LT(globalMaxAsymmetry, 1e-12)
                << "Symmetry test failed for " << kernel_traits::name
                << ": max asymmetry = " << globalMaxAsymmetry;
        }
    }

    // Test 11: Scatter-Gather roundtrip
    void runRoundtripTest(const scatter_config_type& scatterCfg,
                          const gather_config_type& gatherCfg) {
        field_type field(*mesh, *layout, nghost);

        size_t nParticles = 500;
        createUniformParticles(nParticles);
        bunch->update();

        auto weight_view = bunch->weight.getView();
        using RandPool   = Kokkos::Random_XorShift64_Pool<ExecSpace>;
        RandPool randPool(42 + myRank);

        Kokkos::parallel_for(
            "set_random_weights", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) {
                typename RandPool::generator_type gen = randPool.get_state();
                weight_view(i)                        = gen.drand(0.1, 2.0);
                randPool.free_state(gen);
            });
        Kokkos::fence();

        T originalSum = bunch->weight.sum();

        field        = T(0.0);
        auto scatter = ippl::Scatter(kernel, scatterCfg);
        scatter(field, bunch->R, bunch->weight);

        T scatteredSum = ippl::norm(field, 1);

        auto gathered_view = bunch->gathered_scalar.getView();
        Kokkos::parallel_for(
            "zero_gathered", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i) { gathered_view(i) = 0.0; });
        Kokkos::fence();

        auto gather = ippl::Gather(kernel, gatherCfg);
        gather(field, bunch->R, bunch->gathered_scalar);

        T gatheredSum = 0.0;
        Kokkos::parallel_reduce(
            "sum_gathered", Kokkos::RangePolicy<ExecSpace>(0, bunch->getLocalNum()),
            KOKKOS_LAMBDA(size_t i, T& sum) { sum += gathered_view(i); }, gatheredSum);
        Kokkos::fence();

        T globalGatheredSum = 0.0;
        ippl::Comm->allreduce(gatheredSum, globalGatheredSum, 1, std::plus<T>{});

        if (myRank == 0) {
            T scatterError = std::abs(scatteredSum - originalSum) / std::abs(originalSum);
            EXPECT_LT(scatterError, 1e-10)
                << "Scatter didn't conserve for " << kernel_traits::name
                << ": original = " << originalSum << ", scattered = " << scatteredSum;

            EXPECT_GT(globalGatheredSum, 0.0) << "Gathered sum should be positive";
        }
    }

    void runGatherConvergenceTest(const gather_config_type& config) {
        const int order = kernel_traits::order;

        constexpr int N_COARSE = 16;
        constexpr int N_FINE   = 32;

        auto measureGatherError = [&](int nGrid) -> T {
            // Mirror SetUp() exactly, but with nGrid cells per dim
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);

            ippl::Vector<T, Dim> hx_local;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(nGrid);
                hx_local[d] = extent[d] / nGrid;
            }

            auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
            auto layout_local =
                std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel, true, nghost);
            auto mesh_local    = std::make_shared<mesh_type>(owned, hx_local, origin);
            auto playout_local = std::make_shared<playout_type>(*layout_local, *mesh_local);
            auto bunch_local   = std::make_shared<bunch_type>(*playout_local);

            field_type field(*mesh_local, *layout_local, nghost);

            // Fill field with sin at cell centres
            {
                auto view        = field.getView();
                const auto& lDom = layout_local->getLocalNDIndex();
                const int ng     = nghost;
                auto origin_     = origin;
                auto hx_         = hx_local;
                auto extent_     = extent;

                using index_array_type = ippl::RangePolicy<Dim>::index_array_type;
                ippl::parallel_for(
                    "fill_sin_gather", field.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        auto gidx = lDom.first() + args - ng;
                        T val     = T(1);
                        for (unsigned d = 0; d < Dim; ++d) {
                            const T xc = origin_[d] + (static_cast<T>(gidx[d]) + T(0.5)) * hx_[d];
                            val *= Kokkos::sin(T(2) * Kokkos::numbers::pi_v<T> * xc / extent_[d]);
                        }
                        apply(view, args) = val;
                    });
                Kokkos::fence();
            }

            // Create 500 uniform random particles
            constexpr size_t nParticles = 500;
            bunch_local->create(nParticles);
            {
                auto R_view      = bunch_local->R.getView();
                auto weight_view = bunch_local->weight.getView();

                using RandPool = Kokkos::Random_XorShift64_Pool<ExecSpace>;
                RandPool randPool(42 + myRank * 1000);
                auto origin_ = origin;
                auto extent_ = extent;

                Kokkos::parallel_for(
                    "create_particles_gather", Kokkos::RangePolicy<ExecSpace>(0, nParticles),
                    KOKKOS_LAMBDA(size_t i) {
                        typename RandPool::generator_type gen = randPool.get_state();
                        ippl::Vector<T, Dim> pos;
                        for (unsigned d = 0; d < Dim; ++d)
                            pos[d] = origin_[d] + gen.drand() * extent_[d];
                        R_view(i)      = pos;
                        weight_view(i) = T(1);
                        randPool.free_state(gen);
                    });
                Kokkos::fence();
            }
            bunch_local->update();

            auto gathered_view = bunch_local->gathered_scalar.getView();
            Kokkos::parallel_for(
                "zero_gathered_conv", Kokkos::RangePolicy<ExecSpace>(0, bunch_local->getLocalNum()),
                KOKKOS_LAMBDA(size_t i) { gathered_view(i) = T(0); });
            Kokkos::fence();

            auto gather_op = ippl::Gather<decltype(kernel), Dim>(kernel, config);
            gather_op(field, bunch_local->R, bunch_local->gathered_scalar);

            auto R_view   = bunch_local->R.getView();
            auto extent_  = extent;
            size_t localN = bunch_local->getLocalNum();
            T localSum    = T(0);

            Kokkos::parallel_reduce(
                "gather_conv_error", Kokkos::RangePolicy<ExecSpace>(0, localN),
                KOKKOS_LAMBDA(size_t i, T& sum) {
                    T expected = T(1);
                    for (unsigned d = 0; d < Dim; ++d)
                        expected *= Kokkos::sin(T(2) * Kokkos::numbers::pi_v<T>
                                                * R_view(i)[d] / extent_[d]);
                    sum += Kokkos::abs(gathered_view(i) - expected);
                },
                Kokkos::Sum<T>(localSum));
            Kokkos::fence();

            T globalSum    = T(0);
            size_t globalN = 0;
            ippl::Comm->allreduce(localSum, globalSum, 1, std::plus<T>{});
            ippl::Comm->allreduce(localN, globalN, 1, std::plus<size_t>{});
            return globalSum / static_cast<T>(globalN);
        };

        const T err_coarse = measureGatherError(N_COARSE);
        const T err_fine   = measureGatherError(N_FINE);
        const T ratio      = err_coarse / err_fine;
        const T expected   = (order == 0) ? T(2) : T(4);

        if (myRank == 0) {
            std::cout << "  [" << kernel_traits::name << " " << Dim << "D] gather convergence:"
                      << "  err(N=" << N_COARSE << ")=" << err_coarse << "  err(N=" << N_FINE
                      << ")=" << err_fine << "  ratio=" << ratio << "  expected~" << expected
                      << "\n";

            EXPECT_GT(ratio, expected * T(0.5))
                << kernel_traits::name << " " << Dim << "D gather: "
                << "convergence too slow (ratio=" << ratio << ", expected~" << expected << ")";
            EXPECT_LT(ratio, expected * T(4.0))
                << kernel_traits::name << " " << Dim << "D gather: "
                << "convergence ratio suspiciously high (ratio=" << ratio << ", expected~"
                << expected << ")";
        }
    }

    void runScatterConvergenceTest(const scatter_config_type& config) {
        const int order = kernel_traits::order;

        constexpr int N_COARSE = 16;
        constexpr int N_FINE   = 32;

        auto measureScatterError = [&](int nGrid) -> T {
            // Mirror SetUp() exactly, but with nGrid cells per dim
            std::array<ippl::Index, Dim> domains;
            std::array<bool, Dim> isParallel;
            isParallel.fill(true);

            ippl::Vector<T, Dim> hx_local;
            for (unsigned d = 0; d < Dim; ++d) {
                domains[d]  = ippl::Index(nGrid);
                hx_local[d] = extent[d] / nGrid;
            }

            auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(domains);
            auto layout_local =
                std::make_shared<layout_type>(MPI_COMM_WORLD, owned, isParallel, true, nghost);
            auto mesh_local    = std::make_shared<mesh_type>(owned, hx_local, origin);
            auto playout_local = std::make_shared<playout_type>(*layout_local, *mesh_local);
            auto bunch_local   = std::make_shared<bunch_type>(*playout_local);

            field_type field(*mesh_local, *layout_local, nghost);
            field = T(0);

            const auto& lDom = layout_local->getLocalNDIndex();

            // Each rank owns only the cells in its subdomain
            size_t localParticles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                localParticles *= static_cast<size_t>(lDom[d].length());

            bunch_local->create(localParticles);
            {
                auto R_view      = bunch_local->R.getView();
                auto weight_view = bunch_local->weight.getView();
                auto origin_     = origin;
                auto hx_         = hx_local;
                auto extent_     = extent;

                // Encode only the local cells: flatten lDom into a linear index
                Kokkos::Array<int, Dim> local_size;
                Kokkos::Array<int, Dim> local_start;
                for (unsigned d = 0; d < Dim; ++d) {
                    local_size[d]  = static_cast<int>(lDom[d].length());
                    local_start[d] = static_cast<int>(lDom[d].first());
                }

                Kokkos::parallel_for(
                    "create_regular_scatter", Kokkos::RangePolicy<ExecSpace>(0, localParticles),
                    KOKKOS_LAMBDA(size_t idx) {
                        size_t tmp = idx;
                        ippl::Vector<T, Dim> pos;
                        for (unsigned d = 0; d < Dim; ++d) {
                            const int local_cell =
                                static_cast<int>(tmp % static_cast<size_t>(local_size[d]));
                            tmp /= static_cast<size_t>(local_size[d]);
                            const int global_cell = local_start[d] + local_cell;
                            pos[d] = origin_[d] + (static_cast<T>(global_cell) + T(0.75)) * hx_[d];
                        }
                        R_view(idx) = pos;

                        T val = T(1);
                        for (unsigned d = 0; d < Dim; ++d)
                            val *=
                                Kokkos::sin(T(2) * Kokkos::numbers::pi_v<T> * pos[d] / extent_[d]);
                        weight_view(idx) = val;
                    });
                Kokkos::fence();
            }

            auto scatter_op = ippl::Scatter<decltype(kernel), Dim>(kernel, config);
            scatter_op(field, bunch_local->R, bunch_local->weight);

            // Avg |field_j - f(x_j)| over interior cells
            auto view        = field.getView();
            const int ng     = nghost;
            auto origin_     = origin;
            auto hx_         = hx_local;
            auto extent_     = extent;

            T localSum      = T(0);
            size_t localCnt = 0;

            using index_array_type = ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_reduce(
                "scatter_conv_error", field.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args, T& sum, size_t& cnt) {
                    for (unsigned d = 0; d < Dim; ++d) {
                        if (static_cast<int>(args[d]) < ng
                            || static_cast<int>(args[d]) >= ng + nGrid)
                            return;
                    }
                    auto gidx  = lDom.first() + args - ng;
                    T expected = T(1);
                    for (unsigned d = 0; d < Dim; ++d) {
                        const T xc = origin_[d] + (static_cast<T>(gidx[d]) + T(0.5)) * hx_[d];
                        expected *= Kokkos::sin(T(2) * Kokkos::numbers::pi_v<T> * xc / extent_[d]);
                    }
                    sum += Kokkos::abs(apply(view, args) - expected);
                    cnt += 1;
                },
                Kokkos::Sum<T>(localSum), Kokkos::Sum<size_t>(localCnt));
            Kokkos::fence();

            T globalSum      = T(0);
            size_t globalCnt = 0;
            ippl::Comm->allreduce(localSum, globalSum, 1, std::plus<T>{});
            ippl::Comm->allreduce(localCnt, globalCnt, 1, std::plus<size_t>{});
            return globalSum / static_cast<T>(globalCnt);
        };

        const T err_coarse = measureScatterError(N_COARSE);
        const T err_fine   = measureScatterError(N_FINE);
        const T ratio      = err_coarse / err_fine;
        const T expected   = (order == 0) ? T(2) : T(4);

        if (myRank == 0) {
            std::cout << "  [" << kernel_traits::name << " " << Dim << "D] scatter convergence:"
                      << "  err(N=" << N_COARSE << ")=" << err_coarse << "  err(N=" << N_FINE
                      << ")=" << err_fine << "  ratio=" << ratio << "  expected~" << expected
                      << "\n";

            EXPECT_GT(ratio, expected * T(0.5))
                << kernel_traits::name << " " << Dim << "D scatter: "
                << "convergence too slow (ratio=" << ratio << ", expected~" << expected << ")";
            EXPECT_LT(ratio, expected * T(4.0))
                << kernel_traits::name << " " << Dim << "D scatter: "
                << "convergence ratio suspiciously high (ratio=" << ratio << ", expected~"
                << expected << ")";
        }
    }
    // Test 12: Scatter with and without sorting should give identical results
    void runScatterSortComparisonTest() {
        field_type fieldNoSort(*mesh, *layout, nghost);
        field_type fieldWithSort(*mesh, *layout, nghost);

        size_t nParticles = 1000;
        createUniformParticles(nParticles);
        bunch->update();

        // Test with sorting disabled
        {
            scatter_config_type config;
            config.method = ippl::Interpolation::ScatterMethod::Atomic;
            config.sort   = false;

            fieldNoSort  = T(0.0);
            auto scatter = ippl::Scatter(kernel, config);
            scatter(fieldNoSort, bunch->R, bunch->weight);
        }

        // Test with sorting enabled
        {
            scatter_config_type config;
            config.method = ippl::Interpolation::ScatterMethod::Atomic;
            config.sort   = true;

            fieldWithSort = T(0.0);
            auto scatter  = ippl::Scatter(kernel, config);
            scatter(fieldWithSort, bunch->R, bunch->weight);
        }

        // Compare the two fields
        T maxDiff         = 0.0;
        auto viewNoSort   = fieldNoSort.getView();
        auto viewWithSort = fieldWithSort.getView();

        using index_array_type = ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_reduce(
            "compare_fields", fieldNoSort.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args, T& maxVal) {
                T diff = Kokkos::abs(apply(viewNoSort, args) - apply(viewWithSort, args));
                if (diff > maxVal)
                    maxVal = diff;
            },
            Kokkos::Max<T>(maxDiff));
        ippl::fence();

        T globalMaxDiff = 0.0;
        ippl::Comm->allreduce(maxDiff, globalMaxDiff, 1, std::greater<T>{});

        T sumNoSort   = ippl::norm(fieldNoSort, 1);
        T sumWithSort = ippl::norm(fieldWithSort, 1);

        if (myRank == 0) {
            EXPECT_LT(globalMaxDiff, 1e-12)
                << "Scatter with/without sorting produced different results for "
                << kernel_traits::name << ": max diff = " << globalMaxDiff;

            T relDiff = std::abs(sumNoSort - sumWithSort) / std::abs(sumNoSort);
            EXPECT_LT(relDiff, 1e-12)
                << "Scatter sum differs with/without sorting for " << kernel_traits::name
                << ": no sort = " << sumNoSort << ", with sort = " << sumWithSort;
        }
    }

    // Test 13: Single particle at grid point
    void runSingleParticleAtGridPointTest(const scatter_config_type& config) {
        field_type field(*mesh, *layout, nghost);
        field = T(0.0);

        // Only create particle on rank 0
        bunch->create(myRank == 0);
        if (myRank == 0) {
            auto R_view      = bunch->R.getView();
            auto weight_view = bunch->weight.getView();

            // Place particle exactly at a grid point (e.g., center)
            auto R_host      = Kokkos::create_mirror_view(R_view);
            auto weight_host = Kokkos::create_mirror_view(weight_view);

            // Grid point at (16, 16, 16) in grid coordinates
            int gi = gridSize[0] / 2;
            int gj = gridSize[1] / 2;

            if constexpr (Dim == 3) {
                int gk    = gridSize[2] / 2;
                R_host(0) = {origin[0] + (gi + 0.5) * hx[0], origin[1] + (gj + 0.5) * hx[1],
                             origin[2] + (gk + 0.5) * hx[2]};
            } else {
                R_host(0) = {origin[0] + (gi + 0.5) * hx[0], origin[1] + (gj + 0.5) * hx[1]};
            }
            weight_host(0) = 1.0;

            Kokkos::deep_copy(R_view, R_host);
            Kokkos::deep_copy(weight_view, weight_host);
        }

        bunch->update();

        T totalWeight = bunch->weight.sum();

        auto scatter = ippl::Scatter(kernel, config);
        scatter(field, bunch->R, bunch->weight);

        T fieldSum = ippl::norm(field, 1);

        if (myRank == 0) {
            T relError = std::abs(fieldSum - totalWeight) / std::abs(totalWeight);
            EXPECT_LT(relError, 1e-10)
                << "Single particle at grid point failed for " << kernel_traits::name
                << ": field sum = " << fieldSum << ", weight = " << totalWeight;
        }
    }
};

//=============================================================================
// Test Suite Registration
//=============================================================================

TYPED_TEST_SUITE(ScatterGatherTest, TestTypes);

//=============================================================================
// Conservation Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, Conservation_Atomic) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runConservationTest(config);
}

TYPED_TEST(ScatterGatherTest, Conservation_Tiled) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Tiled;
    config.sort   = true;
    this->runConservationTest(config);
}

TYPED_TEST(ScatterGatherTest, Conservation_OutputFocused) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::OutputFocused;
    config.sort   = true;
    this->runConservationTest(config);
}

TYPED_TEST(ScatterGatherTest, Conservation_OutputFocused_ZBatches) {
    typename TestFixture::scatter_config_type config;
    config.method    = ippl::Interpolation::ScatterMethod::OutputFocused;
    config.sort      = true;
    config.z_batches = 2;  // Test z-stencil batching to reduce shared memory pressure
    this->runConservationTest(config);
}

TYPED_TEST(ScatterGatherTest, Conservation_VaryingWeights) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runVaryingWeightConservationTest(config);
}

//=============================================================================
// Periodic Boundary Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, PeriodicBoundary_Atomic) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runPeriodicBoundaryTest(config);
}

TYPED_TEST(ScatterGatherTest, PeriodicBoundary_Tiled) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Tiled;
    config.sort   = true;
    this->runPeriodicBoundaryTest(config);
}

//=============================================================================
// Gather Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, Gather_ConstantField_Atomic) {
    typename TestFixture::gather_config_type config;
    config.method = ippl::Interpolation::GatherMethod::Atomic;
    this->runGatherConstantFieldTest(config);
}

TYPED_TEST(ScatterGatherTest, Gather_ConstantField_AtomicSort) {
    typename TestFixture::gather_config_type config;
    config.method = ippl::Interpolation::GatherMethod::AtomicSort;
    this->runGatherConstantFieldTest(config);
}


TYPED_TEST(ScatterGatherTest, Gather_Convergence_Atomic) {
    typename TestFixture::gather_config_type config;
    config.method = ippl::Interpolation::GatherMethod::Atomic;
    this->runGatherConvergenceTest(config);
}

TYPED_TEST(ScatterGatherTest, Scatter_Convergence_Atomic) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    this->runScatterConvergenceTest(config);
}

//=============================================================================
// Adjointness Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, Adjointness_Atomic_Atomic) {
    typename TestFixture::scatter_config_type scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    typename TestFixture::gather_config_type gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Atomic;

    this->runAdjointnessTest(scatterCfg, gatherCfg);
}

TYPED_TEST(ScatterGatherTest, Adjointness_AtomicSort_AtomicSort) {
    typename TestFixture::scatter_config_type scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    typename TestFixture::gather_config_type gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::AtomicSort;

    this->runAdjointnessTest(scatterCfg, gatherCfg);
}

//=============================================================================
// Edge Case Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, HighDensity_Atomic) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runHighDensityTest(config);
}

TYPED_TEST(ScatterGatherTest, HighDensity_Tiled) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Tiled;
    config.sort   = true;
    this->runHighDensityTest(config);
}

TYPED_TEST(ScatterGatherTest, SingleParticleAtGridPoint) {
    typename TestFixture::scatter_config_type config;
    config.method = ippl::Interpolation::ScatterMethod::Atomic;
    config.sort   = true;
    this->runSingleParticleAtGridPointTest(config);
}

//=============================================================================
// Sorting Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, ScatterSortComparison) {
    this->runScatterSortComparisonTest();
}

//=============================================================================
// Roundtrip Tests
//=============================================================================

TYPED_TEST(ScatterGatherTest, Roundtrip_Atomic) {
    typename TestFixture::scatter_config_type scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    typename TestFixture::gather_config_type gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::Atomic;

    this->runRoundtripTest(scatterCfg, gatherCfg);
}

TYPED_TEST(ScatterGatherTest, Roundtrip_AtomicSort) {
    typename TestFixture::scatter_config_type scatterCfg;
    scatterCfg.method = ippl::Interpolation::ScatterMethod::Atomic;
    scatterCfg.sort   = true;

    typename TestFixture::gather_config_type gatherCfg;
    gatherCfg.method = ippl::Interpolation::GatherMethod::AtomicSort;

    this->runRoundtripTest(scatterCfg, gatherCfg);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
