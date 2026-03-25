#include "Ippl.h"

#include <cmath>
#include <memory>
#include <random>

#include "Manager/datatypes.h"
#include "PoissonSolvers/EvalFunctor.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "ParticleContainer.hpp"
#include "TestUtils.h"
#include "gtest/gtest.h"

const char* TestName = "LoadBalancerRegression";

namespace {
    template <unsigned Dim>
    ippl::NDIndex<Dim> makeDomain(const std::array<int, Dim>& nPoints) {
        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; ++d) {
            args[d] = ippl::Index(nPoints[d]);
        }
        return std::make_from_tuple<ippl::NDIndex<Dim>>(args);
    }

    template <typename T, unsigned Dim>
    T maxParticleDeviationFraction(const std::shared_ptr<ParticleContainer<T, Dim>>& particles) {
        const size_type localParticles = particles->getLocalNum();
        size_type globalParticles      = 0;
        ippl::Comm->allreduce(localParticles, globalParticles, 1, std::plus<size_type>());

        const T idealParticles = static_cast<T>(globalParticles) / ippl::Comm->size();
        const T localDeviation =
            std::abs(static_cast<T>(localParticles) - idealParticles) / globalParticles;

        T globalDeviation = 0;
        ippl::Comm->allreduce(localDeviation, globalDeviation, 1, std::greater<T>());
        return globalDeviation;
    }

    template <typename T, unsigned Dim>
    size_type globalParticleCount(const std::shared_ptr<ParticleContainer<T, Dim>>& particles) {
        const size_type localParticles = particles->getLocalNum();
        size_type globalParticles      = 0;
        ippl::Comm->allreduce(localParticles, globalParticles, 1, std::plus<size_type>());
        return globalParticles;
    }

    template <typename Field>
    int maxExtentMismatch(Field& field) {
        constexpr unsigned Dim = Field::dim;
        const auto& localDomain = field.getLayout().getLocalNDIndex();
        const auto view         = field.getView();
        const int nghost        = field.getNghost();

        int localMismatch = 0;
        for (unsigned d = 0; d < Dim; ++d) {
            const int expectedExtent = localDomain[d].length() + 2 * nghost;
            const int actualExtent   = static_cast<int>(view.extent(d));
            localMismatch            = std::max(localMismatch, std::abs(actualExtent - expectedExtent));
        }

        int globalMismatch = 0;
        ippl::Comm->allreduce(localMismatch, globalMismatch, 1, std::greater<int>());
        return globalMismatch;
    }
}  // namespace

TEST(LoadBalancer, RebalancesArtificialPileUpAndReinitializesFEMOperator) {
    constexpr unsigned Dim      = 3;
    using value_type            = double;
    constexpr size_type nGlobal = 1000000;

    const std::array<int, Dim> nPoints = {17, 17, 17};
    auto domain                         = makeDomain<Dim>(nPoints);

    Vector_t<value_type, Dim> hr;
    Vector_t<value_type, Dim> rmin(0.0);
    Vector_t<value_type, Dim> rmax(1.0);
    Vector_t<value_type, Dim> origin(0.0);
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);

    for (unsigned d = 0; d < Dim; ++d) {
        hr[d] = (rmax[d] - rmin[d]) / nPoints[d];
    }

    auto fieldContainer =
        std::make_shared<FieldContainer<value_type, Dim>>(hr, rmin, rmax, isParallel, domain,
                                                          origin, true);
    fieldContainer->initializeFields("FEM");

    auto particleContainer =
        std::make_shared<ParticleContainer<value_type, Dim>>(fieldContainer->getMesh(),
                                                             fieldContainer->getFL(), true);
    auto fieldSolver = std::make_shared<FieldSolver<value_type, Dim>>(
        "FEM", &fieldContainer->getRho(), &fieldContainer->getE(), &fieldContainer->getPhi());
    fieldSolver->initSolver();
    std::shared_ptr<ippl::FieldSolverBase<value_type, Dim>> fieldSolverBase = fieldSolver;

    auto loadBalancer = std::make_shared<LoadBalancer<value_type, Dim>>(
        0.01, fieldContainer, particleContainer, fieldSolverBase);

    const size_type baseLocal = nGlobal / ippl::Comm->size();
    const size_type remainder = nGlobal % ippl::Comm->size();
    const size_type localNum =
        baseLocal + (static_cast<size_type>(ippl::Comm->rank()) < remainder ? 1 : 0);

    particleContainer->create(localNum);
    particleContainer->q = 1.0 / nGlobal;
    particleContainer->P = Vector_t<value_type, Dim>(0.0);
    particleContainer->E = Vector_t<value_type, Dim>(0.0);

    std::mt19937_64 eng(42);
    eng.discard(localNum * ippl::Comm->rank());
    std::uniform_real_distribution<value_type> xUniform(0.0, 1.0);
    std::uniform_real_distribution<value_type> yzUniform(0.0, 1.0);

    auto positions = particleContainer->R.getHostMirror();
    for (size_type i = 0; i < localNum; ++i) {
        const value_type sample = xUniform(eng);
        positions(i)[0]         = sample * sample;
        positions(i)[1] = yzUniform(eng);
        positions(i)[2] = yzUniform(eng);
    }
    Kokkos::deep_copy(particleContainer->R.getView(), positions);

    particleContainer->update();

    loadBalancer->initializeORB(&fieldContainer->getFL(), &fieldContainer->getMesh());

    const value_type imbalanceBefore = maxParticleDeviationFraction(particleContainer);
    EXPECT_GT(imbalanceBefore, 0.10);
    EXPECT_EQ(globalParticleCount(particleContainer), nGlobal);
    EXPECT_TRUE(loadBalancer->balance(nGlobal, 1));

    bool isFirstRepartition = false;
    loadBalancer->repartition(&fieldContainer->getFL(), &fieldContainer->getMesh(),
                              isFirstRepartition);

    const value_type imbalanceAfter = maxParticleDeviationFraction(particleContainer);
    EXPECT_EQ(globalParticleCount(particleContainer), nGlobal);
    EXPECT_LT(imbalanceAfter, imbalanceBefore);
    EXPECT_LT(imbalanceAfter, 0.10);

    auto& femSolver = std::get<FEMSolver_t<value_type, Dim>>(fieldSolver->getSolver());
    auto& space     = femSolver.getSpace();
    fieldContainer->getRho() = 0.0;
    femSolver.setRhs(fieldContainer->getRho());

    fieldContainer->getPhi() = 1.0;
    fieldContainer->getPhi().fillHalo();

    using fem_solver_type = FEMSolver_t<value_type, Dim>;
    using lagrange_type   = typename fem_solver_type::LagrangeType;
    using element_type    = typename fem_solver_type::ElementType;

    element_type refElement;
    const Vector_t<size_t, Dim> zeroNdIndex(0);
    const auto firstElementVertexPoints = space.getElementMeshVertexPoints(zeroNdIndex);
    const Vector_t<value_type, Dim> DPhiInvT =
        refElement.getInverseTransposeTransformationJacobian(firstElementVertexPoints);
    const value_type absDetDPhi = Kokkos::abs(
        refElement.getDeterminantOfTransformationJacobian(firstElementVertexPoints));
    ippl::EvalFunctor<value_type, Dim, lagrange_type::numElementDOFs> poissonEquationEval(
        DPhiInvT, absDetDPhi);

    auto ax = space.evaluateAx(fieldContainer->getPhi(), poissonEquationEval);
    ax.fillHalo();

    EXPECT_EQ(maxExtentMismatch(ax), 0);
    EXPECT_TRUE(std::isfinite(ax.sum()));
}

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
