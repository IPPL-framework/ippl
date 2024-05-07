#include "Ippl.h"
#include <gtest/gtest.h>

#include "BaseDistributionFunction.hpp"
#include "BaseParticleDistribution.hpp"

TEST(ParticleDistributionBase, InitDistribution) {
    ippl::Vector<double, 2> rmin = 0;
    ippl::Vector<double, 2> rmax = 1;
    ippl::Vector<int, 2> rn      = 10;

    ParticleDistributionBase<double, 2> p_dist(rmin, rmax, new GridPlacement<double, 2>(rn));
    p_dist.generateDistribution();
    
    EXPECT_EQ(p_dist.getNumParticles(), 100);
}

TEST(ParticleDistributionBase, FilterOperation) {
    ippl::Vector<double, 2> rmin = 0;
    ippl::Vector<double, 2> rmax = 1;
    ippl::Vector<int, 2> rn      = 10;

    ParticleDistributionBase<double, 2> p_dist(rmin, rmax, new GridPlacement<double, 2>(rn));

    p_dist.generateDistribution();
    p_dist.applyFilter([](const ippl::Vector<double, 2>& p) -> bool {
        return ((p(0) >= 0.1) && (p(0) <= 0.9) && (p(1) >= 0.1) && p(1) <= 0.9);
    });

    EXPECT_EQ(p_dist.getNumParticles(), 8 * 8);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
