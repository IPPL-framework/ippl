#include "Ippl.h"

#include <cmath>
#include "gtest/gtest.h"

class ParticleBCondsTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    ParticleBCondsTest() {
        setup();
    }

    void setup() {
        bunch = std::make_unique<bunch_type>(pl_m);
    }

    playout_type pl_m;
    std::unique_ptr<bunch_type> bunch;
};



TEST_F(ParticleBCondsTest, PeriodicBC) {
    bunch->create(1);

    typename bunch_type::particle_position_type::HostMirror R_host = Kokkos::create_mirror(bunch->R);

    double xlen = 0.2;
    double shift = 0.1;

    R_host(0) = ippl::Vector<double, dim>({xlen + shift, 0.0, 0.0});

    Kokkos::deep_copy(bunch->R, R_host);

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticlePeriodicBCond<double>, i);
    }

    // domain
    PRegion<double> region(0, xlen);
    NDRegion<double, dim> nr(region, region, region);

    // shift particle
    bunch->getLayout().applyBC(bunch->R, nr);

    // check result
    Kokkos::deep_copy(R_host, bunch->R);

    ippl::Vector<double, dim> expected = {shift, 0.0, 0.0};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], R_host(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, NoBC) {
    bunch->create(1);

    typename bunch_type::particle_position_type::HostMirror R_host = Kokkos::create_mirror(bunch->R);

    double xlen = 0.2;
    double shift = 0.1;

    R_host(0) = ippl::Vector<double, dim>({xlen + shift, 0.0, 0.0});

    Kokkos::deep_copy(bunch->R, R_host);

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleNoBCond<double>, i);
    }

    // domain
    PRegion<double> region(0, xlen);
    NDRegion<double, dim> nr(region, region, region);

    // shift particle
    bunch->getLayout().applyBC(bunch->R, nr);

    // check result
    Kokkos::deep_copy(R_host, bunch->R);

    ippl::Vector<double, dim> expected = {xlen + shift, 0.0, 0.0};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], R_host(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, ReflectiveBC) {
    bunch->create(1);

    typename bunch_type::particle_position_type::HostMirror R_host = Kokkos::create_mirror(bunch->R);

    double xlen = 0.2;
    double shift = 0.1;

    R_host(0) = ippl::Vector<double, dim>({xlen + shift, 0.0, 0.0});

    Kokkos::deep_copy(bunch->R, R_host);

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleReflectiveBCond<double>, i);
    }

    // domain
    PRegion<double> region(0, xlen);
    NDRegion<double, dim> nr(region, region, region);

    // shift particle
    bunch->getLayout().applyBC(bunch->R, nr);

    // check result
    Kokkos::deep_copy(R_host, bunch->R);

    ippl::Vector<double, dim> expected = {xlen - shift, 0.0, 0.0};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], R_host(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, SinkBC) {
    bunch->create(1);

    typename bunch_type::particle_position_type::HostMirror R_host = Kokkos::create_mirror(bunch->R);

    double xlen = 0.2;
    double shift = 0.1;

    R_host(0) = ippl::Vector<double, dim>({xlen + shift, 0.0, 0.0});

    Kokkos::deep_copy(bunch->R, R_host);

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleSinkBCond<double>, i);
    }

    // domain
    PRegion<double> region(0, xlen);
    NDRegion<double, dim> nr(region, region, region);

    // shift particle
    bunch->getLayout().applyBC(bunch->R, nr);

    // check result
    Kokkos::deep_copy(R_host, bunch->R);

    ippl::Vector<double, dim> expected = {xlen, 0.0, 0.0};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], R_host(0)[i]);
    }
}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}