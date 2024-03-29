//
// Unit test Halo
//   Test halo cell functionality and communication, as well as field layout neighbor finding
//
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class HaloTest;

template <typename T, typename ExecSpace, unsigned Dim>
class HaloTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type     = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using layout_type    = ippl::FieldLayout<Dim>;

    HaloTest()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = nPoints[d] / 10;
        }

        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout = layout_type(MPI_COMM_WORLD, owned, isParallel);
        mesh   = mesh_type(owned, hx, origin);
        field  = std::make_shared<field_type>(mesh, layout);
    }

    mesh_type mesh;
    layout_type layout;
    std::shared_ptr<field_type> field;
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_CASE(HaloTest, Tests);

TYPED_TEST(HaloTest, CheckNeighbors) {
    int myRank = ippl::Comm->rank();
    int nRanks = ippl::Comm->size();

    for (int rank = 0; rank < nRanks; ++rank) {
        if (rank == myRank) {
            const auto& neighbors = this->layout.getNeighbors();
            for (unsigned i = 0; i < neighbors.size(); i++) {
                const std::vector<int>& n = neighbors[i];
                if (!n.empty()) {
                    unsigned dim = 0;
                    for (unsigned idx = i; idx > 0; idx /= 3) {
                        dim += idx % 3 == 2;
                    }
                    std::cout << "My rank is " << myRank << " and my neighbors at the";
                    switch (dim) {
                        case 0:
                            std::cout << " vertex ";
                            break;
                        case 1:
                            std::cout << " edge ";
                            break;
                        case 2:
                            std::cout << " face ";
                            break;
                        case 3:
                            std::cout << " cube ";
                            break;
                        default:
                            std::cout << ' ' << dim << "-cube ";
                            break;
                    }
                    std::cout << "with index " << i << " in " << TestFixture::dim
                              << " dimensions are: ";
                    for (const auto& nrank : n) {
                        std::cout << nrank << ' ';
                    }
                    std::cout << std::endl;
                }
            }
        }
        ippl::Comm->barrier();
    }
}

TYPED_TEST(HaloTest, CheckCubes) {
    auto& layout        = this->layout;
    const auto& domains = layout.getHostLocalDomains();

    for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
        if (rank == ippl::Comm->rank()) {
            const auto& neighbors = layout.getNeighbors();

            constexpr static std::array<const char*, 6> cubes = {
                "vertices", "edges", "faces", "cubes", "tesseracts", "peteracts"};
            std::array<int, TestFixture::dim> boundaryCounts{};
            for (unsigned i = 0; i < neighbors.size(); i++) {
                if (neighbors[i].size() > 0) {
                    unsigned dim = 0;
                    for (unsigned idx = i; idx > 0; idx /= 3) {
                        dim += static_cast<unsigned int>(idx % 3 == 2);
                    }
                    boundaryCounts[dim]++;
                }
            }

            std::cout << "Rank " << rank << "'s domain and neighbor components:" << std::endl
                      << " - domain:\t" << domains[rank] << std::endl;
            for (unsigned d = 0; d < TestFixture::dim; d++) {
                std::cout << " - " << cubes[d] << ":\t" << boundaryCounts[d] << std::endl;
            }
            std::cout << "--------------------------------------" << std::endl;
        }
        ippl::Comm->barrier();
    }
}

TYPED_TEST(HaloTest, FillHalo) {
    auto& field = this->field;

    *field = 1;
    field->fillHalo();

    auto view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());
    nestedViewLoop(view, 0, [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(view(args...), 1);
    });
}

TYPED_TEST(HaloTest, AccumulateHalo) {
    constexpr unsigned Dim = TestFixture::dim;

    auto& field  = this->field;
    auto& layout = this->layout;

    *field      = 1;
    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());
    const unsigned int nghost = field->getNghost();

    if (ippl::Comm->size() > 1) {
        const auto& neighbors   = layout.getNeighbors();
        ippl::NDIndex<Dim> lDom = layout.getLocalNDIndex();

        auto arrayToCube = []<size_t... Dims>(const std::index_sequence<Dims...>&,
                                              const std::array<ippl::e_cube_tag, Dim>& tags) {
            return ippl::detail::getCube<Dim>(tags[Dims]...);
        };
        auto indexToTags = [&]<size_t... Dims, typename... Tag>(const std::index_sequence<Dims...>&,
                                                                Tag... tags) {
            return std::array<ippl::e_cube_tag, Dim>{(tags == nghost ? ippl::LOWER
                                                      : tags == lDom[Dims].length() + nghost - 1
                                                          ? ippl::UPPER
                                                          : ippl::IS_PARALLEL)...};
        };

        nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            auto encoding = indexToTags(std::make_index_sequence<Dim>{}, args...);
            auto cube     = arrayToCube(std::make_index_sequence<Dim>{}, encoding);

            // ignore all interior points
            if (cube == ippl::detail::countHypercubes(Dim) - 1) {
                return;
            }

            unsigned int n = 0;
            nestedLoop<Dim>(
                [&](unsigned dl) -> size_t {
                    return encoding[dl] == ippl::IS_PARALLEL ? 0 : (encoding[dl] + 1) * 10;
                },
                [&](unsigned dl) -> size_t {
                    return encoding[dl] == ippl::IS_PARALLEL ? 1 : (encoding[dl] + 1) * 10 + 2;
                },
                [&]<typename... Flag>(const Flag... flags) {
                    auto adjacent = ippl::detail::getCube<Dim>(
                        (flags == 0   ? ippl::IS_PARALLEL
                         : flags < 20 ? (flags & 1 ? ippl::LOWER : ippl::IS_PARALLEL)
                                      : (flags & 1 ? ippl::UPPER : ippl::IS_PARALLEL))...);
                    if (adjacent == ippl::detail::countHypercubes(Dim) - 1) {
                        return;
                    }
                    n += neighbors[adjacent].size();
                });

            if (n > 0) {
                mirror(args...) = 1. / (n + 1);
            }
        });
        Kokkos::deep_copy(field->getView(), mirror);
    }

    field->fillHalo();
    field->accumulateHalo();

    Kokkos::deep_copy(mirror, field->getView());

    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(mirror(args...), 1);
    });
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
