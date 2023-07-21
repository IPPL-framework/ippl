//
// Unit test Halo
//   Test halo cell functionality and communication, as well as field layout neighbor finding
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename T>
class HaloTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<T, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type = ippl::Field<T, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using layout_type = ippl::FieldLayout<Dim>;

    HaloTest() {
        computeGridSizes(nPoints);
        for (unsigned d = 0; d < MaxDim; d++) {
            domain[d] = nPoints[d] / 10;
        }
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        auto& layout = std::get<Idx>(layouts) = layout_type<Dim>(owned, domDec);

        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);

        std::get<Idx>(fields) = std::make_shared<field_type<Dim>>(mesh, layout);
    }

    Collection<mesh_type> meshes;
    Collection<layout_type> layouts;
    PtrCollection<std::shared_ptr, field_type> fields;
    size_t nPoints[MaxDim];
    T domain[MaxDim];
};

using Precisions = ::testing::Types<double, float>;

TYPED_TEST_CASE(HaloTest, Precisions);

TYPED_TEST(HaloTest, CheckNeighbors) {
    auto check = [&]<unsigned Dim>(const typename TestFixture::template layout_type<Dim>& layout) {
        using neighbor_list = typename TestFixture::template layout_type<Dim>::neighbor_list;
        int myRank          = ippl::Comm->rank();
        int nRanks          = ippl::Comm->size();

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == myRank) {
                const neighbor_list& neighbors = layout.getNeighbors();
                for (unsigned i = 0; i < neighbors.size(); i++) {
                    const std::vector<int>& n = neighbors[i];
                    if (n.size() > 0) {
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
                        std::cout << "with index " << i << " in " << Dim << " dimensions are: ";
                        for (const auto& nrank : n) {
                            std::cout << nrank << ' ';
                        }
                        std::cout << std::endl;
                    }
                }
            }
            ippl::Comm->barrier();
        }
    };

    this->apply(check, this->layouts);
}

TYPED_TEST(HaloTest, CheckCubes) {
    auto check = [&]<unsigned Dim>(const typename TestFixture::template layout_type<Dim>& layout) {
        using neighbor_list = typename TestFixture::template layout_type<Dim>::neighbor_list;
        using mirror_type   = typename TestFixture::template layout_type<Dim>::host_mirror_type;
        const mirror_type& domains = layout.getHostLocalDomains();

        for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
            if (rank == ippl::Comm->rank()) {
                const neighbor_list& neighbors = layout.getNeighbors();

                constexpr static const char* cubes[6] = {"vertices", "edges",      "faces",
                                                         "cubes",    "tesseracts", "peteracts"};
                int boundaryCounts[Dim]               = {};
                for (unsigned i = 0; i < neighbors.size(); i++) {
                    if (neighbors[i].size() > 0) {
                        unsigned dim = 0;
                        for (unsigned idx = i; idx > 0; idx /= 3) {
                            dim += idx % 3 == 2;
                        }
                        boundaryCounts[dim]++;
                    }
                }

                std::cout << "Rank " << rank << "'s domain and neighbor components:" << std::endl
                          << " - domain:\t" << domains[rank] << std::endl;
                for (unsigned d = 0; d < Dim; d++) {
                    std::cout << " - " << cubes[d] << ":\t" << boundaryCounts[d] << std::endl;
                }
                std::cout << "--------------------------------------" << std::endl;
            }
            ippl::Comm->barrier();
        }
    };

    this->apply(check, this->layouts);
}

TYPED_TEST(HaloTest, FillHalo) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field = 1;
            field->fillHalo();

            auto view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());
            this->template nestedViewLoop(view, 0, [&]<typename... Idx>(const Idx... args) {
                assertTypeParam<TypeParam>(view(args...), 1);
            });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(HaloTest, AccumulateHalo) {
    auto check = [&]<unsigned Dim>(
                     std::shared_ptr<typename TestFixture::template field_type<Dim>>& field,
                     const typename TestFixture::template layout_type<Dim>& layout) {
        using mirror_type =
            typename TestFixture::template field_type<Dim>::view_type::host_mirror_type;
        using neighbor_list = typename TestFixture::template layout_type<Dim>::neighbor_list;

        *field = 1;
        mirror_type mirror =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());
        const unsigned int nghost = field->getNghost();

        if (ippl::Comm->size() > 1) {
            const neighbor_list& neighbors = layout.getNeighbors();
            ippl::NDIndex<Dim> lDom        = layout.getLocalNDIndex();

            auto arrayToCube = []<size_t... Dims>(const std::index_sequence<Dims...>&,
                                                  const std::array<ippl::e_cube_tag, Dim>& tags) {
                return ippl::detail::getCube<Dim>(tags[Dims]...);
            };
            auto indexToTags = [&]<size_t... Dims, typename... Tag>(
                const std::index_sequence<Dims...>&, Tag... tags) {
                return std::array<ippl::e_cube_tag, Dim>{(tags == nghost ? ippl::LOWER
                                                          : tags == lDom[Dims].length() + nghost - 1
                                                              ? ippl::UPPER
                                                              : ippl::IS_PARALLEL)...};
            };

            this->template nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
                auto encoding = indexToTags(std::make_index_sequence<Dim>{}, args...);
                auto cube     = arrayToCube(std::make_index_sequence<Dim>{}, encoding);

                // ignore all interior points
                if (cube == ippl::detail::countHypercubes(Dim) - 1) {
                    return;
                }

                unsigned int n = 0;
                this->template nestedLoop<Dim>(
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

        this->template nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
            assertTypeParam<TypeParam>(mirror(args...), 1);
        });
    };

    this->apply(check, this->fields, this->layouts);
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
