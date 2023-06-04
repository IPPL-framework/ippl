//
// Unit test FieldBC
//   Test field boundary conditions.
//
// Copyright (c) 2020, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include "Utility/IpplException.h"

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class FieldBCTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type = ippl::Field<double, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using bc_type = ippl::BConds<field_type<Dim>, Dim>;

    FieldBCTest() {
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

        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        auto& layout = std::get<Idx>(layouts) = ippl::FieldLayout<Dim>(owned, domDec);

        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);

        auto field = std::get<Idx>(fields) = std::make_shared<field_type<Dim>>(mesh, layout);
        *field                             = 1.0;
        *field                             = (*field) * 10.0;
        std::get<Idx>(HostFs)              = field->getHostMirror();
    }

    template <unsigned Dim>
    void checkResult(const double expected) {
        constexpr unsigned Idx = dimToIndex(Dim);

        auto& layout = std::get<Idx>(layouts);
        auto& HostF  = std::get<Idx>(HostFs);
        auto field   = std::get<Idx>(fields);

        const auto& lDomains = layout.getHostLocalDomains();
        const auto& domain   = layout.getDomain();
        const int myRank     = Ippl::Comm->rank();

        Kokkos::deep_copy(HostF, field->getView());

        for (size_t face = 0; face < 2 * Dim; ++face) {
            size_t d        = face / 2;
            bool checkUpper = lDomains[myRank][d].max() == domain[d].max();
            bool checkLower = lDomains[myRank][d].min() == domain[d].min();
            if (!checkUpper && !checkLower) {
                continue;
            }
            int N = HostF.extent(d);
            nestedLoop<Dim>(
                [&](unsigned) {
                    return 1;
                },
                [&](unsigned dim) {
                    return dim == d ? 2 : HostF.extent(dim) - 1;
                },
                [&]<typename... Idx>(const Idx... args) {
                    // to avoid ambiguity with MultirankUtils::apply
                    using ippl::apply;
                    using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;

                    index_type coords[Dim] = {args...};
                    if (checkLower) {
                        coords[d] = 0;
                        EXPECT_DOUBLE_EQ(expected, apply(HostF, coords));
                    }
                    if (checkUpper) {
                        coords[d] = N - 1;
                        EXPECT_DOUBLE_EQ(expected, apply(HostF, coords));
                    }
                });
        }
    }

    Collection<ippl::FieldLayout> layouts;
    PtrCollection<std::shared_ptr, field_type> fields;
    Collection<bc_type> bcFields;

    Collection<mesh_type> meshes;

    template <unsigned Dim>
    using mirror_type = typename field_type<Dim>::view_type::host_mirror_type;
    Collection<mirror_type> HostFs;

    size_t nPoints[MaxDim];
    double domain[MaxDim];
};

TEST_F(FieldBCTest, PeriodicBC) {
    double expected = 10.0;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field, bc_type<Dim>& bcField) {
        for (size_t i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type<Dim>>>(i);
        }
        bcField.findBCNeighbors(*field);
        bcField.apply(*field);
        checkResult<Dim>(expected);
    };

    apply(check, fields, bcFields);
}

TEST_F(FieldBCTest, NoBC) {
    double expected = 1.0;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field, bc_type<Dim>& bcField) {
        for (size_t i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::NoBcFace<field_type<Dim>>>(i);
        }
        bcField.findBCNeighbors(*field);
        bcField.apply(*field);
        checkResult<Dim>(expected);
    };

    apply(check, fields, bcFields);
}

TEST_F(FieldBCTest, ZeroBC) {
    double expected = 0.0;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field, bc_type<Dim>& bcField) {
        for (size_t i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<field_type<Dim>>>(i);
        }
        bcField.findBCNeighbors(*field);
        bcField.apply(*field);
        checkResult<Dim>(expected);
    };

    apply(check, fields, bcFields);
}

TEST_F(FieldBCTest, ConstantBC) {
    double constant = 7.0;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field, bc_type<Dim>& bcField) {
        for (size_t i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ConstantFace<field_type<Dim>>>(i, constant);
        }
        bcField.findBCNeighbors(*field);
        bcField.apply(*field);
        checkResult<Dim>(constant);
    };

    apply(check, fields, bcFields);
}

TEST_F(FieldBCTest, ExtrapolateBC) {
    double expected = 10.0;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field, bc_type<Dim>& bcField) {
        for (size_t i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ExtrapolateFace<field_type<Dim>>>(i, 0.0, 1.0);
        }
        bcField.findBCNeighbors(*field);
        bcField.apply(*field);
        checkResult<Dim>(expected);
    };

    apply(check, fields, bcFields);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    ippl::finalize();
    return RUN_ALL_TESTS();
}
