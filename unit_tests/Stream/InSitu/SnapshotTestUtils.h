// Test support: inspect backend-bound Catalyst snapshots and owned field cells.
// Host copies use the field's execution/memory space; expected values follow the
// x-fastest Conduit ordering and exclude ghost cells on each MPI rank.
#pragma once

#include "Ippl.h"
#include "gtest/gtest.h"
#include <catalyst.hpp>
#include <catalyst_conduit.hpp>
#include <vector>

namespace snapshot_test {
inline conduit_cpp::Node captured() {
    conduit_cpp::Node node;
    EXPECT_EQ(catalyst_results(conduit_cpp::c_node(&node)), catalyst_status_ok);
    return node;
}

template <class Field>
std::vector<double> interior(const Field& field) {
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
    const auto local = field.getLayout().getLocalNDIndex();
    const auto g = field.getNghost();
    std::vector<double> result;
    for (int k = 0; k < local[2].length(); ++k)
        for (int j = 0; j < local[1].length(); ++j)
            for (int i = 0; i < local[0].length(); ++i)
                result.push_back(host(i + g, j + g, k + g));
    return result;
}

inline void expectValues(conduit_cpp::Node values, const std::vector<double>& expected) {
    ASSERT_EQ(values.dtype().number_of_elements(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(*static_cast<const double*>(values.element_ptr(i)), expected[i]) << "element " << i;
    }
}
}  // namespace snapshot_test
