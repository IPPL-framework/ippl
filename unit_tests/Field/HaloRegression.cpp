// Halo exchange on thin and strided domains, with independent host expectations.
// Run on 2 or 4 MPI ranks; kernels use the configured execution/memory space.
#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <array>
#include <tuple>

namespace {
    template <typename Value>
    Value rankValue(int rank, int pass) {
        if constexpr (requires { Value::dim; }) {
            Value value;
            for (unsigned component = 0; component < Value::dim; ++component)
                value[component] = 1 + rank * 8 + pass * 64 + component;
            return value;
        } else {
            return Value(1 + rank * 8 + pass * 64);
        }
    }

    template <typename Value>
    void expectEqual(const Value& actual, const Value& expected) {
        if constexpr (requires { Value::dim; }) {
            for (unsigned component = 0; component < Value::dim; ++component)
                EXPECT_EQ(actual[component], expected[component]);
        } else {
            EXPECT_EQ(actual, expected);
        }
    }

    template <typename Value, unsigned Dim>
    void checkExchanges(bool periodic, bool strided, bool crossPartition = false) {
        const int ranks = ippl::Comm->size();
        if (ranks < 2)
            GTEST_SKIP() << "Requires at least two MPI ranks";
        if (crossPartition && ranks != 2 && ranks != 4)
            GTEST_SKIP() << "The 2x2 partition fixture requires two or four MPI ranks";

        using Mesh = ippl::UniformCartesian<double, Dim>;
        using Field = ippl::Field<Value, Dim, Mesh, typename Mesh::DefaultCentering>;
        const unsigned parallelAxis = strided ? 1 : 0;
        std::array<ippl::Index, Dim> axes;
        std::array<bool, Dim> parallel{};
        parallel[parallelAxis] = true;
        if (crossPartition)
            parallel[1] = true;
        for (unsigned d = 0; d < Dim; ++d) {
            const int count = crossPartition && d < 2 ? 2
                              : d == parallelAxis ? (strided ? 4 * ranks : ranks) : 5;
            // Index::split requires unit stride on the partitioned axis. Keep
            // the transverse axes strided to exercise physical-boundary trimming.
            const int stride = strided && d != parallelAxis ? 2 + static_cast<int>(d) : 1;
            axes[d] = ippl::Index(0, (count - 1) * stride, stride);
        }
        const auto domain = std::make_from_tuple<ippl::NDIndex<Dim>>(axes);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, parallel, periodic);
        Mesh mesh(domain, ippl::Vector<double, Dim>(1), ippl::Vector<double, Dim>(0));
        Field field(mesh, layout);
        const auto local = layout.getLocalNDIndex();
        const auto& domains = layout.getHostLocalDomains();
        const int ghost = field.getNghost();
        const int me = ippl::Comm->rank();
        auto expected = Kokkos::create_mirror(field.getView());
        auto actual = Kokkos::create_mirror(field.getView());
        if (!strided) {
            if (crossPartition) {
                ASSERT_TRUE(local[0].length() == 1 || local[1].length() == 1);
            } else {
                ASSERT_EQ(local[parallelAxis].length(), 1u);
            }
        }

        // Convert a source rank's local-view index to global coordinates. The
        // expectation uses domain coordinates, never production neighbor ranges.
        auto coordinates = [&](int source, const auto& indices) {
            std::array<int, Dim> point{};
            for (unsigned d = 0; d < Dim; ++d)
                point[d] = domains[source][d].first()
                           + (static_cast<int>(indices[d]) - ghost) * axes[d].stride();
            return point;
        };
        auto mapPoint = [&](auto& point) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (periodic) {
                    const int period = axes[d].length() * axes[d].stride();
                    point[d] = axes[d].first()
                               + ((point[d] - axes[d].first()) % period + period) % period;
                } else if (point[d] < axes[d].first() || point[d] > axes[d].last()) {
                    return false;
                }
            }
            return true;
        };
        auto contains = [&](int source, const auto& point) {
            for (unsigned d = 0; d < Dim; ++d)
                if (point[d] < domains[source][d].first()
                    || point[d] > domains[source][d].last())
                    return false;
            return true;
        };
        auto compare = [&](bool ownedOnly) {
            Kokkos::deep_copy(actual, field.getView());
            nestedViewLoop(actual, ownedOnly ? ghost : 0, [&](auto... indices) {
                expectEqual(actual(indices...), expected(indices...));
            });
        };

        for (int pass = 0; pass < 3; ++pass) {
            const auto ownValue = rankValue<Value>(me, pass);
            field = ownValue;
            Kokkos::deep_copy(expected, ownValue);
            nestedViewLoop(expected, 0, [&](auto... indices) {
                auto point = coordinates(me, std::array<int, Dim>{static_cast<int>(indices)...});
                if (!mapPoint(point))
                    return;
                for (int source = 0; source < ranks; ++source)
                    if (contains(source, point)) {
                        expected(indices...) = rankValue<Value>(source, pass);
                        break;
                    }
            });
            field.fillHalo();
            compare(false);

            for (bool noGhost : {false, true}) {
                // NOGHOST excludes physical boundary entries and has no periodic
                // serial-axis folding contract; periodic cases test ordinary accumulation.
                if (periodic && noGhost)
                    continue;
                field = ownValue;
                Kokkos::deep_copy(expected, ownValue);
                for (int source = 0; source < ranks; ++source) {
                    nestedLoop<Dim>(
                        [](unsigned) -> size_t { return 0; },
                        [&](unsigned d) -> size_t {
                            return domains[source][d].length() + 2 * ghost;
                        },
                        [&](auto... indices) {
                            const std::array<int, Dim> index{static_cast<int>(indices)...};
                            bool owned = true;
                            for (unsigned d = 0; d < Dim; ++d)
                                owned = owned && index[d] >= ghost
                                        && index[d] < static_cast<int>(domains[source][d].length())
                                                          + ghost;
                            if (owned)
                                return;
                            auto point = coordinates(source, index);
                            if (!mapPoint(point) || !contains(me, point))
                                return;
                            std::array<size_t, Dim> target{};
                            for (unsigned d = 0; d < Dim; ++d) {
                                if (noGhost && (point[d] == axes[d].first()
                                                || point[d] == axes[d].last()))
                                    return;
                                target[d] = (point[d] - local[d].first()) / axes[d].stride() + ghost;
                            }
                            std::apply([&](auto... i) {
                                expected(i...) += rankValue<Value>(source, pass);
                            }, target);
                        });
                }
                if (noGhost)
                    field.accumulateHalo_noghost();
                else
                    field.accumulateHalo();
                compare(true);
            }
        }
    }
}  // namespace

// Each rank owns one x entry. Fill, accumulation and empty NOGHOST messages
// must agree with independent global-coordinate expectations on 2/4 MPI ranks.
TEST(HaloRegression, ThinScalar2D) { checkExchanges<double, 2>(false, false); }

// Periodic one-entry subdomains must distinguish lower/upper images even when
// both messages go to the same rank; serial-axis wrapping must remain correct.
TEST(HaloRegression, ThinPeriodicScalar2D) { checkExchanges<double, 2>(true, false); }

// Repeat the thin nonperiodic check in 3D, preserving all Vector components
// through three buffer-reuse passes on the configured execution/memory space.
TEST(HaloRegression, ThinVector3D) {
    checkExchanges<ippl::Vector<double, 3>, 3>(false, false);
}

// Periodic 3D faces/edges/corners must accumulate all contributions exactly,
// including serial-axis folding and repeated Vector communication buffers.
TEST(HaloRegression, ThinPeriodicVector3D) {
    checkExchanges<ippl::Vector<double, 3>, 3>(true, false);
}

// On four ranks, both partitioned axes have one-entry local domains. A neighbor
// can then be parallel to a one-entry axis: overlap width must not classify it
// as a lower/upper neighbor. Two ranks also exercise a one-entry local axis.
TEST(HaloRegression, ThinParallelAxisScalar2D) {
    checkExchanges<double, 2>(false, false, true);
}

// The 2x2 decomposition must also distinguish periodic images on faces/edges
// when a parallel local axis has one entry. All three Vector components must sum.
TEST(HaloRegression, ThinParallelAxisPeriodicVector3D) {
    checkExchanges<ippl::Vector<double, 3>, 3>(true, false, true);
}

// Non-unit index strides must be applied on both sides of NOGHOST exchange.
// Upper physical boundaries keep their original values; interior contributions sum.
TEST(HaloRegression, StridedNoGhostScalar2D) { checkExchanges<double, 2>(false, true); }

// Exercise the same stride regression with 3D Vector values and two serial
// axes, ensuring boundary trimming preserves message sizes and components.
TEST(HaloRegression, StridedNoGhostVector3D) {
    checkExchanges<ippl::Vector<double, 3>, 3>(false, true);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
