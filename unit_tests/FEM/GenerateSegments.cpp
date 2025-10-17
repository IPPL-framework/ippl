#include <iostream>
#include <gtest/gtest.h>

#include "Ippl.h"
#include "TestUtils.h"


template <typename>
class GridPathSegmenterTest;

template <typename T, unsigned Dim>
class GridPathSegmenterTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
  using value_type = T;
  static constexpr unsigned dim = Dim;

  using Rule = ippl::DefaultCellCrossingRule;

  struct Scenario {
    std::array<T,Dim> A{};
    std::array<T,Dim> B{};
    std::size_t expected_segments = 0; // expected REAL segments (after compacting zero-length)
    const char* name = "";
  };

private:

  static KOKKOS_INLINE_FUNCTION T plane(unsigned a, std::size_t k,
                                        const std::array<T,Dim>& origin,
                                        const std::array<T,Dim>& h) {
    return origin[a] + T(k) * h[a];
  }

  static std::array<T,Dim> basis_scaled(unsigned a, const std::array<T,Dim>& h, T s) {
    std::array<T,Dim> v{}; for (unsigned d=0; d<Dim; ++d) v[d]=T(0);
    v[a] = s * h[a];
    return v;
  }

public:

  // Same cell: no plane crossings → 1 real segment
  Scenario same_cell(const std::array<T,Dim>& origin,
                     const std::array<T,Dim>& h) const {
    std::array<T,Dim> A{}, B{};
    for (unsigned d=0; d<Dim; ++d) {
      A[d] = origin[d] + T(0.25) * h[d];
      B[d] = origin[d] + T(0.75) * h[d];
    }
    return {A, B, 1u, "same_cell"};
  }

  // Single-axis forward cut: cross plane once along +axis → 2 segments
  Scenario single_axis_forward(unsigned axis,
                               const std::array<T,Dim>& origin,
                               const std::array<T,Dim>& h) const {
    std::array<T,Dim> A{}, B{};
    for (unsigned d=0; d<Dim; ++d) A[d] = origin[d] + T(0.2) * h[d];
    // Start near the upper face along 'axis' to guarantee a crossing
    A[axis] = origin[axis] + T(0.9) * h[axis];
    B = A;
    B[axis] += T(0.25) * h[axis]; // crosses plane at k+1 once
    return {A, B, 2u, "single_axis_forward"};
  }

  Scenario two_axes_forward(unsigned axis0, unsigned axis1,
                            const std::array<T,Dim>& origin,
                            const std::array<T,Dim>& h) const {
    static_assert(Dim >= 2, "two_axes_forward requires Dim>=2");
    std::array<T,Dim> A{}, B{};
    for (unsigned d=0; d<Dim; ++d) A[d] = origin[d] + T(0.2) * h[d];
    A[axis0] = origin[axis0] + T(0.9) * h[axis0];
    A[axis1] = origin[axis1] + T(0.85) * h[axis1];
    B = A;
    B[axis0] += T(0.25) * h[axis0];
    B[axis1] += T(0.30) * h[axis1];
    return {A, B, 3u, "two_axes_forward"};
  }

  Scenario three_axes_forward(const std::array<T,Dim>& origin,
                              const std::array<T,Dim>& h) const {
    static_assert(Dim >= 3, "three_axes_forward requires Dim>=3");
    std::array<T,Dim> A{}, B{};

    A[0] = origin[0] + T(0.9)  * h[0];  // dist to plane = 0.1h
    A[1] = origin[1] + T(0.85) * h[1];  // dist to plane = 0.15h
    A[2] = origin[2] + T(0.8)  * h[2];  // dist to plane = 0.20h

    B = A;

    for (unsigned d = 0; d < 3; ++d) {
      B[d] += T(0.30) * h[d];
    }
    return {A, B, 4u, "three_axes_forward"};
  }

  Scenario start_on_plane(unsigned axis,
                          const std::array<T,Dim>& origin,
                          const std::array<T,Dim>& h) const {
    std::array<T,Dim> A{}, B{};
    for (unsigned d=0; d<Dim; ++d) A[d] = origin[d] + T(0.3) * h[d];
    // Put A exactly at plane k=1 on 'axis'
    A[axis] = plane(axis, 1, origin, h);
    B = A;
    B[axis] += T(0.2) * h[axis];
    return {A, B, 1u, "start_on_plane"};
  }

  Scenario vertex_hit(const std::array<T,Dim>& origin,
                      const std::array<T,Dim>& h) const {
    std::array<T,Dim> A{}, B{};
    for (unsigned d = 0; d < Dim; ++d) {
      A[d] = origin[d] + T(0.9) * h[d];
    }
    B = A;
    for (unsigned d = 0; d < Dim; ++d) {
      B[d] += T(0.2) * h[d];
    }
    return {A, B, 2u, "vertex_hit"};
  }

  static std::array<T,Dim> ones() {
    std::array<T,Dim> a{}; for (unsigned d=0; d<Dim; ++d) a[d] = T(1); return a;
  }
  static std::array<T,Dim> zeros() {
    std::array<T,Dim> a{}; for (unsigned d=0; d<Dim; ++d) a[d] = T(0); return a;
  }

  struct OriginH {
    std::array<T,Dim> origin{};
    std::array<T,Dim> h{};
    const char* name{};
  };

  std::vector<Scenario>
  scenario_cases(const std::array<T,Dim>& origin, const std::array<T,Dim>& h) {
    std::vector<Scenario> v;

    // Always useful
    v.push_back(same_cell(origin, h));
    v.push_back(vertex_hit(origin, h));
    v.push_back(single_axis_forward(0, origin, h));
    v.push_back(start_on_plane(0, origin, h));


    // Dim >= 2 scenarios
    if constexpr (Dim >= 2) {
      v.push_back(two_axes_forward(0, 1, origin, h));
    }

    // Dim >= 3 scenarios
    if constexpr (Dim >= 3) {
      v.push_back(three_axes_forward(origin, h));
    }

    return v;
  }

  static std::vector<OriginH> origin_h_cases() {
    return std::vector{ // C++20 CTAD for vectors
      OriginH{ /*origin=*/fill_val(T(0)), /*h=*/fill_val(T(1)), "origin=0, h=1" },
      OriginH{ /*origin=*/fill_seq(T(-0.7), T(0.3)), /*h=*/fill_val(T(1)), "shifted origin, h=1" },
      OriginH{ /*origin=*/fill_val(T(0)), /*h=*/fill_seq(T(0.3), T(0.7)),"origin=0, nonunit h" },
      OriginH{ /*origin=*/fill_seq(T(-1.1), T(0.5)), /*h=*/fill_seq(T(0.25), T(1.1)), "shifted origin, mixed h" },
    };
  }

private:
  static std::array<T,Dim> fill_val(T v) {
    std::array<T,Dim> a{}; a.fill(v); return a;
  }
  static std::array<T,Dim> fill_seq(T base, T step) {
    std::array<T,Dim> a{};
    for (unsigned d=0; d<Dim; ++d) a[d] = base + step*T(d);
    return a;
  }
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<1, 2, 3>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(GridPathSegmenterTest, Tests);

TYPED_TEST(GridPathSegmenterTest, SegmentCountMatchesScenario) {
  using T   = typename TestFixture::value_type;
  constexpr unsigned Dim = TestFixture::dim;

  auto seg_is_real = [](const auto& s) {
    T maxc = T(0);
    for (unsigned d=0; d<Dim; ++d) {
      maxc = std::max({maxc, std::abs(s.p0[d]), std::abs(s.p1[d])});
    }
    const T eps = std::numeric_limits<T>::epsilon() * (T)100; // a bit looser than ulp
    for (unsigned d=0; d<Dim; ++d) {
      if (std::abs(s.p1[d] - s.p0[d]) > eps * (T(1) + maxc)) return true;
    }
    return false;
  };

  for (const auto& oh : TestFixture::origin_h_cases()) {
    auto scenarios = TestFixture::scenario_cases(oh.origin, oh.h);
    for (const auto& sc : scenarios) {
      // run split
      auto segs = ippl::GridPathSegmenter<Dim, T, typename TestFixture::Rule>
                    ::split(sc.A, sc.B, oh.origin, oh.h);

      // compact: count only real (non-zero-length) segments
      std::size_t real_count = 0;
      for (std::size_t i = 0; i < Dim + 1; ++i) {
        if (seg_is_real(segs[i])) ++real_count;
      }

      EXPECT_EQ(real_count, sc.expected_segments)
        << "Mismatch in segment count for scenario='" << sc.name
        << "' with origin/h case='" << oh.name << "'\n"
        << "  expected=" << sc.expected_segments
        << "  got=" << real_count;
    }
  }
}

int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  int result = 0;
  {
    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
  }
  ippl::finalize();
  return result;
}
