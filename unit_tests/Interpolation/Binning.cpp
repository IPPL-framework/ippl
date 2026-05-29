#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include "Interpolation/Binning.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "Interpolation/CoordinateTransform.h"
#include "gtest/gtest.h"

namespace ippl {
    namespace test {

        using size_type = size_t;
        template <unsigned Dim>
        class BinningTest : public ::testing::Test {
        public:
            using T         = double;
            using ExecSpace = Kokkos::DefaultExecutionSpace;
            using MemSpace  = typename ExecSpace::memory_space;

            using Mesh_t    = UniformCartesian<T, Dim>;
            using Layout_t  = FieldLayout<Dim>;
            using PLayout_t = ParticleSpatialLayout<T, Dim, Mesh_t, ExecSpace>;

            // Test parameters
            static constexpr int n_grid_per_dim = 16;
            static constexpr int tile_size_val  = 4;
            static constexpr int kernel_width   = 4;

            Vector<int, Dim> n_grid;
            Vector<int, Dim> tile_size;
            Vector<T, Dim> origin;
            Vector<T, Dim> extent;
            Vector<T, Dim> hx;
            Vector<T, Dim> invdx;

            std::shared_ptr<Layout_t> layout;
            std::shared_ptr<Mesh_t> mesh;

            int myRank;
            int nRanks;

            BinningTest() {
                for (unsigned d = 0; d < Dim; ++d) {
                    n_grid[d]    = n_grid_per_dim;
                    tile_size[d] = tile_size_val;
                    origin[d]    = 0.0;
                    extent[d]    = 2.0 * Kokkos::numbers::pi_v<T>;
                    hx[d]        = extent[d] / n_grid[d];
                    invdx[d]     = 1.0 / hx[d];
                }
            }

            void SetUp() override {
                myRank = Comm->rank();
                nRanks = Comm->size();

                std::array<Index, Dim> domains;
                std::array<bool, Dim> isParallel;
                isParallel.fill(true);

                for (unsigned d = 0; d < Dim; ++d) {
                    domains[d] = Index(n_grid[d]);
                }

                auto owned = std::make_from_tuple<NDIndex<Dim>>(domains);
                layout     = std::make_shared<Layout_t>(MPI_COMM_WORLD, owned, isParallel, true);
                mesh       = std::make_shared<Mesh_t>(owned, hx, origin);
            }

            void TearDown() override {
                mesh.reset();
                layout.reset();
            }

            // Helper to get local domain info
            void getLocalDomainInfo(Vector<int, Dim>& ngrid_global, Vector<int, Dim>& ngrid_local,
                                    Vector<int, Dim>& local_offset) {
                const NDIndex<Dim>& lDom = layout->getLocalNDIndex();
                const NDIndex<Dim>& gDom = layout->getDomain();

                for (unsigned d = 0; d < Dim; ++d) {
                    ngrid_global[d] = gDom[d].length();
                    ngrid_local[d]  = lDom[d].length();
                    local_offset[d] = lDom[d].first();
                }
            }

            // Compute expected bin for a position (host-side reference implementation)
            int computeExpectedBin(const Vector<T, Dim>& pos, const Vector<int, Dim>& ngrid_global,
                                   const Vector<int, Dim>& local_offset,
                                   const Vector<int, Dim>& num_tiles) {
                Interpolation::CoordinateTransform<T, Dim> transform(origin, invdx, ngrid_global);

                int bin_idx = 0;
                int stride  = 1;

                // Row-major ordering: dimension Dim-1 varies fastest
                for (int d = Dim - 1; d >= 0; --d) {
                    T grid_pos  = transform.toGridCoordinate(pos[d], d);
                    int center  = transform.getStencilCenter(grid_pos - T(0.5), kernel_width);
                    int local_c = center - local_offset[d];
                    int tile_d  = Kokkos::clamp(local_c / tile_size[d], 0, num_tiles[d] - 1);

                    bin_idx += tile_d * stride;
                    stride *= num_tiles[d];
                }

                return bin_idx;
            }
        };

        using Dims = ::testing::Types<std::integral_constant<unsigned, 2>,
                                      std::integral_constant<unsigned, 3>>;

        template <typename T>
        class BinningTestTyped : public BinningTest<T::value> {};

        TYPED_TEST_SUITE(BinningTestTyped, Dims);

        // Test 1: Permutation contains all indices exactly once
        TYPED_TEST(BinningTestTyped, PermutationIsValid) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            // Create random positions within local domain
            const size_t n_particles = 1000;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            std::mt19937 rng(42 + this->myRank);
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            for (size_t i = 0; i < n_particles; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max    = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(i)[d] = local_min + dist(rng) * (local_max - local_min);
                }
            }
            Kokkos::deep_copy(positions, pos_host);

            // Compute number of tiles
            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            // Allocate output views
            Kokkos::View<size_t*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_t*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_t*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_t*, MemSpace> cursor("cursor", total_tiles + 1);

            // Run binning
            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            // Check permutation on host
            auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);

            std::vector<bool> seen(n_particles, false);
            for (size_t i = 0; i < n_particles; ++i) {
                size_t idx = permute_host(i);
                ASSERT_LT(idx, n_particles) << "Permutation index out of range: " << idx;
                ASSERT_FALSE(seen[idx]) << "Duplicate index in permutation: " << idx;
                seen[idx] = true;
            }

            for (size_t i = 0; i < n_particles; ++i) {
                ASSERT_TRUE(seen[i]) << "Missing index in permutation: " << i;
            }
        }

        // Test 2: Bin offsets are monotonically increasing and valid
        TYPED_TEST(BinningTestTyped, BinOffsetsAreValid) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 500;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            std::mt19937 rng(123 + this->myRank);
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            for (size_t i = 0; i < n_particles; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max    = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(i)[d] = local_min + dist(rng) * (local_max - local_min);
                }
            }
            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto offsets_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_offsets);

            // First offset should be 0
            EXPECT_EQ(offsets_host(0), 0u) << "First bin offset should be 0";

            // Last offset should be n_particles
            EXPECT_EQ(offsets_host(total_tiles), n_particles)
                << "Last bin offset should be n_particles";

            // Offsets should be monotonically non-decreasing
            for (size_t i = 1; i <= total_tiles; ++i) {
                EXPECT_GE(offsets_host(i), offsets_host(i - 1))
                    << "Bin offsets not monotonic at index " << i;
            }
        }

        // Test 3: Particles are sorted by bin (keys are non-decreasing after sort)
        TYPED_TEST(BinningTestTyped, ParticlesAreSortedByBin) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 500;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            std::mt19937 rng(456 + this->myRank);
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            for (size_t i = 0; i < n_particles; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max    = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(i)[d] = local_min + dist(rng) * (local_max - local_min);
                }
            }
            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto keys_host    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_keys);
            auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);

            for (size_t i = 1; i < n_particles; ++i) {
                const auto k_curr = keys_host(permute_host(i));
                const auto k_prev = keys_host(permute_host(i - 1));
                EXPECT_GE(k_curr, k_prev)
                    << "Particles not grouped by bin at slot " << i << ": "
                    << k_prev << " > " << k_curr;
            }
        }

        // Test 4: Bin offsets correctly partition particles by bin
        TYPED_TEST(BinningTestTyped, BinOffsetsMatchKeys) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 500;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            std::mt19937 rng(789 + this->myRank);
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            for (size_t i = 0; i < n_particles; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max    = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(i)[d] = local_min + dist(rng) * (local_max - local_min);
                }
            }
            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_keys);
            auto offsets_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_offsets);

            // For each bin, verify all particles in [offset[b], offset[b+1])
            // have their bin key == b. With the counting-sort grouping the
            // permute slot at position i references the original particle id;
            // bin_keys is indexed by ORIGINAL particle id, not by sorted slot.
            auto permute_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);
            for (size_t b = 0; b < total_tiles; ++b) {
                size_t start = offsets_host(b);
                size_t end   = offsets_host(b + 1);

                for (size_t i = start; i < end; ++i) {
                    const size_t orig = permute_host(i);
                    EXPECT_EQ(keys_host(orig), b)
                        << "Slot " << i << " in bin range [" << start << ", " << end
                        << ") references particle " << orig << " whose bin key is "
                        << keys_host(orig) << " but expected " << b;
                }
            }
        }

        // Test 5: Empty input handles correctly
        TYPED_TEST(BinningTestTyped, EmptyInput) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 0;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", 1);  // Min size 1

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", 1);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", 1);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            // Should not crash
            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto offsets_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_offsets);

            // All offsets should be 0 (empty)
            for (size_t i = 0; i <= total_tiles; ++i) {
                EXPECT_EQ(offsets_host(i), 0u) << "Empty input: offset[" << i << "] should be 0";
            }
        }

        // Test 6: Single particle
        TYPED_TEST(BinningTestTyped, SingleParticle) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 1;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            // Place particle in center of local domain
            for (unsigned d = 0; d < Dim; ++d) {
                T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                T local_max    = local_min + ngrid_local[d] * this->hx[d];
                pos_host(0)[d] = 0.5 * (local_min + local_max);
            }
            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);
            auto offsets_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_offsets);
            auto keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_keys);

            EXPECT_EQ(permute_host(0), 0u) << "Single particle permutation should be 0";

            // Exactly one bin should have 1 particle
            size_t particle_count = 0;
            for (size_t b = 0; b < total_tiles; ++b) {
                size_t bin_size = offsets_host(b + 1) - offsets_host(b);
                particle_count += bin_size;
            }
            EXPECT_EQ(particle_count, 1u) << "Should have exactly 1 particle in bins";
        }

        // Test 7: Boundary particles (near domain edges)
        TYPED_TEST(BinningTestTyped, BoundaryParticles) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            // Create particles at domain boundaries
            const size_t n_particles = 4 * Dim;  // 2 per boundary per dimension

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            size_t idx = 0;
            T eps      = 1e-10;

            for (unsigned d = 0; d < Dim; ++d) {
                T local_min = this->origin[d] + local_offset[d] * this->hx[d];
                T local_max = local_min + ngrid_local[d] * this->hx[d];
                T mid       = 0.5 * (local_min + local_max);

                // Particle near lower boundary
                for (unsigned dd = 0; dd < Dim; ++dd) {
                    if (dd == d) {
                        pos_host(idx)[dd] = local_min + eps;
                    } else {
                        pos_host(idx)[dd] = mid;
                    }
                }
                idx++;

                // Particle near upper boundary
                for (unsigned dd = 0; dd < Dim; ++dd) {
                    if (dd == d) {
                        pos_host(idx)[dd] = local_max - eps;
                    } else {
                        pos_host(idx)[dd] = mid;
                    }
                }
                idx++;
            }

            // Fill remaining with center particles
            while (idx < n_particles) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min      = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max      = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(idx)[d] = 0.5 * (local_min + local_max);
                }
                idx++;
            }

            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            // Verify permutation is valid
            auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);

            std::vector<bool> seen(n_particles, false);
            for (size_t i = 0; i < n_particles; ++i) {
                size_t p_idx = permute_host(i);
                ASSERT_LT(p_idx, n_particles) << "Boundary particle permutation out of range";
                ASSERT_FALSE(seen[p_idx]) << "Duplicate in boundary particle permutation";
                seen[p_idx] = true;
            }

            // Verify all bins are valid
            auto keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_keys);
            auto offsets_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_offsets);

            for (size_t i = 0; i < n_particles; ++i) {
                EXPECT_GE(keys_host(i), 0) << "Negative bin key for boundary particle";
                EXPECT_LT(static_cast<size_t>(keys_host(i)), total_tiles)
                    << "Bin key out of range for boundary particle";
            }
        }

        // Test 8: Verify bin assignment matches reference implementation
        TYPED_TEST(BinningTestTyped, BinAssignmentCorrectness) {
            constexpr unsigned Dim = TypeParam::value;
            using T                = typename TestFixture::T;
            using ExecSpace        = typename TestFixture::ExecSpace;
            using MemSpace         = typename TestFixture::MemSpace;

            Vector<int, Dim> ngrid_global, ngrid_local, local_offset;
            this->getLocalDomainInfo(ngrid_global, ngrid_local, local_offset);

            const size_t n_particles = 100;

            Kokkos::View<Vector<T, Dim>*, MemSpace> positions("positions", n_particles);
            auto pos_host = Kokkos::create_mirror_view(positions);

            std::mt19937 rng(999 + this->myRank);
            std::uniform_real_distribution<T> dist(0.0, 1.0);

            for (size_t i = 0; i < n_particles; ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    T local_min    = this->origin[d] + local_offset[d] * this->hx[d];
                    T local_max    = local_min + ngrid_local[d] * this->hx[d];
                    pos_host(i)[d] = local_min + dist(rng) * (local_max - local_min);
                }
            }
            Kokkos::deep_copy(positions, pos_host);

            Vector<int, Dim> num_tiles;
            size_t total_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                num_tiles[d] = (ngrid_local[d] + this->tile_size[d] - 1) / this->tile_size[d] + 1;
                total_tiles *= num_tiles[d];
            }

            Kokkos::View<size_type*, MemSpace> permute("permute", n_particles);
            Kokkos::View<size_type*, MemSpace> bin_offsets("bin_offsets", total_tiles + 1);
            Kokkos::View<size_type*, MemSpace> bin_keys("bin_keys", n_particles);
            Kokkos::View<size_type*, MemSpace> cursor("cursor", total_tiles + 1);

            Interpolation::detail::bin_sort<Dim, T, decltype(positions), ExecSpace>(
                positions, ngrid_global, ngrid_local, local_offset, this->tile_size,
                this->kernel_width, this->origin, this->invdx, permute, bin_offsets, bin_keys, cursor,
                n_particles, num_tiles);

            auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), permute);
            auto keys_host    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bin_keys);

            for (size_t i = 0; i < n_particles; ++i) {
                size_t orig_idx  = permute_host(i);
                int actual_bin   = keys_host(orig_idx);
                int expected_bin = this->computeExpectedBin(pos_host(orig_idx), ngrid_global,
                                                            local_offset, num_tiles);

                EXPECT_EQ(actual_bin, expected_bin)
                    << "Particle " << orig_idx << " at position " << pos_host(orig_idx)
                    << " has bin " << actual_bin << " but expected " << expected_bin;
            }
        }

    }  // namespace test
}  // namespace ippl

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
