#include "Ippl.h"

#include <climits>
#include <cstdlib>
#include <limits>
#include <vector>

#include "Communicate/MessageChunks.h"
#include "gtest/gtest.h"

TEST(MessageChunks, CountsAndOffsetsCrossIntBoundary) {
    using size_type       = ippl::detail::size_type;
    const size_type limit = INT_MAX;
    std::vector<std::pair<size_type, int>> chunks;
    ippl::detail::forEachMessageChunk(2 * limit + 7, [&](size_type offset, int count) {
        chunks.emplace_back(offset, count);
    });
    EXPECT_EQ(chunks, (std::vector<std::pair<size_type, int>>{
                          {0, INT_MAX}, {limit, INT_MAX}, {2 * limit, 7}}));
    chunks.clear();
    ippl::detail::forEachMessageChunk(0, [&](size_type offset, int count) {
        chunks.emplace_back(offset, count);
    });
    EXPECT_EQ(chunks, (std::vector<std::pair<size_type, int>>{{0, 0}}));
    EXPECT_THROW(ippl::detail::forEachMessageChunk(1, [](auto, int) {}, 0), std::invalid_argument);
}

class ArchiveMessages : public ::testing::Test {
public:
    using Space   = Kokkos::DefaultExecutionSpace::memory_space;
    using View    = Kokkos::View<int*, Space>;
    using Archive = ippl::detail::Archive<Space>;

    void largeMessage();

    void roundTrip(bool blocking) {
        const int rank            = ippl::Comm->rank();
        const int next            = (rank + 1) % ippl::Comm->size();
        const int previous        = (rank + ippl::Comm->size() - 1) % ippl::Comm->size();
        constexpr int CHUNK_BYTES = 17;  // Also split individual int values across messages.
        for (const std::size_t count : {0, 1, 17, 23}) {
            View input("input", count), output("output", count);
            auto host = Kokkos::create_mirror_view(input);
            for (std::size_t i = 0; i < count; ++i)
                host(i) = rank * 1000 + i;
            if (count)
                Kokkos::deep_copy(input, host);
            Archive send(count * sizeof(int)), recv(count * sizeof(int));
            send.serialize(input, count);
            std::vector<MPI_Request> requests;
            // Two consecutive archives with the same tag must preserve chunk boundaries.
            ippl::Comm->isend(next, 200, send, requests, CHUNK_BYTES);
            ippl::Comm->isend(next, 200, send, requests, CHUNK_BYTES);
            for (int message = 0; message < 2; ++message) {
                if (blocking) {
                    ippl::Comm->recv(previous, 200, recv, count * sizeof(int), CHUNK_BYTES);
                } else {
                    std::vector<MPI_Request> receives;
                    ippl::Comm->irecv(previous, 200, recv, receives, count * sizeof(int),
                                      CHUNK_BYTES);
                    MPI_Waitall(receives.size(), receives.data(), MPI_STATUSES_IGNORE);
                }
                recv.deserialize(output, count);
                recv.resetReadPos();
                auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
                for (std::size_t i = 0; i < count; ++i) {
                    EXPECT_EQ(result(i), previous * 1000 + i);
                }
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
    }
};

TEST_F(ArchiveMessages, NonblockingChunks) {
    roundTrip(false);
}
TEST_F(ArchiveMessages, BlockingChunks) {
    roundTrip(true);
}

void ArchiveMessages::largeMessage() {
    if (!std::getenv("IPPL_TEST_LARGE_MESSAGES") || ippl::Comm->size() != 2) {
        GTEST_SKIP() << "Set IPPL_TEST_LARGE_MESSAGES=1 and use two ranks (about 4 GiB total).";
    }
    const ippl::detail::size_type bytes = static_cast<ippl::detail::size_type>(INT_MAX) + 4097;
    // Exercise the real default chunk boundary without an extra serialization allocation.
    struct SizedArchive {
        Archive storage;
        ippl::detail::size_type bytes;
        char* getBuffer() { return storage.getBuffer(); }
        auto getSize() const { return bytes; }
    } archive{Archive(bytes), bytes};
    using Bytes = Kokkos::View<char*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Bytes buffer(archive.getBuffer(), bytes);
    using Policy = Kokkos::RangePolicy<Kokkos::IndexType<ippl::detail::size_type>>;
    std::vector<MPI_Request> requests;
    if (ippl::Comm->rank() == 0) {
        Kokkos::parallel_for(
            "large archive pattern", Policy(0, bytes),
            KOKKOS_LAMBDA(const ippl::detail::size_type i) {
                buffer(i) = static_cast<char>(i % 127);
            });
        Kokkos::fence();
        ippl::Comm->isend(1, 201, archive, requests);
    } else {
        ippl::Comm->irecv(0, 201, archive, requests, bytes);
    }
    EXPECT_EQ(requests.size(), 2);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    if (ippl::Comm->rank() == 1) {
        ippl::detail::size_type errors = 0;
        Kokkos::parallel_reduce(
            "verify large archive", Policy(0, bytes),
            KOKKOS_LAMBDA(const ippl::detail::size_type i, ippl::detail::size_type& errors) {
                errors += buffer(i) != static_cast<char>(i % 127);
            },
            errors);
        EXPECT_EQ(errors, 0);
    }
}

TEST_F(ArchiveMessages, LargeMessage) { largeMessage(); }

TEST_F(ArchiveMessages, GpuAllocationFailureIsReported) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    EXPECT_THROW(Archive(std::numeric_limits<ippl::detail::size_type>::max() / 2), IpplException);
#else
    GTEST_SKIP() << "Requires a GPU backend.";
#endif
}

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
