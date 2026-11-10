//
// Unit test DLPackExport
//   Round trips ParticleAttribs through DLPack into PyTorch. InferenceBridge
//   loads torch_consumer.py, which imports the exported attribute (host or
//   device memory) with torch.from_dlpack, reads it, and writes into it in-place.
//   The test checks what torch saw and if torch's write is visible on
//   the IPPL side, and that torch tensors keep the allocation alive.
//
#include "Ippl.h"

#include <cstdint>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Interoperability/InferenceBridge.h"
#include "Interoperability/IpplDLPack.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

namespace py      = pybind11;
namespace interop = ippl::Interoperability;

// set in main(), lives for the whole test run
static interop::InferenceBridge* consumer = nullptr;

template <typename>
class DLPackExportTest;

template <typename T, typename ExecSpace, unsigned Dim>
class DLPackExportTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type   = T;
    using playout_type = ippl::detail::ParticleLayout<T, Dim, ExecSpace>;
    using memory_space = typename ExecSpace::memory_space;

    static constexpr unsigned dim = Dim;
    static constexpr bool ON_HOST = std::is_same_v<memory_space, Kokkos::HostSpace>;

    struct Bunch : public ippl::ParticleBase<playout_type> {
        explicit Bunch(playout_type& pl)
            : ippl::ParticleBase<playout_type>(pl) {
            this->addAttribute(Q);
        }
        ippl::ParticleAttrib<T, ExecSpace> Q;
    };

    DLPackExportTest()
        : bunch(playout) {}

    void SetUp() override {
        bunch.create(nParticles);
        auto R = bunch.R.getView();
        auto Q = bunch.Q.getView();
        // R(i)[d] = i * Dim + d and Q(i) = i, so both hold value k at flat position k
        Kokkos::parallel_for(
            "DLPackExportTest::fill", Kokkos::RangePolicy<ExecSpace>(0, nParticles),
            KOKKOS_LAMBDA(const size_t i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    R(i)[d] = static_cast<T>(i * Dim + d);
                }
                Q(i) = static_cast<T>(i);
            });
        Kokkos::fence();  // the export does not fence, the fill must be done before torch reads
    }

    // Sends capsule through torch (reads, then in-place * 2) and checks what torch saw.
    void roundTrip(py::capsule& cap, const std::vector<int64_t>& shape, const void* data) {
        auto info = consumer->call("roundtrip", cap).cast<py::dict>();

        if constexpr (ON_HOST) {
            EXPECT_EQ(info["device"].cast<std::string>(), "cpu");
        } else {
            EXPECT_EQ(info["device"].cast<std::string>(), "cuda");
            EXPECT_EQ(info["device_index"].cast<int>(), Kokkos::device_id());
        }
        const std::string expectedDtype =
            std::is_same_v<T, double> ? "torch.float64" : "torch.float32";
        EXPECT_EQ(info["dtype"].cast<std::string>(), expectedDtype);
        // zero-copy: torch works on the attribute's memory
        EXPECT_EQ(info["data"].cast<std::uintptr_t>(), reinterpret_cast<std::uintptr_t>(data));

        ASSERT_EQ(info["shape"].cast<std::vector<int64_t>>(), shape);
        std::vector<int64_t> strides(shape.size());  // row-major, contiguous
        int64_t stride = 1;
        for (size_t d = shape.size(); d-- > 0;) {
            strides[d] = stride;
            stride *= shape[d];
        }
        EXPECT_EQ(info["strides"].cast<std::vector<int64_t>>(), strides);

        auto seen = info["seen"].cast<std::vector<double>>();
        for (size_t k = 0; k < seen.size(); ++k) {
            EXPECT_EQ(static_cast<T>(seen[k]), static_cast<T>(k));
        }
    }

    const size_t nParticles = 100;
    playout_type playout;
    Bunch bunch;
};

using Tests = TestParams::tests<1, 2, 3>;
TYPED_TEST_SUITE(DLPackExportTest, Tests);

TYPED_TEST(DLPackExportTest, VectorAttribRoundTrip) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;
    auto& R                = this->bunch.R;
    const size_t n         = this->nParticles;

    py::capsule cap =
        interop::attribToDLPackVec(R, interop::ReadOnly::No, interop::DLPackVersion::Legacy);
    this->roundTrip(cap, {static_cast<int64_t>(n), static_cast<int64_t>(Dim)}, R.getView().data());

    // torch's in-place write is visible in the attribute
    auto rHost = R.getHostMirror();
    Kokkos::deep_copy(rHost, R.getView());
    for (size_t i = 0; i < n; ++i) {
        for (unsigned d = 0; d < Dim; ++d) {
            EXPECT_EQ(rHost(i)[d], static_cast<T>(2 * (i * Dim + d)));
        }
    }
}

TYPED_TEST(DLPackExportTest, ScalarAttribRoundTrip) {
    using T        = typename TestFixture::value_type;
    auto& Q        = this->bunch.Q;
    const size_t n = this->nParticles;

    py::capsule cap =
        interop::attribToDLPackScalar(Q, interop::ReadOnly::No, interop::DLPackVersion::Legacy);
    this->roundTrip(cap, {static_cast<int64_t>(n)}, Q.getView().data());

    auto qHost = Q.getHostMirror();
    Kokkos::deep_copy(qHost, Q.getView());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(qHost(i), static_cast<T>(2 * i));
    }
}

TYPED_TEST(DLPackExportTest, TorchTensorKeepsAllocationAlive) {
    auto& R          = this->bunch.R;
    const int before = R.getView().use_count();

    {  // imported by torch: the tensor owns the export until Python frees it
        py::capsule cap =
            interop::attribToDLPackVec(R, interop::ReadOnly::No, interop::DLPackVersion::Legacy);
        consumer->call("hold", cap);
    }
    EXPECT_EQ(consumer->call("held_count").cast<int>(), 1);
    EXPECT_EQ(R.getView().use_count(), before + 1);  // capsule gone, torch tensor still holds it

    consumer->call("drop");
    EXPECT_EQ(R.getView().use_count(), before);  // freeing the tensor called IPPL's deleter

    {  // seen by Python but never imported: the capsule destructor must release it
        py::capsule cap =
            interop::attribToDLPackVec(R, interop::ReadOnly::No, interop::DLPackVersion::Legacy);
        consumer->call("ignore", cap);
        EXPECT_EQ(R.getView().use_count(), before + 1);
    }
    EXPECT_EQ(R.getView().use_count(), before);
}

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        // The interpreter and the bridge live inside IPPL's lifetime,
        // so every exported View is released before ippl::finalize().
        py::scoped_interpreter guard{};
        interop::InferenceBridge bridge(IPPL_TEST_PY_DIR, "torch_consumer");
        consumer = &bridge;

        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();

        bridge.finalize();
        consumer = nullptr;
    }
    ippl::finalize();
    return success;
}
