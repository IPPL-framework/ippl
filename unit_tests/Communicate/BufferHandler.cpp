#include "Ippl.h"

#include "Communicate/BufferHandler.h"

#include "Utility/TypeUtils.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

using MemorySpaces = ippl::detail::TypeForAllSpaces<::testing::Types>::memory_spaces_type;

// DefaultBufferHandler::getBuffer() rounds requests up to a 4 KiB page so the
// GPU-aware MPI registration cache is not churned by tiny allocations. The
// tests below compare against the page-aligned size, not the raw request.
constexpr size_t kBufferPageSize = 4096;
constexpr size_t expectedSize(size_t requested) {
    if (requested < kBufferPageSize) {
        return kBufferPageSize;
    }
    return ((requested + kBufferPageSize - 1) / kBufferPageSize) * kBufferPageSize;
}

template <typename MemorySpace>
class TypedBufferHandlerTest : public ::testing::Test {
protected:
    using memory_space = MemorySpace;

    class TestableBufferHandler : public ippl::DefaultBufferHandler<memory_space> {
    public:
        using ippl::DefaultBufferHandler<memory_space>::deleteAllBuffers;
        using ippl::DefaultBufferHandler<memory_space>::freeAllBuffers;

        size_t usedBuffersSize() const { return this->used_buffers.size(); }

        size_t freeBuffersSize() const { return this->free_buffers.size(); }
    };

    void SetUp() override { handler = std::make_unique<TestableBufferHandler>(); }

    void TearDown() override { handler.reset(); }

    std::unique_ptr<TestableBufferHandler> handler;
};

TYPED_TEST_SUITE(TypedBufferHandlerTest, MemorySpaces);

// Test: Allocating a buffer when no free buffers are available
TYPED_TEST(TypedBufferHandlerTest, GetBuffer_EmptyFreeBuffers) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->getBufferSize(), expectedSize(100));
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: Allocating a buffer when a suitable free buffer is available
TYPED_TEST(TypedBufferHandlerTest, GetBuffer_SuitableBufferAvailable) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    this->handler->freeBuffer(buffer1);

    auto buffer2 = this->handler->getBuffer(40, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), expectedSize(50));
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: Freeing a used buffer moves it to the free buffer pool
TYPED_TEST(TypedBufferHandlerTest, FreeBuffer) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer);

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 1);
}

// Test: Freeing all used buffers moves them to the free buffer pool
TYPED_TEST(TypedBufferHandlerTest, FreeAllBuffers) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    auto buffer2 = this->handler->getBuffer(100, 1.0);

    this->handler->freeAllBuffers();

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 2);
}

// Test: Deleting all buffers removes both used and free buffers
TYPED_TEST(TypedBufferHandlerTest, DeleteAllBuffers) {
    this->handler->getBuffer(50, 1.0);
    auto buffer = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer);

    this->handler->deleteAllBuffers();

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: Allocating a buffer larger than any available free buffer falls back
// to a fresh allocation; the smaller free buffer is retained in the pool so a
// later small request can reuse it (and so we never call free()+alloc() on the
// same virtual address, which would invalidate the GPU-aware MPI registration
// cache).
TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ResizeLargerThanAvailable) {
    constexpr size_t small = 50;
    constexpr size_t large = kBufferPageSize + 1;
    auto smallBuffer = this->handler->getBuffer(small, 1.0);
    this->handler->freeBuffer(smallBuffer);

    auto largeBuffer = this->handler->getBuffer(large, 1.0);
    EXPECT_EQ(largeBuffer->getBufferSize(), expectedSize(large));
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 1);
}

// Test: Allocating a buffer that matches the size of a free buffer exactly
TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ExactSizeMatch) {
    auto buffer1 = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer1);

    auto buffer2 = this->handler->getBuffer(100, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), expectedSize(100));
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: Freeing a buffer that does not exist in the used pool has no effect
TYPED_TEST(TypedBufferHandlerTest, FreeNonExistentBuffer) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    auto newBuffer =
        std::make_shared<ippl::detail::Archive<typename TestFixture::memory_space>>(200);

    this->handler->freeBuffer(newBuffer);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: Repeatedly allocating and freeing buffers should consolidate free buffers
TYPED_TEST(TypedBufferHandlerTest, RepeatedAllocateAndFreeCycle) {
    for (int i = 0; i < 10; ++i) {
        auto buffer = this->handler->getBuffer(100, 1.0);
        this->handler->freeBuffer(buffer);
    }

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 1);
}

// Test: Allocating a zero-size buffer should succeed and result in a non-null buffer
TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ZeroSize) {
    auto buffer = this->handler->getBuffer(0, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_GE(buffer->getBufferSize(), 0);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

// Test: The buffer sizes of an empty BufferHandler are zero
TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_EmptyHandler) {
    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

// Test: Allocating increases used buffer size correctly
TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterBufferAllocation) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    EXPECT_EQ(this->handler->getUsedSize(), expectedSize(100));
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

// Test: Allocating increases used buffer size correctly
TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterFreeBuffer) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer);

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), expectedSize(100));
}

// Test: Correct size is computed after freeing all buffers
TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterFreeAllBuffers) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    auto buffer2 = this->handler->getBuffer(100, 1.0);

    this->handler->freeAllBuffers();

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), expectedSize(50) + expectedSize(100));
}

// Test: Deleting all buffers results in zero free or used buffer sizes
TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterDeleteAllBuffers) {
    this->handler->getBuffer(50, 1.0);
    this->handler->getBuffer(100, 1.0);

    this->handler->deleteAllBuffers();

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_ResizeBufferLargerThanAvailable) {
    constexpr size_t small = 50;
    constexpr size_t large = kBufferPageSize + 1;
    auto smallBuffer = this->handler->getBuffer(small, 1.0);
    this->handler->freeBuffer(smallBuffer);

    auto largeBuffer = this->handler->getBuffer(large, 1.0);

    EXPECT_EQ(this->handler->getUsedSize(), expectedSize(large));
    EXPECT_EQ(this->handler->getFreeSize(), expectedSize(small));
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
