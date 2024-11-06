#include "Ippl.h"
#include "Utility/TypeUtils.h"

#include "Communicate/BufferHandler.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

using MemorySpaces = ippl::detail::TypeForAllSpaces<::testing::Types>::memory_spaces_type;

template <typename MemorySpace>
class TypedBufferHandlerTest : public ::testing::Test {
protected:
    using memory_space = MemorySpace;

    class TestableBufferHandler : public ippl::BufferHandler<memory_space> {
    public:
        using ippl::BufferHandler<memory_space>::deleteAllBuffers;
        using ippl::BufferHandler<memory_space>::freeAllBuffers;

        size_t usedBuffersSize() const { return this->used_buffers.size(); }

        size_t freeBuffersSize() const { return this->free_buffers.size(); }
    };

    void SetUp() override { handler = std::make_unique<TestableBufferHandler>(); }

    void TearDown() override { handler.reset(); }

    std::unique_ptr<TestableBufferHandler> handler;
};

TYPED_TEST_CASE(TypedBufferHandlerTest, MemorySpaces);

TYPED_TEST(TypedBufferHandlerTest, GetBuffer_EmptyFreeBuffers) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->getBufferSize(), 100);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetBuffer_SuitableBufferAvailable) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    this->handler->freeBuffer(buffer1);

    auto buffer2 = this->handler->getBuffer(40, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), 50);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}
TYPED_TEST(TypedBufferHandlerTest, FreeBuffer) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer);

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 1);
}

TYPED_TEST(TypedBufferHandlerTest, FreeAllBuffers) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    auto buffer2 = this->handler->getBuffer(100, 1.0);

    this->handler->freeAllBuffers();

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 2);
}

TYPED_TEST(TypedBufferHandlerTest, DeleteAllBuffers) {
    this->handler->getBuffer(50, 1.0);
    this->handler->getBuffer(100, 1.0);

    this->handler->deleteAllBuffers();

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ResizeLargerThanAvailable) {
    auto smallBuffer = this->handler->getBuffer(50, 1.0);
    this->handler->freeBuffer(smallBuffer);

    auto largeBuffer = this->handler->getBuffer(200, 1.0);
    EXPECT_EQ(largeBuffer->getBufferSize(), 200);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ExactSizeMatch) {
    auto buffer1 = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer1);

    auto buffer2 = this->handler->getBuffer(100, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), 100);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, FreeNonExistentBuffer) {
    auto buffer    = this->handler->getBuffer(100, 1.0);
    auto newBuffer = std::make_shared<ippl::detail::Archive<typename TestFixture::memory_space>>(200);

    this->handler->freeBuffer(newBuffer);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, RepeatedAllocateAndFreeCycle) {
    for (int i = 0; i < 10; ++i) {
        auto buffer = this->handler->getBuffer(100, 1.0);
        this->handler->freeBuffer(buffer);
    }

    EXPECT_EQ(this->handler->usedBuffersSize(), 0);
    EXPECT_EQ(this->handler->freeBuffersSize(), 1);
}

TYPED_TEST(TypedBufferHandlerTest, GetBuffer_ZeroSize) {
    auto buffer = this->handler->getBuffer(0, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_GE(buffer->getBufferSize(), 0);
    EXPECT_EQ(this->handler->usedBuffersSize(), 1);
    EXPECT_EQ(this->handler->freeBuffersSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_EmptyHandler) {
    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterBufferAllocation) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    EXPECT_EQ(this->handler->getUsedSize(), 100);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterFreeBuffer) {
    auto buffer = this->handler->getBuffer(100, 1.0);
    this->handler->freeBuffer(buffer);

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 100);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterFreeAllBuffers) {
    auto buffer1 = this->handler->getBuffer(50, 1.0);
    auto buffer2 = this->handler->getBuffer(100, 1.0);

    this->handler->freeAllBuffers();

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 150);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_AfterDeleteAllBuffers) {
    this->handler->getBuffer(50, 1.0);
    this->handler->getBuffer(100, 1.0);

    this->handler->deleteAllBuffers();

    EXPECT_EQ(this->handler->getUsedSize(), 0);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
}

TYPED_TEST(TypedBufferHandlerTest, GetAllocatedAndFreeSize_ResizeBufferLargerThanAvailable) {
    auto smallBuffer = this->handler->getBuffer(50, 1.0);
    this->handler->freeBuffer(smallBuffer);

    auto largeBuffer = this->handler->getBuffer(200, 1.0);

    EXPECT_EQ(this->handler->getUsedSize(), 200);
    EXPECT_EQ(this->handler->getFreeSize(), 0);
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
