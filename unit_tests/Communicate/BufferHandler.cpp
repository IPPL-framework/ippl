#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

#include "Communicate/BufferHandler.h"

using memory_space = typename Kokkos::View<double***>::memory_space;

class TestableBufferHandler : public BufferHandler<memory_space> {
public:
    using BufferHandler<memory_space>::deleteAllBuffers;
    using BufferHandler<memory_space>::freeAllBuffers;

    size_t usedBuffersSize() const { return used_buffers.size(); }

    size_t freeBuffersSize() const { return free_buffers.size(); }
};

class BufferHandlerTest : public ::testing::Test {
protected:
    void SetUp() override { handler = std::make_unique<TestableBufferHandler>(); }

    void TearDown() override { handler.reset(); }

    std::unique_ptr<TestableBufferHandler> handler;
};

TEST_F(BufferHandlerTest, GetBuffer_EmptyFreeBuffers) {
    auto buffer = handler->getBuffer(100, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->getBufferSize(), 100);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, GetBuffer_SuitableBufferAvailable) {
    auto buffer1 = handler->getBuffer(50, 1.0);
    handler->freeBuffer(buffer1);

    auto buffer2 = handler->getBuffer(40, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), 50);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, FreeBuffer) {
    auto buffer = handler->getBuffer(100, 1.0);
    handler->freeBuffer(buffer);

    EXPECT_EQ(handler->usedBuffersSize(), 0);
    EXPECT_EQ(handler->freeBuffersSize(), 1);
}

TEST_F(BufferHandlerTest, FreeAllBuffers) {
    auto buffer1 = handler->getBuffer(50, 1.0);
    auto buffer2 = handler->getBuffer(100, 1.0);

    handler->freeAllBuffers();

    EXPECT_EQ(handler->usedBuffersSize(), 0);
    EXPECT_EQ(handler->freeBuffersSize(), 2);
}

TEST_F(BufferHandlerTest, DeleteAllBuffers) {
    handler->getBuffer(50, 1.0);
    handler->getBuffer(100, 1.0);

    handler->deleteAllBuffers();

    EXPECT_EQ(handler->usedBuffersSize(), 0);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, GetBuffer_ResizeLargerThanAvailable) {
    auto smallBuffer = handler->getBuffer(50, 1.0);
    handler->freeBuffer(smallBuffer);

    auto largeBuffer = handler->getBuffer(200, 1.0);
    EXPECT_EQ(largeBuffer->getBufferSize(), 200);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, GetBuffer_ExactSizeMatch) {
    auto buffer1 = handler->getBuffer(100, 1.0);
    handler->freeBuffer(buffer1);

    auto buffer2 = handler->getBuffer(100, 1.0);
    EXPECT_EQ(buffer2->getBufferSize(), 100);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, FreeNonExistentBuffer) {
    auto buffer = handler->getBuffer(100, 1.0);
    auto newBuffer = std::make_shared<ippl::detail::Archive<memory_space>>(200);

    handler->freeBuffer(newBuffer);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
}

TEST_F(BufferHandlerTest, RepeatedAllocateAndFreeCycle) {
    for (int i = 0; i < 10; ++i) {
        auto buffer = handler->getBuffer(100, 1.0);
        handler->freeBuffer(buffer);
    }
    
    EXPECT_EQ(handler->usedBuffersSize(), 0);
    EXPECT_EQ(handler->freeBuffersSize(), 1);
}

TEST_F(BufferHandlerTest, GetBuffer_ZeroSize) {
    auto buffer = handler->getBuffer(0, 1.0);
    ASSERT_NE(buffer, nullptr);
    EXPECT_GE(buffer->getBufferSize(), 0);
    EXPECT_EQ(handler->usedBuffersSize(), 1);
    EXPECT_EQ(handler->freeBuffersSize(), 0);
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
