#include "Ippl.h"

#include "Communicate/LoggingBufferHandler.h"

#include "Communicate/BufferHandler.h"
#include "gtest/gtest.h"

using MemorySpaces = ippl::detail::TypeForAllSpaces<::testing::Types>::memory_spaces_type;

template <typename MemorySpace>
class TypedLoggingBufferHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        rank                = 0;
        this->bufferHandler = std::make_shared<ippl::comms::DefaultBufferHandler<MemorySpace>>();
        this->loggingHandler =
            std::make_shared<ippl::comms::LoggingBufferHandler<MemorySpace>>(bufferHandler, rank);
    }

    int rank;
    std::shared_ptr<ippl::comms::DefaultBufferHandler<MemorySpace>> bufferHandler;
    std::shared_ptr<ippl::comms::LoggingBufferHandler<MemorySpace>> loggingHandler;
};

TYPED_TEST_SUITE(TypedLoggingBufferHandlerTest, MemorySpaces);

template <typename T>
void compareNumericParameter(const std::string& paramString, T expectedValue,
                             double tolerance = 1e-6) {
    if constexpr (std::is_floating_point<T>::value) {
        double actualValue = std::stod(paramString);
        EXPECT_NEAR(actualValue, expectedValue, tolerance);
    } else if constexpr (std::is_integral<T>::value) {
        int actualValue = std::stoi(paramString);
        EXPECT_EQ(actualValue, expectedValue);
    } else {
        FAIL() << "Unsupported type for numeric comparison";
    }
}

// Test: The information stored when calling getBuffer is correct
TYPED_TEST(TypedLoggingBufferHandlerTest, GetBufferLogsCorrectly) {
    auto buffer      = this->loggingHandler->getBuffer(100, 1.5);
    const auto& logs = this->loggingHandler->getLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& entry = logs[0];
    EXPECT_EQ(entry.methodName, "getBuffer");

    std::string sizeStr = entry.parameters.at("size");
    compareNumericParameter(sizeStr, 100);

    std::string overallocationStr = entry.parameters.at("overallocation");
    compareNumericParameter(overallocationStr, 1.5);

    EXPECT_EQ(entry.usedSize, this->bufferHandler->getUsedSize());
    EXPECT_EQ(entry.freeSize, this->bufferHandler->getFreeSize());
    EXPECT_EQ(entry.memorySpace, TypeParam::name());
    EXPECT_EQ(entry.rank, this->rank);
}

// Test: The information stored when calling freeBuffer is correct
TYPED_TEST(TypedLoggingBufferHandlerTest, FreeBufferLogsCorrectly) {
    auto buffer = this->loggingHandler->getBuffer(100, 1.0);
    this->loggingHandler->freeBuffer(buffer);

    const auto& logs = this->loggingHandler->getLogs();
    ASSERT_EQ(logs.size(), 2);

    const auto& entry = logs[1];
    EXPECT_EQ(entry.methodName, "freeBuffer");
    EXPECT_EQ(entry.usedSize, this->bufferHandler->getUsedSize());
    EXPECT_EQ(entry.freeSize, this->bufferHandler->getFreeSize());
    EXPECT_EQ(entry.memorySpace, TypeParam::name());
    EXPECT_EQ(entry.rank, this->rank);
}

// Test: The information stored when calling freeAllBuffers is correct
TYPED_TEST(TypedLoggingBufferHandlerTest, FreeAllBuffersLogsCorrectly) {
    this->loggingHandler->getBuffer(100, 1.0);
    this->loggingHandler->getBuffer(200, 1.0);
    this->loggingHandler->freeAllBuffers();

    const auto& logs = this->loggingHandler->getLogs();
    ASSERT_EQ(logs.size(), 3);

    const auto& entry = logs[2];
    EXPECT_EQ(entry.methodName, "freeAllBuffers");
    EXPECT_EQ(entry.usedSize, this->bufferHandler->getUsedSize());
    EXPECT_EQ(entry.freeSize, this->bufferHandler->getFreeSize());
    EXPECT_EQ(entry.memorySpace, TypeParam::name());
    EXPECT_EQ(entry.rank, this->rank);
}

// Test: The information stored when calling deleteAllBuffers is correct
TYPED_TEST(TypedLoggingBufferHandlerTest, DeleteAllBuffersLogsCorrectly) {
    this->loggingHandler->getBuffer(100, 1.0);
    this->loggingHandler->getBuffer(200, 1.0);
    this->loggingHandler->deleteAllBuffers();

    const auto& logs = this->loggingHandler->getLogs();
    ASSERT_EQ(logs.size(), 3);

    const auto& entry = logs[2];
    EXPECT_EQ(entry.methodName, "deleteAllBuffers");
    EXPECT_EQ(entry.usedSize, this->bufferHandler->getUsedSize());
    EXPECT_EQ(entry.freeSize, this->bufferHandler->getFreeSize());
    EXPECT_EQ(entry.memorySpace, TypeParam::name());
    EXPECT_EQ(entry.rank, this->rank);
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
