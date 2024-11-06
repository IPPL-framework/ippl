#include "Ippl.h"

#include "Communicate/LogEntry.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

ippl::LogEntry createSampleLogEntry() {
    ippl::LogEntry logEntry;
    logEntry.methodName  = "TestMethod";
    logEntry.usedSize    = 1024;
    logEntry.freeSize    = 512;
    logEntry.memorySpace = "HostSpace";
    logEntry.rank        = 1;
    logEntry.timestamp   = std::chrono::high_resolution_clock::now();

    logEntry.parameters["overallocation"] = "1.5";
    logEntry.parameters["buffer_size"]    = "2048";

    return logEntry;
}

TEST(LogEntryTest, Serialize) {
    ippl::LogEntry logEntry = createSampleLogEntry();

    std::vector<char> buffer;
    buffer = logEntry.serialize();

    EXPECT_GT(buffer.size(), 0);
}

TEST(LogEntryTest, Deserialize) {
    ippl::LogEntry logEntry = createSampleLogEntry();

    std::vector<char> buffer;
    buffer = logEntry.serialize();

    ippl::LogEntry deserializedLogEntry = ippl::LogEntry::deserialize(buffer);

    EXPECT_EQ(deserializedLogEntry.methodName, logEntry.methodName);
    EXPECT_EQ(deserializedLogEntry.usedSize, logEntry.usedSize);
    EXPECT_EQ(deserializedLogEntry.freeSize, logEntry.freeSize);
    EXPECT_EQ(deserializedLogEntry.memorySpace, logEntry.memorySpace);
    EXPECT_EQ(deserializedLogEntry.rank, logEntry.rank);

    EXPECT_EQ(deserializedLogEntry.parameters.size(), logEntry.parameters.size());
    EXPECT_EQ(deserializedLogEntry.parameters.at("overallocation"),
              logEntry.parameters.at("overallocation"));
    EXPECT_EQ(deserializedLogEntry.parameters.at("buffer_size"),
              logEntry.parameters.at("buffer_size"));

    auto originalTime     = logEntry.timestamp.time_since_epoch().count();
    auto deserializedTime = deserializedLogEntry.timestamp.time_since_epoch().count();
    EXPECT_EQ(originalTime, deserializedTime);
}

TEST(LogEntryTest, RoundTripSerialization) {
    ippl::LogEntry logEntry = createSampleLogEntry();

    std::vector<char> buffer;
    buffer = logEntry.serialize();

    ippl::LogEntry deserializedLogEntry = ippl::LogEntry::deserialize(buffer);

    EXPECT_EQ(deserializedLogEntry.methodName, logEntry.methodName);
    EXPECT_EQ(deserializedLogEntry.usedSize, logEntry.usedSize);
    EXPECT_EQ(deserializedLogEntry.freeSize, logEntry.freeSize);
    EXPECT_EQ(deserializedLogEntry.memorySpace, logEntry.memorySpace);
    EXPECT_EQ(deserializedLogEntry.rank, logEntry.rank);

    EXPECT_EQ(deserializedLogEntry.parameters.size(), logEntry.parameters.size());
    EXPECT_EQ(deserializedLogEntry.parameters.at("overallocation"),
              logEntry.parameters.at("overallocation"));
    EXPECT_EQ(deserializedLogEntry.parameters.at("buffer_size"),
              logEntry.parameters.at("buffer_size"));

    auto originalTime     = logEntry.timestamp.time_since_epoch().count();
    auto deserializedTime = deserializedLogEntry.timestamp.time_since_epoch().count();
    EXPECT_EQ(originalTime, deserializedTime);
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
