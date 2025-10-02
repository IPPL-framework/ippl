# -----------------------------------------------------------------------------
# src/Utility/CMakeLists.txt
#
# Adds utility functions, assertions, timing, and info tools to the IPPL target.
# -----------------------------------------------------------------------------

# Add Utility source files to the ippl library
target_sources(ippl PRIVATE Inform.cpp IpplInfo.cpp IpplTimings.cpp PAssert.cpp Timer.cpp)
