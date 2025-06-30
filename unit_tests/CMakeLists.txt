# -----------------------------------------------------------------------------
# unit_test/CMakeLists.txt
#
# Configures and adds unit tests. This file assumes IPPL_ENABLE_UNIT_TESTS is ON.
# -----------------------------------------------------------------------------

if(NOT IPPL_ENABLE_FFT)
  message(FATAL_ERROR "Unit tests require IPPL_ENABLE_FFT to be ON. Please configure with -DIPPL_ENABLE_FFT=ON.")
endif()

if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR "❌ GoogleTest not found. Check Dependencies.cmake.")
endif()

include(GoogleTest)

# The following line handles automatic test discovery
file(GLOB TEST_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddIpplTest)


add_subdirectory(BareField)
add_subdirectory(Communicate)
add_subdirectory(FEM)
add_subdirectory(FFT)
add_subdirectory(Field)
add_subdirectory(Meshes)
add_subdirectory(Particle)
add_subdirectory(PIC)
add_subdirectory(Utility)
