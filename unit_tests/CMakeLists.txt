# -----------------------------------------------------------------------------
# unit_test/CMakeLists.txt
#
# Configures and adds unit tests. This file assumes IPPL_ENABLE_UNIT_TESTS is ON.
# -----------------------------------------------------------------------------

if(NOT IPPL_ENABLE_FFT)
  message(
    FATAL_ERROR
      "Unit tests require IPPL_ENABLE_FFT to be ON. Please configure with -DIPPL_ENABLE_FFT=ON.")
endif()

if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR "❌ GoogleTest not found. Check Dependencies.cmake.")
endif()

include(GoogleTest)

if(NOT TARGET ippl_test_support)
  add_library(ippl_test_support INTERFACE)
  target_include_directories(ippl_test_support
                             INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
  add_library(ippl::test_support ALIAS ippl_test_support)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/AddIpplTest.cmake")

add_subdirectory(BareField)
add_subdirectory(Communicate)
add_subdirectory(FEM)
add_subdirectory(FFT)
add_subdirectory(Field)
add_subdirectory(Meshes)
add_subdirectory(Particle)
add_subdirectory(PIC)
add_subdirectory(Utility)
