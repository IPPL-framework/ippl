# -----------------------------------------------------------------------------
# unit_test/CMakeLists.txt
#
# Configures and adds unit tests. This file assumes ENABLE_UNIT_TESTS is ON.
# -----------------------------------------------------------------------------

if(NOT ENABLE_FFT)
    message(FATAL_ERROR "Unit tests require ENABLE_FFT to be ON. Please configure with -DENABLE_FFT=ON.")
endif()

if(NOT TARGET GTest::gtest)
    message(FATAL_ERROR "❌ GoogleTest not found. Check Dependencies.cmake.")
endif()

include(GoogleTest)

file(GLOB TEST_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddIpplTest)


add_subdirectory(BareField)
add_subdirectory(Communicate)
add_subdirectory(FFT)
add_subdirectory(Field)
add_subdirectory(Meshes)
add_subdirectory(Particle)
add_subdirectory(PIC)
add_subdirectory(Utility)
