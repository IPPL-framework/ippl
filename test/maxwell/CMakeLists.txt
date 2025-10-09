# -----------------------------------------------------------------------------
# test/maxwell/CMakeLists.txt
#
# Integration tests for IPPL maxwell solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding maxwell integration tests from: ${_relPath}")

# Only needed for these tests
option(IPPL_FETCH_TEST_ASSETS "Download test-only assets (e.g. stb headers)" ON)
set(IPPL_STB_DIR "" CACHE PATH "Provide a directory containing stb_image_write.h to avoid network")

if(IPPL_STB_DIR)
  set(_stb_inc_dir "${IPPL_STB_DIR}")
elseif(IPPL_FETCH_TEST_ASSETS)
  set(_stb_inc_dir "${CMAKE_BINARY_DIR}/_third_party/stb")
  set(_stb_hdr "${_stb_inc_dir}/stb_image_write.h")
  file(MAKE_DIRECTORY "${_stb_inc_dir}")
  if(NOT EXISTS "${_stb_hdr}")
    message(STATUS "⬇️  Fetching stb_image_write.h for integration tests …")
    file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
         "${_stb_hdr}" STATUS _dl_status TLS_VERIFY ON)
    list(GET _dl_status 0 _code)
    if(NOT _code EQUAL 0)
      message(FATAL_ERROR "Failed to download stb_image_write.h: ${_dl_status}")
    endif()
  endif()
else()
  message(
    FATAL_ERROR "stb_image_write.h unavailable. Set IPPL_STB_DIR or enable IPPL_FETCH_TEST_ASSETS.")
endif()

if(NOT TARGET stb_image_write::header)
  add_library(stb_image_write_header INTERFACE)
  add_library(stb_image_write::header ALIAS stb_image_write_header)
  target_include_directories(stb_image_write_header SYSTEM INTERFACE "${_stb_inc_dir}")
endif()

if(NOT TARGET stb_image_write::impl)
  # Generate a tiny wrapper source alongside the header
  set(_stb_wrap "${_stb_inc_dir}/stb_image_write_impl.cpp")
  file(WRITE "${_stb_wrap}"
       "#define STB_IMAGE_WRITE_IMPLEMENTATION\n#include \"stb_image_write.h\"\n")

  add_library(stb_image_write_impl STATIC "${_stb_wrap}")
  add_library(stb_image_write::impl ALIAS stb_image_write_impl)

  # Depend on the header target and keep includes SYSTEM here as well
  target_link_libraries(stb_image_write_impl PUBLIC stb_image_write::header)
  target_include_directories(stb_image_write_impl SYSTEM PRIVATE "${_stb_inc_dir}")

  # Silence the specific warning in this one TU (optional)
  target_compile_options(
    stb_image_write_impl
    PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wno-missing-field-initializers>
            $<$<CXX_COMPILER_ID:MSVC>:/wd4351>)
endif()

if(IPPL_ENABLE_FFT)
  add_ippl_integration_test(TestStandardFDTDSolver LINK_LIBS stb_image_write::impl LABELS solver
                            integration)

  add_ippl_integration_test(TestNonStandardFDTDSolver LINK_LIBS stb_image_write::impl LABELS solver
                            integration)

  add_ippl_integration_test(TestStandardFDTDSolver_convergence LINK_LIBS stb_image_write::impl
                            LABELS solver integration)

  add_ippl_integration_test(TestNonStandardFDTDSolver_convergence LINK_LIBS stb_image_write::header
                            LABELS solver integration)

else()
  message(STATUS "⚠️ FFT disabled; skipping Maxwell FDTD integration tests.")
endif()
