# -----------------------------------------------------------------------------
# Interoperability/CMakeLists.txt
#
# Header-only bridge exposing IPPL/Kokkos device memory to Python via DLPack.
# Consumers link IPPL::python_bridge and get the include paths and pybind11.
# -----------------------------------------------------------------------------
add_library(ippl_python_bridge INTERFACE)
add_library(IPPL::python_bridge ALIAS ippl_python_bridge)

target_include_directories(ippl_python_bridge INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(ippl_python_bridge INTERFACE IPPL::ippl pybind11::embed dlpack::dlpack)