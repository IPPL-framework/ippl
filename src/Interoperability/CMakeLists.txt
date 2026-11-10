# -----------------------------------------------------------------------------
# src/Interoperability/CMakeLists.txt
#
# Header-only bridge exposing IPPL/Kokkos device memory to Python via DLPack. Consumers link
# IPPL::python_bridge and get pybind11 and DLPack. The headers are visible via IPPL::ippl's include
# path: include them as "Interoperability/<header>".
# -----------------------------------------------------------------------------
add_library(ippl_python_bridge INTERFACE)
add_library(IPPL::python_bridge ALIAS ippl_python_bridge)

target_link_libraries(ippl_python_bridge INTERFACE IPPL::ippl pybind11::embed dlpack::dlpack)
