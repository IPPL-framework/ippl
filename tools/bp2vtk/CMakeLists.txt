cmake_minimum_required(VERSION 3.20)
project(bp2vtk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)

if(TARGET adios2::cxx_mpi)
  set(_bp2vtk_adios adios2::cxx_mpi)
else()
  set(_bp2vtk_adios adios2::cxx)
endif()

add_executable(bp2vtk bp2vtk.cxx)
target_link_libraries(bp2vtk PRIVATE ${_bp2vtk_adios} MPI::MPI_CXX)
