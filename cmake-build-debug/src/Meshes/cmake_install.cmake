# Install script for directory: /Users/bobschreiner/ETH/Thesis/ippl/src/Meshes

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Meshes" TYPE FILE FILES
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/CartesianCentering.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/CartesianCentering.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/Cartesian.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/Cartesian.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/CartesianStencilSetup.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/Centering.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/Mesh.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/Mesh.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/UniformCartesian.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Meshes/UniformCartesian.hpp"
    )
endif()

