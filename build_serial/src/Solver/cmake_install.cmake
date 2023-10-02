# Install script for directory: /Users/bobschreiner/ETH/Thesis/ippl/src/Solver

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/bobschreiner/ETH/Thesis")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Solver" TYPE FILE FILES
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/SolverAlgorithm.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/ElectrostaticsCG.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/Electrostatics.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/PCG.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/Solver.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/Preconditioner.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/EnhancedPCG.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/FFTPoissonSolver.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/FFTPoissonSolver.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/FFTPeriodicPoissonSolver.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/FFTPeriodicPoissonSolver.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/P3MSolver.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Solver/P3MSolver.hpp"
    )
endif()

