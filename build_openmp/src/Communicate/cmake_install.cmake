# Install script for directory: /Users/bobschreiner/ETH/Thesis/ippl/src/Communicate

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
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Communicate" TYPE FILE FILES
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/Archive.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/Archive.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/Communicate.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/DataTypes.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/Operations.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/TagMaker.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Communicate/Tags.h"
    )
endif()

