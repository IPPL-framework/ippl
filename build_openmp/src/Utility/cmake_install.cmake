# Install script for directory: /Users/bobschreiner/ETH/Thesis/ippl/src/Utility

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Utility" TYPE FILE FILES
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/Inform.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/IpplException.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/IpplInfo.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/IpplTimings.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/PAssert.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/Timer.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/Unique.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/User.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/UserList.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/my_auto_ptr.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/vmap.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/vmap.hpp"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/ParameterList.h"
    "/Users/bobschreiner/ETH/Thesis/ippl/src/Utility/TypeUtils.h"
    )
endif()

