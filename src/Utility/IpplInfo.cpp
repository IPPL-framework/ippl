//
// Class IpplInfo
//   Global Ippl information.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Utility/IpplInfo.h"
#include "Message/Message.h"
// #include "Message/Communicate.h"
// #include "Message/CommMPI.h"

#include "Ippl.h"

#include "IpplVersions.h"

#include <unistd.h>
#include <cstdio>
#include <csignal>


/////////////////////////////////////////////////////////////////////
// printVersion: print out a version summary.  If the argument is true,
// print out a detailed listing, otherwise a summary.
void IpplInfo::printVersion(void) {
    INFOMSG("IPPL Framework version " << version() << endl);
    INFOMSG("Last build date: " << compileDate() << " by user ");
    INFOMSG(compileUser() << endl);
    INFOMSG("Built for machine: " << compileMachine() << endl);
}


void IpplInfo::printHelp(char** argv) {
    INFOMSG("Usage: " << argv[0] << " [<option> <option> ...]\n");
    INFOMSG("       The possible values for <option> are:\n");
    INFOMSG("   --summary           : Print IPPL lib summary at start.\n");
    INFOMSG("   --time              : Show total time used in execution.\n");
    INFOMSG("   --notime            : Do not show timing info (default).\n");
    INFOMSG("   --info <n>          : Set info message level.  0 = off.\n");
    INFOMSG("   --warn <n>          : Set warning message level.  0 = off.\n");
    INFOMSG("   --error <n>         : Set error message level.  0 = off.\n");
    INFOMSG("   --debug <n>         : Set debug message level.  0 = off.\n");
}

/////////////////////////////////////////////////////////////////////
// version: return the name of this version of Ippl, as a string
// (from Versions.h)
const char *IpplInfo::version() {
    return ippl_version_name;
}



/////////////////////////////////////////////////////////////////////
// compileArch: return the architecture on which this library was built
// (from IpplVersions.h)
const char *IpplInfo::compileArch() {
    return ippl_compile_arch;
}


/////////////////////////////////////////////////////////////////////
// compileDate: return the date on which this library was prepared for
// compilation (from IpplVersions.h)
const char *IpplInfo::compileDate() {
    return ippl_compile_date;
}


/////////////////////////////////////////////////////////////////////
// compileLine: return the compiler command used to compile each source file
// (from IpplVersions.h)
const char *IpplInfo::compileLine() {
    return ippl_compile_line;
}


/////////////////////////////////////////////////////////////////////
// compileMachine: return the machine on which this library was
// compiled (from IpplVersions.h)
const char *IpplInfo::compileMachine() {
    return ippl_compile_machine;
}


/////////////////////////////////////////////////////////////////////
// compileOptions: return the option list used to compile this library
// (from IpplVersions.h)
const char *IpplInfo::compileOptions() {
    return ippl_compile_options;
}


/////////////////////////////////////////////////////////////////////
// compileUser: return the username of the user who compiled this
// library (from IpplVersions.h)
const char *IpplInfo::compileUser() {
    return ippl_compile_user;
}