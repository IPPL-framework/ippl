//
// Class IpplInfo
//   Global Ippl information.
//
#include "Ippl.h"
#include "IpplVersions.h"

#include "Utility/IpplInfo.h"

#include <csignal>
#include <cstdio>
#include <unistd.h>

/////////////////////////////////////////////////////////////////////
// printVersion: print out a version summary.  If the argument is true,
// print out a detailed listing, otherwise a summary.
void IpplInfo::printVersion(void) {
    std::cout << "IPPL Framework version " << version() << std::endl;
    std::cout << "Last build date: " << compileDate() << " by user ";
    std::cout << compileUser() << std::endl;
    std::cout << "Built for machine: " << compileMachine() << std::endl;
}

void IpplInfo::printHelp(char** argv) {
    std::cout << "Usage: " << argv[0] << " [<option> ...]\n";
    std::cout << "The possible values for <option> are:\n";
    std::cout << "   --info <n>                  : Set info message level.  0 = off.\n";
    std::cout << "   --overallocate|-b <factor>  : Set the buffer overallocation factor\n";
    std::cout << "   --timer-fences <on|off>     : Enable or disable timer fences (default enabled "
                 "if only "
                 "one accelerator present)\n";
    std::cout << "   --help                      : Print IPPL help message\n";
    std::cout << "   --kokkos-help               : Print Kokkos help message\n";
}

/////////////////////////////////////////////////////////////////////
// version: return the name of this version of Ippl, as a string
// (from Versions.h)
const char* IpplInfo::version() {
    return ippl_version_name;
}

/////////////////////////////////////////////////////////////////////
// compileArch: return the architecture on which this library was built
// (from IpplVersions.h)
const char* IpplInfo::compileArch() {
    return ippl_compile_arch;
}

/////////////////////////////////////////////////////////////////////
// compileDate: return the date on which this library was prepared for
// compilation (from IpplVersions.h)
const char* IpplInfo::compileDate() {
    return ippl_compile_date;
}

/////////////////////////////////////////////////////////////////////
// compileLine: return the compiler command used to compile each source file
// (from IpplVersions.h)
const char* IpplInfo::compileLine() {
    return ippl_compile_line;
}

/////////////////////////////////////////////////////////////////////
// compileMachine: return the machine on which this library was
// compiled (from IpplVersions.h)
const char* IpplInfo::compileMachine() {
    return ippl_compile_machine;
}

/////////////////////////////////////////////////////////////////////
// compileOptions: return the option list used to compile this library
// (from IpplVersions.h)
const char* IpplInfo::compileOptions() {
    return ippl_compile_options;
}

/////////////////////////////////////////////////////////////////////
// compileUser: return the username of the user who compiled this
// library (from IpplVersions.h)
const char* IpplInfo::compileUser() {
    return ippl_compile_user;
}
