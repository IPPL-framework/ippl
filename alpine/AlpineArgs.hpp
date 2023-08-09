// ALPINE Argument Parser
//   Argument parsing utilities for ALPINE
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include "Ippl.h"

#include <getopt.h>

#include "Types/IpplTypes.h"

template <unsigned Dim>
struct SimulationParameters {
    using size_type = ippl::detail::size_type;

    ippl::Vector<int, Dim> meshRefinement;
    size_type particleCount{};
    unsigned int timeSteps{};
    std::string solver;
    double lbThreshold = 1;

    SimulationParameters()
        : meshRefinement(0)
        , solver("FFT") {}

    void setPPC(uint64_t ppc) {
        size_type cells =
            std::reduce(meshRefinement.begin(), meshRefinement.end(), 1, std::multiplies<>());
        size_type np = ppc * cells;
        if (ppc == 0) {
            *ippl::Warn << "Particle density specified as zero." << endl;
        } else if (np == 0) {
            *ippl::Warn << "Particle density was specified for a mesh with zero cells. Note that "
                           "arguments are parsed in their given order, so mesh refinement must be "
                           "specified before particle density for correct results."
                        << endl;
        }
        particleCount = np;
    }
};

void printArgHelp() {
    std::cout << "ALPINE command line options:\n"
              << "\t--Nx/--Ny/--Nz/--N4/... <int>: sets the mesh refinement along the given axis\n"
              << "\t-N/--Nall <int>: sets the mesh refinement along all axes\n"
              << "\t-p/--particles <int>: sets the particle count\n"
              << "\t-d/--ppc <int>: sets the number of particles per cell (mesh refinement needs "
                 "to be defined first)\n"
              << "\t-s/--solver <name>: set the solver (CG, FFT, OPEN, P3M)\n"
              << "\t-t/--timesteps <int>: set the number of timesteps\n"
              << "\t-l/--lb <real>: sets the load balancing threshold\n"
              << "\t-h/--help: shows this help message" << std::endl;
}

/*!
 * Parses the command line arguments to set up the simulation
 * @param argc number of arguments
 * @param argv argument vector
 * @param params parameter container
 * @return Whether the program should exit immediately
 */
template <unsigned Dim>
bool parseArgs(int argc, char* argv[], SimulationParameters<Dim>& params) {
    // Prevent getopt from warning about non-ALPINE flags
    opterr = 0;

    while (true) {
        // Number of arguments in addition to Nx, Ny, etc
        static constexpr unsigned ARG_COUNT                = 7;
        static const char* shortOpts                       = "hN:p:d:s:t:l:";
        static struct option longOpts[ARG_COUNT + Dim + 1] = {
            {"help", no_argument, NULL, 'h'},
            {"Nall", required_argument, NULL, 'N'},
            {"particles", required_argument, NULL, 'p'},
            {"ppc", required_argument, NULL, 'd'},
            {"solver", required_argument, NULL, 's'},
            {"timesteps", required_argument, NULL, 't'},
            {"lb", required_argument, NULL, 'l'}};
        for (unsigned i = 0; i < Dim; i++) {
            char name[4];
            sprintf(name, "N%u", i + 1);
            longOpts[i + ARG_COUNT] = {name, required_argument, NULL, static_cast<int>(i + 1)};
        }
        // Option list must be terminated with a struct of all zeros
        longOpts[ARG_COUNT + Dim] = {0, 0, 0, 0};

        int idx = 0;
        int opt = getopt_long(argc, argv, shortOpts, longOpts, &idx);

        // No more options
        if (opt == -1) {
            break;
        }

        switch (opt) {
            default:
                // Check for mesh refinement in a single direction with 1-indexed axes
                if (1 <= opt && opt <= static_cast<int>(Dim)) {
                    params.meshRefinement[opt] = std::atoi(optarg);
                } else {
                    // Parsing failed
                    ippl::abort();
                }
                break;
            case 'N':
                params.meshRefinement = std::atoi(optarg);
                break;
            case 'p':
                params.particleCount = std::atoll(optarg);
                break;
            case 'd':
                params.setPPC(std::atoll(optarg));
                break;
            case 's':
                params.solver = optarg;
                break;
            case 't':
                params.timeSteps = std::atoi(optarg);
                break;
            case 'l':
                params.lbThreshold = std::atof(optarg);
                break;
            case 'h':
                printArgHelp();
                return true;
            case '?':
                break;
        }
    }
    return false;
}