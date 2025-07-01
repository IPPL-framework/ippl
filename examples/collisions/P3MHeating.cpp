/***************************************************************************************************
 *                                       P3MHeating
 * =================================================================================================
 * This program tests the P3M Method, applied to the Disorder-Induced-Heating problem.
 * The parameters were chosen according to B.Ulmers Thesis -
 * "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
 * (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
 *
 * This program and corresponding methods were implemented as part of a Bachelors Thesis
 * "A Performance Portable Version of the P3M Algorithm"
 * By Timo Schwab, ETH Zurich (2024)
 *
 * Usage: srun ./P3MHeating <nx> <ny> <nz> <factor> --info 10
 *  nx      =   No. PM grid points in x-direction
 *  ny      =   No. PM grid points in y-direction
 *  nz      =   No. PM grid points in z-direction
 *  factor  =   factor to multiply the grid size in x direction to get the cutoff radius
 */

constexpr unsigned Dim = 3;
using T                = double;

#include "Ippl.h"

#include "P3MHeating.hpp"

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "P3MParticleContainer.hpp"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const T boxlen         = 0.01;
        const T beam_rad       = 0.001774;
        const unsigned np      = 156055;
        const T dt             = 2.15623e-13;
        const unsigned nt      = 1000;
        const T focus_strength = 1.5;

        int arg = 1;
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const unsigned factor = std::atoi(argv[arg]);

        const T rcut  = factor * (boxlen / static_cast<T>(nr[0]));
        const T alpha = 2. / rcut;

        P3MHeatingManager<T, Dim> manager(np, nt, dt, nr, rcut, alpha, beam_rad, focus_strength,
                                          boxlen);
        manager.pre_run();

        manager.setTime(0.0);

        msg << "Starting iterations ..." << endl;

        static IpplTimings::TimerRef simulationTimer =
            IpplTimings::getTimer("Total Simulation Time");
        IpplTimings::startTimer(simulationTimer);
        manager.run(manager.getNt());
        IpplTimings::stopTimer(simulationTimer);
        IpplTimings::print();
    }
    ippl::finalize();

    return 0;
}
