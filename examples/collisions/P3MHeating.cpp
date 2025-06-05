/******************************************************************************************************
 *                                          TestP3MHeating
 * ====================================================================================================
 * This program tests the P3M Method, applied to the Disorder-Induced-Heating problem.
 * The parameters were chosen according to B.Ulmers Thesis -
 * "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
 * (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
 * 
 * This program and corresponding methods were implemented as part of a Bachelors Thesis
 * "A Performance Portable Version of the P3M Algorithm"
 * By Timo Schwab, ETH Zurich (2024)
 * 
 * Usage: srun ./TestP3MHeating <nx> <ny> <nz>
 *  nx  =   No. PM grid points in x-direction
 *  ny  =   No. PM grid points in y-direction
 *  nz  =   No. PM grid points in z-direction
 * 
 * CURRENTLY WORK IN PROGRESS !!!
*/



constexpr unsigned Dim = 3;
using T = double;

#include "Ippl.h"
#include "datatypes.h"
#include "P3MHeatingManager.hpp"
#include "P3MParticleContainer.hpp"

#include "Utility/IpplTimings.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef preTimer = IpplTimings::getTimer("pre run");
        IpplTimings::startTimer(mainTimer);

        // TODO: Change to input
        const double beam_rad       = 0.001774;
        const double boxlen         = 0.01;
        const unsigned int np       = 156055;
        // const double rcut           = 0.0003125;    // 8 * PM grid spacing
        // const double alpha          = 2./rcut;      // choice motivated by B. Ulmer
        const double dt             = 2.15623e-13;
        const double eps            = 0;
        const unsigned int nt       = 1000;
        const double m_e            = 1;
        const double q_e            = 1;
        const double focus_strength = 1.5;

        int arg = 1;
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const double rcut = 4.0 * (boxlen/(double)nr[0]);
        const double alpha = 2./rcut;

        P3M3DHeatingManager<T, Dim> manager(np, nt, dt, nr, rcut, alpha, beam_rad, focus_strength);
        IpplTimings::startTimer(preTimer);
        manager.pre_run();
        IpplTimings::stopTimer(preTimer);

        // set time to 0
        manager.setTime(0.0);

        msg << "Starting iterations ..." << endl;

        manager.run(manager.getNt());

        msg << "End." << endl;
	    IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));

    }
    ippl::finalize();

    return 0;
}
