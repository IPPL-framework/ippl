//
// Copyright (c) 2022, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

//#include "ChargedParticlesPinT.hpp"

void LeapFrogPIF(ChargedParticlesPinT<PLayout_t>& P, ParticleAttrib<Vector_t>& Rtemp,
                 ParticleAttrib<Vector_t>& Ptemp, const unsigned int& nt, 
                 const double& dt, const bool& isConverged, 
                 const double& tStartMySlice) {

    auto& PL = P.getLayout();
    const auto& rmax = P.rmax_m;
    const auto& rmin = P.rmin_m;

    P.time_m = tStartMySlice;

    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        // kick

        Ptemp = Ptemp - 0.5 * dt * P.E;

        //drift
        Rtemp = Rtemp + dt * Ptemp;

        //Apply particle BC
        PL.applyBC(Rtemp, PL.getRegionLayout().getDomain());

        //scatter the charge onto the underlying grid
        P.rhoPIF_m = {0.0, 0.0};
        scatterPIF(P.q, P.rhoPIF_m, Rtemp);

        P.rhoPIF_m = P.rhoPIF_m / ((rmax[0] - rmin[0]) * (rmax[1] - rmin[1]) * (rmax[2] - rmin[2]));

        // Solve for and gather E field
        gatherPIF(P.E, P.rhoPIF_m, Rtemp);

        //kick
        Ptemp = Ptemp - 0.5 * dt * P.E;

        P.time_m += dt;
        if(isConverged) {
            P.dumpLandau(P.getLocalNum());         
            P.dumpEnergy(P.getLocalNum());         
        }

    }
}