//
// Class FDTDSolver
//   Finite Differences Time Domain electromagnetic solver.
//
// Copyright (c) 2022, Sonali Mayani, PSI, Villigen, Switzerland
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

#ifndef FDTD_SOLVER_H_
#define FDTD_SOLVER_H_

#include "FDTDSolver.h"
#include "Field/Field.h"
#include "FieldLayout/FieldLayout.h"
#include "Types/Vector.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename Tfields, unsigned Dim, class M, class C>
    class FDTDSolver<Tfields, Dim, M, C>::FDTDSolver(Field_t charge, VField_t current) {

        // set the rho and J fields to be references to charge and current
        // since charge and current deposition will happen at each timestep
        rhoN_mp = &charge;
        JN_mp = &current;
    
        // call the initialization function
        initialize();

    }

    template <typename Tfields, unsigned Dim, class M, class C>
    class FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver() {};

    template <typename Tfields, unsigned Dim, class M, class C>
    class FDTDSolver<Tfields, Dim, M, C>::solve() { 

        // define some constants
        double a1 = 2.0 * (1.0 - pow(c*dt/hr_m[0], 2) - pow(c*dt/hr_m[1], 2) - pow(c*dt/hr_m[2], 2));
        double a2 = pow(c*dt/hr_m[0], 2); // a3 = a2
        double a4 = pow(c*dt/hr_m[1], 2); // a5 = a4
        double a6 = pow(c*dt/hr_m[2], 2); // a7 = a6
        double a8 = pow(c*dt, 2);

        // preliminaries for Kokkos loops (ghost cells and views)
        auto view_phiN = phiN_m.getView();
        auto view_phiNm1 = phiNm1_m.getView();
        auto view_phiNp1 = phiNp1_m.getView();

        auto view_aN = aN_m.getView();
        auto view_aNm1 = aNm1_m.getView();
        auto view_aNp1 = aNp1_m.getView();

        auto view_rhoN = this->rhoN_mp->getView();
        auto view_JN = this->JN_mp->getView();

        const int nghost_phi = phiN_m.getNghost();
        const int nghost_a = aN_m.getNghost();


        // compute scalar potential at next time-step using Finite Differences
        Kokkos::parallel_for("Scalar potential update",
                mdrange_type({nghost_phi, nghost_phi, nghost_phi},
                {view_phiN.extent(0)-nghost_phi, view_phiN.extent(1)-nghost_phi, view_phiN.extent(2)-nghost_phi}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                view_phiNp1(i,j,k) = -view_phiNm1(i,j,k) + a1*view_phiN(i,j,k) +
                                     a2*(view_phiN(i+1,j,k) + view_phiN(i-1,j,k)) +
                                     a4*(view_phiN(i,j+1,k) + view_phiN(i,j-1,k)) +
                                     a6*(view_phiN(i,j,k+1) + view_phiN(i,j,k-1)) +
                                     a8*(- view_rhoN(i,j,k) / epsilon0);
        });


        // compute vector potential at next time-step
        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for("Vector potential update",
                    mdrange_type({nghost_a, nghost_a, nghost_a},
                    {view_aN.extent(0)-nghost_a, view_aN.extent(1)-nghost_a, view_aN.extent(2)-nghost_a}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        view_aNp1(i,j,k)[gd] = -view_aNm1(i,j,k)[gd] + a1*view_aN(i,j,k)[gd] +
                                             a2*(view_aN(i+1,j,k)[gd] + view_aN(i-1,j,k)[gd]) +
                                             a4*(view_aN(i,j+1,k)[gd] + view_aN(i,j-1,k)[gd]) +
                                             a6*(view_aN(i,j,k+1[gd]) + view_aN(i,j,k-1)[gd]) +
                                             a8*(- view_JN(i,j,k)[gd] * mu0);
            });
        }

        // evaluate E and B fields at N
        field_evaluation();

        // store potentials at N in Nm1, and Np1 in N
        aNm1_m = aN_m;
        aN_m = aNp1_m;
        phiNm1_m = phiN_m;
        phiN_m = phiNp1_m;

    };

    template <typename Tfields, unsigned Dim, class M, class C>
    class FDTDSolver<Tfields, Dim, M, C>::field_evaluation() { 
        
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        Bn_m = 0.5 * (curl(aN_m) + curl(aNp1_m));

        // electric field is the time derivative of the vector potential
        // minus the gradient of the scalar potential
        En_m = -(aNp1_m - aN_m)/dt - grad(phiN_m);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    class FDTDSolver<Tfields, Dim, M, C>::initialize() { 
        //
        //
    };
}

#endif
