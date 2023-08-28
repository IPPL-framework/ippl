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

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FDTDSolver.h"
#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::FDTDSolver(Field_t& charge, VField_t& current, VField_t& E,
                                               VField_t& B, double timestep, bool seed_) {
        // set the rho and J fields to be references to charge and current
        // since charge and current deposition will happen at each timestep
        rhoN_mp = &charge;
        JN_mp   = &current;

        // same for E and B fields
        En_mp = &E;
        Bn_mp = &B;

        // initialize the time-step size
        this->dt = timestep;

        // set the seed flag
        this->seed = seed_;

        // call the initialization function
        initialize();
    }

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver(){};

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::solve() {
        // physical constant
        double c        = 1.0;  // 299792458.0;
        double mu0      = 1.0;  // 1.25663706212e-6;
        double epsilon0 = 1.0 / (c * c * mu0);

        // finite differences constants
        double a1 = 2.0
                    * (1.0 - std::pow(c * dt / hr_m[0], 2) - std::pow(c * dt / hr_m[1], 2)
                       - std::pow(c * dt / hr_m[2], 2));
        double a2 = std::pow(c * dt / hr_m[0], 2);  // a3 = a2
        double a4 = std::pow(c * dt / hr_m[1], 2);  // a5 = a4
        double a6 = std::pow(c * dt / hr_m[2], 2);  // a7 = a6
        double a8 = std::pow(c * dt, 2);

        // 1st order absorbing boundary conditions constants
        double beta0[3] = {(c * dt - hr_m[0]) / (c * dt + hr_m[0]),
                           (c * dt - hr_m[1]) / (c * dt + hr_m[1]),
                           (c * dt - hr_m[2]) / (c * dt + hr_m[2])};
        double beta1[3] = {2.0 * dt * hr_m[0] / (c * dt + hr_m[0]),
                           2.0 * dt * hr_m[1] / (c * dt + hr_m[1]),
                           2.0 * dt * hr_m[2] / (c * dt + hr_m[2])};
        double beta2[3] = {-1.0, -1.0, -1.0};

        // preliminaries for Kokkos loops (ghost cells and views)
        auto view_phiN   = phiN_m.getView();
        auto view_phiNm1 = phiNm1_m.getView();
        auto view_phiNp1 = phiNp1_m.getView();

        auto view_aN   = aN_m.getView();
        auto view_aNm1 = aNm1_m.getView();
        auto view_aNp1 = aNp1_m.getView();

        auto view_rhoN = this->rhoN_mp->getView();
        auto view_JN   = this->JN_mp->getView();

        const int nghost_phi = phiN_m.getNghost();
        const int nghost_a   = aN_m.getNghost();
        const auto& ldom     = layout_mp->getLocalNDIndex();

        // compute scalar potential and vector potential at next time-step
        // first, only the interior points are updated
        // then, if the user has set a seed, the seed is added via TF/SF boundaries
        // (TF/SF = total-field/scattered-field technique)
        // finally, absorbing boundary conditions are imposed

        Kokkos::parallel_for(
            "Scalar potential update", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const int ig = i + ldom[0].first() - nghost_phi;
                const int jg = j + ldom[1].first() - nghost_phi;
                const int kg = k + ldom[2].first() - nghost_phi;

                // interior values
                bool isInterior = ((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                   && (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                double interior = -view_phiNm1(i, j, k) + a1 * view_phiN(i, j, k)
                                  + a2 * (view_phiN(i + 1, j, k) + view_phiN(i - 1, j, k))
                                  + a4 * (view_phiN(i, j + 1, k) + view_phiN(i, j - 1, k))
                                  + a6 * (view_phiN(i, j, k + 1) + view_phiN(i, j, k - 1))
                                  + a8 * (-view_rhoN(i, j, k) / epsilon0);

                view_phiNp1(i, j, k) = isInterior * interior;
            });

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_a;
                    const int jg = j + ldom[1].first() - nghost_a;
                    const int kg = k + ldom[2].first() - nghost_a;

                    // interior values
                    bool isInterior = ((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                       && (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                    double interior = -view_aNm1(i, j, k)[gd] + a1 * view_aN(i, j, k)[gd]
                                      + a2 * (view_aN(i + 1, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                                      + a4 * (view_aN(i, j + 1, k)[gd] + view_aN(i, j - 1, k)[gd])
                                      + a6 * (view_aN(i, j, k + 1)[gd] + view_aN(i, j, k - 1)[gd])
                                      + a8 * (-view_JN(i, j, k)[gd] * mu0);

                    view_aNp1(i, j, k)[gd] = isInterior * interior;
                });
        }

        // interior points need to have been updated before TF/SF seed and ABCs
        Kokkos::fence();

        // add seed field via TF/SF boundaries
        if (seed) {
            iteration++;

            // the scattered field boundary is the 2nd point after the boundary
            // the total field boundary is the 3rd point after the boundary
            for (size_t gd = 0; gd < Dim; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first() - nghost_a;
                        const int jg = j + ldom[1].first() - nghost_a;
                        const int kg = k + ldom[2].first() - nghost_a;

                        // SF boundary in all 3 dimensions
                        bool isXmin_SF = ((ig == 1) && (jg > 1) && (kg > 1) && (jg < nr_m[1] - 2)
                                          && (kg < nr_m[2] - 2));
                        double xmin_SF = a2 * gaussian(iteration);
                        bool isYmin_SF = ((ig > 1) && (jg == 1) && (kg > 1) && (ig < nr_m[0] - 2)
                                          && (kg < nr_m[2] - 2));
                        double ymin_SF = a4 * gaussian(iteration);
                        bool isZmin_SF = ((ig > 1) && (jg > 1) && (kg == 1) && (ig < nr_m[0] - 2)
                                          && (jg < nr_m[1] - 2));
                        double zmin_SF = a6 * gaussian(iteration);
                        bool isXmax_SF = ((ig == nr_m[0] - 2) && (jg > 1) && (kg > 1)
                                          && (jg < nr_m[1] - 2) && (kg < nr_m[2] - 2));
                        double xmax_SF = -a2 * gaussian(iteration);
                        bool isYmax_SF = ((ig > 1) && (jg == nr_m[1] - 2) && (kg > 1)
                                          && (ig < nr_m[0] - 2) && (kg < nr_m[2] - 2));
                        double ymax_SF = -a4 * gaussian(iteration);
                        bool isZmax_SF = ((ig > 1) && (jg > 1) && (kg == nr_m[2] - 2)
                                          && (ig < nr_m[0] - 2) && (jg < nr_m[1] - 2));
                        double zmax_SF = -a6 * gaussian(iteration);

                        // TF boundary
                        bool isXmin_TF = ((ig == 2) && (jg > 2) && (kg > 2) && (jg < nr_m[1] - 3)
                                          && (kg < nr_m[2] - 3));
                        double xmin_TF = -a2 * gaussian(iteration);
                        bool isYmin_TF = ((ig > 2) && (jg == 2) && (kg > 2) && (ig < nr_m[0] - 3)
                                          && (kg < nr_m[2] - 3));
                        double ymin_TF = -a4 * gaussian(iteration);
                        bool isZmin_TF = ((ig > 2) && (jg > 2) && (kg == 2) && (ig < nr_m[0] - 3)
                                          && (jg < nr_m[1] - 3));
                        double zmin_TF = -a6 * gaussian(iteration);
                        bool isXmax_TF = ((ig == nr_m[0] - 3) && (jg > 2) && (kg > 2)
                                          && (jg < nr_m[1] - 3) && (kg < nr_m[2] - 3));
                        double xmax_TF = a2 * gaussian(iteration);
                        bool isYmax_TF = ((ig > 2) && (jg == nr_m[1] - 3) && (kg > 2)
                                          && (ig < nr_m[0] - 3) && (kg < nr_m[2] - 3));
                        double ymax_TF = a4 * gaussian(iteration);
                        bool isZmax_TF = ((ig > 2) && (jg > 2) && (kg == nr_m[2] - 3)
                                          && (ig < nr_m[0] - 3) && (jg < nr_m[1] - 3));
                        double zmax_TF = a6 * gaussian(iteration);

                        // update field (add seed)
                        view_aNp1(i, j, k)[gd] +=
                            isXmin_SF * xmin_SF + isYmin_SF * ymin_SF + isZmin_SF * zmin_SF
                            + isXmax_SF * xmax_SF + isYmax_SF * ymax_SF + isZmax_SF * zmax_SF
                            + isXmin_TF * xmin_TF + isYmin_TF * ymin_TF + isZmin_TF * zmin_TF
                            + isXmax_TF * xmax_TF + isYmax_TF * ymax_TF + isZmax_TF * zmax_TF;
                    });
            }
        }
        Kokkos::fence();

        // apply 1st order Absorbing Boundary Conditions
        // for both scalar and vector potentials
        Kokkos::parallel_for(
            "Scalar potential ABCs", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const int ig = i + ldom[0].first() - nghost_phi;
                const int jg = j + ldom[1].first() - nghost_phi;
                const int kg = k + ldom[2].first() - nghost_phi;

                // boundary values: 1st order Absorbing Boundary Conditions
                bool isXmin =
                    ((ig == 0) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                double xmin = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i + 1, j, k))
                              + beta1[0] * (view_phiN(i, j, k) + view_phiN(i + 1, j, k))
                              + beta2[0] * (view_phiNm1(i + 1, j, k));
                bool isYmin =
                    ((ig > 0) && (jg == 0) && (kg > 0) && (ig < nr_m[0] - 1) && (kg < nr_m[2] - 1));
                double ymin = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j + 1, k))
                              + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j + 1, k))
                              + beta2[1] * (view_phiNm1(i, j + 1, k));
                bool isZmin =
                    ((ig > 0) && (jg > 0) && (kg == 0) && (ig < nr_m[0] - 1) && (jg < nr_m[1] - 1));
                double zmin = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k + 1))
                              + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k + 1))
                              + beta2[2] * (view_phiNm1(i, j, k + 1));
                bool isXmax = ((ig == nr_m[0] - 1) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                               && (kg < nr_m[2] - 1));
                double xmax = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i - 1, j, k))
                              + beta1[0] * (view_phiN(i, j, k) + view_phiN(i - 1, j, k))
                              + beta2[0] * (view_phiNm1(i - 1, j, k));
                bool isYmax = ((ig > 0) && (jg == nr_m[1] - 1) && (kg > 0) && (ig < nr_m[0] - 1)
                               && (kg < nr_m[2] - 1));
                double ymax = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j - 1, k))
                              + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j - 1, k))
                              + beta2[1] * (view_phiNm1(i, j - 1, k));
                bool isZmax = ((ig > 0) && (jg > 0) && (kg == nr_m[2] - 1) && (ig < nr_m[0] - 1)
                               && (jg < nr_m[1] - 1));
                double zmax = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k - 1))
                              + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k - 1))
                              + beta2[2] * (view_phiNm1(i, j, k - 1));

                view_phiNp1(i, j, k) += isXmin * xmin + isYmin * ymin + isZmin * zmin
                                        + isXmax * xmax + isYmax * ymax + isZmax * zmax;
            });

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential ABCs", ippl::getRangePolicy(view_aN, nghost_a),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_a;
                    const int jg = j + ldom[1].first() - nghost_a;
                    const int kg = k + ldom[2].first() - nghost_a;

                    // boundary values: 1st order Absorbing Boundary Conditions
                    bool isXmin = ((ig == 0) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                                   && (kg < nr_m[2] - 1));
                    double xmin = beta0[0] * (view_aNm1(i, j, k)[gd] + view_aNp1(i + 1, j, k)[gd])
                                  + beta1[0] * (view_aN(i, j, k)[gd] + view_aN(i + 1, j, k)[gd])
                                  + beta2[0] * (view_aNm1(i + 1, j, k)[gd]);
                    bool isYmin = ((ig > 0) && (jg == 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                   && (kg < nr_m[2] - 1));
                    double ymin = beta0[1] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j + 1, k)[gd])
                                  + beta1[1] * (view_aN(i, j, k)[gd] + view_aN(i, j + 1, k)[gd])
                                  + beta2[1] * (view_aNm1(i, j + 1, k)[gd]);
                    bool isZmin = ((ig > 0) && (jg > 0) && (kg == 0) && (ig < nr_m[0] - 1)
                                   && (jg < nr_m[1] - 1));
                    double zmin = beta0[2] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j, k + 1)[gd])
                                  + beta1[2] * (view_aN(i, j, k)[gd] + view_aN(i, j, k + 1)[gd])
                                  + beta2[2] * (view_aNm1(i, j, k + 1)[gd]);
                    bool isXmax = ((ig == nr_m[0] - 1) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                                   && (kg < nr_m[2] - 1));
                    double xmax = beta0[0] * (view_aNm1(i, j, k)[gd] + view_aNp1(i - 1, j, k)[gd])
                                  + beta1[0] * (view_aN(i, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                                  + beta2[0] * (view_aNm1(i - 1, j, k)[gd]);
                    bool isYmax = ((ig > 0) && (jg == nr_m[1] - 1) && (kg > 0) && (ig < nr_m[0] - 1)
                                   && (kg < nr_m[2] - 1));
                    double ymax = beta0[1] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j - 1, k)[gd])
                                  + beta1[1] * (view_aN(i, j, k)[gd] + view_aN(i, j - 1, k)[gd])
                                  + beta2[1] * (view_aNm1(i, j - 1, k)[gd]);
                    bool isZmax = ((ig > 0) && (jg > 0) && (kg == nr_m[2] - 1) && (ig < nr_m[0] - 1)
                                   && (jg < nr_m[1] - 1));
                    double zmax = beta0[2] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j, k - 1)[gd])
                                  + beta1[2] * (view_aN(i, j, k)[gd] + view_aN(i, j, k - 1)[gd])
                                  + beta2[2] * (view_aNm1(i, j, k - 1)[gd]);

                    view_aNp1(i, j, k)[gd] += isXmin * xmin + isYmin * ymin + isZmin * zmin
                                              + isXmax * xmax + isYmax * ymax + isZmax * zmax;
                });
        }
        Kokkos::fence();

        // evaluate E and B fields at N
        field_evaluation();

        // store potentials at N in Nm1, and Np1 in N
        Kokkos::deep_copy(aNm1_m.getView(), aN_m.getView());
        Kokkos::deep_copy(aN_m.getView(), aNp1_m.getView());
        Kokkos::deep_copy(phiNm1_m.getView(), phiN_m.getView());
        Kokkos::deep_copy(phiN_m.getView(), phiNp1_m.getView());
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::field_evaluation() {
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        (*Bn_mp) = 0.5 * (curl(aN_m) + curl(aNp1_m));

        // electric field is the time derivative of the vector potential
        // minus the gradient of the scalar potential
        (*En_mp) = -(aNp1_m - aN_m) / dt - grad(phiN_m);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    double FDTDSolver<Tfields, Dim, M, C>::gaussian(size_t it) {
        double arg = Kokkos::pow((it * dt) / 0.1, 2);
        return 100 * Kokkos::exp(-arg);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::initialize() {
        // get layout and mesh
        layout_mp = &(this->rhoN_mp->getLayout());
        mesh_mp   = &(this->rhoN_mp->get_mesh());

        // get mesh spacing, domain, and mesh size
        hr_m     = mesh_mp->getMeshSpacing();
        domain_m = layout_mp->getDomain();
        for (unsigned int i = 0; i < Dim; ++i)
            nr_m[i] = domain_m[i].length();

        // initialize fields
        phiNm1_m.initialize(*mesh_mp, *layout_mp);
        phiN_m.initialize(*mesh_mp, *layout_mp);
        phiNp1_m.initialize(*mesh_mp, *layout_mp);

        aNm1_m.initialize(*mesh_mp, *layout_mp);
        aN_m.initialize(*mesh_mp, *layout_mp);
        aNp1_m.initialize(*mesh_mp, *layout_mp);

        phiNm1_m = 0.0;
        phiN_m   = 0.0;
        phiNp1_m = 0.0;

        aNm1_m = 0.0;
        aN_m   = 0.0;
        aNp1_m = 0.0;
    };
}  // namespace ippl
