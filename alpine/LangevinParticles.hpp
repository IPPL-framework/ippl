#ifndef LANGEVINPARTICLES_HPP
#define LANGEVINPARTICLES_HPP

// Langevin header file
//   Defines a particle attribute for particles existing in velocity space
//   This is used when discretising the velocity distribution of a Langevin
//   type equation with the Cloud-In-Cell (CIC) Method
//   Assumption on the Particle's attributes:
//   - single species particles (same mass and charge)
//   It computes Rosenbluth Potentials which are needed to evolve the
//   velocity particles in velocity space
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <Kokkos_Random.hpp>

#include "Utility/IpplException.h"

#include "LangevinHelpers.hpp"
#include "Solver/FFTPoissonSolver.h"

typedef Field<MatrixD_t, Dim> MField_t;

template <class PLayout, unsigned Dim = 3>
class LangevinParticles : public ChargedParticles<PLayout> {
    using Base = ChargedParticles<PLayout, Dim>;
    using FrictionSolver_t =
        ippl::FFTPoissonSolver<VectorD_t, double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;
    using DiffusionSolver_t =
        ippl::FFTPoissonSolver<VectorD_t, double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

    using KokkosRNG_t = Kokkos::Random_XorShift64_Pool<>;

    // View types (of particle attributes)
    typedef ParticleAttrib<double>::view_type attr_view_t;      // Scalar particle attributes
    typedef ParticleAttrib<VectorD_t>::view_type attr_Dview_t;  // D-dimensional particle attributes
    typedef ParticleAttrib<double>::HostMirror attr_mirror_t;   // Scalar particle attributes
    typedef ParticleAttrib<VectorD_t>::HostMirror
        attr_Dmirror_t;  // D-dimensional particle attributes

    // View types (of Fields)
    typedef ippl::detail::ViewType<double, Dim>::view_type field_view_t;  // Scalar Fields
    typedef ippl::detail::ViewType<VectorD_t, Dim>::view_type
        field_Dview_t;  // D-dimensional Fields

public:
    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    LangevinParticles(PLayout& pl)
        : ChargedParticles<PLayout>(pl) {
        // registerLangevinAttributes();
        setPotentialBCs();
    }

    LangevinParticles(PLayout& pl, VectorD_t hr, VectorD_t rmin, VectorD_t rmax,
                      ippl::e_dim_tag configSpaceDecomp[Dim], std::string solver, double pCharge,
                      double pMass, double epsInv, double Q, size_t globalNumParticles, double dt,
                      size_t nv, double vmax)
        : ChargedParticles<PLayout>(pl, hr, rmin, rmax, configSpaceDecomp, Q, solver)
        , pCharge_m(pCharge)
        , pMass_m(pMass)
        , epsInv_m(epsInv)
        , globParticleNum_m(globalNumParticles)
        , dt_m(dt)
        , nv_m(nv)
        , hv_m(2*vmax / nv)
        , vmin_m(-vmax)
        , vmax_m(vmax)
        , velocitySpaceIdxDomain_m(nv,nv,nv)
        , velocitySpaceMesh_m(velocitySpaceIdxDomain_m, hv_m, 0.0)
        , velocitySpaceFieldLayout_m(velocitySpaceIdxDomain_m, configSpaceDecomp, false) {
        // Compute $\Gamma$ prefactor for Friction and Diffusion coefficients
        double q4          = pCharge_m * pCharge_m * pCharge_m * pCharge_m;
        double eps2Inv     = epsInv_m * epsInv_m;
        double m2_e        = pMass_m * pMass_m;
        double coulomb_log = 10.0;
        gamma_m            = coulomb_log * q4 * eps2Inv / (4.0 * pi * m2_e);

        setupVelocitySpace();
        registerLangevinAttributes();
    }

    void setPotentialBCs() {
        // CG requires explicit periodic boundary conditions while the periodic Poisson solver
        // simply assumes them
        // TODO Set if periodic boundaries are needed (see `ChargedParticles.hpp` for an example)
    }

    void registerLangevinAttributes() {
        // Register particle attributes used to compute Langevin term
        // TODO
        this->addAttribute(p_fv_m);
        this->addAttribute(p_F_m);
    }

    void setupVelocitySpace() {
        Inform msg("setupVelocitySpace");
        fv_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        F_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        msg << "Initialized Velocity Space" << endl;
    }

    void initAllSolvers(std::string frictionSolverName) {
        initSpaceChargeSolver();
        initFrictionSolver(frictionSolverName);
    }

    void initSpaceChargeSolver() {
        // Initializing the solver defined by the ChargedParticles Class
        ChargedParticles<PLayout>::initFFTSolver();
    }

    // Setup Friction Solver ["HOCKNEY", "VICO"]
    void initFrictionSolver(std::string solverName) {
        // Initializing the open-boundary solver for the H potential used
        // to compute the friction coefficient $F$
        // [Hockney Open Poisson Solver]

        ippl::ParameterList sp;
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        frictionSolver_mp =
            std::make_shared<FrictionSolver_t>(F_m, fv_m, sp, solverName, FrictionSolver_t::GRAD);
    }

    size_type getGlobParticleNum() const { return globParticleNum_m; }

    VectorD_t compAvgSCForce(double beamRadius) {
        Inform m("computeAvgSpaceChargeForces ");

        VectorD_t avgEF;
        double locEFsum[Dim] = {};
        double globEFsum[Dim];
        double locQ, globQ;
        double locCheck, globCheck;

        attr_view_t pq_view  = this->q.getView();
        attr_Dview_t pE_view = this->E.getView();
        attr_Dview_t pR_view = this->R.getView();

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local EField sum", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& lefsum) { lefsum += fabs(pE_view(i)[d]); },
                Kokkos::Sum<double>(locEFsum[d]));
        }
        Kokkos::parallel_reduce(
            "check charge", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& qsum) { qsum += pq_view[i]; },
            Kokkos::Sum<double>(locQ));
        Kokkos::parallel_reduce(
            "check  positioning", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& check) {
                check += int(L2Norm(pR_view[i]) <= beamRadius);
            },
            Kokkos::Sum<double>(locCheck));

        Kokkos::fence();
        MPI_Allreduce(locEFsum, globEFsum, Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&locQ, &globQ, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&locCheck, &globCheck, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        for (unsigned d = 0; d < Dim; ++d) {
            avgEF[d] = globEFsum[d] / this->getGlobParticleNum();
        }

        m << "Position Check = " << globCheck << endl;
        m << "globQ = " << globQ << endl;
        m << "globSumEF = " << globEFsum[0] << " " << globEFsum[1] << " " << globEFsum[2] << endl;
        m << "AVG Electric SC Force = " << avgEF << endl;

        return avgEF;
    }

    void runSolver() { throw IpplException("LangevinParticles::runSolver", "Not implemented. Run `runSpaceChargeSolver` or `runFrictionSolver` instead."); }

    void runSpaceChargeSolver() {
        // Multiply by inverse vacuum permittivity
        // (due to differential formulation of Green's function)
        this->rho_m = this->rho_m * epsInv_m;
        // Call SpaceCharge Solver of `ChargeParticles.hpp`
        Base::runSolver();
    }
    
    void velocityParticleCheck(){
        Inform msg("velocityParticleStats");
        attr_Dview_t pP_view = this->P.getView();

        double avgVel = 0.0;
        double minVelComponent = std::numeric_limits<double>::max();
        double maxVelComponent = std::numeric_limits<double>::min();
        double minVel = std::numeric_limits<double>::max();
        double maxVel = std::numeric_limits<double>::min();

        Kokkos::parallel_reduce(
            "check charge", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& loc_avgVel,
                          double& loc_minVelComponent, double& loc_maxVelComponent,
                          double& loc_minVel, double& loc_maxVel) {
                double velNorm = L2Norm(pP_view[i]);
                loc_avgVel += velNorm;

                loc_minVelComponent = pP_view[i][0] < loc_minVelComponent ? velNorm : loc_minVelComponent;
                loc_minVelComponent = pP_view[i][1] < loc_minVelComponent ? velNorm : loc_minVelComponent;
                loc_minVelComponent = pP_view[i][2] < loc_minVelComponent ? velNorm : loc_minVelComponent;

                loc_maxVelComponent = pP_view[i][0] > loc_maxVelComponent ? velNorm : loc_maxVelComponent;
                loc_maxVelComponent = pP_view[i][1] > loc_maxVelComponent ? velNorm : loc_maxVelComponent;
                loc_maxVelComponent = pP_view[i][2] > loc_maxVelComponent ? velNorm : loc_maxVelComponent;

                loc_minVel = velNorm < loc_minVel ? velNorm : loc_minVel;
                loc_maxVel = velNorm > loc_maxVel ? velNorm : loc_maxVel;
            },
            Kokkos::Sum<double>(avgVel),
            Kokkos::Min<double>(minVelComponent),
            Kokkos::Max<double>(maxVelComponent),
            Kokkos::Min<double>(minVel),
            Kokkos::Max<double>(maxVel));
        avgVel /= this->getGlobParticleNum();
        msg << "avgVel = " << avgVel << endl;
        msg << "minVelComponent = " << minVelComponent << endl;
        msg << "maxVelComponent = " << maxVelComponent << endl;
        msg << "minVel = " << minVel << endl;
        msg << "maxVel = " << maxVel << endl;
    }

    void runFrictionSolver() {
        Inform msg("runFrictionSolver");

        velocityParticleCheck();

        // Scatter velocity density on grid
        fv_m = 0.0;
        // Scattered quantity should be a density ($\sum_i fv_i = 1$)
        p_fv_m = 1.0 / globParticleNum_m;
        scatter(p_fv_m, fv_m, this->P);
        // Normalize with dV
        double cellVolume =
            std::reduce(hv_m.begin(), hv_m.end(), 1., std::multiplies<double>());
        fv_m = fv_m / cellVolume;

        // Multiply with prefactors defined in RHS of Rosenbluth equations
        // FFTPoissonSolver already returns $- \nabla H(\vec v)$, so `-` was omitted here
        fv_m = 8.0 * pi_m * gamma_m * fv_m;

        // Set origin of velocity space mesh to zero (for FFT)
        velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\nabla H(\vec v)$, is stored in `F_m`
        frictionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        velocitySpaceMesh_m.setOrigin(vmin_m);

        // Gather Friction coefficients to particles attribute
        gather(p_F_m, F_m, this->P);
        msg << "Friction computation done." << endl;
    }

    void applyConstantFocusing(const double focus_fct, const double beamRadius,
                               const VectorD_t avgEF) {
        attr_Dview_t pE_view  = this->E.getView();
        attr_Dview_t pR_view  = this->R.getView();
        double focusingEnergy = L2Norm(avgEF) * focus_fct / beamRadius;
        Kokkos::parallel_for(
            "Apply Constant Focusing", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i) { pE_view(i) += pR_view(i) * focusingEnergy; });
        Kokkos::fence();
    }

    void dumpBeamStatistics(unsigned int iteration, std::string folder) {
        Inform m("DUMPLangevin");

        // Usefull constants
        const size_t pN_glob = getGlobParticleNum();
        const double c_inv   = 1.0 / c_m;
        const double c2_inv  = c_inv * c_inv;

        // Views/Mirrors for Particle Attributes
        // attr_view_t  pq_view = this->q.getView();
        attr_Dview_t pR_view = this->R.getView();
        attr_Dview_t pP_view = this->P.getView();
        attr_Dview_t pE_view = this->E.getView();

        // attr_mirror_t  pq_mirror = this->q.getHostMirror();
        // attr_Dmirror_t pE_mirror = this->E.getHostMirror();
        // attr_Dmirror_t pR_mirror = this->R.getHostMirror();

        // Kokkos::deep_copy(pE_mirror, pE_view);
        // Kokkos::deep_copy(pR_mirror, pR_view);

        // Views/Mirrors for Fields
        field_Dview_t E_view = this->E_m.getView();

        ////////////////////////////////////////////////////////////
        // Gather E-field in x-direction (via particle attribute) //
        ////////////////////////////////////////////////////////////

        // Compute the Avg E-Field over the particle Attribute
        VectorD_t avgEF_particle;
        double locEFsum[Dim] = {};
        double globEFsum[Dim];

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local EField sum", static_cast<size_t>(this->getLocalNum()),
                KOKKOS_LAMBDA(const int i, double& lefsum) { lefsum += std::fabs(pE_view(i)[d]); },
                Kokkos::Sum<double>(locEFsum[d]));
        }

        Kokkos::fence();
        MPI_Allreduce(locEFsum, globEFsum, Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        for (unsigned d = 0; d < Dim; ++d)
            avgEF_particle[d] = globEFsum[d] / pN_glob;

        /////////////////////////////////////////////////////////////
        // Gather E-field in x-direction (via Field datastructure) //
        /////////////////////////////////////////////////////////////

        const int nghostE = this->E_m.getNghost();
        double ExAmp;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        double temp = 0.0;
        Kokkos::parallel_reduce(
            "Ex inner product",
            mdrange_type({nghostE, nghostE, nghostE},
                         {E_view.extent(0) - nghostE, E_view.extent(1) - nghostE,
                          E_view.extent(2) - nghostE}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                double myVal = std::pow(E_view(i, j, k)[0], 2);
                valL += myVal;
            },
            Kokkos::Sum<double>(temp));
        Kokkos::fence();
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        double fieldEnergy = globaltemp * this->hr_m[0] * this->hr_m[1] * this->hr_m[2];

        /////////////////////////////////////////////
        // Gather E-Field amplitude in x-direction //
        /////////////////////////////////////////////

        double tempMax = 0.0;
        Kokkos::parallel_reduce(
            "Ex max norm",
            mdrange_type({nghostE, nghostE, nghostE},
                         {E_view.extent(0) - nghostE, E_view.extent(1) - nghostE,
                          E_view.extent(2) - nghostE}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& valL) {
                double myVal = std::fabs(E_view(i, j, k)[0]);
                if (myVal > valL)
                    valL = myVal;
            },
            Kokkos::Max<double>(tempMax));
        Kokkos::fence();
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());

        //////////////////////////////////
        // Calculate Global Temperature //
        //////////////////////////////////

        double locVELsum[Dim] = {0.0, 0.0, 0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];
        double locT[Dim] = {0.0, 0.0, 0.0};
        double globT[Dim];
        VectorD_t temperature;

        const size_t locNp = static_cast<size_t>(this->getLocalNum());

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local velocity sum", locNp,
                KOKKOS_LAMBDA(const int k, double& valL) {
                    double myVal = pP_view(k)[d];
                    valL += myVal;
                },
                Kokkos::Sum<double>(locVELsum[d]));
            Kokkos::fence();
        }
        MPI_Allreduce(locVELsum, globVELsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        for (unsigned d = 0; d < Dim; ++d)
            avgVEL[d] = globVELsum[d] / pN_glob;

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local velocity sum", locNp,
                KOKKOS_LAMBDA(const int k, double& valL) {
                    double myVal = (pP_view(k)[d] - avgVEL[d]) * (pP_view(k)[d] - avgVEL[d]);
                    valL += myVal;
                },
                Kokkos::Sum<double>(locT[d]));
            Kokkos::fence();
        }

        MPI_Reduce(locT, globT, 3, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        if (Ippl::Comm->rank() == 0)
            for (unsigned d = 0; d < Dim; ++d)
                temperature[d] = globT[d] / pN_glob;

        //////////////////////////////////
        //// Calculate Global Lorentz  ///
        //////////////////////////////////

        double lorentzAvg     = 0.0;
        double lorentzMax     = 0.0;
        double loc_lorentzAvg = 0.0;
        double loc_lorentzMax = 0.0;

        Kokkos::parallel_reduce(
            "Lorentz reductions (avg, max)", locNp,
            KOKKOS_LAMBDA(const int k, double& lAvg, double& lMax) {
                double lorentz =
                    1.0
                    / sqrt(1.0
                           - c2_inv
                                 * (pP_view(k)[0] * pP_view(k)[0] + pP_view(k)[1] * pP_view(k)[1]
                                    + pP_view(k)[2] * pP_view(k)[2]));
                lAvg += lorentz / pN_glob;
                lMax = lorentz > lMax ? lorentz : lMax;
            },
            Kokkos::Sum<double>(loc_lorentzAvg), Kokkos::Max<double>(loc_lorentzMax));

        MPI_Allreduce(&loc_lorentzAvg, &lorentzAvg, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(&loc_lorentzMax, &lorentzMax, 1, MPI_DOUBLE, MPI_MAX, Ippl::getComm());

        ////////////////////////////////////
        //// Calculate Moments of R and V //
        ////////////////////////////////////

        m << "Moments" << endl;
        const double zero = 0.0;

        double centroid[2 * Dim]         = {};
        double moment[2 * Dim][2 * Dim]  = {};
        double Ncentroid[2 * Dim]        = {};
        double Nmoment[2 * Dim][2 * Dim] = {};

        double loc_centroid[2 * Dim]        = {};
        double loc_moment[2 * Dim][2 * Dim] = {};

        for (unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for (unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = 0.0;
            }
        }

        for (unsigned i = 0; i < 2 * Dim; ++i) {
            Kokkos::parallel_reduce(
                "write Emittance 1 redcution", locNp,
                KOKKOS_LAMBDA(const int k, double& cent, double& mom0, double& mom1, double& mom2,
                              double& mom3, double& mom4, double& mom5) {
                    double part[2 * Dim];
                    part[0] = pR_view(k)[0];
                    part[1] = pP_view(k)[0];
                    part[2] = pR_view(k)[1];
                    part[3] = pP_view(k)[1];
                    part[4] = pR_view(k)[2];
                    part[5] = pP_view(k)[2];

                    cent += part[i];
                    mom0 += part[i] * part[0];
                    mom1 += part[i] * part[1];
                    mom2 += part[i] * part[2];
                    mom3 += part[i] * part[3];
                    mom4 += part[i] * part[4];
                    mom5 += part[i] * part[5];
                },
                Kokkos::Sum<double>(loc_centroid[i]), Kokkos::Sum<double>(loc_moment[i][0]),
                Kokkos::Sum<double>(loc_moment[i][1]), Kokkos::Sum<double>(loc_moment[i][2]),
                Kokkos::Sum<double>(loc_moment[i][3]), Kokkos::Sum<double>(loc_moment[i][4]),
                Kokkos::Sum<double>(loc_moment[i][5]));
            Kokkos::fence();
        }
        Ippl::Comm->barrier();
        MPI_Allreduce(loc_moment, moment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(loc_centroid, centroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        //////////////////////////////////////
        //// Calculate Normalized Emittance //
        //////////////////////////////////////

        for (unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for (unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = 0.0;
            }
        }

        for (unsigned i = 0; i < 2 * Dim; ++i) {
            Kokkos::parallel_reduce(
                "write Emittance 1 redcution", locNp,
                KOKKOS_LAMBDA(const int k, double& cent, double& mom0, double& mom1, double& mom2,
                              double& mom3, double& mom4, double& mom5) {
                    double v2 = pP_view(k)[0] * pP_view(k)[0] + pP_view(k)[1] * pP_view(k)[1]
                                + pP_view(k)[2] * pP_view(k)[2];
                    double lorentz = 1.0 / (sqrt(1.0 - v2 * c2_inv));

                    double part[2 * Dim];
                    part[0] = pR_view(k)[0];
                    part[1] = (pP_view(k)[0] * c_inv) * lorentz;
                    part[2] = pR_view(k)[1];
                    part[3] = (pP_view(k)[1] * c_inv) * lorentz;
                    part[4] = pR_view(k)[2];
                    part[5] = (pP_view(k)[2] * c_inv) * lorentz;

                    cent += part[i];
                    mom0 += part[i] * part[0];
                    mom1 += part[i] * part[1];
                    mom2 += part[i] * part[2];
                    mom3 += part[i] * part[3];
                    mom4 += part[i] * part[4];
                    mom5 += part[i] * part[5];
                },
                Kokkos::Sum<double>(loc_centroid[i]), Kokkos::Sum<double>(loc_moment[i][0]),
                Kokkos::Sum<double>(loc_moment[i][1]), Kokkos::Sum<double>(loc_moment[i][2]),
                Kokkos::Sum<double>(loc_moment[i][3]), Kokkos::Sum<double>(loc_moment[i][4]),
                Kokkos::Sum<double>(loc_moment[i][5]));
            Kokkos::fence();
        }

        MPI_Allreduce(loc_moment, Nmoment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        MPI_Allreduce(loc_centroid, Ncentroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

        VectorD_t eps2, fac;
        VectorD_t rsqsum, vsqsum, rvsum;
        VectorD_t rmean, vmean, rrms, vrms, eps, rvrms;
        VectorD_t norm;

        VectorD_t Neps2, Nfac;
        VectorD_t Nrsqsum, Nvsqsum, Nrvsum;
        VectorD_t Nrmean, Nvmean, Nrrms, Nvrms, Neps, Nrvrms;
        VectorD_t Nnorm;

        if (Ippl::Comm->rank() == 0) {
            for (unsigned int i = 0; i < Dim; i++) {
                rmean(i)  = centroid[2 * i] / pN_glob;
                vmean(i)  = centroid[(2 * i) + 1] / pN_glob;
                rsqsum(i) = moment[2 * i][2 * i] - pN_glob * rmean(i) * rmean(i);
                vsqsum(i) = moment[(2 * i) + 1][(2 * i) + 1] - pN_glob * vmean(i) * vmean(i);
                if (vsqsum(i) < 0)
                    vsqsum(i) = 0;
                rvsum(i) = (moment[(2 * i)][(2 * i) + 1] - pN_glob * rmean(i) * vmean(i));

                Nrmean(i)  = Ncentroid[2 * i] / pN_glob;
                Nvmean(i)  = Ncentroid[(2 * i) + 1] / pN_glob;
                Nrsqsum(i) = Nmoment[2 * i][2 * i] - pN_glob * Nrmean(i) * Nrmean(i);
                Nvsqsum(i) = Nmoment[(2 * i) + 1][(2 * i) + 1] - pN_glob * Nvmean(i) * Nvmean(i);
                if (Nvsqsum(i) < 0)
                    Nvsqsum(i) = 0;
                Nrvsum(i) = (Nmoment[(2 * i)][(2 * i) + 1] - pN_glob * Nrmean(i) * Nvmean(i));
            }

            // Coefficient wise calculation
            eps2  = (rsqsum * vsqsum - rvsum * rvsum) / (pN_glob * pN_glob);
            rvsum = rvsum / pN_glob;

            Neps2  = (Nrsqsum * Nvsqsum - Nrvsum * Nrvsum) / (pN_glob * pN_glob);
            Nrvsum = Nrvsum / pN_glob;

            for (unsigned int i = 0; i < Dim; i++) {
                rrms(i) = sqrt(rsqsum(i) / pN_glob);
                vrms(i) = sqrt(vsqsum(i) / pN_glob);

                eps(i)       = std::sqrt(std::max(eps2(i), zero));
                double tmpry = rrms(i) * vrms(i);
                fac(i)       = (tmpry == 0.0) ? zero : 1.0 / tmpry;

                Nrrms(i) = sqrt(rsqsum(i) / pN_glob);
                Nvrms(i) = sqrt(vsqsum(i) / pN_glob);

                Neps(i) = std::sqrt(std::max(Neps2(i), zero));
                tmpry   = Nrrms(i) * Nvrms(i);
                Nfac(i) = (tmpry == 0.0) ? zero : 1.0 / tmpry;
            }
            rvrms  = rvsum * fac;
            Nrvrms = Nrvsum * Nfac;
        }

        /////////////////////////////////
        //// Calculate Velocity Bounds //
        /////////////////////////////////

        double vmax_loc[Dim];
        double vmin_loc[Dim];
        double vmax[Dim];
        double vmin[Dim];

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "vel max", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& mm) {
                    double tmp_vel = pP_view(i)[d];
                    mm             = tmp_vel > mm ? tmp_vel : mm;
                },
                Kokkos::Max<double>(vmax_loc[d]));

            Kokkos::parallel_reduce(
                "vel min", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& mm) {
                    double tmp_vel = pP_view(i)[d];
                    mm             = tmp_vel < mm ? tmp_vel : mm;
                },
                Kokkos::Min<double>(vmin_loc[d]));
        }
        Kokkos::fence();
        MPI_Allreduce(vmax_loc, vmax, Dim, MPI_DOUBLE, MPI_MAX, Ippl::getComm());
        MPI_Allreduce(vmin_loc, vmin, Dim, MPI_DOUBLE, MPI_MIN, Ippl::getComm());

        /////////////////////////////////
        //// Calculate Position Bounds //
        /////////////////////////////////

        double rmax_loc[Dim];
        double rmin_loc[Dim];
        double rmax[Dim];
        double rmin[Dim];

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "rel max", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& mm) {
                    double tmp_vel = pR_view(i)[d];
                    mm             = tmp_vel > mm ? tmp_vel : mm;
                },
                Kokkos::Max<double>(rmax_loc[d]));

            Kokkos::parallel_reduce(
                "rel min", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& mm) {
                    double tmp_vel = pR_view(i)[d];
                    mm             = tmp_vel < mm ? tmp_vel : mm;
                },
                Kokkos::Min<double>(rmin_loc[d]));
        }
        Kokkos::fence();
        MPI_Allreduce(rmax_loc, rmax, Dim, MPI_DOUBLE, MPI_MAX, Ippl::getComm());
        MPI_Allreduce(rmin_loc, rmin, Dim, MPI_DOUBLE, MPI_MIN, Ippl::getComm());

        ////////////////////////////
        //// Write to output file //
        ////////////////////////////
        if (Ippl::Comm->rank() == 0) {

            std::stringstream fname;
            fname << "/All_FieldLangevin_";
            fname << Ippl::Comm->rank();
            fname << ".csv";
            Inform csvout(NULL, (folder + fname.str()).c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (iteration == 0) {
                csvout << "iteration,"
                        << "vmaxX,vmaxY,vmaxZ,"
                        << "vminX,vminY,vminZ,"
                        << "rmaxX,rmaxY,rmaxZ,"
                        << "rminX,rminY,rminZ,"
                        << "vrmsX,vrmsY,vrmsZ,"
                        << "Tx,Ty,Tz,"
                        << "epsX,epsY,epsZ,"
                        << "NepsX,NepsY,NepsZ,"
                        << "epsX2,epsY2,epsZ2,"
                        << "rvrmsX,rvrmsY,rvrmsZ,"
                        << "rrmsX,rrmsY,rrmsZ,"
                        << "rmeanX,rmeanY,rmeanZ,"
                        << "vmeanX,vmeanY,vmeanZ,"
                        << "time,"
                        << "Ex_field_energy,"
                        << "Ex_max_norm,"
                        << "lorentz_avg,"
                        << "lorentz_max,"
                        //<< "avgPotential,"         <<
                        //<< "avgEfield_x,"          <<
                        //<< "avgEfield_y,"          <<
                        //<< "avgEfield_z,"          <<
                        << "avgEfield_particle_x,"
                        << "avgEfield_particle_y,"
                        << "avgEfield_particle_z" << endl;
            }

            // clang-format off
            csvout << iteration << "," << vmax[0] << "," << vmax[1] << "," << vmax[2] << ","
                    << vmin[0] << "," << vmin[1] << "," << vmin[2] << "," << rmax[0] << ","
                    << rmax[1] << "," << rmax[2] << "," << rmin[0] << "," << rmin[1] << ","
                    << rmin[2] << "," << vrms(0) << "," << vrms(1) << "," << vrms(2) << ","
                    << temperature(0) << "," << temperature(1) << "," << temperature(2) << ","
                    << eps(0) << "," << eps(1) << "," << eps(2) << ","
                    << Neps(0) << "," << Neps(1) << "," << Neps(2) << ","
                    << eps2(0) << "," << eps2(1) << "," << eps2(2) << ","
                    << rvrms(0) << "," << rvrms(1) << "," << rvrms(2) << ","
                    << rrms(0) << "," << rrms(1) << "," << rrms(2) << ","
                    << rmean(0) << "," << rmean(1) << "," << rmean(2) << ","
                    << vmean(0) << "," << vmean(1) << "," << vmean(2) << ","
                    << iteration * dt_m << ","
                    << fieldEnergy << ","
                    << ExAmp << ","
                    << lorentzAvg << ","
                    << lorentzMax << ","
                    // avgPot      <<","<<
                    // avgEF[0]    <<","<<
                    // avgEF[1]    <<","<<
                    // avgEF[2]    <<","<<
                    << avgEF_particle[0] << "," << avgEF_particle[1] << "," << avgEF_particle[2] << endl;
            // clang-format on
        }

        Ippl::Comm->barrier();
    }

public:
    // Friction Coefficient
    Field_t<Dim> fv_m;
    VField_t<Dim> F_m;

    // Velocity density
    ParticleAttrib<double> p_fv_m;
    ParticleAttrib<VectorD_t> p_F_m;

public:
    // Particle Charge
    double pCharge_m;
    // Mass of the individual particles
    double pMass_m;
    // $\frac{1}{\epsilon_0}$ Inverse vacuum permittivity
    double epsInv_m;
    // Total number of global particles
    double globParticleNum_m;
    // Simulation timestep, used to dump the time in `dumpBeamStatistics()`
    double dt_m;

    ////////////////////////////////////////////
    // MEMBERS USED FOR LANGEVIN RELATED CODE //
    ////////////////////////////////////////////

    // Number of cells per dim in velocity space
    Vector<size_t> nv_m;
    // Mesh-Spacing of velocity space grid `fv_m`
    VectorD_t hv_m;

    // Extents of velocity space grid `fv_m`
    VectorD_t vmin_m;
    VectorD_t vmax_m;

    ippl::NDIndex<Dim> velocitySpaceIdxDomain_m;
    Mesh_t<Dim> velocitySpaceMesh_m;
    FieldLayout_t<Dim> velocitySpaceFieldLayout_m;

    // $\Gamma$ prefactor for Friction and Diffusion coefficients
    double gamma_m;
    // Speed of light in [cm/s]
    double c_m = 2.99792458e10;
    // $\pi$ needed for Rosenbluth potentials
    double pi_m = std::acos(-1.0);

    // Solver in velocity space
    // Solves $\Delta H(\vec v) = -8 \pi f(\vec v)$ and
    // directly stores $ - \nabla H(\vec v)$ in-place in LHS
    std::shared_ptr<FrictionSolver_t> frictionSolver_mp;
};

#endif /* LANGEVINPARTICLES_HPP */
