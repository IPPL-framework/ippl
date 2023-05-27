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

template <unsigned Dim = 3>
using MField_t = Field<MatrixD_t, Dim>;

template <class PLayout, unsigned Dim = 3>
class LangevinParticles : public ChargedParticles<PLayout> {
    using Base = ChargedParticles<PLayout, Dim>;

    // Solver types acting on Fields in velocity space
    using FrictionSolver_t =
        ippl::FFTPoissonSolver<VectorD_t, double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;
    using DiffusionSolver_t =
        ippl::FFTPoissonSolver<VectorD_t, double, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

    // Kokkos Random Number Generator types (for stochastic diffusion coefficients)
    using KokkosRNGPool_t = Kokkos::Random_XorShift64_Pool<>;
    using KokkosRNG_t     = KokkosRNGPool_t::generator_type;

    // View types (of particle attributes)
    // typedef ParticleAttrib<double>::view_type attr_view_t;
    // typedef ParticleAttrib<VectorD_t>::view_type attr_Dview_t;
    // typedef ParticleAttrib<MatrixD_t>::view_type attr_DMatrixView_t;
    // typedef ParticleAttrib<double>::HostMirror attr_mirror_t;
    // typedef ParticleAttrib<VectorD_t>::HostMirror attr_Dmirror_t;

    // // View types (of Fields)
    // typedef typename ippl::detail::ViewType<double, Dim>::view_type Field_view_t;
    // typedef typename ippl::detail::ViewType<VectorD_t, Dim>::view_type VField_view_t;
    // typedef typename ippl::detail::ViewType<MatrixD_t, Dim>::view_type MField_view_t;

public:
    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the bunch buffer uses this
    */
    LangevinParticles(PLayout& pl)
        : ChargedParticles<PLayout>(pl) {
        registerLangevinAttributes();
        setPotentialBCs();
    }

    LangevinParticles(PLayout& pl, VectorD_t hr, VectorD_t rmin, VectorD_t rmax,
                      ippl::e_dim_tag configSpaceDecomp[Dim], std::string solver, double pCharge,
                      double pMass, double epsInv, double Q, size_type globalNumParticles,
                      double dt, size_type nv, double vmax)
        : ChargedParticles<PLayout>(pl, hr, rmin, rmax, configSpaceDecomp, Q, solver)
        , rank_m(Ippl::Comm->rank())
        , pCharge_m(pCharge)
        , pMass_m(pMass)
        , epsInv_m(epsInv)
        , globParticleNum_m(globalNumParticles)
        , configSpaceIntegral_m(globParticleNum_m)
        , dt_m(dt)
        , nv_m(nv)
        , hv_m(2 * vmax / nv)
        , vmin_m(-vmax)
        , vmax_m(vmax)
        , velocitySpaceIdxDomain_m(nv, nv, nv)
        , velocitySpaceMesh_m(velocitySpaceIdxDomain_m, hv_m, 0.0)
        , velocitySpaceFieldLayout_m(velocitySpaceIdxDomain_m, configSpaceDecomp, false)
        , randPool_m((size_type)(42 + 100 * rank_m)) {
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
        this->addAttribute(p_fv_m);
        this->addAttribute(p_Fd_m);

        this->addAttribute(p_D0_m);
        this->addAttribute(p_D1_m);
        this->addAttribute(p_D2_m);
        this->addAttribute(p_Q0_m);
        this->addAttribute(p_Q1_m);
        this->addAttribute(p_Q2_m);
        this->addAttribute(p_QdW_m);
    }

    void setupVelocitySpace() {
        Inform msg("setupVelocitySpace");
        // Velocity density (with open boundaries)
        fv_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);

        // Velocity friction coefficient
        Fd_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);

        // Diffusion Coefficients
        D_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        D0_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        D1_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        D2_m.initialize(velocitySpaceMesh_m, velocitySpaceFieldLayout_m);
        msg << "Finished velocity space setup." << endl;
    }

    void initAllSolvers(std::string frictionSolverName) {
        Inform msg("initAllSolvers");
        initSpaceChargeSolver();
        initFrictionSolver(frictionSolverName);
        initDiffusionSolver();
        msg << "Finished solver setup." << endl;
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

        frictionSolver_mp = std::make_shared<FrictionSolver_t>(Fd_m, fv_m, sp, solverName,
                                                               FrictionSolver_t::SOL_AND_GRAD);
    }

    // Setup Diffusion Solver ["BIHARMONIC"]
    void initDiffusionSolver() {
        // Initializing the open-boundary solver for the G potential used
        // to compute the diffusion coefficient $D$

        ippl::ParameterList sp;
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        // Saves the potential $g(\vec v)$ in `fv_m`
        diffusionSolver_mp = std::make_shared<DiffusionSolver_t>(fv_m, sp, "BIHARMONIC");
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

    void runSolver() {
        throw IpplException(
            "LangevinParticles::runSolver",
            "Not implemented. Run `runSpaceChargeSolver` or `runFrictionSolver` instead.");
    }

    void runSpaceChargeSolver(size_type iteration) {
        Base::scatterCIC(this->getGlobParticleNum(), iteration, this->hr_m);
        // Multiply by inverse vacuum permittivity
        // (due to differential formulation of Green's function)
        this->rho_m = this->rho_m * epsInv_m;
        // Call SpaceCharge Solver of `ChargeParticles.hpp`
        Base::runSolver();
        Base::gatherCIC();
    }

    void velocityParticleCheck() {
        Inform msg("velocityParticleStats");
        attr_Dview_t pP_view = this->P.getView();

        double avgVel          = 0.0;
        double minVelComponent = std::numeric_limits<double>::max();
        double maxVelComponent = std::numeric_limits<double>::min();
        double minVel          = std::numeric_limits<double>::max();
        double maxVel          = std::numeric_limits<double>::min();

        Kokkos::parallel_reduce(
            "check charge", this->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& loc_avgVel, double& loc_minVelComponent,
                          double& loc_maxVelComponent, double& loc_minVel, double& loc_maxVel) {
                double velNorm = L2Norm(pP_view[i]);
                double velX    = pP_view[i][0];
                double velY    = pP_view[i][1];
                double velZ    = pP_view[i][2];

                loc_avgVel += velNorm;

                loc_minVelComponent = velX < loc_minVelComponent ? velX : loc_minVelComponent;
                loc_minVelComponent = velY < loc_minVelComponent ? velY : loc_minVelComponent;
                loc_minVelComponent = velZ < loc_minVelComponent ? velZ : loc_minVelComponent;

                loc_maxVelComponent = velX > loc_maxVelComponent ? velX : loc_maxVelComponent;
                loc_maxVelComponent = velY > loc_maxVelComponent ? velY : loc_maxVelComponent;
                loc_maxVelComponent = velZ > loc_maxVelComponent ? velZ : loc_maxVelComponent;

                loc_minVel = velNorm < loc_minVel ? velNorm : loc_minVel;
                loc_maxVel = velNorm > loc_maxVel ? velNorm : loc_maxVel;
            },
            Kokkos::Sum<double>(avgVel), Kokkos::Min<double>(minVelComponent),
            Kokkos::Max<double>(maxVelComponent), Kokkos::Min<double>(minVel),
            Kokkos::Max<double>(maxVel));

        MPI_Reduce(rank_m == 0 ? MPI_IN_PLACE : &avgVel, &avgVel, 1, MPI_DOUBLE, MPI_SUM, 0,
                   Ippl::getComm());
        MPI_Reduce(rank_m == 0 ? MPI_IN_PLACE : &minVelComponent, &minVelComponent, 1, MPI_DOUBLE,
                   MPI_MIN, 0, Ippl::getComm());
        MPI_Reduce(rank_m == 0 ? MPI_IN_PLACE : &maxVelComponent, &maxVelComponent, 1, MPI_DOUBLE,
                   MPI_MAX, 0, Ippl::getComm());
        MPI_Reduce(rank_m == 0 ? MPI_IN_PLACE : &minVel, &minVel, 1, MPI_DOUBLE, MPI_MIN, 0,
                   Ippl::getComm());
        MPI_Reduce(rank_m == 0 ? MPI_IN_PLACE : &maxVel, &maxVel, 1, MPI_DOUBLE, MPI_MAX, 0,
                   Ippl::getComm());

        avgVel /= this->getGlobParticleNum();
        msg << "avgVel = " << avgVel << endl;
        msg << "minVelComponent = " << minVelComponent << endl;
        msg << "maxVelComponent = " << maxVelComponent << endl;
        msg << "minVel = " << minVel << endl;
        msg << "maxVel = " << maxVel << endl;
    }

    void dumpFdField(unsigned int iteration, std::string folder) {
        // Gather from particle attributes
        gather(p_Fd_m, Fd_m, this->P);

        double L2vec;
        double L2Fd;
        VectorD_t vVec;
        VField_view_t FdView = Fd_m.getView();

        const int nghost               = Fd_m.getNghost();
        const ippl::NDIndex<Dim>& lDom = velocitySpaceFieldLayout_m.getLocalNDIndex();

        typename VField_view_t::host_mirror_type hostView = Fd_m.getHostMirror();
        Kokkos::deep_copy(hostView, FdView);

        std::stringstream fname;
        fname << folder;
        fname << "/FdNorm_it";
        fname << std::setw(4) << std::setfill('0') << iteration;
        fname << ".csv";

        Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        // Write header
        csvout << "v,Fd,Fd/v" << endl;

        // Compute $||F_d(\vec v)||^2 / ||\vec{v}||^2$
        // And dump into file
        for (unsigned z = nghost; z < nv_m[2] + nghost; z++) {
            for (unsigned y = nghost; y < nv_m[1] + nghost; y++) {
                for (unsigned x = nghost; x < nv_m[0] + nghost; x++) {
                    vVec = {double(x), double(y), double(z)};
                    // Construct velocity vector at this cell
                    for (unsigned d = 0; d < Dim; d++) {
                        vVec[d] = (vVec[d] + lDom[d].first() - nghost + 0.5) * hv_m[d] + vmin_m[d];
                    }
                    L2vec = L2Norm(vVec);
                    L2Fd  = L2Norm(hostView(x, y, z));
                    csvout << L2vec << "," << L2Fd << "," << L2Fd / L2vec << endl;
                }
            }
        }
    }

    void scatterVelSpace() {
        // Scatter velocity density on grid
        fv_m = 0.0;
        // Scattered quantity should be a density ($\sum_i fv_i = 1$)
        p_fv_m = 1.0 / globParticleNum_m;
        scatter(p_fv_m, fv_m, this->P);
        // Normalize with dV
        double cellVolume = std::reduce(hv_m.begin(), hv_m.end(), 1., std::multiplies<double>());
        fv_m              = fv_m / cellVolume;
    }

    void gatherFd() {
        // Gather Friction coefficients to particles attribute
        gather(p_Fd_m, Fd_m, this->P);
    }

    void extractRows(MField_t<Dim>& M, VField_t<Dim>& V0, VField_t<Dim>& V1, VField_t<Dim>& V2) {
        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        const int nghost       = M.getNghost();
        MField_view_t Mview    = M.getView();
        VField_view_t V0view   = V0.getView();
        VField_view_t V1view   = V1.getView();
        VField_view_t V2view   = V2.getView();
        ippl::parallel_for(
            "Extract rows into separate Fields", ippl::getRangePolicy<Dim>(Mview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                ippl::apply<Dim>(V0view, args) = ippl::apply<Dim>(Mview, args)[0];
                ippl::apply<Dim>(V1view, args) = ippl::apply<Dim>(Mview, args)[1];
                ippl::apply<Dim>(V2view, args) = ippl::apply<Dim>(Mview, args)[2];
            });
        Kokkos::fence();
    }

    void gatherHessian() {
        extractRows(D_m, D0_m, D1_m, D2_m);
        gather(p_D0_m, D0_m, this->P);
        gather(p_D1_m, D1_m, this->P);
        gather(p_D2_m, D2_m, this->P);
    }

    void choleskyMultiply() {
        attr_Dview_t pD0_view = p_D0_m.getView();
        attr_Dview_t pD1_view = p_D1_m.getView();
        attr_Dview_t pD2_view = p_D2_m.getView();

        attr_Dview_t pQ0_view = p_Q0_m.getView();
        attr_Dview_t pQ1_view = p_Q1_m.getView();
        attr_Dview_t pQ2_view = p_Q2_m.getView();

        attr_Dview_t pQdW_view = p_QdW_m.getView();

        // Have to create references
        // as we want to avoid passing member variables to Kokkos kernels
        KokkosRNGPool_t& randPoolRef = randPool_m;
        double dt                    = dt_m;

        // First compute cholesky decomposition of D: $Q^T Q = D$
        // Then multiply with Gaussian noise vector $dW$ to get $Q \cdot dW$
        Kokkos::parallel_for(
            "Apply Constant Focusing", this->getLocalNum(), KOKKOS_LAMBDA(const int i) {
                KokkosRNG_t rand_gen = randPoolRef.get_state();
                MatrixD_t Q  = LDLtCholesky3x3(MatrixD_t({pD0_view(i), pD1_view(i), pD2_view(i)}));
                pQ0_view(i)  = Q[0];
                pQ1_view(i)  = Q[1];
                pQ2_view(i)  = Q[2];
                VectorD_t dW = VectorD_t(
                    {rand_gen.normal(0.0, dt), rand_gen.normal(0.0, dt), rand_gen.normal(0.0, dt)});
                pQdW_view(i) = matrixVectorMul3x3(Q, dW);

                // Give the state back, which will allow another thread to acquire it
                randPoolRef.free_state(rand_gen);
            });
        Kokkos::fence();
    }

    void runFrictionSolver() {
        Inform msg("runFrictionSolver");

        scatterVelSpace();

        // Multiply velSpaceDensity `fv_m` with prefactors defined in RHS of Rosenbluth equations
        // `-1.0` prefactor is because the solver returns $- \nabla H(\vec v)$
        // Multiply with prob. density in configuration space $f(\vec r)$
        fv_m = -1.0 * (-8.0 * pi_m * gamma_m * fv_m * configSpaceIntegral_m);

        // Set origin of velocity space mesh to zero (for FFT)
        velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\Delta_v H(\vec v)$. Its gradient is stored in `Fd_m`
        frictionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        velocitySpaceMesh_m.setOrigin(vmin_m);

        gatherFd();

        msg << "Friction computation done." << endl;
    }

    void runDiffusionSolver() {
        Inform msg("runDiffusionSolver");

        scatterVelSpace();

        // Multiply with prefactors defined in RHS of Rosenbluth equations
        // FFTPoissonSolver returns $ \Delta_v \Delta_v G(\vec v)$ in `fv_m`
        fv_m = -8.0 * pi_m * gamma_m * fv_m * configSpaceIntegral_m;

        // Set origin of velocity space mesh to zero (for FFT)
        velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\Delta_v \Delta_v G(\vec v)$ and store it in `fv_m`
        diffusionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        velocitySpaceMesh_m.setOrigin(vmin_m);

        // Compute Hessian of $g(\vec v)$
        D_m = hess(fv_m);

        // Gather Hessian to particle attributes
        gatherHessian();

        // Do Cholesky decomposition of $D$
        // and directly multiply with Gaussian random vector
        choleskyMultiply();

        msg << "Diffusion computation done." << endl;
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

    void dumpCollisionStatistics(unsigned int iteration, std::string folder) {
        Inform m("dumpCollisionStatistics");

        //////////////////////
        // Calculate Fd Avg //
        //////////////////////

        attr_Dview_t pFd_view = this->p_Fd_m.getView();
        VectorD_t FdAvg;

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "rel max", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& FdLoc) { FdLoc += pFd_view(i)[d]; },
                Kokkos::Sum<double>(FdAvg(d)));
        }

        FdAvg = FdAvg / globParticleNum_m;

        //////////////////////
        // Calculate D Avg //
        //////////////////////

        // Get views on the particle attributes
        attr_Dview_t pD0_view = p_D0_m.getView();
        attr_Dview_t pD1_view = p_D1_m.getView();
        attr_Dview_t pD2_view = p_D2_m.getView();

        attr_Dview_t pQ0_view = p_Q0_m.getView();
        attr_Dview_t pQ1_view = p_Q1_m.getView();
        attr_Dview_t pQ2_view = p_Q2_m.getView();

        attr_Dview_t pQdW_view = p_QdW_m.getView();

        // Vectors to store the avg statistics in
        VectorD_t D0Avg, D1Avg, D2Avg;
        VectorD_t Q0Avg, Q1Avg, Q2Avg;
        VectorD_t QdWAvg;

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "gather `D` and `Q` statistics", this->getLocalNum(),
                KOKKOS_LAMBDA(const int i, double& D0Loc, double& D1Loc, double& D2Loc,
                              double& Q0Loc, double& Q1Loc, double& Q2Loc, double& QdWLoc) {
                    D0Loc += pD0_view(i)[d];
                    D1Loc += pD1_view(i)[d];
                    D2Loc += pD2_view(i)[d];
                    Q0Loc += pQ0_view(i)[d];
                    Q1Loc += pQ1_view(i)[d];
                    Q2Loc += pQ2_view(i)[d];
                    QdWLoc += pQdW_view(i)[d];
                },
                Kokkos::Sum<double>(D0Avg(d)), Kokkos::Sum<double>(D1Avg(d)),
                Kokkos::Sum<double>(D2Avg(d)), Kokkos::Sum<double>(Q0Avg(d)),
                Kokkos::Sum<double>(Q1Avg(d)), Kokkos::Sum<double>(Q2Avg(d)),
                Kokkos::Sum<double>(QdWAvg(d)));
        }

        FdAvg  = FdAvg / globParticleNum_m;
        D0Avg  = D0Avg / globParticleNum_m;
        D1Avg  = D1Avg / globParticleNum_m;
        D2Avg  = D2Avg / globParticleNum_m;
        Q0Avg  = Q0Avg / globParticleNum_m;
        Q1Avg  = Q1Avg / globParticleNum_m;
        Q2Avg  = Q2Avg / globParticleNum_m;
        QdWAvg = QdWAvg / globParticleNum_m;

        if (Ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "/collision_statistics_";
            fname << Ippl::Comm->rank();
            fname << ".csv";
            Inform csvout(NULL, (folder + fname.str()).c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            // clang-format off
            if (iteration == 0) {
                csvout << "iteration,"
                       << "FdAvg_x,"  << "FdAvg_y,"  << "FdAvg_z,"
                       << "D0Avg_x,"  << "D0Avg_y,"  << "D0Avg_z,"
                       << "D1Avg_x,"  << "D1Avg_y,"  << "D1Avg_z,"
                       << "D2Avg_x,"  << "D2Avg_y,"  << "D2Avg_z,"
                       << "Q0Avg_x,"  << "Q0Avg_y,"  << "Q0Avg_z,"
                       << "Q1Avg_x,"  << "Q1Avg_y,"  << "Q1Avg_z,"
                       << "Q2Avg_x,"  << "Q2Avg_y,"  << "Q2Avg_z,"
                       << "QdWAvg_x," << "QdWAvg_y," << "QdWAvg_z" << endl;
            }

            csvout << iteration << ","
                   << FdAvg(0)  << "," << FdAvg(1)  << "," << FdAvg(2)  << ","
                   << D0Avg(0)  << "," << D0Avg(1)  << "," << D0Avg(2)  << ","
                   << D1Avg(0)  << "," << D1Avg(1)  << "," << D1Avg(2)  << ","
                   << D2Avg(0)  << "," << D2Avg(1)  << "," << D2Avg(2)  << ","
                   << Q0Avg(0)  << "," << Q0Avg(1)  << "," << Q0Avg(2)  << ","
                   << Q1Avg(0)  << "," << Q1Avg(1)  << "," << Q1Avg(2)  << ","
                   << Q2Avg(0)  << "," << Q2Avg(1)  << "," << Q2Avg(2)  << ","
                   << QdWAvg(0) << "," << QdWAvg(1) << "," << QdWAvg(2) << endl;
            // clang-format on
        }
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
        VField_view_t E_view = this->E_m.getView();

        ////////////////////////////////////////////////////////////
        // Gather E-field in x-direction (via particle attribute) //
        ////////////////////////////////////////////////////////////

        // Compute the Avg E-Field over the particle Attribute
        VectorD_t avgEF_particle;
        double locEFsum[Dim] = {};
        double globEFsum[Dim];

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::parallel_reduce(
                "get local EField sum", static_cast<size_type>(this->getLocalNum()),
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
            KOKKOS_LAMBDA(const size_type i, const size_type j, const size_type k, double& valL) {
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
            KOKKOS_LAMBDA(const size_type i, const size_type j, const size_type k, double& valL) {
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

        const size_type locNp = static_cast<size_type>(this->getLocalNum());

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
    Field_t<Dim> fv_m;
    // Friction Coefficients
    VField_t<Dim> Fd_m;
    // Diffusion Coefficients
    MField_t<Dim> D_m;
    // Separate rows, as gathering of Matrices is not yet possible
    VField_t<Dim> D0_m;
    VField_t<Dim> D1_m;
    VField_t<Dim> D2_m;

    // Velocity density
    ParticleAttrib<double> p_fv_m;
    // Friction Coefficient
    ParticleAttrib<VectorD_t> p_Fd_m;
    // Rows of Diffusion Coefficient
    ParticleAttrib<VectorD_t> p_D0_m;
    ParticleAttrib<VectorD_t> p_D1_m;
    ParticleAttrib<VectorD_t> p_D2_m;
    // Cholesky decomposition of `D` TODO Remove as it can be stored as a temporary matrix in the
    // diffusion kernel
    ParticleAttrib<VectorD_t> p_Q0_m;
    ParticleAttrib<VectorD_t> p_Q1_m;
    ParticleAttrib<VectorD_t> p_Q2_m;
    // Diffusion coefficient multiplied with Gaussian random vectors
    ParticleAttrib<VectorD_t> p_QdW_m;

public:
    // MPI Rank
    int rank_m;

    // Particle Charge
    double pCharge_m;
    // Mass of the individual particles
    double pMass_m;
    // $\frac{1}{\epsilon_0}$ Inverse vacuum permittivity
    double epsInv_m;
    // Total number of global particles
    double globParticleNum_m;

    // $\int f_r(r) dr^3 = N$
    double configSpaceIntegral_m;

    // Simulation timestep
    // Used to dump the time in `choleskyMultiply()` and `dumpBeamStatistics()`
    double dt_m;

    ////////////////////////////////////////////
    // MEMBERS USED FOR LANGEVIN RELATED CODE //
    ////////////////////////////////////////////

    // Number of cells per dim in velocity space
    Vector<size_type> nv_m;
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

    // Random number generator for random gaussians that are multiplied with $Q$
    KokkosRNGPool_t randPool_m;

    ///////////////////////////////
    // SOLVERS IN VELOCITY SPACE //
    ///////////////////////////////

    // Solves $\Delta_v H(\vec v) = -8 \pi f(\vec v)$ and
    // directly stores $ - \nabla H(\vec v)$ in-place in LHS
    std::shared_ptr<FrictionSolver_t> frictionSolver_mp;

    // Solves $\Delta_v \Delta_v G(\vec v) = -8 \pi f(\vec v)$
    std::shared_ptr<DiffusionSolver_t> diffusionSolver_mp;
};

#endif /* LANGEVINPARTICLES_HPP */
