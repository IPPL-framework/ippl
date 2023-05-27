#include <Kokkos_MathematicalConstants.hpp>
#include <cmath>

#include "LangevinParticles.hpp"

const char* TestName = "LangevinPotentials";

KOKKOS_INLINE_FUNCTION double maxwellianPDF(const VectorD_t& v, const double& numberDensity,
                                            const double& vth) {
    double vNorm   = L2Norm(v);
    double expTerm = Kokkos::exp(-vNorm * vNorm / (2.0 * vth * vth));
    return (numberDensity / Kokkos::pow(2.0 * Kokkos::numbers::pi_v<double> * vth * vth, 1.5))
           * expTerm;
}

KOKKOS_INLINE_FUNCTION double HexactDistribution(const VectorD_t& v, const double& numberDensity,
                                                 const double& vth) {
    double vNorm = L2Norm(v);
    return (2.0 * numberDensity / vNorm) * Kokkos::erf(vNorm / (Kokkos::sqrt(2.0) * vth));
}

KOKKOS_INLINE_FUNCTION double GexactDistribution(const VectorD_t& v, const double& numberDensity,
                                                 const double& vth) {
    double vNorm   = L2Norm(v);
    double sqrt2   = Kokkos::sqrt(2.0);
    double expTerm = Kokkos::exp(-vNorm * vNorm / (2.0 * vth * vth))
                     / Kokkos::sqrt(Kokkos::numbers::pi_v<double>);
    double erfTerm   = Kokkos::erf(vNorm / (sqrt2 * vth));
    double erfFactor = (vth / (sqrt2 * vNorm)) + (vNorm / (sqrt2 * vth));
    return sqrt2 * numberDensity * vth * (expTerm + erfTerm * erfFactor);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    Inform msg("TestLangevinPotentials");
    Inform msg2all("TestLangevinPotentials", INFORM_ALL_NODES);

    auto start = std::chrono::high_resolution_clock::now();

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");

    int rank      = Ippl::Comm->rank();
    int comm_size = Ippl::Comm->size();

    ///////////////////////
    // Read Cmdline Args //
    ///////////////////////

    Ippl::Comm->setDefaultOverallocation(std::atof(argv[1]));

    const std::string SOLVER_T = argv[2];
    const double LB_THRESHOLD  = std::atof(argv[3]);
    const size_type NR         = std::atoll(argv[4]);
    // const double BEAM_RADIUS     = std::atof(argv[5]);
    const double BOXL  = std::atof(argv[6]);
    const size_type NP = std::atoll(argv[7]);
    const double DT    = std::atof(argv[8]);
    // const size_type NT           = std::atoll(argv[9]);
    const double PARTICLE_CHARGE = std::atof(argv[10]);
    const double PARTICLE_MASS   = std::atof(argv[11]);
    // const double FOCUS_FORCE     = std::atof(argv[12]);
    // const int PRINT_INTERVAL     = std::atoi(argv[13]);
    const double EPS_INV = std::atof(argv[14]);
    const size_t NV_MAX  = std::atoi(argv[15]);
    const double VMAX    = std::atof(argv[16]);
    // const double REL_BUFF        = std::atof(argv[17]);
    // const bool VMESH_ADAPT_B     = std::atoi(argv[18]);
    // const bool SCATTER_PHASE_B   = std::atoi(argv[19]);
    // const double FCT             = std::atof(argv[20]);
    // const double DRAG_FCT_B      = std::atof(argv[21]);
    // const double DIFF_FCT_B      = std::atof(argv[22]);
    // const double DRAG_B          = std::atof(argv[23]);
    // const double DIFFUSION_B     = std::atof(argv[24]);
    // const bool PRINT             = std::atoi(argv[25]);
    // const bool COLLISION         = std::atoi(argv[26]);
    const std::string OUT_DIR = argv[27];

    using bunch_type = LangevinParticles<PLayout_t<Dim>, Dim>;

    /////////////////////////////
    // CONSTANTS FOR MAXELLIAN //
    /////////////////////////////
    double vth           = 1.0;
    double numberDensity = 1.0;
    // double numberDensity = NP / (BOXL*BOXL*BOXL);

    for (size_t nv = 8; nv <= NV_MAX; nv *= 2) {
        /////////////////////////
        // CONFIGURATION SPACE //
        /////////////////////////

        const ippl::NDIndex<Dim> configSpaceIdxDomain(NR, NR, NR);
        ippl::e_dim_tag configSpaceDecomp[Dim] = {ippl::PARALLEL, ippl::PARALLEL, ippl::PARALLEL};

        const double L = BOXL * 0.5;
        const VectorD_t configSpaceLowerBound({-L, -L, -L});
        const VectorD_t configSpaceUpperBound({L, L, L});
        const VectorD_t configSpaceOrigin({-L, -L, -L});
        VectorD_t hr({BOXL / NR, BOXL / NR, BOXL / NR});  // spacing
        VectorD<size_t> nr({NR, NR, NR});

        Mesh_t<Dim> configSpaceMesh(configSpaceIdxDomain, hr, configSpaceOrigin);
        const bool isAllPeriodic = true;
        FieldLayout_t<Dim> configSpaceFieldLayout(configSpaceIdxDomain, configSpaceDecomp,
                                                  isAllPeriodic);
        PLayout_t<Dim> PL(configSpaceFieldLayout, configSpaceMesh);

        const double Q = NP * PARTICLE_CHARGE;

        msg << "Initialized Configuration Space" << endl;

        /////////////////////////////////
        // LANGEVIN PARTICLE CONTAINER //
        /////////////////////////////////

        std::shared_ptr P = std::make_shared<bunch_type>(
            PL, hr, configSpaceLowerBound, configSpaceUpperBound, configSpaceDecomp, SOLVER_T,
            PARTICLE_CHARGE, PARTICLE_MASS, EPS_INV, Q, NP, DT, nv, VMAX);

        // Initialize Particle Fields in Particles Class
        P->nr_m = {int(NR), int(NR), int(NR)};
        P->E_m.initialize(configSpaceMesh, configSpaceFieldLayout);
        P->rho_m.initialize(configSpaceMesh, configSpaceFieldLayout);

        // Set Periodic BCs for rho
        typedef ippl::BConds<double, Dim, Mesh_t<Dim>, Centering_t<Dim>> bc_type;

        bc_type bcField;
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] =
                std::make_shared<ippl::PeriodicFace<double, Dim, Mesh_t<Dim>, Centering_t<Dim>>>(i);
        }
        P->rho_m.setFieldBC(bcField);

        bunch_type bunchBuffer(PL);
        std::string frictionSolverName = "HOCKNEY";
        P->initAllSolvers(frictionSolverName);

        P->loadbalancethreshold_m = LB_THRESHOLD;

        msg << "Initialized Particle Bunch" << endl;

        //////////////////////////////////////////////
        // REFERENCE SOLUTION FIELDS FOR MAXWELLIAN //
        //////////////////////////////////////////////

        // Create scalar Field for Rosenbluth Potential
        auto HfieldExact = P->fv_m.deepCopy();
        auto GfieldExact = P->fv_m.deepCopy();

        //////////////////////////////////////////////
        // PARTICLE CREATION & INITIAL SPACE CHARGE //
        //////////////////////////////////////////////

        size_t nloc = NP / comm_size;

        // Last rank might have to initialize more Particles if `NP` not evenly divisble
        if (rank == comm_size - 1) {
            nloc = NP - (comm_size - 1) * nloc;
        }

        P->create(nloc);

        // Initialize Cold Sphere (positions only)
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * rank));
        Kokkos::parallel_for(
            nloc, GenerateRandomBoxPositions<VectorD_t, Kokkos::Random_XorShift64_Pool<>>(
                      P->R.getView(), BOXL, rand_pool64));

        // Initialize constant particle attributes
        P->q = PARTICLE_CHARGE;

        Kokkos::fence();
        Ippl::Comm->barrier();

        // Initialize Maxwellian Velocity Distribution
        const ippl::NDIndex<Dim>& lDom = P->velocitySpaceFieldLayout_m.getLocalNDIndex();
        const int nghost               = P->fv_m.getNghost();
        auto fvView                    = P->fv_m.getView();
        auto HviewExact                = HfieldExact.getView();
        auto GviewExact                = GfieldExact.getView();
        VectorD_t hv                   = P->hv_m;
        VectorD_t vOrigin              = P->vmin_m;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign initial velocity PDF and reference solution for H",
            ippl::getRangePolicy<Dim>(fvView, nghost), KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t<Dim> xvec = args;
                for (unsigned d = 0; d < Dim; d++) {
                    xvec[d] = (xvec[d] + lDom[d].first() - nghost + 0.5) * hv[d] + vOrigin[d];
                }

                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                ippl::apply<Dim>(fvView, args) =
                    maxwellianPDF(xvec, numberDensity, vth) * P->configSpaceIntegral_m;
                ippl::apply<Dim>(HviewExact, args) =
                    HexactDistribution(xvec, numberDensity, vth) * P->configSpaceIntegral_m;
            });

        Kokkos::fence();

        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "fvInit");

        //////////////////////////////////////
        // COMPUTE 1st ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        // Multiply with prefactor
        P->fv_m = -8.0 * P->pi_m * P->fv_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\nabla_v H(\vec v)$, is stored in `Fd_m`
        // $H(\vec v)$, is stored in `fv_m`
        P->frictionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Dump all data of the potentials
        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Happr");
        dumpVTKScalar(HfieldExact, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Hexact");
        dumpVTKVector(P->Fd_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "gradHappr");
        auto Hdiff = P->fv_m.deepCopy();
        Hdiff      = Hdiff - HfieldExact;
        dumpVTKScalar(Hdiff, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Hdiff");

        P->gatherFd();

        // Multiply with $\Gamma$
        P->p_Fd_m = P->p_Fd_m * P->gamma_m;

        // Dump actual friction coefficients
        P->dumpFdField(0, OUT_DIR);

        //////////////////////////////////////
        // COMPUTE 2nd ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign initial velocity PDF and reference solution for G",
            ippl::getRangePolicy<Dim>(fvView, nghost), KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t<Dim> xvec = args;
                for (unsigned d = 0; d < Dim; d++) {
                    xvec[d] = (xvec[d] + lDom[d].first() - nghost + 0.5) * hv[d] + vOrigin[d];
                }

                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                ippl::apply<Dim>(fvView, args) =
                    maxwellianPDF(xvec, numberDensity, vth) * P->configSpaceIntegral_m;
                ippl::apply<Dim>(GviewExact, args) =
                    GexactDistribution(xvec, numberDensity, vth) * P->configSpaceIntegral_m;
            });

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        // Multiply with prefactors defined in RHS of Rosenbluth equations
        // FFTPoissonSolver returns $ \Delta_v \Delta_v G(\vec v)$ in `fv_m`
        P->fv_m = -8.0 * P->pi_m * P->fv_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\Delta_v \Delta_v G(\vec v)$, is stored in `fv_m`
        P->diffusionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Dump all data of the potentials
        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Gappr");
        dumpVTKScalar(GfieldExact, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Gexact");
        auto Gdiff = P->fv_m.deepCopy();
        Gdiff      = Gdiff - GfieldExact;
        dumpVTKScalar(Gdiff, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "Gdiff");

        // Compute Hessian of $g(\vec v)$
        P->D_m = hess(P->fv_m);

        // Extract rows to separate fields
        P->extractRows(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        dumpCSVMatrixField(P->D0_m, P->D1_m, P->D2_m, P->hv_m, "D", 0, OUT_DIR);

        // Do Cholesky decomposition of $D$
        // and directly multiply with Gaussian random vector
        // P->choleskyMultiply();

        ////////////////////////////
        // COMPUTE RELATIVE ERROR //
        ////////////////////////////

        double HrelError = norm(Hdiff) / norm(HfieldExact);
        double GrelError = norm(Gdiff) / norm(GfieldExact);
        msg << "h(v) rel. error (" << nv << "^3)"
            << ": " << HrelError << endl;
        msg << "g(v) rel. error (" << nv << "^3)"
            << ": " << GrelError << endl;
    }

    msg << "LangevinPotentials: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
