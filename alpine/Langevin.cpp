#include "LangevinParticles.hpp"

const char* TestName = "LangevinDIH";

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    Inform msg("Langevin");
    Inform msg2all("Langevin", INFORM_ALL_NODES);

    auto start = std::chrono::high_resolution_clock::now();

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");

    int rank      = Ippl::Comm->rank();
    int comm_size = Ippl::Comm->size();
    msg << "Running on " << comm_size << " MPI ranks" << endl;

    ///////////////////////
    // Read Cmdline Args //
    ///////////////////////

    Ippl::Comm->setDefaultOverallocation(std::atof(argv[1]));

    const std::string SOLVER_T   = argv[2];
    const double LB_THRESHOLD    = std::atof(argv[3]);
    const size_type NR           = std::atoll(argv[4]);
    const double BEAM_RADIUS     = std::atof(argv[5]);
    const double BOXL            = std::atof(argv[6]);
    const size_type NP           = std::atoll(argv[7]);
    const double DT              = std::atof(argv[8]);
    const size_type NT           = std::atoll(argv[9]);
    const double PARTICLE_CHARGE = std::atof(argv[10]);
    const double PARTICLE_MASS   = std::atof(argv[11]);
    const double FOCUS_FORCE     = std::atof(argv[12]);
    const int PRINT_INTERVAL     = std::atoi(argv[13]);
    const double EPS_INV         = std::atof(argv[14]);
    const size_t NV              = std::atoi(argv[15]);
    const double VMAX            = std::atof(argv[16]);
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

    ////////////////////////
    // PARTICLE CONTAINER //
    ////////////////////////

    std::shared_ptr P = std::make_shared<bunch_type>(
        PL, hr, configSpaceLowerBound, configSpaceUpperBound, configSpaceDecomp, SOLVER_T,
        PARTICLE_CHARGE, PARTICLE_MASS, EPS_INV, Q, NP, DT, NV, VMAX);

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
    Kokkos::parallel_for(nloc, GenerateBoxMuller<VectorD_t, Kokkos::Random_XorShift64_Pool<>>(
                                   P->R.getView(), BEAM_RADIUS, rand_pool64));

    // Initialize constant particle attributes
    P->q = PARTICLE_CHARGE;

    Kokkos::fence();
    Ippl::Comm->barrier();

    msg << "Created " << NP << " Particles" << endl;

    // Distribute Particles to respective Processor according to SpatialLayout
    PL.update(*P, bunchBuffer);

    P->runSpaceChargeSolver(0);
    VectorD_t avgEF(P->compAvgSCForce(BEAM_RADIUS));

    P->applyConstantFocusing(FOCUS_FORCE, BEAM_RADIUS, avgEF);

    // dumpVTKScalar(P->rho_m, hr, nr, P->rmin_m, 0, 1.0, OUT_DIR, "Rho");
    // dumpVTKVector(P->E_m, hr, nr, P->rmin_m, 0, 1.0, OUT_DIR, "E");

    P->dumpBeamStatistics(0, OUT_DIR);
    P->dumpCollisionStatistics(0, OUT_DIR);

    for (size_t it = 1; it < NT; ++it) {
        P->R = P->R + 0.5 * DT * P->P;
        P->P = P->P + 0.5 * DT * P->E * PARTICLE_CHARGE / PARTICLE_MASS;

        // Field Solve
        P->runSpaceChargeSolver(it);

        // Add constant focusing term
        P->applyConstantFocusing(FOCUS_FORCE, BEAM_RADIUS, avgEF);

        P->runFrictionSolver();

        P->runDiffusionSolver();

        P->dumpCollisionStatistics(it, OUT_DIR);

        // Add dynamic friction & stochastic diffusion coefficients
        P->P = P->P + DT * P->p_Fd_m + P->p_QdW_m;
        // Add friction contribution
        // P->P = P->P + DT * P->p_Fd_m;
        //// Add velocity Diffusion contribution
        // P->P = P->P + P->p_QdW_m;

        P->P = P->P + 0.5 * DT * P->E * PARTICLE_CHARGE / PARTICLE_MASS;
        P->R = P->R + 0.5 * DT * P->P;

        // Dump Statistics every PRINT_INTERVAL iteration
        if (it % PRINT_INTERVAL == 0) {
            P->dumpBeamStatistics(it, OUT_DIR);
            P->velocityParticleCheck();

            //if (it % 200 == 0) {
                // dumpVTKVector(P->Fd_m, P->hv_m, P->nv_m, P->vmin_m, it, 1.0, OUT_DIR, "F_d");
                // dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, it, 1.0, OUT_DIR, "H(v)");
                // P->dumpFdStatistics(it, OUT_DIR);
            //}
            P->dumpDfield(it%2, OUT_DIR);

            msg << "Finished iteration " << it << endl;
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> time_chrono =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
        }
    }
    P->dumpFdStatistics(NT - 1, OUT_DIR);

    msg << "LangevinDIH: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
