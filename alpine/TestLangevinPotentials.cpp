#include <Kokkos_MathematicalConstants.hpp>
#include <cmath>

#include "LangevinHelpers.hpp"
#include "LangevinParticles.hpp"

const char* TestName = "LangevinPotentials";
enum TestCase {
    MAXWELLIAN = 0b01,
    GAUSSIAN   = 0b10
};

KOKKOS_INLINE_FUNCTION double gaussianPDF(const VectorD_t& v, const double& sigma,
                                          const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double sigma3 = sigma * sigma * sigma;
    const double pi     = Kokkos::numbers::pi_v<double>;
    return prefactor * Kokkos::exp(-1.0 * vNorm * vNorm / (2.0 * sigma * sigma))
           / (Kokkos::sqrt(8.0 * pi * pi * pi) * sigma3);
}

KOKKOS_INLINE_FUNCTION double gaussianHexact(const VectorD_t& v, const double& sigma,
                                             const double& prefactor) {
    const double vNorm = L2Norm(v);
    return prefactor * 2.0 / vNorm * Kokkos::erf(vNorm / (Kokkos::sqrt(2.0) * sigma));
}

KOKKOS_INLINE_FUNCTION VectorD_t gaussianFdExact(const VectorD_t& v, const double& gamma,
                                                 const double& sigma, const double& prefactor) {
    const double vNorm     = L2Norm(v);
    const double vNorm2    = vNorm * vNorm;
    const double pi        = Kokkos::numbers::pi_v<double>;
    const double preFactor = 2.0 / vNorm2;
    const double expTerm =
        Kokkos::sqrt(2.0 / pi) * 1.0 / sigma * Kokkos::exp(-1.0 * vNorm2 / (2.0 * sigma * sigma));
    const double erfTerm = 1.0 / vNorm * Kokkos::erf(vNorm / (Kokkos::sqrt(2.0) * sigma));
    return prefactor * gamma * preFactor * (expTerm - erfTerm) * v;
}

KOKKOS_INLINE_FUNCTION double gaussianGexact(const VectorD_t& v, const double& sigma,
                                             const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double sigma2 = sigma * sigma;
    const double pi     = Kokkos::numbers::pi_v<double>;
    const double expTerm =
        sigma * Kokkos::sqrt(2.0 / pi) * Kokkos::exp(-1.0 * vNorm * vNorm / (2.0 * sigma2));
    const double erfTerm =
        (vNorm + sigma2 / vNorm) * Kokkos::erf(vNorm / (sigma * Kokkos::sqrt(2.0)));
    return prefactor * (expTerm + erfTerm);
}

KOKKOS_INLINE_FUNCTION double gaussianDiagEntryExact(const VectorD_t& v, const size_type colIdx,
                                                     const double& gamma, const double& sigma,
                                                     const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double vNorm2 = vNorm * vNorm;
    const double sigma2 = sigma * sigma;
    const double vHat2  = v[colIdx] * v[colIdx];
    const double pi     = Kokkos::numbers::pi_v<double>;

    const double expFactor = (vHat2 - sigma2) / sigma2
                             + (sigma2 / vNorm + vNorm)
                                   * (sigma2 * vNorm2 - vHat2 * vNorm2 - sigma2 * vHat2)
                                   / (sigma2 * vNorm2 * vNorm)
                             + 2.0 * vHat2 / (vNorm2 * vNorm2) * (vNorm2 - sigma2);
    const double expTerm =
        Kokkos::sqrt(2.0 / pi) * 1.0 / sigma * Kokkos::exp(-vNorm2 / (2.0 * sigma2)) * expFactor;
    const double erfTerm = 1.0 / (vNorm2 * vNorm2 * vNorm)
                           * Kokkos::erf(vNorm / (sigma * Kokkos::sqrt(2.0)))
                           * (vNorm2 * vNorm2 - vNorm2 * (vHat2 + sigma2) + 3.0 * sigma2 * vHat2);
    return gamma * prefactor * (expTerm + erfTerm);
}

KOKKOS_INLINE_FUNCTION double gaussianD01exact(const VectorD_t& v, const double& gamma,
                                               const double& sigma, const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double vNorm2 = vNorm * vNorm;
    const double pi     = Kokkos::numbers::pi_v<double>;
    const double sigma2 = sigma * sigma;
    return gamma * prefactor
           * (-3 * sigma * sqrt(2 / pi) * v[0] * v[1] / (vNorm2 * vNorm2)
                  * Kokkos::exp(-vNorm2 / (2 * sigma2))
              + Kokkos::erf(vNorm / (sigma * Kokkos::sqrt(2)))
                    * (v[0] * v[1] / (vNorm * vNorm2) * (3 * sigma2 / vNorm2 - 1)));
}

KOKKOS_INLINE_FUNCTION double gaussianD02exact(const VectorD_t& v, const double& gamma,
                                               const double& sigma, const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double vNorm2 = vNorm * vNorm;
    const double pi     = Kokkos::numbers::pi_v<double>;
    const double sigma2 = sigma * sigma;
    return gamma * prefactor
           * (-3 * sigma * sqrt(2 / pi) * v[0] * v[2] / (vNorm2 * vNorm2)
                  * Kokkos::exp(-vNorm2 / (2 * sigma2))
              + Kokkos::erf(vNorm / (sigma * Kokkos::sqrt(2)))
                    * (v[0] * v[2] / (vNorm * vNorm2) * (3 * sigma2 / vNorm2 - 1)));
}

KOKKOS_INLINE_FUNCTION double gaussianD12exact(const VectorD_t& v, const double& gamma,
                                               const double& sigma, const double& prefactor) {
    const double vNorm  = L2Norm(v);
    const double vNorm2 = vNorm * vNorm;
    const double pi     = Kokkos::numbers::pi_v<double>;
    const double sigma2 = sigma * sigma;
    return gamma * prefactor
           * (-3 * sigma * sqrt(2 / pi) * v[1] * v[2] / (vNorm2 * vNorm2)
                  * Kokkos::exp(-vNorm2 / (2 * sigma2))
              + Kokkos::erf(vNorm / (sigma * Kokkos::sqrt(2)))
                    * (v[1] * v[2] / (vNorm * vNorm2) * (3 * sigma2 / vNorm2 - 1)));
}

KOKKOS_INLINE_FUNCTION MatrixD_t gaussianFullDexact(const VectorD_t& v, const double& gamma,
                                                    const double& sigma, const double& prefactor) {
    MatrixD_t D;
    // Diagonal Entries
    D[0][0] = gaussianDiagEntryExact(v, 0, gamma, sigma, prefactor);
    D[1][1] = gaussianDiagEntryExact(v, 1, gamma, sigma, prefactor);
    D[2][2] = gaussianDiagEntryExact(v, 2, gamma, sigma, prefactor);

    // Off-Diagonals
    D[0][1] = gaussianD01exact(v, gamma, sigma, prefactor);
    D[0][2] = gaussianD02exact(v, gamma, sigma, prefactor);
    D[1][2] = gaussianD12exact(v, gamma, sigma, prefactor);

    // Mirror along diagonal
    D[1][0] = D[0][1];
    D[2][0] = D[0][2];
    D[2][1] = D[1][2];

    return D;
}

template <unsigned Test, class Bunch>
class GenerateTestData {
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;

public:
    GenerateTestData(Field_view_t fvView, Field_view_t HviewExact, Field_view_t GviewExact,
                     VField_view_t FdViewExact, MField_view_t DviewExact, double numberDensity,
                     double vth, const Bunch& P)
        : fvView_m(fvView)
        , HviewExact_m(HviewExact)
        , GviewExact_m(GviewExact)
        , FdViewExact_m(FdViewExact)
        , DviewExact_m(DviewExact)
        , numberDensity_m(numberDensity)
        , vth_m(vth)
        , lDom_m(P->velocitySpaceFieldLayout_m.getLocalNDIndex())
        , P_m(P) {}

    KOKKOS_INLINE_FUNCTION void operator()(const index_array_type& args) const {
        VectorD_t v = args;
        for (unsigned d = 0; d < Dim; d++) {
            v[d] = (v[d] + lDom_m[d].first() + 0.5) * P_m->hv_m[d] + P_m->vmin_m[d] - P_m->hv_m[d];
        }

        double prefactor;
        double sigma;

        // Set specific factors depending on Test-Case
        if constexpr (Test == TestCase::MAXWELLIAN) {
            prefactor = numberDensity_m;
            sigma     = vth_m;
        } else if constexpr (Test == TestCase::GAUSSIAN) {
            prefactor = 1.0;
            sigma     = 1.0;
        }

        // Multiply Integral over configuration-space
        prefactor *= P_m->configSpaceIntegral_m;

        // Initialize I.C. and all resulting potentials, etc.
        ippl::apply(fvView_m, args)      = gaussianPDF(v, sigma, prefactor);
        ippl::apply(HviewExact_m, args)  = gaussianHexact(v, sigma, prefactor);
        ippl::apply(GviewExact_m, args)  = gaussianGexact(v, sigma, prefactor);
        ippl::apply(FdViewExact_m, args) = gaussianFdExact(v, P_m->gamma_m, sigma, prefactor);

        // Diffusion Coefficient $D$
        ippl::apply(DviewExact_m, args) = gaussianFullDexact(v, P_m->gamma_m, sigma, prefactor);
    }

private:
    const Field_view_t fvView_m;
    const Field_view_t HviewExact_m;
    const Field_view_t GviewExact_m;
    const VField_view_t FdViewExact_m;
    const MField_view_t DviewExact_m;
    const double numberDensity_m;
    const double vth_m;
    const ippl::NDIndex<Dim>& lDom_m;
    const Bunch& P_m;
};

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

    const std::string SOLVER_T        = argv[2];
    const double LB_THRESHOLD         = std::atof(argv[3]);
    const size_type NR                = std::atoll(argv[4]);
    const double BOXL                 = std::atof(argv[5]);
    const size_type NP                = std::atoll(argv[6]);
    const double DT                   = std::atof(argv[7]);
    const double PARTICLE_CHARGE      = std::atof(argv[8]);
    const double PARTICLE_MASS        = std::atof(argv[9]);
    const double EPS_INV              = std::atof(argv[10]);
    const size_t NV_MAX               = std::atoi(argv[11]);
    const double V_MAX                = std::atof(argv[12]);
    const std::string FRICTION_SOLVER = argv[13];
    const std::string OUT_DIR         = argv[14];

    using bunch_type = LangevinParticles<PLayout_t<double, Dim>, double, Dim>;

    /////////////////////////////
    // CONSTANTS FOR MAXELLIAN //
    /////////////////////////////

    constexpr TestCase testType = TestCase::MAXWELLIAN;
    const double vth            = 1.0;
    const double numberDensity  = 1.0;
    // double numberDensity = NP / (BOXL*BOXL*BOXL);

    const size_t nvMin = 8;
    for (size_t nv = nvMin; nv <= NV_MAX; nv *= 2) {
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
        PLayout_t<double, Dim> PL(configSpaceFieldLayout, configSpaceMesh);

        const double Q = NP * PARTICLE_CHARGE;

        msg << "Initialized Configuration Space" << endl;

        /////////////////////////////////
        // LANGEVIN PARTICLE CONTAINER //
        /////////////////////////////////

        std::shared_ptr P = std::make_shared<bunch_type>(
            PL, hr, configSpaceLowerBound, configSpaceUpperBound, configSpaceDecomp, SOLVER_T,
            PARTICLE_CHARGE, PARTICLE_MASS, EPS_INV, Q, NP, DT, nv, V_MAX);

        // Initialize Particle Fields in Particles Class
        P->nr_m = {double(NR), double(NR), double(NR)};
        P->E_m.initialize(configSpaceMesh, configSpaceFieldLayout);
        P->rho_m.initialize(configSpaceMesh, configSpaceFieldLayout);

        // Set Periodic BCs for rho
        typedef ippl::BConds<Field_t<Dim>, Dim> bc_type;

        bc_type bcField;
        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<Field_t<Dim>>>(i);
        }
        P->rho_m.setFieldBC(bcField);

        bunch_type bunchBuffer(PL);
        P->initAllSolvers(FRICTION_SOLVER);

        P->loadbalancethreshold_m = LB_THRESHOLD;

        msg << "Initialized Particle Bunch" << endl;

        //////////////////////////////////////////////
        // REFERENCE SOLUTION FIELDS FOR MAXWELLIAN //
        //////////////////////////////////////////////

        // Gamma prefactor used for multiplying the potentials to obtain friction and diffusion
        // coefficients
        const double gamma = P->gamma_m;

        // Create scalar Field for Rosenbluth Potentials
        Field_t<Dim> HfieldExact      = P->fv_m.deepCopy();
        Field_t<Dim> GfieldExact      = P->fv_m.deepCopy();
        MField_t<Dim> DfieldExact     = P->D_m.deepCopy();
        VField_t<double, Dim> FdExact = P->Fd_m.deepCopy();

        // Fields for identities that must hold
        Field_t<Dim> Dtrace     = P->fv_m.deepCopy();
        Field_t<Dim> DtraceDiff = P->fv_m.deepCopy();

        Field_t<Dim> D0div             = P->fv_m.deepCopy();
        Field_t<Dim> D1div             = P->fv_m.deepCopy();
        Field_t<Dim> D2div             = P->fv_m.deepCopy();
        VField_t<double, Dim> Ddiv     = P->Fd_m.deepCopy();
        VField_t<double, Dim> DdivDiff = P->Fd_m.deepCopy();

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

        /////////////////////////////////////////////////
        // INITIALIZE MAXWELLIAN VELOCITY DISTRIBUTION //
        /////////////////////////////////////////////////

        const int nghost  = P->fv_m.getNghost();
        VectorD_t hv      = P->hv_m;
        VectorD_t vOrigin = P->vmin_m;

        Field_view_t fvView         = P->fv_m.getView();
        Field_view_t HviewExact     = HfieldExact.getView();
        Field_view_t GviewExact     = GfieldExact.getView();
        VField_view_t FdViewExact   = FdExact.getView();
        MField_view_t DviewExact    = DfieldExact.getView();
        Field_view_t DtraceView     = Dtrace.getView();
        Field_view_t DtraceDiffView = DtraceDiff.getView();

        GenerateTestData<testType, std::shared_ptr<bunch_type>> dataGenerator(
            fvView, HviewExact, GviewExact, FdViewExact, DviewExact, numberDensity, vth, P);

        ippl::parallel_for("Assign initial velocity PDF and reference solutions",
                           ippl::getRangePolicy(fvView, 0), dataGenerator);
        Kokkos::fence();

        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "fvInit");

        //////////////////////////////////////
        // COMPUTE 1st ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        /*
         * Multiply velSpaceDensity `fv_m` with prefactors defined in RHS of Rosenbluth equations
         * `-1.0` prefactor is because the solver computes $- \Delta H(\vec v) = - rhs(v)$
         * Multiply with prob. density in configuration space $f(\vec r)$
         * Prob. density in configuration space $f(\vec r)$ already added in in initial
         * condition
         */
        P->fv_m = 8.0 * P->pi_m * P->fv_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\nabla_v H(\vec v)$, is stored in `Fd_m`
        // $H(\vec v)$, is stored in `fv_m`
        P->frictionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Dump all data of the potentials
        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Happr");
        dumpVTKScalar(HfieldExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Hexact");
        auto Hdiff = P->fv_m.deepCopy();
        Hdiff      = Hdiff - HfieldExact;
        dumpVTKScalar(Hdiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Hdiff");

        // Sign change needed as solver returns $- \nabla H(\vec)$
        P->Fd_m = -1.0 * gamma * P->Fd_m;
        P->gatherFd();

        auto FdDiff = P->Fd_m.deepCopy();
        FdDiff      = FdDiff - FdExact;

        dumpVTKVector(P->Fd_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Fdappr");
        dumpVTKVector(FdExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Fdexact");
        dumpVTKVector(FdExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "FdDiff");

        // Dump actual friction coefficients
        P->dumpFdField(nv, OUT_DIR);

        //////////////////////////////////////
        // COMPUTE 2nd ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        ippl::parallel_for("Assign initial velocity PDF and reference solutions",
                           ippl::getRangePolicy(fvView, 0), dataGenerator);
        Kokkos::fence();

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        /*
         * Multiply with prefactors defined in RHS of Rosenbluth equations
         * FFTPoissonSolver returns $G(\vec v)$ in `fv_m`
         * Density multiplied with `-1.0` as the solver computes $\Delta \Delta G(\vec v) = -
         * rhs(v)$
         */
        P->fv_m = 8.0 * P->pi_m * P->fv_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\Delta_v \Delta_v G(\vec v)$, is stored in `fv_m`
        P->diffusionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Dump all data of the potentials
        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gappr");
        dumpVTKScalar(GfieldExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gexact");
        auto Gdiff = P->fv_m.deepCopy();
        Gdiff      = Gdiff - GfieldExact;
        dumpVTKScalar(Gdiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gdiff");

        // Compute Hessian of $g(\vec v)$
        P->D_m = gamma * hess(P->fv_m);

        // Extract rows of exact field to separate Vector-Fields
        P->extractRows(DfieldExact, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        // dumpCSVMatrixField(P->D0_m, P->D1_m, P->D2_m, P->nv_m, "D", nv, OUT_DIR);
        dumpVTKVector(P->D0_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D0exact");
        dumpVTKVector(P->D1_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D1exact");
        dumpVTKVector(P->D2_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D2exact");

        // Extract rows of approximation to separate Vector-Fields
        P->extractRows(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        dumpVTKVector(P->D0_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D0");
        dumpVTKVector(P->D1_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D1");
        dumpVTKVector(P->D2_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D2");

        ///////////////////////////////////////
        // COMPUTE IDENTITIES THAT MUST HOLD //
        //                                   //
        // $Tr(\boldsymbol D) / \Gamma = h$  //
        ///////////////////////////////////////

        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
        Kokkos::parallel_for(
            "Gather trace of $D$",
            mdrange_type({nghost, nghost, nghost},
                         {DtraceView.extent(0) - nghost, DtraceView.extent(1) - nghost,
                          DtraceView.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                DtraceView(i, j, k) =
                    P->D0_m(i, j, k)[0] + P->D1_m(i, j, k)[1] + P->D2_m(i, j, k)[2];
            });

        Kokkos::fence();

        DtraceDiff = Dtrace / gamma - HfieldExact;

        dumpVTKScalar(Dtrace, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Dtrace");
        dumpVTKScalar(DtraceDiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "DtraceDiff");

        ///////////////////////////////////////
        // $\nabla \cdot \boldsymbol D = Fd$ //
        ///////////////////////////////////////

        P->extractCols(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        D0div = div(P->D0_m);
        D1div = div(P->D1_m);
        D2div = div(P->D2_m);

        constructVFieldFromFields(Ddiv, D0div, D1div, D2div);

        // DdivDiff = Ddiv - P->Fd_m;
        DdivDiff = Ddiv - FdExact;
        dumpVTKVector(Ddiv, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Ddiv");

        dumpVTKVector(DdivDiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "DdivDiff");

        ///////////////////////////////////////////////
        // COMPUTE RELATIVE ERRORS AND DUMP TO FILES //
        ///////////////////////////////////////////////

        const int shift     = nghost;
        double HrelError    = subfieldNorm(Hdiff, shift) / subfieldNorm(HfieldExact, shift);
        double GrelError    = subfieldNorm(Gdiff, shift) / subfieldNorm(GfieldExact, shift);
        double FdRelError   = L2VectorNorm(FdDiff, shift) / L2VectorNorm(FdExact, shift);
        MatrixD_t DrelError = MFieldRelError(P->D_m, DfieldExact, 2 * shift);
        double DtraceRelError =
            subfieldNorm(DtraceDiff, 2 * shift) / subfieldNorm(HfieldExact, 2 * shift);
        VectorD_t DdivDiffRelError =
            L2VectorNorm(DdivDiff, 2 * shift) / L2VectorNorm(FdExact, 2 * shift);

        std::string convergenceOutDir = OUT_DIR + "/convergenceStats";
        bool writeHeader              = (nv == nvMin);
        dumpCSVScalar(HrelError, "H", nv, writeHeader, convergenceOutDir);
        dumpCSVScalar(GrelError, "G", nv, writeHeader, convergenceOutDir);
        dumpCSVScalar(FdRelError, "Fd", nv, writeHeader, convergenceOutDir);
        dumpCSVMatrix(DrelError, "D", nv, writeHeader, convergenceOutDir);
        dumpCSVScalar(DtraceRelError, "Dtrace", nv, writeHeader, convergenceOutDir);
        dumpCSVVector(DdivDiffRelError, "Ddiv", nv, writeHeader, convergenceOutDir);

        /////////////////////////////
        // WRITE RESULTS TO STDOUT //
        /////////////////////////////

        msg << "h(v) rel. error (" << nv << "^3)"
            << ": " << HrelError << endl;
        msg << "g(v) rel. error (" << nv << "^3)"
            << ": " << GrelError << endl;
        msg << "Fd(v) rel. error (" << nv << "^3)"
            << ": " << FdRelError << endl;
        msg << "D(v) rel. error (" << nv << "^3)"
            << ": " << endl;
        msg << DrelError[0] << endl;
        msg << DrelError[1] << endl;
        msg << DrelError[2] << endl;
        msg << "Tr(D) - h = 0 rel. error (" << nv << "^3)"
            << ": " << DtraceRelError << endl;
        msg << "div(D) - Fd = 0 rel. error (" << nv << "^3)"
            << ": " << DdivDiffRelError << endl;
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
