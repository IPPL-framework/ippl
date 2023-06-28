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
    GenerateTestData(Field_view_t HviewExact, Field_view_t GviewExact, VField_view_t FdViewExact,
                     MField_view_t DviewExact, double numberDensity, double sigma, const Bunch& P)
        : HviewExact_m(HviewExact)
        , GviewExact_m(GviewExact)
        , FdViewExact_m(FdViewExact)
        , DviewExact_m(DviewExact)
        , numberDensity_m(numberDensity)
        , sigma_m(sigma)
        , lDom_m(P->velocitySpaceFieldLayout_m.getLocalNDIndex())
        , P_m(P) {}

    KOKKOS_INLINE_FUNCTION void operator()(const index_array_type& args) const {
        // Use mesh-spacing on original domain for computation of reference solution ([-VMAX,VMAX])
        VectorD_t v = args;
        for (unsigned d = 0; d < Dim; d++) {
            v[d] = (v[d] + lDom_m[d].first() + 0.5) * P_m->hvInit_m[d] + P_m->vminInit_m[d]
                   - P_m->hvInit_m[d];
        }

        double prefactor;

        // Set specific factors depending on Test-Case
        if constexpr (Test == TestCase::MAXWELLIAN) {
            prefactor = numberDensity_m;
        } else if constexpr (Test == TestCase::GAUSSIAN) {
            prefactor = 1.0;
        }

        // Multiply Integral over configuration-space
        prefactor *= P_m->configSpaceIntegral_m;

        // Initialize all needed analytical potentials and their gradient, etc.
        ippl::apply(HviewExact_m, args)  = gaussianHexact(v, sigma_m, prefactor);
        ippl::apply(GviewExact_m, args)  = gaussianGexact(v, sigma_m, prefactor);
        ippl::apply(FdViewExact_m, args) = gaussianFdExact(v, P_m->gamma_m, sigma_m, prefactor);

        // Diffusion Coefficient $D$
        ippl::apply(DviewExact_m, args) = gaussianFullDexact(v, P_m->gamma_m, sigma_m, prefactor);
    }

private:
    const Field_view_t HviewExact_m;
    const Field_view_t GviewExact_m;
    const VField_view_t FdViewExact_m;
    const MField_view_t DviewExact_m;
    const double numberDensity_m;
    const double sigma_m;
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
    const std::string FRICTION_SOLVER = argv[12];
    const std::string OUT_DIR         = argv[13];

    using bunch_type = LangevinParticles<PLayout_t<double, Dim>, double, Dim>;

    /////////////////////////////
    // CONSTANTS FOR MAXELLIAN //
    /////////////////////////////

    constexpr TestCase testType = TestCase::MAXWELLIAN;
    const double vth            = 1.0;
    const double numberDensity  = 1.0;

    const size_t nvMin = 8;
    for (size_t nv = nvMin; nv <= NV_MAX; nv *= 2) {
        /////////////////////////
        // CONFIGURATION SPACE //
        /////////////////////////

        const ippl::NDIndex<Dim> configSpaceIdxDomain(NR, NR, NR);
        ippl::e_dim_tag configSpaceDecomp[Dim] = {ippl::PARALLEL, ippl::PARALLEL, ippl::PARALLEL};

        // Define \sigma of initial distribution
        double vMax;
        double sigma;
        if constexpr (testType == TestCase::MAXWELLIAN) {
            sigma = vth;
            vMax  = 5.0 * sigma;
        } else if constexpr (testType == TestCase::GAUSSIAN) {
            vMax = 5e7;
            // Domain is [-5\sigma, 5\sigma]^3
            sigma = 0.2 * vMax;
        }

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

        ////////////////////
        // VELOCITY SPACE //
        ////////////////////

        /////////////////////////////////
        // LANGEVIN PARTICLE CONTAINER //
        /////////////////////////////////

        std::shared_ptr P = std::make_shared<bunch_type>(
            PL, hr, configSpaceLowerBound, configSpaceUpperBound, configSpaceDecomp, SOLVER_T,
            PARTICLE_CHARGE, PARTICLE_MASS, EPS_INV, Q, NP, DT, nv, vMax);

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

        // Initialize Random Positions of particles in configuration space
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * rank));
        Kokkos::parallel_for(
            nloc, GenerateMaxwellian<VectorD_t, Kokkos::Random_XorShift64_Pool<>>(
                      P->R.getView(), P->P.getView(), 0.0, sigma, BOXL, 2.0 * vMax, rand_pool64));

        // Initialize constant particle attributes
        P->q = PARTICLE_CHARGE;

        Kokkos::fence();
        Ippl::Comm->barrier();

        /////////////////////////////////////////////////
        // INITIALIZE MAXWELLIAN VELOCITY DISTRIBUTION //
        /////////////////////////////////////////////////

        const int nghost = P->fv_m.getNghost();

        Field_view_t fvView         = P->fv_m.getView();
        Field_view_t HviewExact     = HfieldExact.getView();
        Field_view_t GviewExact     = GfieldExact.getView();
        VField_view_t FdViewExact   = FdExact.getView();
        MField_view_t DviewExact    = DfieldExact.getView();
        Field_view_t DtraceView     = Dtrace.getView();
        Field_view_t DtraceDiffView = DtraceDiff.getView();

        GenerateTestData<testType, std::shared_ptr<bunch_type>> dataGenerator(
            HviewExact, GviewExact, FdViewExact, DviewExact, numberDensity, sigma, P);

        ippl::parallel_for("Assign initial velocity PDF and reference solutions",
                           ippl::getRangePolicy(HviewExact, 0), dataGenerator);
        Kokkos::fence();

        //////////////////////////////////////
        // COMPUTE 1st ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        // Need to scatter rho as we use it as $f(\vec r)$
        // P->runSpaceChargeSolver(0);

        // Initialize `fv_m`
        // Scattered quantity should be a density ($\sum_i fv_i = 1$)
        P->p_fv_m = 1.0 / P->globParticleNum_m;
        P->scatterVelSpace();

        // Need to rescale and dump `fv_m` before the solver overwrites it with the potential
        if (nv == 64) {
            P->fv_m = P->fv_m * P->vScalingFactor_m;
            dumpVTKScalar(P->fv_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "fv");
            P->fv_m = P->fv_m / P->vScalingFactor_m;
        }

        /*
         * Multiply velSpaceDensity `fv_m` with prefactors defined in RHS of Rosenbluth equations
         * `-1.0` prefactor is because the solver computes $- \Delta H(\vec v) = - rhs(v)$
         * Multiply with prob. density in configuration space $f(\vec r)$
         */
        P->fv_m = 8.0 * P->pi_m * P->fv_m * P->configSpaceIntegral_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\nabla_v H(\vec v)$, is stored in `Fd_m`
        // $H(\vec v)$, is stored in `fv_m`
        P->frictionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Rescale magnitude of computed $h(v)$ and $\nabla h(v)$ as they are computed on [-1,1]^3
        P->fv_m = P->fv_m * P->vScalingFactor_m;
        P->Fd_m = P->Fd_m * P->vScalingFactor_m * P->vScalingFactor_m;

        // Dump all data of the potentials
        auto Hdiff = P->fv_m.deepCopy();
        Hdiff      = Hdiff - HfieldExact;

        // Sign change needed as solver returns $- \nabla H(\vec)$
        P->Fd_m = -1.0 * P->gamma_m * P->Fd_m;
        // P->gatherFd();

        // // Scale potential back to [-VMAX,VMAX]
        // P->p_Fd_m = P->p_Fd_m * P->vScalingFactor_m * P->vScalingFactor_m;

        auto FdDiff = P->Fd_m.deepCopy();
        FdDiff      = FdDiff - FdExact;

        if (nv == 64) {
            dumpVTKScalar(P->fv_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Happr");
            dumpVTKScalar(HfieldExact, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Hexact");
            dumpVTKScalar(Hdiff, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Hdiff");
            dumpVTKVector(P->Fd_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Fdappr");
            dumpVTKVector(FdExact, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Fdexact");
            dumpVTKVector(FdDiff, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "FdDiff");
        }

        // Dump actual friction coefficients
        // P->dumpFdField(nv, OUT_DIR);

        //////////////////////////////////////
        // COMPUTE 2nd ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        // Initialize `fv_m`
        // Scattered quantity should be a density ($\sum_i fv_i = 1$)
        P->p_fv_m = 1.0 / P->globParticleNum_m;
        P->scatterVelSpace();

        /*
         * Multiply with prefactors defined in RHS of Rosenbluth equations
         * FFTPoissonSolver returns $G(\vec v)$ in `fv_m`
         * Density multiplied with `-1.0` as the solver computes $\Delta \Delta G(\vec v) = -
         * rhs(v)$
         */
        P->fv_m = 8.0 * P->pi_m * P->fv_m * P->configSpaceIntegral_m;

        // Set origin of velocity space mesh to zero (for FFT)
        P->velocitySpaceMesh_m.setOrigin(0.0);

        // Solve for $\Delta_v \Delta_v G(\vec v)$, is stored in `fv_m`
        P->diffusionSolver_mp->solve();

        // Set origin of velocity space mesh to vmin (for scatter / gather)
        P->velocitySpaceMesh_m.setOrigin(P->vmin_m);

        // Rescale magnitude of computed $g(v)$ for dumping as it is computed on [-1,1]^3
        // P->fv_m = P->fv_m * P->vScalingFactor_m / P->vScalingFactor_m / P->vScalingFactor_m;
        P->fv_m = P->fv_m / P->vScalingFactor_m;

        // Dump all data of the potentials
        auto Gdiff = P->fv_m.deepCopy();
        Gdiff      = Gdiff - GfieldExact;

        if (nv == 64) {
            dumpVTKScalar(P->fv_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Gappr");
            dumpVTKScalar(GfieldExact, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Gexact");
            dumpVTKScalar(Gdiff, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "Gdiff");
        }

        // Reset scaling before computing the hessian
        P->fv_m = P->fv_m * P->vScalingFactor_m;

        P->D_m = P->gamma_m * hess(P->fv_m);

        // Apply rescaling
        P->D_m = P->D_m * P->vScalingFactor_m;

        // Gather Hessian to particle attributes
        // P->velocityParticleCheck();
        // P->gatherHessian();

        // P->p_D0_m = P->p_D0_m * P->vScalingFactor_m / P->vScalingFactor_m;
        // P->p_D0_m = P->p_D0_m * P->vScalingFactor_m / P->vScalingFactor_m;
        // P->p_D0_m = P->p_D0_m * P->vScalingFactor_m / P->vScalingFactor_m;

        // Dump Collisional Coefficient avg. from particle attributes
        P->dumpCollisionStatistics(nv, nv == nvMin, OUT_DIR);

        // Extract rows of exact field to separate Vector-Fields
        P->extractRows(DfieldExact, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        if (nv == 64) {
            // dumpCSVMatrixField(P->D0_m, P->D1_m, P->D2_m, P->nv_m, "D", nv, OUT_DIR);
            dumpVTKVector(P->D0_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D0exact");
            dumpVTKVector(P->D1_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D1exact");
            dumpVTKVector(P->D2_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D2exact");
        }

        // Extract rows of approximation to separate Vector-Fields
        P->extractRows(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        if (nv == 64) {
            dumpVTKVector(P->D0_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D0");
            dumpVTKVector(P->D1_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D1");
            dumpVTKVector(P->D2_m, P->hvInit_m, P->nv_m, P->vminInit_m, nv, 1.0, OUT_DIR, "D2");
        }

        ///////////////////////////////////////
        // COMPUTE IDENTITIES THAT MUST HOLD //
        //                                   //
        // $Tr(\boldsymbol D) / \Gamma = h$  //
        ///////////////////////////////////////

        // using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
        // Kokkos::parallel_for(
        //     "Gather trace of $D$",
        //     mdrange_type({nghost, nghost, nghost},
        //                  {DtraceView.extent(0) - nghost, DtraceView.extent(1) - nghost,
        //                   DtraceView.extent(2) - nghost}),
        //     KOKKOS_LAMBDA(const int i, const int j, const int k) {
        //         DtraceView(i, j, k) =
        //             P->D0_m(i, j, k)[0] + P->D1_m(i, j, k)[1] + P->D2_m(i, j, k)[2];
        //     });

        // Kokkos::fence();

        // DtraceDiff = Dtrace / P->gamma_m - HfieldExact;

        //if (nv == 64) {
        // dumpVTKScalar(Dtrace, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Dtrace");
        // dumpVTKScalar(DtraceDiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "DtraceDiff");
        // }

        // ///////////////////////////////////////
        // // $\nabla \cdot \boldsymbol D = Fd$ //
        // ///////////////////////////////////////

        // P->extractCols(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        // D0div = div(P->D0_m);
        // D1div = div(P->D1_m);
        // D2div = div(P->D2_m);

        // constructVFieldFromFields(Ddiv, D0div, D1div, D2div);

        // DdivDiff = Ddiv - FdExact;

        //if (nv == 64) {
        // dumpVTKVector(Ddiv, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Ddiv");
        // dumpVTKVector(DdivDiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "DdivDiff");
        // }

        ///////////////////////////////////////////////
        // COMPUTE RELATIVE ERRORS AND DUMP TO FILES //
        ///////////////////////////////////////////////

        const int shift     = nghost;
        double HrelError    = subfieldNorm(Hdiff, shift) / subfieldNorm(HfieldExact, shift);
        double GrelError    = subfieldNorm(Gdiff, shift) / subfieldNorm(GfieldExact, shift);
        double FdRelError   = L2VectorNorm(FdDiff, shift) / L2VectorNorm(FdExact, shift);
        MatrixD_t DrelError = MFieldRelError(P->D_m, DfieldExact, 2 * shift);
        // double DtraceRelError =
        //     subfieldNorm(DtraceDiff, 2 * shift) / subfieldNorm(HfieldExact, 2 * shift);
        // VectorD_t DdivDiffRelError =
        //     L2VectorNorm(DdivDiff, 2 * shift) / L2VectorNorm(FdExact, 2 * shift);

        std::string convergenceOutDir = OUT_DIR + "/convergenceStats";
        bool writeHeader              = (nv == nvMin);
        dumpCSVScalar(HrelError, "H", nv, writeHeader, convergenceOutDir);
        dumpCSVScalar(GrelError, "G", nv, writeHeader, convergenceOutDir);
        dumpCSVScalar(FdRelError, "Fd", nv, writeHeader, convergenceOutDir);
        dumpCSVMatrix(DrelError, "D", nv, writeHeader, convergenceOutDir);
        // dumpCSVScalar(DtraceRelError, "Dtrace", nv, writeHeader, convergenceOutDir);
        // dumpCSVVector(DdivDiffRelError, "Ddiv", nv, writeHeader, convergenceOutDir);

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
        // msg << "Tr(D) - h = 0 rel. error (" << nv << "^3)"
        //     << ": " << DtraceRelError << endl;
        // msg << "div(D) - Fd = 0 rel. error (" << nv << "^3)"
        //     << ": " << DdivDiffRelError << endl;
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
