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

KOKKOS_INLINE_FUNCTION double gaussianPDF(const VectorD_t& v) {
    double vNorm = L2Norm(v);
    double pi    = Kokkos::numbers::pi_v<double>;
    return -1.0 * Kokkos::exp(vNorm * vNorm) / Kokkos::sqrt(8 * pi * pi * pi);
}

KOKKOS_INLINE_FUNCTION double maxwellianHexact(const VectorD_t& v, const double& numberDensity,
                                               const double& vth) {
    double vNorm = L2Norm(v);
    return (2.0 * numberDensity / vNorm) * Kokkos::erf(vNorm / (Kokkos::sqrt(2.0) * vth));
}

KOKKOS_INLINE_FUNCTION double gaussianHexact(const VectorD_t& v) {
    double vNorm = L2Norm(v);
    double pi    = Kokkos::numbers::pi_v<double>;
    return -1.0 * Kokkos::erf(vNorm / Kokkos::sqrt(2.0)) / (32.0 * pi * pi * vNorm);
}

KOKKOS_INLINE_FUNCTION double maxwellianGexact(const VectorD_t& v, const double& numberDensity,
                                               const double& vth) {
    double vNorm   = L2Norm(v);
    double sqrt2   = Kokkos::sqrt(2.0);
    double expTerm = Kokkos::exp(-vNorm * vNorm / (2.0 * vth * vth))
                     / Kokkos::sqrt(Kokkos::numbers::pi_v<double>);
    double erfTerm   = Kokkos::erf(vNorm / (sqrt2 * vth));
    double erfFactor = (vth / (sqrt2 * vNorm)) + (vNorm / (sqrt2 * vth));
    return sqrt2 * numberDensity * vth * (expTerm + erfTerm * erfFactor);
}

KOKKOS_INLINE_FUNCTION double gaussianGexact(const VectorD_t& v) {
    double vNorm   = L2Norm(v);
    double pi      = Kokkos::numbers::pi_v<double>;
    double expTerm = Kokkos::sqrt(2.0 / pi) * Kokkos::exp(-0.5 * vNorm * vNorm);
    double erfTerm = (vNorm + 1.0 / vNorm) * Kokkos::erf(vNorm / Kokkos::sqrt(2.0));
    return (-1.0 / (64.0 * pi * pi)) * (expTerm + erfTerm);
}

KOKKOS_INLINE_FUNCTION double maxwellianD00exact(const VectorD_t& v, const double& gamma,
                                                 const double& numberDensity, const double& vth) {
    double vNorm = L2Norm(v);
    double v2    = vNorm * vNorm;
    return Kokkos::sqrt(2) * gamma * numberDensity * vth
               * (-(1 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * Kokkos::pow(vth, 2)))
                  + Kokkos::pow(v[0], 2)
                        / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * Kokkos::pow(vth, 4)))
               / Kokkos::sqrt(Kokkos::numbers::pi_v<double>)
           + (2 * Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * v[0]
              * (-((vth * v[0]) / (Kokkos::sqrt(2) * Kokkos::pow(v2, 1.5)))
                 + v[0] / (Kokkos::sqrt(2) * vth * vNorm)))
                 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * vNorm)
           + (-((Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * Kokkos::pow(v[0], 2))
                / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * Kokkos::pow(v2, 1.5)))
              + Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>)
                    / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * vNorm)
              - (Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * Kokkos::pow(v[0], 2))
                    / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * Kokkos::pow(vth, 3)
                       * vNorm))
                 * (vth / (Kokkos::sqrt(2) * vNorm) + vNorm / (Kokkos::sqrt(2) * vth))
           + ((vth * ((3 * Kokkos::pow(v[0], 2)) / Kokkos::pow(v2, 2.5) - Kokkos::pow(v2, -1.5)))
                  / Kokkos::sqrt(2)
              + (-(Kokkos::pow(v[0], 2) / Kokkos::pow(v2, 1.5)) + 1 / vNorm)
                    / (Kokkos::sqrt(2) * vth))
                 * Kokkos::erf(vNorm / (Kokkos::sqrt(2) * vth));
}

KOKKOS_INLINE_FUNCTION double maxwellianD01exact(const VectorD_t& v, const double& gamma,
                                                 const double& numberDensity, const double& vth) {
    double vNorm = L2Norm(v);
    double v2    = vNorm * vNorm;
    return Kokkos::sqrt(2.0) * gamma * numberDensity * vth * (v[0] * v[1])
               / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2)))
                  * Kokkos::sqrt(Kokkos::numbers::pi_v<double>) * Kokkos::pow(vth, 4))
           + (Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * v[1]
              * (-((vth * v[0]) / (Kokkos::sqrt(2) * Kokkos::pow(v2, 1.5)))
                 + v[0] / (Kokkos::sqrt(2) * vth * vNorm)))
                 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * vNorm)
           + (Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * v[0]
              * (-((vth * v[1]) / (Kokkos::sqrt(2) * Kokkos::pow(v2, 1.5)))
                 + v[1] / (Kokkos::sqrt(2) * vth * vNorm)))
                 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * vNorm)
           - (Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * v[0] * v[1]
              * (vth / (Kokkos::sqrt(2) * vNorm) + Kokkos::sqrt(v2) / (Kokkos::sqrt(2) * vth)))
                 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * vth * Kokkos::pow(v2, 1.5))
           - (Kokkos::sqrt(2 / Kokkos::numbers::pi_v<double>) * v[0] * v[1]
              * (vth / (Kokkos::sqrt(2) * vNorm) + Kokkos::sqrt(v2) / (Kokkos::sqrt(2) * vth)))
                 / (Kokkos::exp((v2) / (2. * Kokkos::pow(vth, 2))) * Kokkos::pow(vth, 3) * vNorm)
           + ((3 * vth * v[0] * v[1]) / (Kokkos::sqrt(2) * Kokkos::pow(v2, 2.5))
              - (v[0] * v[1]) / (Kokkos::sqrt(2) * vth * Kokkos::pow(v2, 1.5)))
                 * Kokkos::erf(vNorm / (Kokkos::sqrt(2) * vth));
}

KOKKOS_INLINE_FUNCTION double gaussianD00exact(const VectorD_t& v, const double& gamma) {
    // double vNorm = L2Norm(v);
    // double v2    = vNorm * vNorm;
    double pi = Kokkos::numbers::pi_v<double>;
    return gamma * (-1.0 / (64 * pi * pi)) * Kokkos::sqrt(2 / pi)
               * (-Kokkos::exp((-Kokkos::pow(v[0], 2) - Kokkos::pow(v[1], 2) - Kokkos::pow(v[2], 2))
                               / 2.)
                  + Kokkos::exp(
                        (-Kokkos::pow(v[0], 2) - Kokkos::pow(v[1], 2) - Kokkos::pow(v[2], 2)) / 2.)
                        * Kokkos::pow(v[0], 2))
           + (4 * Kokkos::sqrt(2 / pi) * v[0]
              * (-(v[0]
                   / Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 1.5))
                 + v[0]
                       / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                      + Kokkos::pow(v[2], 2))))
                 / Kokkos::exp(
                     Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 2)
                     / 2.)
           + (1 / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2))
              + Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2)))
                 * ((2 * Kokkos::sqrt(2 / pi))
                        / Kokkos::exp(Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                                      + Kokkos::pow(v[2], 2),
                                                  2)
                                      / 2.)
                    - (4 * Kokkos::sqrt(2 / pi) * Kokkos::pow(v[0], 2)
                       * (Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2)))
                          / Kokkos::exp(Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                                        + Kokkos::pow(v[2], 2),
                                                    2)
                                        / 2.))
           + ((3 * Kokkos::pow(v[0], 2))
                  / Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                2.5)
              - Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                            -1.5)
              - Kokkos::pow(v[0], 2)
                    / Kokkos::pow(
                        Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2), 1.5)
              + 1
                    / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                   + Kokkos::pow(v[2], 2)))
                 * Kokkos::erf((Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2))
                               / Kokkos::sqrt(2));
}

KOKKOS_INLINE_FUNCTION double gaussianD01exact(const VectorD_t& v, const double& gamma) {
    double pi = Kokkos::numbers::pi_v<double>;
    return gamma * (-1.0 / (64 * pi * pi))
               * Kokkos::exp((-Kokkos::pow(v[0], 2) - Kokkos::pow(v[1], 2) - Kokkos::pow(v[2], 2))
                             / 2.)
               * Kokkos::sqrt(2 / pi) * v[0] * v[1]
           + (2 * Kokkos::sqrt(2 / pi) * v[1]
              * (-(v[0]
                   / Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 1.5))
                 + v[0]
                       / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                      + Kokkos::pow(v[2], 2))))
                 / Kokkos::exp(
                     Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 2)
                     / 2.)
           + (2 * Kokkos::sqrt(2 / pi) * v[0]
              * (-(v[1]
                   / Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 1.5))
                 + v[1]
                       / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                      + Kokkos::pow(v[2], 2))))
                 / Kokkos::exp(
                     Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 2)
                     / 2.)
           - (4 * Kokkos::sqrt(2 / pi) * v[0] * v[1]
              * (Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2))
              * (1
                     / Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                    + Kokkos::pow(v[2], 2))
                 + Kokkos::sqrt(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2)
                                + Kokkos::pow(v[2], 2))))
                 / Kokkos::exp(
                     Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                 2)
                     / 2.)
           + ((3 * v[0] * v[1])
                  / Kokkos::pow(Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2),
                                2.5)
              - (v[0] * v[1])
                    / Kokkos::pow(
                        Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2), 1.5))
                 * Kokkos::erf((Kokkos::pow(v[0], 2) + Kokkos::pow(v[1], 2) + Kokkos::pow(v[2], 2))
                               / Kokkos::sqrt(2));
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

    constexpr std::string_view testCase = "Gaussian";
    double vth                          = 1.0;
    double numberDensity                = 1.0;
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
        double gamma = P->gamma_m;

        // Create scalar Field for Rosenbluth Potentials
        Field_t<Dim> HfieldExact   = P->fv_m.deepCopy();
        Field_t<Dim> GfieldExact   = P->fv_m.deepCopy();
        Field_t<Dim> D00fieldExact = P->fv_m.deepCopy();
        Field_t<Dim> D01fieldExact = P->fv_m.deepCopy();

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

        // Initialize Maxwellian Velocity Distribution
        const ippl::NDIndex<Dim>& lDom = P->velocitySpaceFieldLayout_m.getLocalNDIndex();
        const int nghost               = P->fv_m.getNghost();
        VectorD_t hv                   = P->hv_m;
        VectorD_t vOrigin              = P->vmin_m;

        Field_view_t fvView         = P->fv_m.getView();
        Field_view_t HviewExact     = HfieldExact.getView();
        Field_view_t GviewExact     = GfieldExact.getView();
        Field_view_t D00viewExact   = D00fieldExact.getView();
        Field_view_t D01viewExact   = D01fieldExact.getView();
        Field_view_t DtraceView     = Dtrace.getView();
        Field_view_t DtraceDiffView = DtraceDiff.getView();

        // Define initial condition and analytical solution as lambda function
        auto initialPDF = [testCase, numberDensity, vth](const VectorD_t& v) {
            if constexpr (testCase == "Maxwellian") {
                return maxwellianPDF(v, numberDensity, vth);
            } else if constexpr (testCase == "Gaussian") {
                return gaussianPDF(v);
            }
        };

        auto Hexact = [testCase, numberDensity, vth](const VectorD_t& v) {
            if constexpr (testCase == "Maxwellian") {
                return maxwellianHexact(v, numberDensity, vth);
            } else if constexpr (testCase == "Gaussian") {
                return gaussianHexact(v);
            }
        };

        auto Gexact = [testCase, numberDensity, vth](const VectorD_t& v) {
            if constexpr (testCase == "Maxwellian") {
                return maxwellianGexact(v, numberDensity, vth);
            } else if constexpr (testCase == "Gaussian") {
                return gaussianGexact(v);
            }
        };

        auto D00exact = [testCase, gamma, numberDensity, vth](const VectorD_t& v) {
            if constexpr (testCase == "Maxwellian") {
                return maxwellianD00exact(v, gamma, numberDensity, vth);
            } else if constexpr (testCase == "Gaussian") {
                return gaussianD00exact(v, gamma);
            }
        };

        auto D01exact = [testCase, gamma, numberDensity, vth](const VectorD_t& v) {
            if constexpr (testCase == "Maxwellian") {
                return maxwellianD01exact(v, gamma, numberDensity, vth);
            } else if constexpr (testCase == "Gaussian") {
                return gaussianD01exact(v, gamma);
            }
        };

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign initial velocity PDF and reference solution for H",
            ippl::getRangePolicy(fvView, 0), KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t<double, Dim> xvec = args;
                for (unsigned d = 0; d < Dim; d++) {
                    xvec[d] = (xvec[d] + lDom[d].first() + 0.5) * hv[d] + vOrigin[d] - hv[d];
                }

                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                ippl::apply(fvView, args)     = initialPDF(xvec) * P->configSpaceIntegral_m;
                ippl::apply(HviewExact, args) = Hexact(xvec) * P->configSpaceIntegral_m;
            });

        Kokkos::fence();

        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, 0, 1.0, OUT_DIR, "fvInit");

        //////////////////////////////////////
        // COMPUTE 1st ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        // Need to scatter rho as we use it as $f(\vec r)$
        P->runSpaceChargeSolver(0);

        // Multiply with prefactor
        // Multiply velSpaceDensity `fv_m` with prefactors defined in RHS of Rosenbluth
        // equations
        // `-1.0` prefactor is **not** needed because we need SOL and not GRAD output of solver
        // Prob. density in configuration space $f(\vec r)$ already added in in initial
        // condition
        P->fv_m = -8.0 * P->pi_m * P->fv_m;

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
        dumpVTKVector(P->Fd_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "gradHappr");
        auto Hdiff = P->fv_m.deepCopy();
        Hdiff      = Hdiff - HfieldExact;
        dumpVTKScalar(Hdiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Hdiff");

        // Multiply with `-1.0 \Gamma` as solver returns $- \nabla H(\vec v)$
        P->Fd_m = -1.0 * gamma * P->Fd_m;
        P->gatherFd();

        // Dump actual friction coefficients
        P->dumpFdField(nv, OUT_DIR);

        //////////////////////////////////////
        // COMPUTE 2nd ROSENBLUTH POTENTIAL //
        //////////////////////////////////////

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign initial velocity PDF and reference solution for G",
            ippl::getRangePolicy(fvView, 0), KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t<double, Dim> xvec = args;
                for (unsigned d = 0; d < Dim; d++) {
                    xvec[d] = (xvec[d] + lDom[d].first() + 0.5) * hv[d] + vOrigin[d] - hv[d];
                }

                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                ippl::apply(fvView, args)     = initialPDF(xvec) * P->configSpaceIntegral_m;
                ippl::apply(GviewExact, args) = Gexact(xvec) * P->configSpaceIntegral_m;

                // First diagonal and off-diagonal entries of Hessian
                ippl::apply(D00viewExact, args) = D00exact(xvec) * P->configSpaceIntegral_m;
                ippl::apply(D01viewExact, args) = D01exact(xvec) * P->configSpaceIntegral_m;
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
        dumpVTKScalar(P->fv_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gappr");
        dumpVTKScalar(GfieldExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gexact");
        dumpVTKScalar(D00fieldExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR,
                      "D00Analytical");
        dumpVTKScalar(D01fieldExact, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR,
                      "D01Analytical");
        auto Gdiff = P->fv_m.deepCopy();
        Gdiff      = Gdiff - GfieldExact;
        dumpVTKScalar(Gdiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Gdiff");

        // Compute Hessian of $g(\vec v)$
        P->D_m = gamma * hess(P->fv_m);

        // Extract rows to separate Vector-Fields
        P->extractRows(P->D_m, P->D0_m, P->D1_m, P->D2_m);

        // Dump actual diffusion coefficients
        // dumpCSVMatrixField(P->D0_m, P->D1_m, P->D2_m, P->nv_m, "D", nv, OUT_DIR);
        dumpVTKVector(P->D0_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D0");
        dumpVTKVector(P->D1_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D1");
        dumpVTKVector(P->D2_m, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "D2");

        ///////////////////////////////////////
        // COMPUTE IDENTITIES THAT MUST HOLD //
        ///////////////////////////////////////

        ////////////////////////////////////
        // $Tr(\boldsymbol D) / \Gamma = h$ //
        ////////////////////////////////////

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

        DdivDiff = Ddiv - P->Fd_m;

        dumpVTKVector(Ddiv, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "Ddiv");
        dumpVTKVector(DdivDiff, P->hv_m, P->nv_m, P->vmin_m, nv, 1.0, OUT_DIR, "DdivDiff");

        /////////////////////////////
        // COMPUTE RELATIVE ERRORS //
        /////////////////////////////

        const int shift  = nghost;
        double HrelError = subfieldNorm(Hdiff, shift) / subfieldNorm(HfieldExact, shift);
        double GrelError = subfieldNorm(Gdiff, shift) / subfieldNorm(GfieldExact, shift);
        double DtraceRelError =
            subfieldNorm(DtraceDiff, 2 * shift) / subfieldNorm(HfieldExact, 2 * shift);
        VectorD_t DdivDiffRelError =
            L2VectorNorm(DdivDiff, shift + 2 * shift) / L2VectorNorm(P->Fd_m, 2 * shift);
        msg << "h(v) rel. error (" << nv << "^3)"
            << ": " << HrelError << endl;
        msg << "g(v) rel. error (" << nv << "^3)"
            << ": " << GrelError << endl;
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
