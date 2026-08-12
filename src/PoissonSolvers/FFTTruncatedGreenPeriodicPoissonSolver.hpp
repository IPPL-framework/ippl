
namespace ippl {

    /////////////////////////////////////////////////////////////////////////
    // constructor and destructor

    template <typename FieldLHS, typename FieldRHS>
    FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS,
                                           FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver()
        : Base()
        , mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        FFTTruncatedGreenPeriodicPoissonSolver::setDefaultParameters();
    }

    template <typename FieldLHS, typename FieldRHS>
    FFTTruncatedGreenPeriodicPoissonSolver<
        FieldLHS, FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver(rhs_type& rhs,
                                                                    ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        FFTTruncatedGreenPeriodicPoissonSolver::setDefaultParameters();

        this->params_m.merge(params);
        this->params_m.update("output_type", Base::SOL);

        FFTTruncatedGreenPeriodicPoissonSolver::setRhs(rhs);
    }

    template <typename FieldLHS, typename FieldRHS>
    FFTTruncatedGreenPeriodicPoissonSolver<
        FieldLHS, FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver(lhs_type& lhs, rhs_type& rhs,
                                                                    ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        FFTTruncatedGreenPeriodicPoissonSolver::setDefaultParameters();

        this->params_m.merge(params);

        this->setLhs(lhs);
        FFTTruncatedGreenPeriodicPoissonSolver::setRhs(rhs);
    }

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::setRhs(rhs_type& rhs) {
        Base::setRhs(rhs);
        initializeFields();
    }

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::setLhs(lhs_type& lhs) {
        Base::setLhs(lhs);
        if (openSolver_m) {
            openSolver_m->setLhs(lhs);
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // initializeFields method, called in constructor

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::initializeFields() {
        static_assert(
            Dim == 3,
            "Dimension other than 3 not supported in FFTTruncatedGreenPeriodicPoissonSolver!");

        const int configuredBoundary = this->params_m.template get<int>("boundary_type");
        if ((configuredBoundary != BoundaryType::OPEN)
            && (configuredBoundary != BoundaryType::PERIODIC)) {
            throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::initializeFields",
                                "boundary_type must be OPEN or PERIODIC");
        }

        const auto boundaryType = static_cast<BoundaryType>(configuredBoundary);
        if (boundaryInitialized_m && boundaryType != boundaryType_m) {
            throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::initializeFields",
                                "boundary_type cannot be changed after solver initialization");
        }
        boundaryType_m        = boundaryType;
        boundaryInitialized_m = true;

        if (this->params_m.template get<Trhs>("alpha") <= 0) {
            throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::initializeFields",
                                "alpha must be greater than zero");
        }

        if (boundaryType_m == BoundaryType::OPEN) {
            openSolver_m = std::make_unique<OpenSolver_t>();
            openSolver_m->mergeParameters(this->params_m);
            openSolver_m->updateParameter("algorithm", OpenSolver_t::HOCKNEY);
            openSolver_m->updateParameter("greens_function", OpenSolver_t::TRUNCATED);
            openSolver_m->setRhs(*this->rhs_mp);
            if (this->lhs_mp != nullptr) {
                openSolver_m->setLhs(*this->lhs_mp);
            }
            return;
        }

        openSolver_m.reset();

        // get layout and mesh
        layout_mp              = &(this->rhs_mp->getLayout());
        mesh_mp                = &(this->rhs_mp->get_mesh());
        mpi::Communicator comm = layout_mp->comm;

        // get mesh spacing
        hr_m = mesh_mp->getMeshSpacing();

        // get origin
        Vector_t origin = mesh_mp->getOrigin();

        // create domain for the real fields
        domain_m = layout_mp->getDomain();

        // get the mesh spacings and sizes for each dimension
        for (unsigned int i = 0; i < Dim; ++i) {
            nr_m[i] = domain_m[i].length();
        }

        // define decomposition (parallel / serial)
        std::array<bool, Dim> isParallel = layout_mp->isParallel();

        // create the domain for the transformed (complex) fields
        // since we use HeFFTe for the transforms it doesn't require permuting to the right
        // one of the dimensions has only (n/2 +1) as our original fields are fully real
        // the dimension is given by the user via r2c_direction
        unsigned int RCDirection = this->params_m.template get<int>("r2c_direction");
        for (unsigned int i = 0; i < Dim; ++i) {
            if (i == RCDirection)
                domainComplex_m[RCDirection] = Index(nr_m[RCDirection] / 2 + 1);
            else
                domainComplex_m[i] = Index(nr_m[i]);
        }

        // create mesh and layout for the real to complex FFT transformed fields
        using mesh_type = typename lhs_type::Mesh_t;
        meshComplex_m   = std::unique_ptr<mesh_type>(new mesh_type(domainComplex_m, hr_m, origin));
        layoutComplex_m =
            std::unique_ptr<FieldLayout_t>(new FieldLayout_t(comm, domainComplex_m, isParallel));

        // initialize fields
        rhotr_m.initialize(*meshComplex_m, *layoutComplex_m);
        grntr_m.initialize(*meshComplex_m, *layoutComplex_m);
        tempFieldComplex_m.initialize(*meshComplex_m, *layoutComplex_m);

        // create the FFT object
        fft_m = std::make_unique<FFT_t>(*layout_mp, *layoutComplex_m, this->params_m);
        fft_m->warmup(*(this->rhs_mp), rhotr_m);

        // call greensFunction and we will get the transformed G in the class attribute
        // this is done in initialization so that we already have the precomputed fct
        // for all timesteps (green's fct will only change if mesh size changes)

        greensFunction();
    };

    /////////////////////////////////////////////////////////////////////////
    // compute the periodic long-range Ewald potential and field from rho
    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::solve() {
        if (boundaryType_m == BoundaryType::OPEN) {
            if (!openSolver_m) {
                throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::solve",
                                    "Open-boundary solver was not initialized");
            }
            openSolver_m->solve();
            return;
        }

        for (unsigned d = 0; d < Dim; ++d) {
            if ((nr_m[d] % 2) != 0) {
                throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::solve",
                                    "Odd mesh sizes are unsupported by the spectral gradient");
            }
        }

        // get the output type (sol, grad, or sol & grad)
        const int out = this->params_m.template get<int>("output_type");

        // set the mesh & spacing, which may change each timestep
        mesh_mp = &(this->rhs_mp->get_mesh());

        // check whether the mesh spacing has changed with respect to the old one
        // if yes, update and set green flag to true
        bool green = false;
        for (unsigned int i = 0; i < Dim; ++i) {
            if (hr_m[i] != mesh_mp->getMeshSpacing(i)) {
                hr_m[i] = mesh_mp->getMeshSpacing(i);
                green   = true;
            }
        }

        // set mesh spacing on the other grids again
        meshComplex_m->setMeshSpacing(hr_m);

        // forward FFT of the charge density field
        rhotr_m = 0.0;
        fft_m->transform(FORWARD, *(this->rhs_mp), rhotr_m);

        // call greensFunction to recompute if the mesh spacing has changed
        if (green) {
            greensFunction();
        }

        // The minus sign preserves the convention of the real-space kernel:
        // phi_hat = -rho_hat * G_hat.
        rhotr_m = -rhotr_m * grntr_m;

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        if ((out == Base::GRAD) || (out == Base::SOL_AND_GRAD)) {
            // Compute gradient in Fourier space and then
            // take inverse FFT.

            const Trhs pi              = Kokkos::numbers::pi_v<Trhs>;
            Kokkos::complex<Trhs> imag = {0.0, 1.0};

            auto view               = rhotr_m.getView();
            const int nghost        = rhotr_m.getNghost();
            auto tempview           = tempFieldComplex_m.getView();
            auto viewRhs            = this->rhs_mp->getView();
            auto viewLhs            = this->lhs_mp->getView();
            const int nghostL       = this->lhs_mp->getNghost();
            const auto& lDomComplex = layoutComplex_m->getLocalNDIndex();

            // define some member variables in local scope for the parallel_for
            Vector_t hsize     = hr_m;
            Vector<int, Dim> N = nr_m;

            for (size_t gd = 0; gd < Dim; ++gd) {
                ippl::parallel_for(
                    "Gradient FFTPeriodicPoissonSolver", getRangePolicy(view, nghost),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        Vector<int, Dim> iVec = args - nghost;

                        for (unsigned d = 0; d < Dim; ++d) {
                            iVec[d] += lDomComplex[d].first();
                        }

                        Vector_t kVec;

                        for (size_t d = 0; d < Dim; ++d) {
                            const Trhs Len = N[d] * hsize[d];
                            bool shift     = (iVec[d] > (N[d] / 2));
                            bool notMid    = (iVec[d] != (N[d] / 2));
                            // For the noMid part see
                            // https://math.mit.edu/~stevenj/fft-deriv.pdf Algorithm 1
                            kVec[d] = notMid * 2 * pi / Len * (iVec[d] - shift * N[d]);
                        }

                        Trhs Dr = 0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            Dr += kVec[d] * kVec[d];
                        }

                        apply(tempview, args) = apply(view, args);

                        bool isNotZero = (Dr != 0.0);

                        apply(tempview, args) *= -(isNotZero * imag * kVec[gd]);
                    });

                fft_m->transform(BACKWARD, *this->rhs_mp, tempFieldComplex_m);

                ippl::parallel_for(
                    "Assign Gradient FFTPeriodicPoissonSolver", getRangePolicy(viewLhs, nghostL),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        apply(viewLhs, args)[gd] = apply(viewRhs, args);
                    });
            }
        }

        if ((out == Base::SOL) || (out == Base::SOL_AND_GRAD)) {
            // inverse FFT of the product and store the electrostatic potential in rho2_mr
            fft_m->transform(BACKWARD, *(this->rhs_mp), rhotr_m);
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // calculate the periodic Ewald Green's function in Fourier space

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::greensFunction() {
        const Trhs alpha         = this->params_m.template get<Trhs>("alpha");
        const Trhs forceConstant = this->params_m.template get<Trhs>("force_constant");
        const Trhs pi            = Kokkos::numbers::pi_v<Trhs>;
        auto view                = grntr_m.getView();
        const int nghost         = grntr_m.getNghost();
        const auto& lDomComplex  = layoutComplex_m->getLocalNDIndex();
        const Vector_t hsize     = hr_m;
        const Vector<int, Dim> N = nr_m;
        using index_array_type   = typename RangePolicy<Dim>::index_array_type;

        ippl::parallel_for(
            "Assign periodic Ewald Green's function", getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                Vector<int, Dim> iVec = args - nghost;
                for (unsigned d = 0; d < Dim; ++d) {
                    iVec[d] += lDomComplex[d].first();
                }

                Trhs k2 = 0.0;
                for (unsigned d = 0; d < Dim; ++d) {
                    const Trhs length = N[d] * hsize[d];
                    const bool shift  = (iVec[d] > (N[d] / 2));
                    const Trhs kd     = 2.0 * pi / length * (iVec[d] - shift * N[d]);
                    k2 += kd * kd;
                }

                const bool nonzero = (k2 != 0.0);
                const Trhs safeK2  = k2 + ((!nonzero) * 1.0);
                apply(view, args)  = nonzero * 4.0 * pi * forceConstant
                                    * Kokkos::exp(-k2 / (4.0 * alpha * alpha)) / safeK2;
            });
    };

}  // namespace ippl
