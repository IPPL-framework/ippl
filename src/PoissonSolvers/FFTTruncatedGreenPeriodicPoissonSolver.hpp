//
// Class FFTTruncatedGreenPeriodicPoissonSolver
//   Poisson solver for periodic boundaries, based on FFTs.
//   Solves laplace(phi) = -rho, and E = -grad(phi).
//
//   Uses a convolution with a Green's function given by:
//      G(r) = forceConstant * erf(alpha * r) / r,
//         alpha = controls long-range interaction.
//
//

namespace ippl {

    /////////////////////////////////////////////////////////////////////////
    // constructor and destructor

    template <typename FieldLHS, typename FieldRHS>
    FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver()
        : Base()
        , mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        FFTTruncatedGreenPeriodicPoissonSolver::setDefaultParameters();
    }

    template <typename FieldLHS, typename FieldRHS>
    FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver(rhs_type& rhs, ParameterList& params)
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
    FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::FFTTruncatedGreenPeriodicPoissonSolver(lhs_type& lhs, rhs_type& rhs, ParameterList& params)
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

    /////////////////////////////////////////////////////////////////////////
    // initializeFields method, called in constructor

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::initializeFields() {
        static_assert(Dim == 3, "Dimension other than 3 not supported in FFTTruncatedGreenPeriodicPoissonSolver!");

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
        grn_m.initialize(*mesh_mp, *layout_mp);
        rhotr_m.initialize(*meshComplex_m, *layoutComplex_m);
        grntr_m.initialize(*meshComplex_m, *layoutComplex_m);
        tempFieldComplex_m.initialize(*meshComplex_m, *layoutComplex_m);

        // create the FFT object
        fft_m = std::make_unique<FFT_t>(*layout_mp, *layoutComplex_m, this->params_m);
        fft_m->warmup(grn_m, grntr_m);  // warmup the FFT object

        // these are fields that are used for calculating the Green's function
        for (unsigned int d = 0; d < Dim; ++d) {
            grnIField_m[d].initialize(*mesh_mp, *layout_mp);

            // get number of ghost points and the Kokkos view to iterate over field
            auto view        = grnIField_m[d].getView();
            const int nghost = grnIField_m[d].getNghost();
            const auto& ldom = layout_mp->getLocalNDIndex();

            // the length of the physical domain
            const int size = nr_m[d];

            // Kokkos parallel for loop to initialize grnIField[d]
            switch (d) {
                case 0:
                    Kokkos::parallel_for(
                        "Helper index Green field initialization",
                        ippl::getRangePolicy(view, nghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            // go from local indices to global
                            const int ig = i + ldom[0].first() - nghost;
                            const int jg = j + ldom[1].first() - nghost;
                            const int kg = k + ldom[2].first() - nghost;

                            // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                            const bool outsideN = (ig >= size / 2);
                            view(i, j, k)       = (size * outsideN - ig) * (size * outsideN - ig);

                            // add 1.0 if at (0,0,0) to avoid singularity
                            const bool isOrig = ((ig == 0) && (jg == 0) && (kg == 0));
                            view(i, j, k) += isOrig * 1.0;
                        });
                    break;
                case 1:
                    Kokkos::parallel_for(
                        "Helper index Green field initialization",
                        ippl::getRangePolicy(view, nghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            // go from local indices to global
                            const int jg = j + ldom[1].first() - nghost;

                            // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                            const bool outsideN = (jg >= size / 2);
                            view(i, j, k)       = (size * outsideN - jg) * (size * outsideN - jg);
                        });
                    break;
                case 2:
                    Kokkos::parallel_for(
                        "Helper index Green field initialization",
                        ippl::getRangePolicy(view, nghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            // go from local indices to global
                            const int kg = k + ldom[2].first() - nghost;

                            // assign (index)^2 if 0 <= index < N, and (2N-index)^2 elsewhere
                            const bool outsideN = (kg >= size / 2);
                            view(i, j, k)       = (size * outsideN - kg) * (size * outsideN - kg);
                        });
                    break;
            }
        }

        // call greensFunction and we will get the transformed G in the class attribute
        // this is done in initialization so that we already have the precomputed fct
        // for all timesteps (green's fct will only change if mesh size changes)

        greensFunction();
    };

    /////////////////////////////////////////////////////////////////////////
    // compute electric potential by solving Poisson's eq given a field rho and mesh spacings hr
    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::solve() {
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

        // forward FFT of the charge density field on doubled grid
        rhotr_m = 0.0;
        fft_m->transform(FORWARD, *(this->rhs_mp), rhotr_m);

        // call greensFunction to recompute if the mesh spacing has changed
        if (green) {
            greensFunction();
        }

        // multiply FFT(rho2)*FFT(green)
        // convolution becomes multiplication in FFT
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
            Vector_t origin    = mesh_mp->getOrigin();

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

            // normalization is double counted due to 2 transforms
            *(this->lhs_mp) = *(this->lhs_mp) * nr_m[0] * nr_m[1] * nr_m[2];
            // discretization of integral requires h^3 factor
            *(this->lhs_mp) = *(this->lhs_mp) * hr_m[0] * hr_m[1] * hr_m[2];
        }

        if ((out == Base::SOL) || (out == Base::SOL_AND_GRAD)) {
            // inverse FFT of the product and store the electrostatic potential in rho2_mr
            fft_m->transform(BACKWARD, *(this->rhs_mp), rhotr_m);

            // normalization is double counted due to 2 transforms
            *(this->rhs_mp) = *(this->rhs_mp) * nr_m[0] * nr_m[1] * nr_m[2];
            // discretization of integral requires h^3 factor
            *(this->rhs_mp) = *(this->rhs_mp) * hr_m[0] * hr_m[1] * hr_m[2];
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // calculate FFT of the Green's function

    template <typename FieldLHS, typename FieldRHS>
    void FFTTruncatedGreenPeriodicPoissonSolver<FieldLHS, FieldRHS>::greensFunction() {
        grn_m = 0.0;

        // This alpha parameter is a choice for the Green's function
        // it controls the "range" of the Green's function (e.g.
        // for the collision modelling method, it indicates
        // the splitting between Particle-Particle interactions
        // and the Particle-Mesh computations).
        const Trhs alpha = this->params_m. template get<Trhs>("alpha");
        const Trhs forceConstant = this->params_m. template get<Trhs>("force_constant");

        // calculate square of the mesh spacing for each dimension
        Vector_t hrsq(hr_m * hr_m);

        // use the grnIField_m helper field to compute Green's function
        for (unsigned int i = 0; i < Dim; ++i) {
            grn_m = grn_m + grnIField_m[i] * hrsq[i];
        }

        typename Field_t::view_type view = grn_m.getView();
        const int nghost                 = grn_m.getNghost();
        const auto& ldom                 = layout_mp->getLocalNDIndex();

        // Kokkos parallel for loop to find (0,0,0) point and regularize
        Kokkos::parallel_for(
            "Assign Green's function ", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local indices to global
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                const bool isOrig = (ig == 0 && jg == 0 && kg == 0);

                Trhs r        = Kokkos::real(Kokkos::sqrt(view(i, j, k)));
                view(i, j, k) = (!isOrig) * forceConstant * (Kokkos::erf(alpha * r) / r);
            });

        // perform the FFT of the Green's function for the convolution
        fft_m->transform(FORWARD, grn_m, grntr_m);
    };

}  // namespace ippl
