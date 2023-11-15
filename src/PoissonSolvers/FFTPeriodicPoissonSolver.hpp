//
// Class FFTPeriodicPoissonSolver
//   Solves the periodic Poisson problem using Fourier transforms
//   cf. https://math.mit.edu/~stevenj/fft-deriv.pdf Algorithm 5
//
//

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    void FFTPeriodicPoissonSolver<FieldLHS, FieldRHS>::setRhs(rhs_type& rhs) {
        Base::setRhs(rhs);
        initialize();
    }

    template <typename FieldLHS, typename FieldRHS>
    void FFTPeriodicPoissonSolver<FieldLHS, FieldRHS>::initialize() {
        const Layout_t& layout_r = this->rhs_mp->getLayout();
        domain_m                 = layout_r.getDomain();

        NDIndex<Dim> domainComplex;

        vector_type hComplex;
        vector_type originComplex;

        std::array<bool, Dim> isParallel = layout_r.isParallel();
        for (unsigned d = 0; d < Dim; ++d) {
            hComplex[d]      = 1.0;
            originComplex[d] = 0.0;

            if (this->params_m.template get<int>("r2c_direction") == (int)d) {
                domainComplex[d] = Index(domain_m[d].length() / 2 + 1);
            } else {
                domainComplex[d] = Index(domain_m[d].length());
            }
        }

        layoutComplex_mp = std::make_shared<Layout_t>(layout_r.comm, domainComplex, isParallel);

        mesh_type meshComplex(domainComplex, hComplex, originComplex);

        fieldComplex_m.initialize(meshComplex, *layoutComplex_mp);

        if (this->params_m.template get<int>("output_type") == Base::GRAD) {
            tempFieldComplex_m.initialize(meshComplex, *layoutComplex_mp);
        }

        fft_mp = std::make_shared<FFT_t>(layout_r, *layoutComplex_mp, this->params_m);
    }

    template <typename FieldLHS, typename FieldRHS>
    void FFTPeriodicPoissonSolver<FieldLHS, FieldRHS>::solve() {
        fft_mp->transform(FORWARD, *this->rhs_mp, fieldComplex_m);

        auto view        = fieldComplex_m.getView();
        const int nghost = fieldComplex_m.getNghost();

        scalar_type pi            = Kokkos::numbers::pi_v<scalar_type>;
        const mesh_type& mesh     = this->rhs_mp->get_mesh();
        const auto& lDomComplex   = layoutComplex_mp->getLocalNDIndex();
        using vector_type         = typename mesh_type::vector_type;
        const vector_type& origin = mesh.getOrigin();
        const vector_type& hx     = mesh.getMeshSpacing();

        vector_type rmax;
        Vector<int, Dim> N;
        for (size_t d = 0; d < Dim; ++d) {
            N[d]    = domain_m[d].length();
            rmax[d] = origin[d] + (N[d] * hx[d]);
        }

        // Based on output_type calculate either solution
        // or gradient

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        switch (this->params_m.template get<int>("output_type")) {
            case Base::SOL: {
                ippl::parallel_for(
                    "Solution FFTPeriodicPoissonSolver", getRangePolicy(view, nghost),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        Vector<int, Dim> iVec = args - nghost;
                        for (unsigned d = 0; d < Dim; ++d) {
                            iVec[d] += lDomComplex[d].first();
                        }

                        Vector_t kVec;

                        for (size_t d = 0; d < Dim; ++d) {
                            const scalar_type Len = rmax[d] - origin[d];
                            bool shift            = (iVec[d] > (N[d] / 2));
                            kVec[d]               = 2 * pi / Len * (iVec[d] - shift * N[d]);
                        }

                        scalar_type Dr = 0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            Dr += kVec[d] * kVec[d];
                        }

                        bool isNotZero     = (Dr != 0.0);
                        scalar_type factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0)));

                        apply(view, args) *= factor;
                    });

                fft_mp->transform(BACKWARD, *this->rhs_mp, fieldComplex_m);

                break;
            }
            case Base::GRAD: {
                // Compute gradient in Fourier space and then
                // take inverse FFT.

                Complex_t imag    = {0.0, 1.0};
                auto tempview     = tempFieldComplex_m.getView();
                auto viewRhs      = this->rhs_mp->getView();
                auto viewLhs      = this->lhs_mp->getView();
                const int nghostL = this->lhs_mp->getNghost();

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
                                const scalar_type Len = rmax[d] - origin[d];
                                bool shift            = (iVec[d] > (N[d] / 2));
                                bool notMid           = (iVec[d] != (N[d] / 2));
                                // For the noMid part see
                                // https://math.mit.edu/~stevenj/fft-deriv.pdf Algorithm 1
                                kVec[d] = notMid * 2 * pi / Len * (iVec[d] - shift * N[d]);
                            }

                            scalar_type Dr = 0;
                            for (unsigned d = 0; d < Dim; ++d) {
                                Dr += kVec[d] * kVec[d];
                            }

                            apply(tempview, args) = apply(view, args);

                            bool isNotZero     = (Dr != 0.0);
                            scalar_type factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0)));

                            apply(tempview, args) *= -(imag * kVec[gd] * factor);
                        });

                    fft_mp->transform(BACKWARD, *this->rhs_mp, tempFieldComplex_m);

                    ippl::parallel_for(
                        "Assign Gradient FFTPeriodicPoissonSolver",
                        getRangePolicy(viewLhs, nghostL),
                        KOKKOS_LAMBDA(const index_array_type& args) {
                            apply(viewLhs, args)[gd] = apply(viewRhs, args);
                        });
                }

                break;
            }

            default:
                throw IpplException("FFTPeriodicPoissonSolver::solve", "Unrecognized output_type");
        }
    }
}  // namespace ippl
