//
// Class FFTPeriodicPoissonSolver
//   Solves periodic electrostatics problems using Fourier transforms
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen, Switzerland
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

namespace ippl {

    template <typename Tl, typename Tr, unsigned Dim, class Mesh, class Centering>
    void FFTPeriodicPoissonSolver<Tl, Tr, Dim, Mesh, Centering>::setRhs(rhs_type& rhs) {
        Base::setRhs(rhs);
        initialize();
    }

    template <typename Tl, typename Tr, unsigned Dim, class Mesh, class Centering>
    void FFTPeriodicPoissonSolver<Tl, Tr, Dim, Mesh, Centering>::initialize() {
        const Layout_t& layout_r = this->rhs_mp->getLayout();
        domain_m                 = layout_r.getDomain();

        e_dim_tag decomp[Dim];

        NDIndex<Dim> domainComplex;

        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = layout_r.getRequestedDistribution(d);
            if (this->params_m.template get<int>("r2c_direction") == (int)d)
                domainComplex[d] = Index(domain_m[d].length() / 2 + 1);
            else
                domainComplex[d] = Index(domain_m[d].length());
        }

        layoutComplex_mp = std::make_shared<Layout_t>(domainComplex, decomp);

        Vector<double, 3> hComplex      = {1.0, 1.0, 1.0};
        Vector<double, 3> originComplex = {0.0, 0.0, 0.0};
        M meshComplex(domainComplex, hComplex, originComplex);

        fieldComplex_m.initialize(meshComplex, *layoutComplex_mp);

        if (this->params_m.template get<int>("output_type") == Base::GRAD)
            tempFieldComplex_m.initialize(meshComplex, *layoutComplex_mp);

        fft_mp = std::make_shared<FFT_t>(layout_r, *layoutComplex_mp, this->params_m);
    }

    template <typename Tl, typename Tr, unsigned Dim, class Mesh, class Centering>
    void FFTPeriodicPoissonSolver<Tl, Tr, Dim, Mesh, Centering>::solve() {
        fft_mp->transform(1, *this->rhs_mp, fieldComplex_m);

        auto view          = fieldComplex_m.getView();
        const int nghost   = fieldComplex_m.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;

        double pi                 = std::acos(-1.0);
        const M& mesh             = this->rhs_mp->get_mesh();
        const auto& lDomComplex   = layoutComplex_mp->getLocalNDIndex();
        using vector_type         = typename M::vector_type;
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

        switch (this->params_m.template get<int>("output_type")) {
            case Base::SOL: {
                Kokkos::parallel_for(
                    "Solution FFTPeriodicPoissonSolver",
                    mdrange_type({nghost, nghost, nghost},
                                 {view.extent(0) - nghost, view.extent(1) - nghost,
                                  view.extent(2) - nghost}),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        const int ig = i + lDomComplex[0].first() - nghost;
                        const int jg = j + lDomComplex[1].first() - nghost;
                        const int kg = k + lDomComplex[2].first() - nghost;

                        Vector<int, 3> iVec = {ig, jg, kg};
                        Vector_t kVec;

                        for (size_t d = 0; d < Dim; ++d) {
                            const double Len = rmax[d] - origin[d];
                            bool shift       = (iVec[d] > (N[d] / 2));
                            kVec[d]          = 2 * pi / Len * (iVec[d] - shift * N[d]);
                        }

                        double Dr = kVec[0] * kVec[0] + kVec[1] * kVec[1] + kVec[2] * kVec[2];

                        // It would be great if we can remove this conditional
                        if (Dr != 0.0)
                            view(i, j, k) *= 1 / Dr;
                        else
                            view(i, j, k) = 0.0;
                    });

                fft_mp->transform(-1, *this->rhs_mp, fieldComplex_m);

                break;
            }
            case Base::GRAD: {
                // Compute gradient in Fourier space and then
                // take inverse FFT.

                Kokkos::complex<double> imag = {0.0, 1.0};
                auto tempview                = tempFieldComplex_m.getView();
                auto viewRhs                 = this->rhs_mp->getView();
                auto viewLhs                 = this->lhs_mp->getView();
                const int nghostL            = this->lhs_mp->getNghost();

                for (size_t gd = 0; gd < Dim; ++gd) {
                    Kokkos::parallel_for(
                        "Gradient FFTPeriodicPoissonSolver",
                        mdrange_type({nghost, nghost, nghost},
                                     {view.extent(0) - nghost, view.extent(1) - nghost,
                                      view.extent(2) - nghost}),
                        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                            const int ig = i + lDomComplex[0].first() - nghost;
                            const int jg = j + lDomComplex[1].first() - nghost;
                            const int kg = k + lDomComplex[2].first() - nghost;

                            Vector<int, 3> iVec = {ig, jg, kg};
                            Vector_t kVec;

                            for (size_t d = 0; d < Dim; ++d) {
                                const double Len = rmax[d] - origin[d];
                                bool shift       = (iVec[d] > (N[d] / 2));
                                bool notMid      = (iVec[d] != (N[d] / 2));
                                kVec[d]          = notMid * 2 * pi / Len * (iVec[d] - shift * N[d]);
                            }

                            double Dr = kVec[0] * kVec[0] + kVec[1] * kVec[1] + kVec[2] * kVec[2];

                            tempview(i, j, k) = view(i, j, k);

                            // It would be great if we can remove this conditional
                            if (Dr != 0.0)
                                tempview(i, j, k) *= -(imag * kVec[gd] / Dr);
                            else
                                tempview(i, j, k) = 0.0;
                        });

                    fft_mp->transform(-1, *this->rhs_mp, tempFieldComplex_m);

                    Kokkos::parallel_for(
                        "Assign Gradient FFTPeriodicPoissonSolver",
                        mdrange_type({nghostL, nghostL, nghostL},
                                     {viewLhs.extent(0) - nghostL, viewLhs.extent(1) - nghostL,
                                      viewLhs.extent(2) - nghostL}),
                        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                            viewLhs(i, j, k)[gd] = viewRhs(i, j, k);
                        });
                }

                break;
            }

            default:
                throw IpplException("FFTPeriodicPoissonSolver::solve", "Unrecognized output_type");
        }
    }
}  // namespace ippl
