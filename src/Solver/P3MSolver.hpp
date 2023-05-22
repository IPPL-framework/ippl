//
//// Class P3MSolver
////   FFT-based Poisson Solver class.
////
//// This file is part of IPPL.
////
//// IPPL is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// You should have received a copy of the GNU General Public License
//// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
////
//

#include <Kokkos_MathematicalFunctions.hpp>
#include <algorithm>

#include "Utility/IpplException.h"

#include "Field/HaloCells.h"
#include "P3MSolver.h"

namespace ippl {

    /////////////////////////////////////////////////////////////////////////
    // constructor and destructor

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::P3MSolver()
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        setDefaultParameters();
    }

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::P3MSolver(rhs_type& rhs, ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        setDefaultParameters();

        this->params_m.merge(params);
        this->params_m.update("output_type", Base::SOL);

        this->setRhs(rhs);
    }

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::P3MSolver(lhs_type& lhs, rhs_type& rhs,
                                                           ParameterList& params)
        : mesh_mp(nullptr)
        , layout_mp(nullptr)
        , meshComplex_m(nullptr)
        , layoutComplex_m(nullptr) {
        setDefaultParameters();

        this->params_m.merge(params);

        this->setLhs(lhs);
        this->setRhs(rhs);
    }

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    void P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::setRhs(rhs_type& rhs) {
        Base::setRhs(rhs);
        initializeFields();
    }

    /////////////////////////////////////////////////////////////////////////
    // initializeFields method, called in constructor

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    void P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::initializeFields() {
        static_assert(Dim == 3, "Dimension other than 3 not supported in P3MSolver!");

        // get layout and mesh
        layout_mp = &(this->rhs_mp->getLayout());
        mesh_mp   = &(this->rhs_mp->get_mesh());

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
        e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; ++d) {
            decomp[d] = layout_mp->getRequestedDistribution(d);
        }

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
        meshComplex_m = std::unique_ptr<Mesh>(new Mesh(domainComplex_m, hr_m, origin));
        layoutComplex_m =
            std::unique_ptr<FieldLayout_t>(new FieldLayout_t(domainComplex_m, decomp));

        // initialize fields
        grn_m.initialize(*mesh_mp, *layout_mp);
        rhotr_m.initialize(*meshComplex_m, *layoutComplex_m);
        grntr_m.initialize(*meshComplex_m, *layoutComplex_m);

        // create the FFT object
        fft_m = std::make_unique<FFT_t>(*layout_mp, *layoutComplex_m, this->params_m);

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
                        ippl::getRangePolicy<3>(view, nghost),
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
                        ippl::getRangePolicy<3>(view, nghost),
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
                        ippl::getRangePolicy<3>(view, nghost),
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
    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    void P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::solve() {
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
        fft_m->transform(+1, *(this->rhs_mp), rhotr_m);

        // call greensFunction to recompute if the mesh spacing has changed
        if (green) {
            greensFunction();
        }

        // multiply FFT(rho2)*FFT(green)
        // convolution becomes multiplication in FFT
        rhotr_m = rhotr_m * grntr_m;

        // if output_type is SOL or SOL_AND_GRAD, we caculate solution
        if ((out == Base::SOL) || (out == Base::SOL_AND_GRAD)) {
            // inverse FFT of the product and store the electrostatic potential in rho2_mr
            fft_m->transform(-1, *(this->rhs_mp), rhotr_m);
        }

        // normalization is double counted due to 2 transforms
        *(this->rhs_mp) = *(this->rhs_mp) * nr_m[0] * nr_m[1] * nr_m[2];
        // discretization of integral requires h^3 factor
        *(this->rhs_mp) = *(this->rhs_mp) * hr_m[0] * hr_m[1] * hr_m[2];

        // if we want gradient of phi = Efield instead of doing grad in Fourier domain
        // this is only possible if SOL_AND_GRAD is output type
        if (out == Base::SOL_AND_GRAD) {
            *(this->lhs_mp) = -grad(*this->rhs_mp);
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // calculate FFT of the Green's function

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    void P3MSolver<Tlhs, Trhs, Dim, Mesh, Centering>::greensFunction() {
        grn_m = 0.0;

        // This alpha parameter is a choice for the Green's function
        // it controls the "range" of the Green's function (e.g.
        // for the P3M collision modelling method, it indicates
        // the splitting between Particle-Particle interactions
        // and the Particle-Mesh computations).
        double alpha = 1e6;

        // calculate square of the mesh spacing for each dimension
        Vector_t hrsq(hr_m * hr_m);

        // use the grnIField_m helper field to compute Green's function
        for (unsigned int i = 0; i < Dim; ++i) {
            grn_m = grn_m + grnIField_m[i] * hrsq[i];
        }

        typename Field_t::view_type view = grn_m.getView();
        const int nghost                 = grn_m.getNghost();
        const auto& ldom                 = layout_mp->getLocalNDIndex();

        constexpr double ke = 2.532638e8;

        // Kokkos parallel for loop to find (0,0,0) point and regularize
        Kokkos::parallel_for(
            "Assign Green's function ", ippl::getRangePolicy<3>(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local indices to global
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                const bool isOrig = (ig == 0 && jg == 0 && kg == 0);

                double r      = Kokkos::real(Kokkos::sqrt(view(i, j, k)));
                view(i, j, k) = (!isOrig) * ke * (Kokkos::erf(alpha * r) / r);
            });

        // perform the FFT of the Green's function for the convolution
        fft_m->transform(+1, grn_m, grntr_m);
    };

}  // namespace ippl
