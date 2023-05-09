//
//// Class P3MSolver
////   Poisson Solver for preiodic boundaries.
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

#ifndef P3M_SOLVER_H_
#define P3M_SOLVER_H_

#include "Types/Vector.h"

#include "Field/Field.h"

#include "Electrostatics.h"
#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {
    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    class P3MSolver : public Electrostatics<Tlhs, Trhs, Dim, Mesh, Centering> {
    public:
        // types for LHS and RHS
        using lhs_type = typename Solver<Tlhs, Trhs, Dim, Mesh, Centering>::lhs_type;
        using rhs_type = typename Solver<Tlhs, Trhs, Dim, Mesh, Centering>::rhs_type;

        // type of output
        using Base = Electrostatics<Tlhs, Trhs, Dim, Mesh, Centering>;

        // define a type for a 3 dimensional field (e.g. charge density field)
        // define a type of Field with integers to be used for the helper Green's function
        // also define a type for the Fourier transformed complex valued fields
        typedef Field<Trhs, Dim, Mesh, Centering> Field_t;
        typedef Field<int, Dim, Mesh, Centering> IField_t;
        typedef Field<Kokkos::complex<Trhs>, Dim, Mesh, Centering> CxField_t;
        typedef Vector<Trhs, Dim> Vector_t;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // define a type for the 3 dimensional real to complex Fourier transform
        typedef FFT<RCTransform, Dim, Trhs, Mesh, Centering> FFT_t;

        // constructor and destructor
        P3MSolver(rhs_type& rhs, ParameterList& fftparams);
        P3MSolver(lhs_type& lhs, rhs_type& rhs, ParameterList& fftparams,
                  int sol = Base::SOL_AND_GRAD);
        ~P3MSolver() = default;

        // solve the Poisson equation
        // more specifically, compute the scalar potential given a density field rho
        void solve() override;

        // function called in the constructor to initialize the fields
        void initializeFields();

        // compute standard Green's function
        void greensFunction();

    private:
        Field_t grn_m;  // the Green's function

        CxField_t rhotr_m;
        CxField_t grntr_m;

        // fields that facilitate the calculation in greensFunction()
        IField_t grnIField_m[Dim];

        // the FFT object
        std::unique_ptr<FFT_t> fft_m;

        // mesh and layout objects for rho_m (RHS)
        Mesh* mesh_mp;
        FieldLayout_t* layout_mp;

        // mesh and layout objects for the Fourier transformed Complex fields
        std::unique_ptr<Mesh> meshComplex_m;
        std::unique_ptr<FieldLayout_t> layoutComplex_m;

        // domains for the various fields
        NDIndex<Dim> domain_m;         // physical domain
        NDIndex<Dim> domainComplex_m;  // Fourier domain

        // mesh spacing and mesh size
        Vector_t hr_m;
        Vector<int, Dim> nr_m;

    protected:
        virtual void setDefaultParameters() override {
            using heffteBackend       = typename FFT_t::heffteBackend;
            heffte::plan_options opts = heffte::default_options<heffteBackend>();
            this->params_m.add("use_pencils", opts.use_pencils);
            this->params_m.add("use_reorder", opts.use_reorder);
            this->params_m.add("use_gpu_aware", opts.use_gpu_aware);
            this->params_m.add("r2c_direction", 0);

            switch (opts.algorithm) {
                case heffte::reshape_algorithm::alltoall:
                    this->params_m.add("comm", a2a);
                    break;
                case heffte::reshape_algorithm::alltoallv:
                    this->params_m.add("comm", a2av);
                    break;
                case heffte::reshape_algorithm::p2p:
                    this->params_m.add("comm", p2p);
                    break;
                case heffte::reshape_algorithm::p2p_plined:
                    this->params_m.add("comm", p2p_pl);
                    break;
                default:
                    throw IpplException("P3MSolver::setDefaultParameters",
                                        "Unrecognized heffte communication type");
            }
        }
    };
}  // namespace ippl

#include "P3MSolver.hpp"

#endif
