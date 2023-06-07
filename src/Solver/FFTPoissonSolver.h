//
// Class FFTPoissonSolver
//   FFT-based Poisson Solver for open boundaries.
//   Solves laplace(phi) = -rho, and E = -grad(phi).
//
// Copyright (c) 2023, Sonali Mayani,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Types/Vector.h"

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Field/Field.h"

#include "Communicate/Archive.h"
#include "Electrostatics.h"
#include "FFT/FFT.h"
#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    namespace detail {
        /*!
         * Access a view that either contains a vector field or a scalar field
         * in such a way that the correct element access is determined at compile
         * time, reducing the number of functions needed to achieve the same
         * behavior for both kinds of fields
         * @tparam isVec whether the field is a vector field
         * @tparam - the view type
         */
        template <bool isVec, typename>
        struct ViewAccess;

        template <typename View>
        struct ViewAccess<true, View> {
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view, unsigned dim, size_t i,
                                                              size_t j, size_t k) {
                return view(i, j, k)[dim];
            }
        };

        template <typename View>
        struct ViewAccess<false, View> {
            KOKKOS_INLINE_FUNCTION constexpr static auto& get(View&& view,
                                                              [[maybe_unused]] unsigned dim,
                                                              size_t i, size_t j, size_t k) {
                return view(i, j, k);
            }
        };
    }  // namespace detail

    template <typename FieldLHS, typename FieldRHS>
    class FFTPoissonSolver : public Electrostatics<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Trhs                    = typename FieldRHS::value_type;
        using mesh_type               = typename FieldLHS::Mesh_t;
        using Tg                      = typename FieldLHS::value_type::value_type;

    public:
        // type of output
        using Base = Electrostatics<FieldLHS, FieldRHS>;

        // types for LHS and RHS
        using typename Base::lhs_type, typename Base::rhs_type;

        // define a type for the 3 dimensional real to complex Fourier transform
        typedef FFT<RCTransform, FieldRHS> FFT_t;

        // enum type for the algorithm
        enum Algorithm {
            HOCKNEY    = 0b01,
            VICO       = 0b10,
            BIHARMONIC = 0b11
        };

        // define a type for a 3 dimensional field (e.g. charge density field)
        // define a type of Field with integers to be used for the helper Green's function
        // also define a type for the Fourier transformed complex valued fields
        typedef FieldRHS Field_t;
        typedef Field<int, Dim, mesh_type, typename FieldLHS::Centering_t> IField_t;
        typedef Field<Tg, Dim, mesh_type, Centering> Field_gt;
        typedef Field<Kokkos::complex<Tg>, Dim, mesh_type, Centering> CxField_gt;
        typedef typename FFT_t::ComplexField CxField_t;
        typedef Vector<Trhs, Dim> Vector_t;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // type for communication buffers
        using buffer_type = mpi::Communicator::buffer_type;

        // types of mesh and mesh spacing
        using vector_type = typename mesh_type::vector_type;
        using scalar_type = typename mesh_type::value_type;

        // constructor and destructor
        FFTPoissonSolver();
        FFTPoissonSolver(rhs_type& rhs, ParameterList& params);
        FFTPoissonSolver(lhs_type& lhs, rhs_type& rhs, ParameterList& params);
        ~FFTPoissonSolver() = default;

        // override the setRhs function of the Solver class
        // since we need to call initializeFields()
        void setRhs(rhs_type& rhs) override;

        // allows user to set gradient of phi = Efield instead of spectral
        // calculation of Efield (which uses FFTs)
        void setGradFD();

        // solve the Poisson equation using FFT;
        // more specifically, compute the scalar potential given a density field rho using
        void solve() override;

        // compute standard Green's function
        void greensFunction();

        // function called in the constructor to initialize the fields
        void initializeFields();

        // communication used for multi-rank Vico-Greengard's Green's function
        void communicateVico(Vector<int, Dim> size, typename CxField_gt::view_type view_g,
                             const ippl::NDIndex<Dim> ldom_g, const int nghost_g,
                             typename Field_t::view_type view, const ippl::NDIndex<Dim> ldom,
                             const int nghost);

    private:
        // create a field to use as temporary storage
        // references to it can be created to make the code where it is used readable
        Field_t storage_field;

        Field_t& rho2_mr =
            storage_field;  // the charge-density field with mesh doubled in each dimension
        Field_t& grn_mr = storage_field;  // the Green's function

        // rho2tr_m is the Fourier transformed charge-density field
        // domain3_m and mesh3_m are used
        CxField_t rho2tr_m;

        // grntr_m is the Fourier transformed Green's function
        // domain3_m and mesh3_m are used
        CxField_t grntr_m;

        // temp_m field for the E-field computation
        CxField_t temp_m;

        // fields that facilitate the calculation in greensFunction()
        IField_t grnIField_m[Dim];

        // the FFT object
        std::unique_ptr<FFT_t> fft_m;

        // mesh and layout objects for rho_m (RHS)
        mesh_type* mesh_mp;
        FieldLayout_t* layout_mp;

        // mesh and layout objects for rho2_m
        std::unique_ptr<mesh_type> mesh2_m;
        std::unique_ptr<FieldLayout_t> layout2_m;

        // mesh and layout objects for the Fourier transformed Complex fields
        std::unique_ptr<mesh_type> meshComplex_m;
        std::unique_ptr<FieldLayout_t> layoutComplex_m;

        // domains for the various fields
        NDIndex<Dim> domain_m;         // original domain, gridsize
        NDIndex<Dim> domain2_m;        // doubled gridsize (2*Nx,2*Ny,2*Nz)
        NDIndex<Dim> domainComplex_m;  // field for the complex values of the RC transformation

        // mesh spacing and mesh size
        vector_type hr_m;
        Vector<int, Dim> nr_m;

        // string specifying algorithm: Hockney or Vico-Greengard
        std::string alg_m;

        // members for Vico-Greengard
        CxField_gt grnL_m;

        std::unique_ptr<FFT<CCTransform, CxField_gt>> fft4n_m;

        std::unique_ptr<mesh_type> mesh4_m;
        std::unique_ptr<FieldLayout_t> layout4_m;

        NDIndex<Dim> domain4_m;

        // bool indicating whether we want gradient of solution to calculate E field
        bool isGradFD_m;

        // buffer for communication
        detail::FieldBufferData<Trhs> fd_m;

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
                    throw IpplException("FFTPoissonSolver::setDefaultParameters",
                                        "Unrecognized heffte communication type");
            }

            this->params_m.add("algorithm", HOCKNEY);
        }
    };
}  // namespace ippl

#include "Solver/FFTPoissonSolver.hpp"
#endif
