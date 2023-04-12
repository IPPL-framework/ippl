//
//// Class FFTPoissonSolver
////   FFT-based Poisson Solver for open boundaries.
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

#ifndef FFT_POISSON_SOLVER_H_
#define FFT_POISSON_SOLVER_H_

#include "Types/Vector.h"

#include "Field/Field.h"

#include "Electrostatics.h"
#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {
    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    class FFTPoissonSolver : public Electrostatics<Tlhs, Trhs, Dim, Mesh, Centering> {
    public:
        // types for LHS and RHS
        using lhs_type = typename Solver<Tlhs, Trhs, Dim, Mesh, Centering>::lhs_type;
        using rhs_type = typename Solver<Tlhs, Trhs, Dim, Mesh, Centering>::rhs_type;
	using Tg = typename Tlhs::value_type;
        // type of output
        using Base = Electrostatics<Tlhs, Trhs, Dim, Mesh, Centering>;

        // define a type for a 3 dimensional field (e.g. charge density field)
        // define a type of Field with integers to be used for the helper Green's function
        // also define a type for the Fourier transformed complex valued fields
        typedef Field<Trhs, Dim, Mesh, Centering> Field_t;
        typedef Field<Tg, Dim, Mesh, Centering> Field_gt;
        typedef Field<int, Dim, Mesh, Centering> IField_t;
        typedef Field<Kokkos::complex<Trhs>, Dim, Mesh, Centering> CxField_t;
        typedef Field<Kokkos::complex<Tg>, Dim, Mesh, Centering> CxField_gt;
        typedef Vector<Trhs, Dim> Vector_t;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

	// define a type for the 3 dimensional real to complex Fourier transform
        typedef FFT<RCTransform, Dim, Trhs, Mesh, Centering> FFT_t;

        // type for communication buffers
        using buffer_type = Communicate::buffer_type;

        // constructor and destructor
        FFTPoissonSolver(rhs_type& rhs, ParameterList& fftparams, std::string alg);
        FFTPoissonSolver(lhs_type& lhs, rhs_type& rhs, ParameterList& fftparams, std::string alg,
                         int sol = Base::SOL_AND_GRAD);
        ~FFTPoissonSolver();

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
        Mesh* mesh_mp;
        FieldLayout_t* layout_mp;

        // mesh and layout objects for rho2_m
        std::unique_ptr<Mesh> mesh2_m;
        std::unique_ptr<FieldLayout_t> layout2_m;

        // mesh and layout objects for the Fourier transformed Complex fields
        std::unique_ptr<Mesh> meshComplex_m;
        std::unique_ptr<FieldLayout_t> layoutComplex_m;

        // domains for the various fields
        NDIndex<Dim> domain_m;         // original domain, gridsize
        NDIndex<Dim> domain2_m;        // doubled gridsize (2*Nx,2*Ny,2*Nz)
        NDIndex<Dim> domainComplex_m;  // field for the complex values of the RC transformation

        // mesh spacing and mesh size
        Vector_t hr_m;
        Vector<int, Dim> nr_m;

        // string specifying algorithm: Hockney or Vico-Greengard
        std::string alg_m;

        // members for Vico-Greengard
        CxField_gt grnL_m;

        std::unique_ptr<FFT<CCTransform, Dim, Tg, Mesh, Centering>> fft4n_m;

        std::unique_ptr<Mesh> mesh4_m;
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
        }
    };
}  // namespace ippl

#include "FFTPoissonSolver.hpp"

#endif
