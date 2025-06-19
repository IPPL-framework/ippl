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

#ifndef IPPL_FFT_TRUNCATED_GREEN_PERIODIC_POISSON_SOLVER_H_SOLVER_H_
#define IPPL_FFT_TRUNCATED_GREEN_PERIODIC_POISSON_SOLVER_H_SOLVER_H_

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
#include "Poisson.h"

namespace ippl {
    template <typename FieldLHS, typename FieldRHS>
    class FFTTruncatedGreenPeriodicPoissonSolver : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Trhs                    = typename FieldRHS::value_type;
        using mesh_type               = typename FieldRHS::Mesh_t;

    public:
        // type of output
        using Base = Poisson<FieldLHS, FieldRHS>;

        // types for LHS and RHS
        using typename Base::lhs_type, typename Base::rhs_type;

        // define a type for the 3 dimensional real to complex Fourier transform
        typedef FFT<RCTransform, FieldRHS> FFT_t;

        // define a type for a 3 dimensional field (e.g. charge density field)
        // define a type of Field with integers to be used for the helper Green's function
        // also define a type for the Fourier transformed complex valued fields
        typedef FieldRHS Field_t;
        typedef Field<int, Dim, mesh_type, typename FieldLHS::Centering_t> IField_t;
        typedef typename FFT_t::ComplexField CxField_t;
        typedef Vector<Trhs, Dim> Vector_t;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // constructor and destructor
        FFTTruncatedGreenPeriodicPoissonSolver();
        FFTTruncatedGreenPeriodicPoissonSolver(rhs_type& rhs, ParameterList& params);
        FFTTruncatedGreenPeriodicPoissonSolver(lhs_type& lhs, rhs_type& rhs, ParameterList& params);
        ~FFTTruncatedGreenPeriodicPoissonSolver() = default;

        // override the setRhs function of the Solver class
        // since we need to call initializeFields()
        void setRhs(rhs_type& rhs) override;

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
        CxField_t tempFieldComplex_m;

        // fields that facilitate the calculation in greensFunction()
        IField_t grnIField_m[Dim];

        // the FFT object
        std::unique_ptr<FFT_t> fft_m;

        // mesh and layout objects for rho_m (RHS)
        mesh_type* mesh_mp;
        FieldLayout_t* layout_mp;

        // mesh and layout objects for the Fourier transformed Complex fields
        std::unique_ptr<mesh_type> meshComplex_m;
        std::unique_ptr<FieldLayout_t> layoutComplex_m;

        // domains for the various fields
        NDIndex<Dim> domain_m;         // physical domain
        NDIndex<Dim> domainComplex_m;  // Fourier domain

        // mesh spacing and mesh size
        Vector_t hr_m;
        Vector<int, Dim> nr_m;

    protected:
        void setDefaultParameters() override {
            using heffteBackend       = typename FFT_t::heffteBackend;
            heffte::plan_options opts = heffte::default_options<heffteBackend>();
            this->params_m.add("use_pencils", opts.use_pencils);
            this->params_m.add("use_reorder", opts.use_reorder);
            this->params_m.add("use_gpu_aware", opts.use_gpu_aware);
            this->params_m.add("r2c_direction", 0);
            this->params_m.template add<Trhs>("alpha", 1);
            this->params_m.template add<Trhs>("force_constant", 1);

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
                    throw IpplException("FFTTruncatedGreenPeriodicPoissonSolver::setDefaultParameters",
                                        "Unrecognized heffte communication type");
            }
        }
    };
}  // namespace ippl

#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.hpp"
#endif // IPPL_FFT_TRUNCATED_GREEN_PERIODIC_POISSON_SOLVER_H_SOLVER_H_
