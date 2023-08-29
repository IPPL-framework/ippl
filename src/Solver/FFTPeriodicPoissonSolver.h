//
// Class FFTPeriodicPoissonSolver
//   Solves periodic electrostatics problems using Fourier transforms
//
//

#ifndef IPPL_FFT_PERIODIC_POISSON_SOLVER_H
#define IPPL_FFT_PERIODIC_POISSON_SOLVER_H

#include <Kokkos_MathematicalConstants.hpp>

#include "Types/ViewTypes.h"

#include "Electrostatics.h"
#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class FFTPeriodicPoissonSolver : public Electrostatics<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Trhs                    = typename FieldRHS::value_type;
        using mesh_type               = typename FieldRHS::Mesh_t;

    public:
        using Field_t   = FieldRHS;
        using FFT_t     = FFT<RCTransform, FieldRHS>;
        using Complex_t = typename FFT_t::Complex_t;
        using CxField_t = typename FFT_t::ComplexField;
        using Layout_t  = FieldLayout<Dim>;
        using Vector_t  = Vector<Trhs, Dim>;

        using Base = Electrostatics<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using scalar_type = typename FieldLHS::Mesh_t::value_type;
        using vector_type = typename FieldLHS::Mesh_t::vector_type;

        FFTPeriodicPoissonSolver()
            : Base() {
            using T = typename FieldLHS::value_type::value_type;
            static_assert(std::is_floating_point<T>::value, "Not a floating point type");

            setDefaultParameters();
        }

        FFTPeriodicPoissonSolver(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs) {
            using T = typename FieldLHS::value_type::value_type;
            static_assert(std::is_floating_point<T>::value, "Not a floating point type");

            setDefaultParameters();
        }

        //~FFTPeriodicPoissonSolver() {}

        void setRhs(rhs_type& rhs) override;

        void solve() override;

    private:
        void initialize();

        std::shared_ptr<FFT_t> fft_mp;
        CxField_t fieldComplex_m;
        CxField_t tempFieldComplex_m;
        NDIndex<Dim> domain_m;
        std::shared_ptr<Layout_t> layoutComplex_mp;

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
                    throw IpplException("FFTPeriodicPoissonSolver::setDefaultParameters",
                                        "Unrecognized heffte communication type");
            }
        }
    };
}  // namespace ippl

#include "Solver/FFTPeriodicPoissonSolver.hpp"
#endif
