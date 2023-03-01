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

#ifndef IPPL_FFT_PERIODIC_POISSON_SOLVER_H
#define IPPL_FFT_PERIODIC_POISSON_SOLVER_H

#include "Electrostatics.h"
#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"

namespace ippl {

    template <
        typename Tlhs, typename Trhs, unsigned Dim, class M = UniformCartesian<double, Dim>,
        class C = typename M::DefaultCentering>
    class FFTPeriodicPoissonSolver : public Electrostatics<Tlhs, Trhs, Dim, M, C> {
    public:
        using Field_t   = Field<Trhs, Dim>;
        using FFT_t     = FFT<RCTransform, Dim, Trhs>;
        using Complex_t = Kokkos::complex<Trhs>;
        using CxField_t = Field<Complex_t, Dim>;
        using Layout_t  = FieldLayout<Dim>;
        using Vector_t  = Vector<Trhs, Dim>;

        using Base     = Electrostatics<Tlhs, Trhs, Dim, M, C>;
        using lhs_type = typename Solver<Tlhs, Trhs, Dim, M, C>::lhs_type;
        using rhs_type = typename Solver<Tlhs, Trhs, Dim, M, C>::rhs_type;

        FFTPeriodicPoissonSolver() : Base() {
            setDefaultParameters();
        }

        FFTPeriodicPoissonSolver(lhs_type& lhs, rhs_type& rhs) : Base(lhs, rhs) {
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
                    throw IpplException(
                        "FFTPeriodicPoissonSolver::setDefaultParameters",
                        "Unrecognized heffte communication type");
            }
        }
    };
}  // namespace ippl

#include "Solver/FFTPeriodicPoissonSolver.hpp"
#endif
