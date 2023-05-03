//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//
// Copyright (c) 2021 Alessandro Vinciguerra, ETH Zürich, Zurich, Switzerland
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

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "SolverAlgorithm.h"

namespace ippl {

    template <typename Tlhs, typename Trhs, unsigned Dim, typename OpRet, class Mesh,
              class Centering>
    class PCG : public SolverAlgorithm<Tlhs, Trhs, Dim, Mesh, Centering> {
    public:
        using Base = SolverAlgorithm<Tlhs, Trhs, Dim, Mesh, Centering>;
        using typename Base::lhs_type;
        using typename Base::rhs_type;
        using operator_type = std::function<OpRet(lhs_type)>;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setOperator(operator_type op) { op_m = std::move(op); }

        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return iterations_m; }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            typedef typename lhs_type::type T;

            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r(mesh, layout), d(mesh, layout);

            using bc_type  = BConds<T, lhs_type::dimension, Mesh, Centering>;
            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;

            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] =
                        std::make_shared<PeriodicFace<T, lhs_type::dimension, Mesh, Centering>>(i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i] = std::make_shared<ZeroFace<T, lhs_type::dimension, Mesh, Centering>>(i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }
            d.setFieldBC(bc);

            r = rhs - op_m(lhs);
            // The d field should be a copy of the r field, but deep copies have
            // not yet been implemented for fields, so we need a dummy operation
            // to get an expression, which is then copyable
            // https://gitlab.psi.ch/OPAL/Libraries/ippl/-/issues/80
            d = r * 1;

            T delta1          = innerProduct(r, r);
            T rNorm           = std::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q(mesh, layout);

            while (iterations_m < maxIterations && rNorm > tolerance) {
                q       = op_m(d);
                T alpha = delta1 / innerProduct(d, q);
                lhs     = lhs + alpha * d;

                // The exact residue is given by
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r = r - alpha * q;

                T delta0 = delta1;
                delta1   = innerProduct(r, r);
                T beta   = delta1 / delta0;

                rNorm = std::sqrt(delta1);

                d = r + beta * d;

                ++iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

    protected:
        operator_type op_m;
        int iterations_m = 0;
    };

}  // namespace ippl

#endif
