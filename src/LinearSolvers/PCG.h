//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "SolverAlgorithm.h"

namespace ippl {

    template <typename OpRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class PCG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
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
            constexpr unsigned Dim = lhs_type::dim;

            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r(mesh, layout);

            using bc_type  = BConds<lhs_type, Dim>;
            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;

            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] = std::make_shared<PeriodicFace<lhs_type>>(i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i]            = std::make_shared<ZeroFace<lhs_type>>(i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }

            r = rhs - op_m(lhs);

            lhs_type d = r.deepCopy();
            d.setFieldBC(bc);

            T delta1          = innerProduct(r, r);
            residueNorm       = std::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q(mesh, layout);

            while (iterations_m < maxIterations && residueNorm > tolerance) {
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

                residueNorm = std::sqrt(delta1);

                d = r + beta * d;

                ++iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

        T getResidue() const { return residueNorm; }

    protected:
        operator_type op_m;
        T residueNorm    = 0;
        int iterations_m = 0;
    };

}  // namespace ippl

#endif
