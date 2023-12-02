//
// Class Multigrid
//      Multigrid Solver for Linear System of Equations
//

#ifndef IPPL_MULTIGRID_H
#define IPPL_MULRIGRID_H

#include "SolverAlgorithm.h"
#include "Preconditioner.h"

namespace ippl {
    template <typename OpRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class Multigrid : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using operator_type = std::function<OpRet(lhs_type)>;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setOperator(operator_type op) {op_m = std::move(op); }
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return iterations_m; }

        rhs_type restrict(rhs_type& rhs_old){}

        rhs_type prolongate(rhs_type& rhs_old){}

        //Implements one V-cycle
        void recursive_step(lhs_type& lhs, rhs_type& rhs,int level){
            if (level == 0){
                solver_m(lhs , rhs);
            }
            else{
                //Initialize temporary Fields
                rhs_type residual;
                rhs_type restricted_residual;
                lhs_type restricted_lhs;
                // Pre-smoothening
                smoothening_function_m(lhs , rhs);
                //Restriction
                residual = op_m(lhs);
                residual = rhs - residual;
                restricted_residual = restrict(residual);
                restricted_lhs = 0* restricted_residual;
                // Recursive Call
                recursive_step(mu , restricted_residual, level-1);
                // Prolongation
                lhs = lhs + prolongate(restricted_lhs);
                //Post-smoothening
                smoothening_function_m(lhs , rhs);
            }
        }
        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {

            constexpr unsigned Dim = lhs_type::dim;
            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            lhs_type r(mesh, layout);
            lhs_type d(mesh, layout);

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
            d = r.deepCopy();
            d.setFieldBC(bc);

            T delta1          = innerProduct(r, d);
            T delta0          = delta1;
            residueNorm       = std::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q(mesh, layout);

            while (iterations_m < maxIterations && residueNorm > tolerance) {
                q       = op_m(d);
                T alpha = delta1 / innerProduct(d, q);
                lhs     = lhs + alpha * d;

                // The exact residue is given by
                r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                //r = r - alpha * q;
                delta0 = delta1;
                delta1   = innerProduct(r,r);
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
        operator_type smoothening_function_m;
        operator_type solver_m;

        T residueNorm    = 0;
        int iterations_m = 0;
        int levels_m;

    };
} // namespace ippl

#endif //IPPL_MULRIGRID_H