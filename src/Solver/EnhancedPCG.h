//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm for matrix-free HPC
//   Follows the algorithm of Kronbichler, Martin & Sashko, Dmytro & Munch, Peter. (2022).
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "SolverAlgorithm.h"
#include "Preconditioner.h"


namespace ippl {

    template <typename OpRet, typename PreRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class PCG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using preconditioner_type = std::function<PreRet(lhs_type)>;
        using operator_type = std::function<OpRet(lhs_type)>;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setOperator(operator_type op) { op_m = std::move(op); }
        void setPreconditioner(preconditioner_type op) {op_preconditioner = std::move(op); }

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
            lhs_type     r(mesh, layout);
            lhs_type     p(mesh, layout);
            lhs_type     v(mesh, layout);
            lhs_type   x_2(mesh , layout);
            lhs_type Minvr(mesh , layout);
            lhs_type Minvv(mesh , layout);

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
            p.setFieldBC(bc);

            T gamma;
            T a,b,c,d,e,f;
            T alpha           = 0;
            T alpha_2         = 0;
            T beta            = 0;
            T beta_2          = 0;

            residueNorm       = std::sqrt(innerProduct(r,r));
            const T tolerance = params.get<T>("tolerance")* norm(rhs);

            lhs_type q(mesh, layout);

            while (iterations_m < maxIterations && residueNorm > tolerance) {
                ++iterations_m;
                Minvr = op_preconditioner(r);
                if (iterations_m>1 && iterations_m%2){
                    lhs = lhs + alpha*p + alpha_2/beta_2*(p - Minvr);
                }
                r = r - alpha*v;
                p = op_preconditioner(r) + beta*p;
                v = op_m(p);
                Minvv = op_preconditioner(v);
                gamma = innerProduct(r,r);
                a = innerProduct(p,v);
                b = innerProduct(r,v);
                c = innerProduct(v,v);
                d = innerProduct(r,Minvr);
                e = innerProduct(r,Minvv);
                f = innerProduct(v,Minvv);
                alpha_2 = alpha;
                alpha = d/a;
                if(std::sqrt(gamma - 2*alpha*b + alpha*alpha*c) < tolerance){
                    if (iterations_m%2){
                        lhs = lhs + alpha*p;
                    }
                    else{
                        lhs = lhs + alpha*p + alpha_2/beta_2*(p - Minvr);
                    }
                    break;
                }
                beta_2 = beta;
                beta = (d - (2*alpha*e)+alpha*alpha*f)/d;
                residueNorm = std::sqrt(innerProduct(r,r));
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

        T getResidue() const { return residueNorm; }

    protected:
        operator_type op_m;
        preconditioner_type op_preconditioner;
        T residueNorm    = 0;
        int iterations_m = 0;
    };

}  // namespace ippl

#endif


