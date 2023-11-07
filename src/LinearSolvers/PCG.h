//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "SolverAlgorithm.h"
#include "Preconditioner.h"

namespace ippl {
    template <typename OpRet, typename PreRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class CG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using operator_type = std::function<OpRet(lhs_type)>;
        using preconditioner_type = std::function<PreRet(lhs_type)>;

        virtual ~CG() = default;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        virtual void setOperator(operator_type op) {op_m = std::move(op); }
        virtual void setPreconditioner([[maybe_unused]] std::string preconditioner_type="", [[maybe_unused]] unsigned level = 3, [[maybe_unused]] unsigned depth = 7) {}
                /*!
                 * Query how many iterations were required to obtain the solution
                 * the last time this solver was used
                 * @return Iteration count of last solve
                 */
         virtual int getIterationCount() { return iterations_m; }

        virtual void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {

            constexpr unsigned Dim = lhs_type::dim;
            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
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
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r = r - alpha * q;
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

        virtual T getResidue() const { return residueNorm; }

    protected:
        operator_type op_m;
        T residueNorm    = 0;
        int iterations_m = 0;
    };

    template <typename OpRet, typename PreRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class PCG : public CG<OpRet , PreRet, FieldLHS, FieldRHS>{
        using BaseCG = CG<OpRet , PreRet, FieldLHS, FieldRHS>;
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using operator_type = std::function<OpRet(lhs_type)>;

        PCG(): preconditioner_m(nullptr){};
        ~PCG(){
            if (preconditioner_m != nullptr){
                delete preconditioner_m;
                preconditioner_m = nullptr;
           }
        }

        PCG(const PCG& other): preconditioner_m(other.preconditioner_m){}

        PCG& operator=(const PCG& other){
            return *this = PCG(other);
        }


        /*!
        * Sets the differential operator for the conjugate gradient algorithm
        * @param op A function that returns OpRet and takes a field of the LHS type
        */
        void setOperator(operator_type op) override { BaseCG::op_m = std::move(op); }
        virtual void setPreconditioner(std::string preconditioner_type="" , unsigned level = 4, unsigned degree = 15) override{
                    if (preconditioner_type == "jacobi"){
                        preconditioner_m = new jacobi_preconditioner<FieldLHS>();
                    }
                    else if (preconditioner_type == "newton"){
                        preconditioner_m = new polynomial_newton_preconditioner<FieldLHS>(level);
                    }
                    else if (preconditioner_type == "chebyshev"){
                        preconditioner_m = new polynomial_chebyshev_preconditioner<FieldLHS>(degree);
                    }
                    /*else if (preconditioner_type == "incomplete_poisson"){
                        preconditioner_m = new incomplete_poisson_preconditioner<FieldLHS>();
                    }*/
                    else{
                        preconditioner_m = new preconditioner<FieldLHS>();
                    }
            }

        int getIterationCount() override { return BaseCG::iterations_m; }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            BaseCG::iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r(mesh, layout);
            lhs_type d(mesh, layout);
            lhs_type s(mesh, layout);
            lhs_type q(mesh, layout);

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

            r = rhs - BaseCG::op_m(lhs);
            d = preconditioner_m->operator()(r);
            d.setFieldBC(bc);

            T delta1          = innerProduct(r, d);
            T delta0          = delta1;
            BaseCG::residueNorm       = std::sqrt(innerProduct(r,r));
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            while (BaseCG::iterations_m < maxIterations && BaseCG::residueNorm > tolerance) {
            q       = BaseCG::op_m(d);
            T alpha = delta1 / innerProduct(d, q);
            lhs     = lhs + alpha * d;

            // The exact residue is given by
            // r = rhs - BaseCG::op_m(lhs);
            // This correction is generally not used in practice because
            // applying the Laplacian is computationally expensive and
            // the correction does not have a significant effect on accuracy;
            // in some implementations, the correction may be applied every few
            // iterations to offset accumulated floating point errors
            r = r - alpha * q;
            s = preconditioner_m->operator()(r);

            delta0 = delta1;
            delta1   = innerProduct(r, s);

            T beta   = delta1 / delta0;
            BaseCG::residueNorm = std::sqrt(innerProduct(r,r));

            d = s + beta * d;

            ++BaseCG::iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

        T getResidue() const override { return BaseCG::residueNorm; }

    protected:
        preconditioner<FieldLHS>* preconditioner_m;
    };

    template <typename OpRet, typename PreRet, typename FieldLHS, typename FieldRHS = FieldLHS>
    class EnhancedPCG : public PCG<OpRet, PreRet, FieldLHS, FieldRHS> {
        using BaseCG = CG<OpRet , PreRet, FieldLHS, FieldRHS>;
        using BasePCG = PCG<OpRet, PreRet, FieldLHS, FieldRHS>;
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using operator_type = std::function<OpRet(lhs_type)>;

        void setPreconditioner(std::string preconditioner_type , unsigned level = 3, unsigned depth = 7) override {
            if (preconditioner_type == "jacobi"){
                BasePCG::preconditioner_m = jacobi_preconditioner<FieldLHS>();
            }
            else if (preconditioner_type == "newton"){
                BasePCG::preconditioner_m = polynomial_newton_preconditioner<FieldLHS>(level);
            }
            else if (preconditioner_type == "chebyshev"){
                BasePCG::preconditioner_m = polynomial_chebyshev_preconditioner<FieldLHS>(depth);
            }
            else{
                BasePCG::preconditioner_m = preconditioner<FieldLHS>();
            }
        }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            typename lhs_type::Mesh_t mesh = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            BaseCG::iterations_m = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // Enhancing_data_locality_of_the_conjugate_gradient_method_for_high-order_
            // matrix-free_finite-element_implementations
            // https://www.researchgate.net/publication/361849071
            lhs_type r(mesh, layout);
            lhs_type p(mesh, layout);
            lhs_type v(mesh, layout);
            lhs_type Minvr(mesh, layout);
            lhs_type Minvv(mesh, layout);

            using bc_type = BConds<lhs_type, Dim>;
            bc_type lhsBCs = lhs.getFieldBC();
            bc_type bc;

            bool allFacesPeriodic = true;
            for (unsigned int i = 0; i < 2 * Dim; ++i) {
                FieldBC bcType = lhsBCs[i]->getBCType();
                if (bcType == PERIODIC_FACE) {
                    // If the LHS has periodic BCs, so does the residue
                    bc[i] = std::make_shared < PeriodicFace < lhs_type >> (i);
                } else if (bcType & CONSTANT_FACE) {
                    // If the LHS has constant BCs, the residue is zero on the BCs
                    // Bitwise AND with CONSTANT_FACE will succeed for ZeroFace or ConstantFace
                    bc[i] = std::make_shared < ZeroFace < lhs_type >> (i);
                    allFacesPeriodic = false;
                } else {
                    throw IpplException("PCG::operator()",
                                        "Only periodic or constant BCs for LHS supported.");
                    return;
                }
            }

            r = rhs - BaseCG::op_m(lhs);
            p.setFieldBC(bc);
            v.setFieldBC(bc);

            T gamma = 0.;
            T a, b, c, d, e, f;
            T alpha = 0.;
            T alpha_2 = 0.;
            T beta = 0.;
            T beta_2 = 0.;

            //BaseCG::residueNorm = std::sqrt(innerProduct(r, r));
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q(mesh, layout);

            while (BaseCG::iterations_m < maxIterations /*&& BaseCG::residueNorm > tolerance*/) {
                ++BaseCG::iterations_m;
                if (BaseCG::iterations_m > 1 && BaseCG::iterations_m % 2) {
                    lhs = lhs + alpha * p + (alpha_2 / beta_2) * (p - Minvr);
                }
                r = r - alpha * v;
                Minvr = BasePCG::preconditioner_m(r);
                p = Minvr + beta * p;
                v = BaseCG::op_m(p);
                Minvv = BasePCG::preconditioner_m(v);
                gamma = innerProduct(r, r);
                a = innerProduct(p, v);
                b = innerProduct(r, v);
                c = innerProduct(v, v);
                d = innerProduct(r, Minvr);
                e = innerProduct(r, Minvv);
                f = innerProduct(v, Minvv);
                alpha_2 = alpha;
                alpha = d / a;
                if (std::sqrt(gamma - 2 * alpha * b + alpha * alpha * c) < tolerance) {
                    if (BaseCG::iterations_m % 2) {
                        lhs = lhs + alpha * p;
                    } else {
                        lhs = lhs + alpha * p + (alpha_2 / beta_2) * (p - Minvr);
                    }
                    break;
                }
                beta_2 = beta;
                beta = (d - (2 * alpha * e) + alpha * alpha * f) / d;

            }
            BaseCG::residueNorm = std::sqrt(gamma - 2 * alpha * b + alpha * alpha * c);
            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs = lhs - avg;
            }
        }
    };
};// namespace ippl

#endif

