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
        virtual void setPreconditioner([[maybe_unused]] std::string preconditioner_type="",
                                       [[maybe_unused]] int level = 0,
                                       [[maybe_unused]] int degree = 0,
                                       [[maybe_unused]] int richardson_iterations = 0,
                                       [[maybe_unused]] int inner = 0,
                                       [[maybe_unused]] int outer = 0
                                        ) {}
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
                std::cout << "CG residue " << residueNorm << " iteration " << iterations_m << std::endl;
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

        PCG(): CG<OpRet , PreRet , FieldLHS , FieldRHS>() , preconditioner_m(nullptr){};

        /*!
        * Sets the differential operator for the conjugate gradient algorithm
        * @param op A function that returns OpRet and takes a field of the LHS type
        */
        virtual void setPreconditioner(std::string preconditioner_type="",
                                       int level = 5, // Dummy default parameters true default parameters need to be set in main
                                       int degree = 31,  // Dummy default parameters true default parameters need to be set in main
                                       int richardson_iterations = 1, // Dummy default parameters true default parameters need to be set in main
                                       int inner = 5, // Dummy default parameters true default parameters need to be set in main
                                       int outer = 1  // Dummy default parameters true default parameters need to be set in main
                                               ) override{
                    if (preconditioner_type == "jacobi"){
                        preconditioner_m = std::move(std::make_unique<jacobi_preconditioner<FieldLHS>>());
                    }
                    else if (preconditioner_type == "newton"){
                        preconditioner_m = std::move(std::make_unique<polynomial_newton_preconditioner<FieldLHS>>(level , 1e-3));
                    }
                    else if (preconditioner_type == "chebyshev"){
                        preconditioner_m = std::move(std::make_unique<polynomial_chebyshev_preconditioner<FieldLHS>>(degree , 1e-3));
                    }
                    else if (preconditioner_type == "richardson"){
                        preconditioner_m = std::move(std::make_unique<richardson_preconditioner<FieldLHS>>(richardson_iterations));
                    }
                    else if (preconditioner_type == "gauss-seidel"){
                        preconditioner_m = std::move(std::make_unique<gs_preconditioner<FieldLHS>>(inner , outer));
                    }
                    else{
                        preconditioner_m = std::move(std::make_unique<preconditioner<FieldLHS>>());
                    }
            }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            this->iterations_m            = 0;
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

            r = rhs - this->op_m(lhs);
            d = preconditioner_m->operator()(r);
            d.setFieldBC(bc);

            T delta1          = innerProduct(r, d);
            T delta0          = delta1;
            this->residueNorm       = std::sqrt(std::abs(delta1));
            const T tolerance = params.get<T>("tolerance") * delta1;

            while (this->iterations_m < maxIterations && this->residueNorm > tolerance) {
            q       = this->op_m(d);
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
            this->residueNorm = std::sqrt(std::abs(delta1));

            d = s + beta * d;
            std::cout << "PCG residue " << this->residueNorm << " iteration " << this->iterations_m << std::endl;

            ++this->iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

    protected:
        std::unique_ptr<preconditioner<FieldLHS>> preconditioner_m;
    };

    /* COMPLETELY OUTDATED NEEDS TO BE REWRITTEN FROM SCRATCH
     * Prototype of new PCG algorithm https://www.researchgate.net/publication/361849071
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

            while (BaseCG::iterations_m < maxIterations) {
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
    };*/

};// namespace ippl

#endif

