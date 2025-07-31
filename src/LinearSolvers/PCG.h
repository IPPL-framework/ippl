//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include "Preconditioner.h"
#include "SolverAlgorithm.h"

namespace ippl {
    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename DiagRet, typename FieldLHS,
              typename FieldRHS = FieldLHS>
    class CG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type)>;
        using LowerF       = std::function<LowerRet(lhs_type)>;
        using UpperF       = std::function<UpperRet(lhs_type)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type)>;
        using DiagF        = std::function<DiagRet(lhs_type)>;

        virtual ~CG() = default;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        virtual void setOperator(OperatorF op) { op_m = std::move(op); }
        virtual void setPreconditioner(
            [[maybe_unused]] OperatorF&& op,  // Operator passed to chebyshev and newton
            [[maybe_unused]] LowerF&& lower,  // Operator passed to 2-step gauss-seidel and ssor
            [[maybe_unused]] UpperF&& upper,  // Operator passed to 2-step gauss-seidel and ssor
            [[maybe_unused]] UpperLowerF&&
                upper_and_lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] InverseDiagF&&
                inverse_diagonal,  // Operator passed to jacobi, 2-step gauss-seidel and ssor
            [[maybe_unused]] DiagF&& diagonal,  // Operator passed to SSOR
            [[maybe_unused]] double alpha,      // smallest eigenvalue of the operator
            [[maybe_unused]] double beta,       // largest eigenvalue of the operator
            [[maybe_unused]] std::string preconditioner_type =
                "",  // Name of the preconditioner that should be used
            [[maybe_unused]] int level =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int degree =
                31,  // This is a dummy default parameter, actual default parameter should
            // be set in main
            [[maybe_unused]] int richardson_iterations =
                1,  // This is a dummy default parameter, actual default
            // parameter should be set in main
            [[maybe_unused]] int inner =
                5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            [[maybe_unused]] int outer =
                1,  // This is a dummy default parameter, actual default parameter should be
            [[maybe_unused]] double omega =
                1  // This is a dummy default parameter, actual default parameter should be
                   // set in main
        ) {}
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        virtual int getIterationCount() { return iterations_m; }

        virtual void operator()(lhs_type& lhs, rhs_type& rhs,
                                const ParameterList& params) override {
            constexpr unsigned Dim             = lhs_type::dim;
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
                q = op_m(d);

                T alpha = delta1 / innerProduct(d, q);
                lhs     = lhs + alpha * d;

                // The exact residue is given by
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r      = r - alpha * q;
                delta0 = delta1;
                delta1 = innerProduct(r, r);
                T beta = delta1 / delta0;

                residueNorm = std::sqrt(delta1);
                d           = r + beta * d;
                ++iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

        virtual T getResidue() const { return residueNorm; }

    protected:
        OperatorF op_m;
        T residueNorm    = 0;
        int iterations_m = 0;
    };

    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename DiagRet, typename FieldLHS,
              typename FieldRHS = FieldLHS>
    class PCG : public CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet,
                          FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

    public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type)>;
        using LowerF       = std::function<LowerRet(lhs_type)>;
        using UpperF       = std::function<UpperRet(lhs_type)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type)>;
        using DiagF        = std::function<DiagRet(lhs_type)>;

        PCG()
            : CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet, FieldLHS,
                 FieldRHS>()
            , preconditioner_m(nullptr){};

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        void setPreconditioner(
            OperatorF&& op,                   // Operator passed to chebyshev and newton
            LowerF&& lower,                   // Operator passed to 2-step gauss-seidel
            UpperF&& upper,                   // Operator passed to 2-step gauss-seidel
            UpperLowerF&& upper_and_lower,    // Operator passed to 2-step gauss-seidel
            InverseDiagF&& inverse_diagonal,  // Operator passed to jacobi and 2-step gauss-seidel
            DiagF&& diagonal,                 // Operator passed to ssor
            double alpha,                     // smallest eigenvalue of the operator
            double beta,                      // largest eigenvalue of the operator
            std::string preconditioner_type = "",  // Name of the preconditioner that should be used
            int level = 5,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            int degree = 31,  // This is a dummy default parameter, actual default parameter should
            // be set in main
            int richardson_iterations = 4,  // This is a dummy default parameter, actual default
            // parameter should be set in main
            int inner = 2,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            int outer = 2,  // This is a dummy default parameter, actual default parameter should be
            // set in main
            double omega = 1.57079632679  // This is a dummy default parameter, actual default
            // parameter should be set in main
            // default = pi/2 as this was found optimal during hyperparameter scan for test case 
            // (see https://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/BSc-mbolliger.pdf)
            ) override {
            if (preconditioner_type == "jacobi") {
                // Turn on damping parameter
                /*
                double w = 2.0 / ((alpha + beta));
                preconditioner_m = std::move(std::make_unique<jacobi_preconditioner<FieldLHS ,
                InverseDiagF>>(std::move(inverse_diagonal), w));
                */
                preconditioner_m =
                    std::move(std::make_unique<jacobi_preconditioner<FieldLHS, InverseDiagF>>(
                        std::move(inverse_diagonal)));
            } else if (preconditioner_type == "newton") {
                preconditioner_m = std::move(
                    std::make_unique<polynomial_newton_preconditioner<FieldLHS, OperatorF>>(
                        std::move(op), alpha, beta, level, 1e-3));
            } else if (preconditioner_type == "chebyshev") {
                preconditioner_m = std::move(
                    std::make_unique<polynomial_chebyshev_preconditioner<FieldLHS, OperatorF>>(
                        std::move(op), alpha, beta, degree, 1e-3));
            } else if (preconditioner_type == "richardson") {
                preconditioner_m =
                    std::move(std::make_unique<
                              richardson_preconditioner<FieldLHS, UpperLowerF, InverseDiagF>>(
                        std::move(upper_and_lower), std::move(inverse_diagonal),
                        richardson_iterations));
            } else if (preconditioner_type == "gauss-seidel") {
                preconditioner_m = std::move(
                    std::make_unique<gs_preconditioner<FieldLHS, LowerF, UpperF, InverseDiagF>>(
                        std::move(lower), std::move(upper), std::move(inverse_diagonal), inner,
                        outer));
            } else if (preconditioner_type == "ssor") {
                preconditioner_m =
                    std::move(std::make_unique<
                              ssor_preconditioner<FieldLHS, LowerF, UpperF, InverseDiagF, DiagF>>(
                        std::move(lower), std::move(upper), std::move(inverse_diagonal),
                        std::move(diagonal), inner, outer, omega));
            } else {
                preconditioner_m = std::move(std::make_unique<preconditioner<FieldLHS>>());
            }
        }

        void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            if (preconditioner_m == nullptr) {
                throw IpplException("PCG::operator()",
                                    "Preconditioner has not been set for PCG solver");
            }

            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();

            this->iterations_m      = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r(mesh, layout);
            lhs_type d(mesh, layout);
            lhs_type s(mesh, layout);
            lhs_type q(mesh, layout);

            preconditioner_m->init_fields(lhs);

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
            this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));
            const T tolerance = params.get<T>("tolerance") * this->residueNorm;

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
                delta1 = innerProduct(r, s);

                T beta            = delta1 / delta0;
                this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));

                d = s + beta * d;
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

};  // namespace ippl

#endif
