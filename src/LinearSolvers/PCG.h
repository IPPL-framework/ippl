//
// Class PCG
//   Preconditioned Conjugate Gradient solver algorithm
//

#ifndef IPPL_PCG_H
#define IPPL_PCG_H

#include <algorithm>
#include <array>

#include "FEM/FEMVector.h"
#include "Multigrid.h"
#include "Preconditioner.h"
#include "SolverAlgorithm.h"

namespace ippl {
    namespace pcg_preconditioner_defaults {
        inline constexpr int newton_level          = 5;
        inline constexpr int chebyshev_degree      = 31;
        inline constexpr int richardson_iterations = 4;
        inline constexpr int gauss_seidel_inner    = 2;
        inline constexpr int gauss_seidel_outer    = 2;
        inline constexpr int communication         = 0;
        inline constexpr double ssor_omega         = 1.57079632679;

        // Multigrid preconditioner defaults
        inline constexpr int mg_pre_smooth     = 2;
        inline constexpr int mg_post_smooth    = 2;
        inline constexpr double mg_omega       = 0.8;
        inline constexpr int mg_min_cells = 4;
        inline constexpr bool mg_communication = false;

        inline constexpr std::array<const char*, 8> valid_types = {
            "jacobi",         "newton",       "chebyshev", "richardson",
            "richardson_alt", "gauss-seidel", "ssor",      "multigrid"};

        inline bool is_valid_type(const std::string& type) {
            return std::find(valid_types.begin(), valid_types.end(), type) != valid_types.end();
        }
    }  // namespace pcg_preconditioner_defaults

    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename DiagRet, typename FieldLHS,
              typename FieldRHS = FieldLHS>
    class CG : public SolverAlgorithm<FieldLHS, FieldRHS> {
        using Base = SolverAlgorithm<FieldLHS, FieldRHS>;
        typedef typename Base::lhs_type::value_type T;

      public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type&)>;
        using LowerF       = std::function<LowerRet(lhs_type&)>;
        using UpperF       = std::function<UpperRet(lhs_type&)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type&)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type&)>;
        using DiagF        = std::function<DiagRet(lhs_type&)>;
        using mesh_type    = typename lhs_type::Mesh_t;
        using layout_type  = typename lhs_type::Layout_t;

        virtual ~CG() = default;

        /*
         * Initializes the fields needed for CG operations
         * and avoids allocating them at each solve step.
         * @param mesh The mesh to initialize the field with
         * @param layout The layout to initialize the field with
         */
        virtual void initializeFields(mesh_type& mesh, layout_type& layout) {
            r.initialize(mesh, layout);
            d.initialize(mesh, layout);
            q.initialize(mesh, layout);
        }

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
                pcg_preconditioner_defaults::newton_level,  // This is a dummy default parameter,
                                                            // actual default parameter should be
            // set in main
            [[maybe_unused]] int degree =
                pcg_preconditioner_defaults::chebyshev_degree,  // This is a dummy default
                                                                // parameter, actual default
                                                                // parameter should
            // be set in main
            [[maybe_unused]] int richardson_iterations =
                pcg_preconditioner_defaults::richardson_iterations,  // This is a dummy default
                                                                     // parameter, actual default
            // parameter should be set in main
            [[maybe_unused]] int inner =
                pcg_preconditioner_defaults::gauss_seidel_inner,  // This is a dummy default
                                                                  // parameter, actual default
                                                                  // parameter should be
            // set in main
            [[maybe_unused]] int outer =
                pcg_preconditioner_defaults::gauss_seidel_outer,  // This is a dummy default
                                                                  // parameter, actual default
                                                                  // parameter should be set in main
            [[maybe_unused]] double omega =
                pcg_preconditioner_defaults::ssor_omega,  // This is a dummy default parameter,
                                                          // actual default parameter should be set
                                                          // in main
            [[maybe_unused]] int mg_pre =
                pcg_preconditioner_defaults::mg_pre_smooth,  // This is a dummy default parameter,
                                                             // actual default parameter should be
                                                             // set in main
            [[maybe_unused]] int mg_post =
                pcg_preconditioner_defaults::mg_post_smooth,  // This is a dummy default parameter,
                                                              // actual default parameter should be
                                                              // set in main
            [[maybe_unused]] double mg_omega =
                pcg_preconditioner_defaults::mg_omega,  // This is a dummy default parameter, actual
                                                        // default parameter should be set in main
            [[maybe_unused]] int mg_min_cells_per_rank_per_dim =
                pcg_preconditioner_defaults::mg_min_cells,
            [[maybe_unused]] bool mg_communication =
                pcg_preconditioner_defaults::mg_communication) {}
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        virtual int getIterationCount() { return iterations_m; }

        virtual void operator()(lhs_type& lhs, rhs_type& rhs,
                                const ParameterList& params) override {
            constexpr unsigned Dim = lhs_type::dim;

            static IpplTimings::TimerRef cg_ops    = IpplTimings::getTimer("CG");
            static IpplTimings::TimerRef up_layout = IpplTimings::getTimer("updateLayout");
            static IpplTimings::TimerRef apply     = IpplTimings::getTimer("applyOp");
            static IpplTimings::TimerRef inner     = IpplTimings::getTimer("innerProduct");

            IpplTimings::startTimer(cg_ops);

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            IpplTimings::startTimer(up_layout);
            r.updateLayout(lhs.getLayout());
            d.updateLayout(lhs.getLayout());
            q.updateLayout(lhs.getLayout());
            IpplTimings::stopTimer(up_layout);

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

            IpplTimings::startTimer(apply);
            r = rhs - op_m(lhs);
            IpplTimings::stopTimer(apply);

            d = r.deepCopy();
            d.setFieldBC(bc);

            IpplTimings::startTimer(inner);
            T delta1 = innerProduct(r, d);
            IpplTimings::stopTimer(inner);
            T delta0          = delta1;
            residueNorm       = Kokkos::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            while (iterations_m < maxIterations && residueNorm > tolerance) {
                IpplTimings::startTimer(apply);
                q = op_m(d);
                IpplTimings::stopTimer(apply);

                IpplTimings::startTimer(inner);
                T alpha = delta1 / innerProduct(d, q);
                IpplTimings::stopTimer(inner);
                lhs = lhs + alpha * d;

                // The exact residue is given by
                // r = rhs - op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                r      = r - alpha * q;
                delta0 = delta1;
                IpplTimings::startTimer(inner);
                delta1 = innerProduct(r, r);
                IpplTimings::stopTimer(inner);
                T beta = delta1 / delta0;

                residueNorm = Kokkos::sqrt(delta1);
                d           = r + beta * d;
                ++iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
            IpplTimings::stopTimer(cg_ops);
        }

        virtual T getResidue() const { return residueNorm; }

      protected:
        OperatorF op_m;
        T residueNorm    = 0;
        int iterations_m = 0;

        // Workspaces, allocated once via initializeFields() and reused across
        // solves. Protected so derived solvers (e.g. PCG) can extend the
        // workspace set without redeclaring r, d, q as locals on every
        // operator() call.
        lhs_type r;
        lhs_type d;
        lhs_type q;
    };

    template <typename OperatorRet, typename LowerRet, typename UpperRet, typename UpperLowerRet,
              typename InverseDiagRet, typename T>
    class CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, FEMVector<T>,
             FEMVector<T>> : public SolverAlgorithm<FEMVector<T>, FEMVector<T>> {
        using Base = SolverAlgorithm<FEMVector<T>, FEMVector<T>>;

      public:
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorF    = std::function<OperatorRet(lhs_type&)>;
        using LowerF       = std::function<LowerRet(lhs_type&)>;
        using UpperF       = std::function<UpperRet(lhs_type&)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type&)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type&)>;

        virtual ~CG() = default;

        /*!
         * Sets the differential operator for the conjugate gradient algorithm
         * @param op A function that returns OpRet and takes a field of the LHS type
         */
        virtual void setOperator(OperatorF op) { op_m = std::move(op); }
        virtual void setPreconditioner(
            [[maybe_unused]] OperatorF&& op,  // Operator passed to chebyshev and newton
            [[maybe_unused]] LowerF&& lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] UpperF&& upper,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] UpperLowerF&&
                upper_and_lower,  // Operator passed to 2-step gauss-seidel
            [[maybe_unused]] InverseDiagF&&
                inverse_diagonal,           // Operator passed to jacobi and 2-step gauss-seidel
            [[maybe_unused]] double alpha,  // smallest eigenvalue of the operator
            [[maybe_unused]] double beta,   // largest eigenvalue of the operator
            [[maybe_unused]] std::string preconditioner_type =
                "",  // Name of the preconditioner that should be used
            [[maybe_unused]] int level =
                pcg_preconditioner_defaults::newton_level,  // This is a dummy default parameter,
                                                            // actual default parameter should be
            // set in main
            [[maybe_unused]] int degree =
                pcg_preconditioner_defaults::chebyshev_degree,  // This is a dummy default
                                                                // parameter, actual default
                                                                // parameter should
            // be set in main
            [[maybe_unused]] int richardson_iterations =
                pcg_preconditioner_defaults::richardson_iterations,  // This is a dummy default
                                                                     // parameter, actual default
            // parameter should be set in main
            [[maybe_unused]] int inner =
                pcg_preconditioner_defaults::gauss_seidel_inner,  // This is a dummy default
                                                                  // parameter, actual default
                                                                  // parameter should be
            // set in main
            [[maybe_unused]] int outer =
                pcg_preconditioner_defaults::gauss_seidel_outer  // This is a dummy default
                                                                 // parameter, actual default
                                                                 // parameter should be set in main
        ) {}
        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        virtual int getIterationCount() { return iterations_m; }

        virtual void operator()(lhs_type& lhs, rhs_type& rhs,
                                const ParameterList& params) override {
            // constexpr unsigned Dim             = lhs_type::dim;
            // typename lhs_type::Mesh_t& mesh     = lhs.get_mesh();
            // typename lhs_type::Layout_t& layout = lhs.getLayout();

            iterations_m            = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            lhs_type r = lhs.deepCopy();
            r          = 0;
            lhs_type d = lhs.deepCopy();
            d          = 0;

            r = rhs - op_m(lhs);
            r.setHalo(0);
            d = r;  //.deepCopy();
            // d.setFieldBC(bc);
            T delta1          = innerProduct(r, d);
            T delta0          = delta1;
            residueNorm       = Kokkos::sqrt(delta1);
            const T tolerance = params.get<T>("tolerance") * norm(rhs);

            lhs_type q = lhs.deepCopy();
            q          = 0;

            while (iterations_m < maxIterations && residueNorm > tolerance) {
                q = op_m(d);
                d.setHalo(0);
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
                r.setHalo(0);
                delta1 = innerProduct(r, r);
                T beta = delta1 / delta0;

                residueNorm = Kokkos::sqrt(delta1);
                d           = r + beta * d;
                ++iterations_m;
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
        using OperatorF    = std::function<OperatorRet(lhs_type&)>;
        using LowerF       = std::function<LowerRet(lhs_type&)>;
        using UpperF       = std::function<UpperRet(lhs_type&)>;
        using UpperLowerF  = std::function<UpperLowerRet(lhs_type&)>;
        using InverseDiagF = std::function<InverseDiagRet(lhs_type&)>;
        using DiagF        = std::function<DiagRet(lhs_type&)>;

        using mesh_type   = typename lhs_type::Mesh_t;
        using layout_type = typename lhs_type::Layout_t;

        PCG()
            : CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet, FieldLHS,
                 FieldRHS>()
            , preconditioner_m(nullptr) {};

        /*
         * Allocate the extra PCG work fields once, together with the CG
         * work fields r, d, and q.
         *
         * s and pcond_out are scratch buffers for M^{-1} r. They intentionally
         * keep the default NoBcFace boundary conditions. Only the search
         * direction d gets the physical BCs after the preconditioner result has
         * been copied into it. If these scratch fields inherited periodic BCs,
         * assignments inside the preconditioner would call PeriodicFace::apply
         * and add halo MPI exchanges that did not exist when the preconditioner
         * returned fresh temporary NoBcFace fields.
         */
        void initializeFields(mesh_type& mesh, layout_type& layout) override {
            CG<OperatorRet, LowerRet, UpperRet, UpperLowerRet, InverseDiagRet, DiagRet, FieldLHS,
               FieldRHS>::initializeFields(mesh, layout);
            s.initialize(mesh, layout);
            pcond_out.initialize(mesh, layout);
        }

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
            int level =
                pcg_preconditioner_defaults::newton_level,  // This is a dummy default parameter,
                                                            // actual default parameter should be
            // set in main
            int degree = pcg_preconditioner_defaults::chebyshev_degree,  // This is a dummy default
                                                                         // parameter, actual
                                                                         // default parameter should
            // be set in main
            int richardson_iterations =
                pcg_preconditioner_defaults::richardson_iterations,  // This is a dummy default
                                                                     // parameter, actual default
            // parameter should be set in main
            int inner =
                pcg_preconditioner_defaults::gauss_seidel_inner,  // This is a dummy default
                                                                  // parameter, actual default
                                                                  // parameter should be
            // set in main
            int outer =
                pcg_preconditioner_defaults::gauss_seidel_outer,  // This is a dummy default
                                                                  // parameter, actual default
                                                                  // parameter should be
            // set in main
            double omega = pcg_preconditioner_defaults::ssor_omega,  // This is a dummy default
                                                                     // parameter, actual default
            // parameter should be set in main
            // default = pi/2 as this was found optimal during hyperparameter scan for test case
            // (see
            // https://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/BSc-mbolliger.pdf)
            int mg_pre =
                pcg_preconditioner_defaults::mg_pre_smooth,  // This is a dummy default parameter,
                                                             // actual default parameter should be
                                                             // set in main
            int mg_post =
                pcg_preconditioner_defaults::mg_post_smooth,  // This is a dummy default parameter,
                                                              // actual default parameter should be
                                                              // set in main
            double mg_omega =
                pcg_preconditioner_defaults::mg_omega,  // This is a dummy default parameter, actual
                                                        // default parameter should be set in main
            int mg_min_cells_per_rank_per_dim = pcg_preconditioner_defaults::mg_min_cells,
            bool mg_communication = pcg_preconditioner_defaults::mg_communication) override {
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
            } else if (preconditioner_type == "richardson_alt") {
                preconditioner_m =
                    std::move(std::make_unique<
                              richardson_preconditioner_alt<FieldLHS, OperatorF, InverseDiagF>>(
                        std::move(op), std::move(inverse_diagonal), richardson_iterations));
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
            } else if (preconditioner_type == "multigrid") {
                preconditioner_m =
                    std::move(std::make_unique<multigrid_preconditioner<FieldLHS, OperatorF>>(
                        std::move(op), mg_pre, mg_post, mg_omega, mg_min_cells_per_rank_per_dim,
                        mg_communication));
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

            this->iterations_m      = 0;
            const int maxIterations = params.get<int>("max_iterations");

            // Variable names mostly based on description in
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            // Field layouts are updated such that we don't keep a stale domain decomposition
            // if the lhs layout has been changed during the simulation, for example by the load
            // balancer.
            this->r.updateLayout(lhs.getLayout());
            this->d.updateLayout(lhs.getLayout());
            s.updateLayout(lhs.getLayout());
            pcond_out.updateLayout(lhs.getLayout());
            this->q.updateLayout(lhs.getLayout());

            // Each preconditioner's init_fields() is responsible for being
            // cheap on the steady-state path (refreshing layout in-place, not
            // reallocating).
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

            this->r = rhs - this->op_m(lhs);
            (*preconditioner_m)(this->r, pcond_out);
            this->d = T(1) * pcond_out;
            this->d.setFieldBC(bc);

            T delta1          = innerProduct(this->r, this->d);
            T delta0          = delta1;
            this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));
            const T tolerance = params.get<T>("tolerance") * this->residueNorm;

            while (this->iterations_m<maxIterations&& this->residueNorm> tolerance) {
                // op_m(d) writes its expression into q's existing storage; no
                // allocation, no per-iteration deep copy.
                this->q = this->op_m(this->d);
                T alpha = delta1 / innerProduct(this->d, this->q);
                lhs     = lhs + alpha * this->d;

                // The exact residue is given by
                // r = rhs - BaseCG::op_m(lhs);
                // This correction is generally not used in practice because
                // applying the Laplacian is computationally expensive and
                // the correction does not have a significant effect on accuracy;
                // in some implementations, the correction may be applied every few
                // iterations to offset accumulated floating point errors
                this->r = this->r - alpha * this->q;
                // s := M^{-1} r; preconditioner writes into s (NoBcFace).
                (*preconditioner_m)(this->r, s);

                delta0 = delta1;
                delta1 = innerProduct(this->r, s);

                T beta            = delta1 / delta0;
                this->residueNorm = Kokkos::sqrt(Kokkos::abs(delta1));

                this->d = s + beta * this->d;
                ++this->iterations_m;
            }

            if (allFacesPeriodic) {
                T avg = lhs.getVolumeAverage();
                lhs   = lhs - avg;
            }
        }

      protected:
        std::unique_ptr<preconditioner<FieldLHS>> preconditioner_m;

        /*
         * Persistent preconditioner output buffers. They are allocated in
         * initializeFields() and reused on every solve to avoid per-iteration
         * Field allocation.
         *
         * These buffers are scratch storage, not physical solution fields, so
         * they must keep NoBcFace BCs. The physically meaningful BCs are applied
         * only to d after copying pcond_out into it.
         */

        lhs_type s;
        lhs_type pcond_out;
    };

};  // namespace ippl

#endif
