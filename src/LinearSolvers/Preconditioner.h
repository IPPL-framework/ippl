//
// General pre-conditioners for the pre-conditioned Conjugate Gradient Solver
// such as Jacobi, Polynomial Newton, Polynomial Chebyshev, Richardson, Gauss-Seidel
// provides a function operator() that returns the preconditioned field
//

#ifndef IPPL_PRECONDITIONER_H
#define IPPL_PRECONDITIONER_H

#include "Expression/IpplOperations.h"  // get the function apply()

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type arg) {                              \
        return fun(arg);                        \
    }

namespace ippl {
    template <typename Field>
    struct preconditioner {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        preconditioner()
            : type_m("Identity") {}

        preconditioner(std::string name)
            : type_m(name) {}

        virtual ~preconditioner() = default;

        // Placeholder for the function operator, actually implemented in the derived classes
        virtual Field operator()(Field& u) {
            Field res = u.deepCopy();
            return res;
        }

        // Placeholder for setting additional fields, actually implemented in the derived classes
        virtual void init_fields(Field& b) {
            Field res = b.deepCopy();
            return;
        }

        std::string get_type() { return type_m; };

    protected:
        std::string type_m;
    };

    /*!
     * Jacobi preconditioner
     * M = w*diag{A} // w is a damping factor
     */
    template <typename Field, typename InvDiagF>
    struct jacobi_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        jacobi_preconditioner(InvDiagF&& inverse_diagonal, double w = 1.0)
            : preconditioner<Field>("jacobi")
            , w_m(w) {
            inverse_diagonal_m = std::move(inverse_diagonal);
        }

        Field operator()(Field& u) override {
            mesh_type& mesh     = u.get_mesh();
            layout_type& layout = u.getLayout();
            Field res(mesh, layout);

            res = inverse_diagonal_m(u);
            res = w_m * res;
            return res;
        }

    protected:
        InvDiagF inverse_diagonal_m;
        double w_m;  // Damping factor
    };

    /*!
     * Polynomial Newton Preconditioner
     * Computes iteratively approximations for A^-1
     */
    template <typename Field, typename OperatorF>
    struct polynomial_newton_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        polynomial_newton_preconditioner(OperatorF&& op, double alpha, double beta,
                                         unsigned int max_level = 6, double zeta = 1e-3,
                                         double* eta = nullptr)
            : preconditioner<Field>("polynomial_newton")
            , alpha_m(alpha)
            , beta_m(beta)
            , level_m(max_level)
            , zeta_m(zeta)
            , eta_m(eta) {
            op_m = std::move(op);
        }

        // Memory management is needed because of runtime defined eta_m
        ~polynomial_newton_preconditioner() {
            if (eta_m != nullptr) {
                delete[] eta_m;
                eta_m = nullptr;
            }
        }

        polynomial_newton_preconditioner(const polynomial_newton_preconditioner& other)
            : preconditioner<Field>("polynomial_newton")
            , level_m(other.level_m)
            , alpha_m(other.alpha_m)
            , beta_m(other.beta_m)
            , zeta_m(other.zeta_m)
            , eta_m(other.eta_m) {
            op_m = std::move(other.op_m);
        }

        polynomial_newton_preconditioner& operator=(const polynomial_newton_preconditioner& other) {
            return *this = polynomial_newton_preconditioner(other);
        }

        Field recursive_preconditioner(Field& u, unsigned int level) {
            mesh_type& mesh     = u.get_mesh();
            layout_type& layout = u.getLayout();
            // Define etas if not defined yet
            if (eta_m == nullptr) {
                // Precompute the etas for later use
                eta_m    = new double[level_m + 1];
                eta_m[0] = 2.0 / ((alpha_m + beta_m) * (1.0 + zeta_m));
                if (level_m > 0) {
                    eta_m[1] =
                        2.0
                        / (1.0 + 2 * alpha_m * eta_m[0] - alpha_m * eta_m[0] * alpha_m * eta_m[0]);
                }
                for (unsigned int i = 2; i < level_m + 1; ++i) {
                    eta_m[i] = 2.0 / (1.0 + 2 * eta_m[i - 1] - eta_m[i - 1] * eta_m[i - 1]);
                }
            }

            Field res(mesh, layout);
            // Base case
            if (level == 0) {
                res = eta_m[0] * u;
                return res;
            }
            // Recursive case
            Field PAPr(mesh, layout);
            Field Pr(mesh, layout);

            Pr   = recursive_preconditioner(u, level - 1);
            PAPr = op_m(Pr);
            PAPr = recursive_preconditioner(PAPr, level - 1);
            res  = eta_m[level] * (2.0 * Pr - PAPr);
            return res;
        }

        Field operator()(Field& u) override { return recursive_preconditioner(u, level_m); }

    protected:
        OperatorF op_m;        // Operator to be preconditioned
        double alpha_m;        // Smallest Eigenvalue
        double beta_m;         // Largest Eigenvalue
        unsigned int level_m;  // Number of recursive calls
        double zeta_m;  // smallest (alpha + beta) is multiplied by (1+zeta) to avoid clustering of
                        // Eigenvalues
        double* eta_m = nullptr;  // Size is determined at runtime
    };

    /*!
     * Polynomial Chebyshev Preconditioner
     * Computes iteratively approximations for A^-1
     */
    template <typename Field, typename OperatorF>
    struct polynomial_chebyshev_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        polynomial_chebyshev_preconditioner(OperatorF&& op, double alpha, double beta,
                                            unsigned int degree = 63, double zeta = 1e-3)
            : preconditioner<Field>("polynomial_chebyshev")
            , alpha_m(alpha)
            , beta_m(beta)
            , degree_m(degree)
            , zeta_m(zeta)
            , rho_m(nullptr) {
            op_m = std::move(op);
        }

        // Memory management is needed because of runtime defined rho_m
        ~polynomial_chebyshev_preconditioner() {
            if (rho_m != nullptr) {
                delete[] rho_m;
                rho_m = nullptr;
            }
        }

        polynomial_chebyshev_preconditioner(const polynomial_chebyshev_preconditioner& other)
            : preconditioner<Field>("polynomial_chebyshev")
            , degree_m(other.degree_m)
            , theta_m(other.theta_m)
            , sigma_m(other.sigma_m)
            , delta_m(other.delta_m)
            , alpha_m(other.delta_m)
            , beta_m(other.delta_m)
            , zeta_m(other.zeta_m)
            , rho_m(other.rho_m) {
            op_m = std::move(other.op_m);
        }

        polynomial_chebyshev_preconditioner& operator=(
            const polynomial_chebyshev_preconditioner& other) {
            return *this = polynomial_chebyshev_preconditioner(other);
        }

        Field operator()(Field& r) override {
            mesh_type& mesh     = r.get_mesh();
            layout_type& layout = r.getLayout();

            Field res(mesh, layout);
            Field x(mesh, layout);
            Field x_old(mesh, layout);
            Field A(mesh, layout);
            Field z(mesh, layout);

            // Precompute the coefficients if not done yet
            if (rho_m == nullptr) {
                // Start precomputing the coefficients
                theta_m = (beta_m + alpha_m) / 2.0 * (1.0 + zeta_m);
                delta_m = (beta_m - alpha_m) / 2.0;
                sigma_m = theta_m / delta_m;

                rho_m    = new double[degree_m + 1];
                rho_m[0] = 1.0 / sigma_m;
                for (unsigned int i = 1; i < degree_m + 1; ++i) {
                    rho_m[i] = 1.0 / (2.0 * sigma_m - rho_m[i - 1]);
                }
            }  // End of precomputing the coefficients

            res = r.deepCopy();

            x_old = r / theta_m;
            A     = op_m(r);
            x     = 2.0 * rho_m[1] / delta_m * (2.0 * r - A / theta_m);

            if (degree_m == 0) {
                return x_old;
            }

            if (degree_m == 1) {
                return x;
            }
            for (unsigned int i = 2; i < degree_m + 1; ++i) {
                A     = op_m(x);
                z     = 2.0 / delta_m * (r - A);
                res   = rho_m[i] * (2 * sigma_m * x - rho_m[i - 1] * x_old + z);
                x_old = x.deepCopy();
                x     = res.deepCopy();
            }
            return res;
        }

    protected:
        OperatorF op_m;
        double alpha_m;
        double beta_m;
        double delta_m;
        double theta_m;
        double sigma_m;
        unsigned degree_m;
        double zeta_m;
        double* rho_m = nullptr;  // Size is determined at runtime
    };

    /*!
     * Richardson preconditioner
     */
    template <typename Field, typename UpperAndLowerF, typename InvDiagF>
    struct richardson_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        richardson_preconditioner(UpperAndLowerF&& upper_and_lower, InvDiagF&& inverse_diagonal,
                                  unsigned innerloops = 5)
            : preconditioner<Field>("Richardson")
            , innerloops_m(innerloops) {
            upper_and_lower_m  = std::move(upper_and_lower);
            inverse_diagonal_m = std::move(inverse_diagonal);
        }

        Field operator()(Field& r) override {
            mesh_type& mesh     = r.get_mesh();
            layout_type& layout = r.getLayout();
            Field g(mesh, layout);

            g = 0;
            for (unsigned int j = 0; j < innerloops_m; ++j) {
                ULg_m = upper_and_lower_m(g);
                g     = r - ULg_m;
         
                // The inverse diagonal is applied to the
                // vector itself to return the result usually.
                // However, the operator for FEM already
                // returns the result of inv_diag * itself
                // due to the matrix-free evaluation.
                // Therefore, we need this if to differentiate
                // the two cases.
                if constexpr (std::is_same_v<InvDiagF, double>) {
                    g = inverse_diagonal_m(g) * g;
                } else {
                    g = inverse_diagonal_m(g);
                }
            }
            return g;
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();

            ULg_m = Field(mesh, layout);
        }

    protected:
        UpperAndLowerF upper_and_lower_m;
        InvDiagF inverse_diagonal_m;
        unsigned innerloops_m;
        Field ULg_m;
    };

    /*!
     * 2-step Gauss-Seidel preconditioner
     */
    template <typename Field, typename LowerF, typename UpperF, typename InvDiagF>
    struct gs_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        gs_preconditioner(LowerF&& lower, UpperF&& upper, InvDiagF&& inverse_diagonal,
                          unsigned innerloops, unsigned outerloops)
            : preconditioner<Field>("Gauss-Seidel")
            , innerloops_m(innerloops)
            , outerloops_m(outerloops) {
            lower_m            = std::move(lower);
            upper_m            = std::move(upper);
            inverse_diagonal_m = std::move(inverse_diagonal);
        }

        Field operator()(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();

            Field x(mesh, layout);

            x = 0;  // Initial guess

            for (unsigned int k = 0; k < outerloops_m; ++k) {
                UL_m = upper_m(x);
                r_m  = b - UL_m;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    UL_m = lower_m(x);
                    x    = r_m - UL_m;
                    // The inverse diagonal is applied to the
                    // vector itself to return the result usually.
                    // However, the operator for FEM already
                    // returns the result of inv_diag * itself
                    // due to the matrix-free evaluation.
                    // Therefore, we need this if to differentiate
                    // the two cases.
                    if constexpr (std::is_same_v<InvDiagF, double>) {
                        x = inverse_diagonal_m(x) * x;
                    } else {
                        x = inverse_diagonal_m(x);
                    }
                }
                UL_m = lower_m(x);
                r_m  = b - UL_m;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    UL_m = upper_m(x);
                    x    = r_m - UL_m;
                    // The inverse diagonal is applied to the
                    // vector itself to return the result usually.
                    // However, the operator for FEM already
                    // returns the result of inv_diag * itself
                    // due to the matrix-free evaluation.
                    // Therefore, we need this if to differentiate
                    // the two cases.
                    if constexpr (std::is_same_v<InvDiagF, double>) {
                        x = inverse_diagonal_m(x) * x;
                    } else {
                        x = inverse_diagonal_m(x);
                    }
                }
            }
            return x;
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();

            UL_m = Field(mesh, layout);
            r_m  = Field(mesh, layout);
        }

    protected:
        LowerF lower_m;
        UpperF upper_m;
        InvDiagF inverse_diagonal_m;
        unsigned innerloops_m;
        unsigned outerloops_m;
        Field UL_m;
        Field r_m;
    };

    /*!
     * Symmetric successive over-relaxation
     */
    template <typename Field, typename LowerF, typename UpperF, typename InvDiagF, typename DiagF>
    struct ssor_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        ssor_preconditioner(LowerF&& lower, UpperF&& upper, InvDiagF&& inverse_diagonal,
                            DiagF&& diagonal, unsigned innerloops, unsigned outerloops,
                            double omega)
            : preconditioner<Field>("ssor")
            , innerloops_m(innerloops)
            , outerloops_m(outerloops)
            , omega_m(omega) {
            lower_m            = std::move(lower);
            upper_m            = std::move(upper);
            inverse_diagonal_m = std::move(inverse_diagonal);
            diagonal_m         = std::move(diagonal);
        }

        Field operator()(Field& b) override {
            static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("SSOR Init");
            IpplTimings::startTimer(initTimer);

            double D;

            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();

            Field x(mesh, layout);

            x = 0;  // Initial guess

            IpplTimings::stopTimer(initTimer);

            static IpplTimings::TimerRef loopTimer = IpplTimings::getTimer("SSOR loop");
            IpplTimings::startTimer(loopTimer);

            // The inverse diagonal is applied to the
            // vector itself to return the result usually.
            // However, the operator for FEM already
            // returns the result of inv_diag * itself
            // due to the matrix-free evaluation.
            // Therefore, we need this if to differentiate
            // the two cases.
            for (unsigned int k = 0; k < outerloops_m; ++k) {
                if constexpr (std::is_same_v<DiagF, double>) {
                    UL_m = upper_m(x);
                    D    = diagonal_m(x);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * D * x;

                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m = lower_m(x);
                        x    = r_m - omega_m * UL_m;
                        x    = inverse_diagonal_m(x) * x;
                    }
                    UL_m = lower_m(x);
                    D    = diagonal_m(x);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * D * x;
                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m = upper_m(x);
                        x    = r_m - omega_m * UL_m;
                        x    = inverse_diagonal_m(x) * x;
                    }
                } else {
                    UL_m = upper_m(x);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * diagonal_m(x);

                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m = lower_m(x);
                        x    = r_m - omega_m * UL_m;
                        x    = inverse_diagonal_m(x);
                    }
                    UL_m = lower_m(x);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * diagonal_m(x);
                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m = upper_m(x);
                        x    = r_m - omega_m * UL_m;
                        x    = inverse_diagonal_m(x);
                    }
                }
            }
            IpplTimings::stopTimer(loopTimer);
            return x;
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();

            UL_m = Field(mesh, layout);
            r_m  = Field(mesh, layout);
        }

    protected:
        LowerF lower_m;
        UpperF upper_m;
        InvDiagF inverse_diagonal_m;
        DiagF diagonal_m;
        unsigned innerloops_m;
        unsigned outerloops_m;
        double omega_m;
        Field UL_m;
        Field r_m;
    };

    /*!
     * Computes the largest Eigenvalue of the Functor f
     * @param f Functor
     * @param x_0 initial guess
     * @param max_iter maximum number of iterations
     * @param tol tolerance
     */
    template <typename Field, typename Functor>
    double powermethod(Functor&& f, Field& x_0, unsigned int max_iter = 5000, double tol = 1e-3) {
        unsigned int i      = 0;
        using mesh_type     = typename Field::Mesh_t;
        using layout_type   = typename Field::Layout_t;
        mesh_type& mesh     = x_0.get_mesh();
        layout_type& layout = x_0.getLayout();
        Field x_new(mesh, layout);
        Field x_diff(mesh, layout);
        double error  = 1.0;
        double lambda = 1.0;
        while (error > tol && i < max_iter) {
            x_new  = f(x_0);
            lambda = norm(x_new);
            x_diff = x_new - lambda * x_0;
            error  = norm(x_diff);
            x_new  = x_new / lambda;
            x_0    = x_new.deepCopy();
            ++i;
        }
        if (i == max_iter) {
            std::cerr << "Powermethod did not converge, lambda_max : " << lambda
                      << ", error : " << error << std::endl;
        }
        return lambda;
    }

    /*!
     * Computes the smallest Eigenvalue of the Functor f (f must be symmetric positive definite)
     * @param f Functor
     * @param x_0 initial guess
     * @param lambda_max largest Eigenvalue
     * @param max_iter maximum number of iterations
     * @param tol tolerance
     */
    template <typename Field, typename Functor>
    double adapted_powermethod(Functor&& f, Field& x_0, double lambda_max,
                               unsigned int max_iter = 5000, double tol = 1e-3) {
        unsigned int i      = 0;
        using mesh_type     = typename Field::Mesh_t;
        using layout_type   = typename Field::Layout_t;
        mesh_type& mesh     = x_0.get_mesh();
        layout_type& layout = x_0.getLayout();
        Field x_new(mesh, layout);
        Field x_diff(mesh, layout);
        double error  = 1.0;
        double lambda = 1.0;
        while (error > tol && i < max_iter) {
            x_new  = f(x_0);
            x_new  = x_new - lambda_max * x_0;
            lambda = -norm(x_new);  // We know that lambda < 0;
            x_diff = x_new - lambda * x_0;
            error  = norm(x_diff);
            x_new  = x_new / -lambda;
            x_0    = x_new.deepCopy();
            ++i;
        }
        lambda = lambda + lambda_max;
        if (i == max_iter) {
            std::cerr << "Powermethod did not converge, lambda_min : " << lambda
                      << ", error : " << error << std::endl;
        }
        return lambda;
    }

    /*
    // Use the powermethod to compute the eigenvalues if no analytical solution is known
    beta = powermethod(std::move(op_m), x_0);
    // Trick for computing the smallest Eigenvalue of SPD Matrix
    alpha = adapted_powermethod(std::move(op_m), x_0, beta);
     */

}  // namespace ippl

#endif  // IPPL_PRECONDITIONER_H
