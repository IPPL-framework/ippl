//
// General pre-conditioners for the pre-conditioned Conjugate Gradient Solver
// such as Jacobi, Polynomial Newton, Polynomial Chebyshev, Richardson, Gauss-Seidel
// provides a function operator() that returns the preconditioned field
//

#ifndef IPPL_PRECONDITIONER_H
#define IPPL_PRECONDITIONER_H

#include <vector>

#include "Expression/IpplOperations.h"  // get the function apply()

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
//
// IMPORTANT: takes the field by *reference*, not by value. A by-value
// signature would copy the Field on every call -- BareField has shared-
// ownership Kokkos::View members, so the data isn't actually copied, but
// the embedded HaloCells::haloData_m buffer that the halo exchange grows
// via Kokkos::realloc only grows on the copy and never propagates back to
// the original. The result was a fresh cudaMalloc per op_m() call in
// multi-rank runs (see nsys profile in ~/prof/pif-pr-verify).
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type& arg) {                             \
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

        // Apply the preconditioner: result = M^{-1} u. Concrete preconditioners
        // override this to write into the caller-provided result buffer; this
        // avoids per-call Field allocations and per-call deep copies in PCG.
        // The default (identity) preconditioner copies u into result.
        virtual void operator()(Field& u, Field& result) {
            Kokkos::deep_copy(result.getView(), u.getView());
        }

        // Allocate any scratch fields the preconditioner needs. Called once
        // by the owning solver after the layout is known. Concrete
        // preconditioners that need scratch override this.
        virtual void init_fields(Field& /*b*/) {}

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

        void operator()(Field& u, Field& result) override {
            // result = w * D^{-1} * u, written element-wise into the caller-
            // provided result buffer. Two forms of inverse_diagonal_m are
            // supported, matching the convention used by the other
            // preconditioners in this file:
            //   - returns a scalar (e.g. uniform-mesh Poisson Laplacian):
            //       result = w * scalar * u
            //   - returns a Field / expression equal to D^{-1} * u
            //     (e.g. matrix-free FEM operators):
            //       result = w * inverse_diagonal_m(u)
            if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                const double scale = w_m * inverse_diagonal_m(u);
                result             = scale * u;
            } else {
                result = inverse_diagonal_m(u);
                result = w_m * result;
            }
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

        // Recursive Newton expansion P_k(u) where:
        //   P_0(u) = eta_0 * u
        //   P_k(u) = eta_k * (2 P_{k-1}(u) - P_{k-1}(A P_{k-1}(u)))
        // Writes the result of level `level` into `out`. Uses one scratch
        // field per recursion depth (Pr, PA, PAPr) so that no Field is
        // allocated per call. The two recursive calls at each level execute
        // sequentially and may reuse scratch at lower depths because the
        // first call's lower-depth values have already been folded into
        // Pr_scratch[level] before the second call begins.
        void recursive_preconditioner(Field& u, unsigned int level, Field& out) {
            // Timers accumulated across all levels of the recursion: useful
            // for "how much of pcond is laplace vs combine vs leaf-assign".
            // Counts include every call at every level (so the call-count
            // column equals 2^(level_m+1)-1, not the number of operator()
            // invocations).
            if (level == 0) {
                out = eta_m[0] * u;
                return;
            }
            Field& Pr   = Pr_scratch_m[level];
            Field& PA   = PA_scratch_m[level];
            Field& PAPr = PAPr_scratch_m[level];

            // Inner recursive call owns its own timing; we don't double-count
            // it here.
            recursive_preconditioner(u, level - 1, Pr);
            PA = op_m(Pr);
            recursive_preconditioner(PA, level - 1, PAPr);
            out = eta_m[level] * (2.0 * Pr - PAPr);
        }

        void operator()(Field& u, Field& result) override {
            // One outer timer per polynomial_newton apply, complementing the
            // per-call recursion timers. Lets us see how many outer pcond
            // calls a solve does vs. the time inside each.
            recursive_preconditioner(u, level_m, result);
        }

        void init_fields(Field& b) override {
            // One-shot precomputation of the eta coefficients.
            if (eta_m == nullptr) {
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

            // First call: allocate one scratch field per recursion depth.
            // Subsequent calls: refresh layout in place so a load-balance
            // repartition tracks correctly without freeing/reallocating
            // when the local extents are unchanged.
            mesh_type& mesh     = b.get_mesh();
            layout_type& layout = b.getLayout();
            if (!fields_initialized_m) {
                Pr_scratch_m.resize(level_m + 1);
                PA_scratch_m.resize(level_m + 1);
                PAPr_scratch_m.resize(level_m + 1);
                for (unsigned int i = 1; i <= level_m; ++i) {
                    Pr_scratch_m[i]   = Field(mesh, layout);
                    PA_scratch_m[i]   = Field(mesh, layout);
                    PAPr_scratch_m[i] = Field(mesh, layout);
                }
                fields_initialized_m = true;
            } else {
                for (unsigned int i = 1; i <= level_m; ++i) {
                    Pr_scratch_m[i].updateLayout(layout);
                    PA_scratch_m[i].updateLayout(layout);
                    PAPr_scratch_m[i].updateLayout(layout);
                }
            }
        }

    protected:
        OperatorF op_m;        // Operator to be preconditioned
        double alpha_m;        // Smallest Eigenvalue
        double beta_m;         // Largest Eigenvalue
        unsigned int level_m;  // Number of recursive calls
        double zeta_m;  // smallest (alpha + beta) is multiplied by (1+zeta) to avoid clustering of
                        // Eigenvalues
        double* eta_m = nullptr;  // Size is determined at runtime
        std::vector<Field> Pr_scratch_m;
        std::vector<Field> PA_scratch_m;
        std::vector<Field> PAPr_scratch_m;
        bool fields_initialized_m = false;
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

        void operator()(Field& r, Field& result) override {
            // x_m, x_old_m, A_m, z_m are pre-allocated scratch (init_fields).
            // Coefficients in rho_m are also computed once.
            x_old_m = r / theta_m;
            A_m     = op_m(r);
            x_m     = 2.0 * rho_m[1] / delta_m * (2.0 * r - A_m / theta_m);
            if (degree_m == 0) {
                // result = x_old
                Kokkos::deep_copy(result.getView(), x_old_m.getView());
                return;
            }

            if (degree_m == 1) {
                // result = x
                Kokkos::deep_copy(result.getView(), x_m.getView());
                return;
            }
            for (unsigned int i = 2; i < degree_m + 1; ++i) {
                A_m = op_m(x_m);
                z_m = 2.0 / delta_m * (r - A_m);
                // Write the new x value into result (the caller's buffer);
                // x_old gets a deep copy of the previous x.
                result = rho_m[i] * (2 * sigma_m * x_m - rho_m[i - 1] * x_old_m + z_m);
                Kokkos::deep_copy(x_old_m.getView(), x_m.getView());
                Kokkos::deep_copy(x_m.getView(), result.getView());
            }
        }

        void init_fields(Field& b) override {
            // One-shot precomputation of the rho coefficients.
            if (rho_m == nullptr) {
                theta_m = (beta_m + alpha_m) / 2.0 * (1.0 + zeta_m);
                delta_m = (beta_m - alpha_m) / 2.0;
                sigma_m = theta_m / delta_m;

                rho_m    = new double[degree_m + 1];
                rho_m[0] = 1.0 / sigma_m;
                for (unsigned int i = 1; i < degree_m + 1; ++i) {
                    rho_m[i] = 1.0 / (2.0 * sigma_m - rho_m[i - 1]);
                }
            }
            mesh_type& mesh     = b.get_mesh();
            layout_type& layout = b.getLayout();
            // First call allocates; subsequent calls refresh the layout so a
            // repartition is tracked without throwing the storage away.
            if (!fields_initialized_m) {
                x_m                  = Field(mesh, layout);
                x_old_m              = Field(mesh, layout);
                A_m                  = Field(mesh, layout);
                z_m                  = Field(mesh, layout);
                fields_initialized_m = true;
            } else {
                x_m.updateLayout(layout);
                x_old_m.updateLayout(layout);
                A_m.updateLayout(layout);
                z_m.updateLayout(layout);
            }
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
        Field x_m;
        Field x_old_m;
        Field A_m;
        Field z_m;
        bool fields_initialized_m = false;
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

        void operator()(Field& r, Field& result) override {
            // Richardson iteration in-place on the caller-provided result
            // buffer. ULg_m stays as a member scratch; the inner deep copies
            // remain because the (upper, diag, inverse, lower) operators may
            // return Field-valued expressions that alias their input.

            result = 0;
            for (unsigned int j = 0; j < innerloops_m; ++j) {
                ULg_m  = upper_and_lower_m(result);
                ULg_m  = ULg_m.deepCopy();
                result = r - ULg_m;

                // The inverse diagonal is applied to the
                // vector itself to return the result usually.
                // However, the operator for FEM already
                // returns the result of inv_diag * itself
                // due to the matrix-free evaluation.
                // Therefore, we need this if to differentiate
                // the two cases.
                if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                    result = inverse_diagonal_m(result) * result;
                } else {
                    result = inverse_diagonal_m(result).deepCopy();
                }
            }
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();
            if (!fields_initialized_m) {
                ULg_m                = Field(mesh, layout);
                fields_initialized_m = true;
            } else {
                ULg_m.updateLayout(layout);
            }
        }

    protected:
        UpperAndLowerF upper_and_lower_m;
        InvDiagF inverse_diagonal_m;
        unsigned innerloops_m;
        Field ULg_m;
        bool fields_initialized_m = false;
    };

    /*!
     * Alternative Richardson preconditioner
     * Given the linear system of equations Ax=b the update steps are performed as follows:
     * x_{k+1} = x_k + D^{-1}*(b-A*x_k)
     * A derivation of this preconditioner can be found at:
     * https://jschoeberl.github.io/iFEM/iterative/preconditioning.html
     */
    template <typename Field, typename OperatorF, typename InvDiagF>
    struct richardson_preconditioner_alt : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

        richardson_preconditioner_alt(OperatorF&& op, InvDiagF&& inverse_diagonal,
                                      unsigned innerloops = 5)
            : preconditioner<Field>("Richardson_alt")
            , innerloops_m(innerloops) {
            op_m               = std::move(op);
            inverse_diagonal_m = std::move(inverse_diagonal);
        }

        void operator()(Field& r, Field& result) override {
            // result holds the running iterate; Ag_m and g_old_m are scratch
            // members. The inner deep copies remain because the operators
            // (op_m, inverse_diagonal_m) may return Field-valued expressions
            // that alias their input.

            result  = 0;
            g_old_m = 0;

            for (unsigned int j = 0; j < innerloops_m; ++j) {
                Ag_m   = op_m(result);
                Ag_m   = Ag_m.deepCopy();
                result = r - Ag_m;

                // The inverse diagonal is applied to the
                // vector itself to return the result usually.
                // However, the operator for FEM already
                // returns the result of inv_diag * itself
                // due to the matrix-free evaluation.
                // Therefore, we need this if to differentiate
                // the two cases.
                if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                    result = g_old_m + inverse_diagonal_m(result) * result;
                } else {
                    result = g_old_m + inverse_diagonal_m(result);
                }
                Kokkos::deep_copy(g_old_m.getView(), result.getView());
            }
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();
            if (!fields_initialized_m) {
                Ag_m                 = Field(mesh, layout);
                g_old_m              = Field(mesh, layout);
                fields_initialized_m = true;
            } else {
                Ag_m.updateLayout(layout);
                g_old_m.updateLayout(layout);
            }
        }

    protected:
        OperatorF op_m;
        InvDiagF inverse_diagonal_m;
        unsigned innerloops_m;
        Field Ag_m;
        Field g_old_m;
        bool fields_initialized_m = false;
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

        void operator()(Field& b, Field& result) override {
            // The running iterate lives in result; UL_m and r_m are scratch
            // members. The inner deep copies remain because the (upper, lower,
            // inverse) operators may return Field-valued expressions that
            // alias their input.

            result = 0;  // Initial guess

            for (unsigned int k = 0; k < outerloops_m; ++k) {
                UL_m = upper_m(result);
                UL_m = UL_m.deepCopy();
                r_m  = b - UL_m;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    UL_m   = lower_m(result);
                    UL_m   = UL_m.deepCopy();
                    result = r_m - UL_m;
                    if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                        result = inverse_diagonal_m(result) * result;
                    } else {
                        result = inverse_diagonal_m(result).deepCopy();
                    }
                    // The inverse diagonal is applied to the
                    // vector itself to return the result usually.
                    // However, the operator for FEM already
                    // returns the result of inv_diag * itself
                    // due to the matrix-free evaluation.
                    // Therefore, we need this if to differentiate
                    // the two cases.
                }
                UL_m = lower_m(result);
                UL_m = UL_m.deepCopy();
                r_m  = b - UL_m;
                for (unsigned int j = 0; j < innerloops_m; ++j) {
                    UL_m   = upper_m(result);
                    UL_m   = UL_m.deepCopy();
                    result = r_m - UL_m;
                    if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                        result = inverse_diagonal_m(result) * result;
                    } else {
                        result = inverse_diagonal_m(result).deepCopy();
                    }
                    // The inverse diagonal is applied to the
                    // vector itself to return the result usually.
                    // However, the operator for FEM already
                    // returns the result of inv_diag * itself
                    // due to the matrix-free evaluation.
                    // Therefore, we need this if to differentiate
                    // the two cases.
                }
            }
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();
            if (!fields_initialized_m) {
                UL_m                 = Field(mesh, layout);
                r_m                  = Field(mesh, layout);
                fields_initialized_m = true;
            } else {
                UL_m.updateLayout(layout);
                r_m.updateLayout(layout);
            }
        }

    protected:
        LowerF lower_m;
        UpperF upper_m;
        InvDiagF inverse_diagonal_m;
        unsigned innerloops_m;
        unsigned outerloops_m;
        Field UL_m;
        Field r_m;
        bool fields_initialized_m = false;
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

        void operator()(Field& b, Field& result) override {
            // In the FEM solver, which uses the preconditioner,
            // we re-use a resultField to avoid allocating new
            // memory at every iteration.
            // In order for the operator calls to not rewrite
            // on this same field over and over when calling
            // the operators (upper, diag, inverse, lower, etc)
            // we need deep copies to the preconditioner fields.

            // The inverse diagonal is applied to the
            // vector itself to return the result usually.
            // However, the operator for FEM already
            // returns the result of inv_diag * itself
            // due to the matrix-free evaluation.
            // Therefore, we need this if to differentiate
            // the two cases.

            double D;

            // In order for the operator calls to not rewrite on this same field
            // over and over when calling the operators (upper, diag, inverse,
            // lower, etc) we need deep copies to the preconditioner fields."

            result = 0;  // Initial guess

            for (unsigned int k = 0; k < outerloops_m; ++k) {
                if constexpr (std::is_same_v<InvDiagF, std::function<double(Field&)>>) {
                    UL_m = upper_m(result);
                    D    = diagonal_m(result);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * D * result;

                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m   = lower_m(result);
                        result = r_m - omega_m * UL_m;
                        result = inverse_diagonal_m(result) * result;
                    }
                    UL_m = lower_m(result);
                    D    = diagonal_m(result);
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * D * result;
                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m   = upper_m(result);
                        result = r_m - omega_m * UL_m;
                        result = inverse_diagonal_m(result) * result;
                    }
                } else {
                    UL_m = upper_m(result).deepCopy();
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * diagonal_m(result);

                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m   = lower_m(result).deepCopy();
                        result = r_m - omega_m * UL_m;
                        result = inverse_diagonal_m(result).deepCopy();
                    }
                    UL_m = lower_m(result).deepCopy();
                    r_m  = omega_m * (b - UL_m) + (1.0 - omega_m) * diagonal_m(result);
                    for (unsigned int j = 0; j < innerloops_m; ++j) {
                        UL_m   = upper_m(result).deepCopy();
                        result = r_m - omega_m * UL_m;
                        result = inverse_diagonal_m(result).deepCopy();
                    }
                }
            }
        }

        void init_fields(Field& b) override {
            layout_type& layout = b.getLayout();
            mesh_type& mesh     = b.get_mesh();
            if (!fields_initialized_m) {
                UL_m                 = Field(mesh, layout);
                r_m                  = Field(mesh, layout);
                fields_initialized_m = true;
            } else {
                UL_m.updateLayout(layout);
                r_m.updateLayout(layout);
            }
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
        bool fields_initialized_m = false;
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
