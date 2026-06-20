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
            // One-shot precomputation of the rho
