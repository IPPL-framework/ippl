//
// Preconditioners for various operators.
//

#ifndef IPPL_PRECONDITIONER_H
#define IPPL_PRECONDITIONER_H

#include "Expression/IpplOperations.h" // get the function apply()

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type arg) {                              \
        return fun(arg);                        \
    }


namespace ippl{
    namespace detail {

        template <typename E>
        struct meta_poisson: public Expression<meta_poisson<E>,
                sizeof(E)> {
            constexpr static unsigned dim = E::dim;

            KOKKOS_FUNCTION
            meta_poisson(const E& u)
                    : u_m(u) {}

            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;
                T res            = 0;
                index_type coords[dim] = {args...};
                auto &&center = apply(u_m, coords);
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};

                    coords[d] -= 1;
                    auto &&left = apply(u_m, coords);

                    coords[d] += 2;
                    auto &&right = apply(u_m, coords);
                    res += (2.0* center - left - right);
                }
                return res;
            }

        private:
            const E u_m;
        };
    }// namespace detail

    /*!
     * User interface of poisson
     * @param u field
     */
    template <typename Field>
    detail::meta_poisson<Field> poisson(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        return detail::meta_poisson<Field>(u);
    }

    template <typename Field>
    struct preconditioner {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;

        preconditioner() : type_m("Identity") {}
        virtual ~preconditioner() = default;

        virtual Field operator()(Field &u) {
            u.fillHalo();
            BConds <Field, Dim> &bcField = u.getFieldBC();
            bcField.apply(u);
            Field res = u.deepCopy();
            return res;
        }
        virtual std::string get_type(){return type_m;};

    protected :
        std::string type_m;
    };

    /*!
    * Jacobi preconditioner
    * M = diag{A}
    */

    template<typename Field>
    struct jacobi_preconditioner: public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;

        jacobi_preconditioner(unsigned s = 1) : type_m("jacobi"), s_m(s){}

        Field operator()(Field &u) override {
            Field res = u.deepCopy();
            res.fillHalo();
            BConds<Field, Dim>& bcField = res.getFieldBC();
            bcField.apply(res);
            double sum = 0.0;
            mesh_type mesh = u.get_mesh();
            typename mesh_type::vector_type hvector(0);
            for (unsigned d = 0; d < Dim; ++d) {
                hvector[d] = 1./std::pow(mesh.getMeshSpacing(d), 2);
                sum += hvector[d];
            }
            for (unsigned i = 0; i<s_m; ++i)
                res = res*(0.5/sum);
            res = u/(Dim*2.0); // For testing purposes
            return res;
        }
        std::string get_type() override {return type_m;};
    protected:
        std::string type_m;
        unsigned s_m;
    };

    /*!
    * Polynomial Newton Preconditioner
    * Computes iteratively approximations for A^-1
    */
    template <typename Field>
    struct polynomial_newton_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;

        polynomial_newton_preconditioner(unsigned int max_level = 6 , double zeta = 1e-2, double* eta = nullptr) :
        type_m("polynomial_newton"),
        level_m(max_level),
        zeta_m(zeta),
        eta_m(eta){}

        ~polynomial_newton_preconditioner(){
            if (eta_m != nullptr){
                delete[] eta_m;
                eta_m = nullptr;
            }
        }

        polynomial_newton_preconditioner(const polynomial_newton_preconditioner& other): type_m("polynomial_newton"),
                                                                                               level_m(other.level_m),
                                                                                               alpha_m(other.alpha_m),
                                                                                               beta_m(other.beta_m),
                                                                                               zeta_m(other.zeta_m),
                                                                                               eta_m(other.eta_m){}

        polynomial_newton_preconditioner& operator=(const polynomial_newton_preconditioner& other){
            return *this = polynomial_newton_preconditioner(other);
        }

        Field recursive_preconditioner(Field& u,unsigned int level) {
            mesh_type &mesh = u.get_mesh();
            layout_type &layout = u.getLayout();
            //Define Etas if not defined yet
            if (eta_m == nullptr){
                if(analytical_m) {
                    //Analytical eigenvalues for the 3 dimensional laplace operator
                    beta_m = 0;
                    alpha_m = 0;
                    unsigned long n;
                    double h;
                    for (unsigned int d = 0; d < Dim; ++d) {
                        n = mesh.getGridsize(d);
                        h = mesh.getMeshSpacing(d);
                        double local_min = 4/std::pow(h,2); // theoretical maximum
                        double local_max = 0;
                        double test;
                        for (unsigned int i = 1; i<n; ++i) {
                            test = 4. / std::pow(h, 2) * std::pow(std::sin(i* M_PI * h/2.), 2);
                            if (test > local_max){
                                local_max = test;
                            }
                            if (test < local_min){
                                local_min = test;
                            }
                        }
                        beta_m += local_max;
                        alpha_m += local_min;
                    }
                    std::cout << "Analytical results: alpha : " << alpha_m << std::endl;
                    std::cout << "Analytical results: beta : " << beta_m << std::endl;
                }
                else{
                    Field x_0(mesh , layout);
                    x_0 = u.deepCopy() + 0.1;
                    beta_m = powermethod(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, Field), x_0);
                    x_0 = u.deepCopy() + 0.1;
                    // Trick for computing the smallest Eigenvalue of an SPD Matrix
                    alpha_m = adapted_powermethod(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace , Field) , x_0 , beta_m);
                }

                eta_m = new double[level_m+1];
                eta_m[0] = 2.0/((alpha_m + beta_m) * (1.0+zeta_m));
                if (level_m > 0){
                    eta_m[1] = 2.0/(1.0+2*alpha_m*eta_m[0] - alpha_m*eta_m[0]*alpha_m*eta_m[0]);
                }
                for (unsigned int i = 2; i<level_m+1; ++i){
                    eta_m[i] = 2.0/(1.0+2*eta_m[i-1] - eta_m[i-1]*eta_m[i-1]);
                }
            }
            Field res(mesh, layout);
            if (level == 0) {
                res = eta_m[0]* u;
                return res;
            }
            Field PAPr(mesh, layout);
            Field Pr(mesh, layout);

            Pr = recursive_preconditioner(u, level - 1);
            PAPr = -laplace(Pr);
            PAPr = recursive_preconditioner(PAPr, level - 1);
            res = eta_m[level] * (2.0 * Pr - PAPr);
            return res;
        }

        Field operator()(Field &u) override {
            return recursive_preconditioner(u, level_m);
        }

        std::string get_type() override {return type_m;};

    protected:
        std::string type_m;
        unsigned int level_m;
        double alpha_m;
        double beta_m;
        double zeta_m;
        double* eta_m = nullptr;
        bool analytical_m = true;

    };
    /*!
    * Polynomial Chebyshev Preconditioner
    * Computes iteratively approximations for A^-1
    */
    template <typename Field>
    struct polynomial_chebyshev_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;

        polynomial_chebyshev_preconditioner(unsigned int degree = 63 , double zeta=1e-3) :
                type_m("polynomial_chebyshev"),
                degree_m(degree),
                zeta_m(zeta),
                rho_m(nullptr)
                {}
        ~polynomial_chebyshev_preconditioner(){
            if(rho_m != nullptr) {
                delete[] rho_m;
                rho_m = nullptr;
            }
        }

        polynomial_chebyshev_preconditioner(const polynomial_chebyshev_preconditioner& other): type_m("polynomial_chebyshev"),
                                                                                               degree_m(other.degree_m),
                                                                                               theta_m(other.theta_m),
                                                                                               sigma_m(other.sigma_m),
                                                                                               delta_m(other.delta_m),
                                                                                               alpha_m(other.delta_m),
                                                                                               beta_m(other.delta_m),
                                                                                               zeta_m(other.zeta_m),
                                                                                               rho_m(other.rho_m){}

        polynomial_chebyshev_preconditioner& operator=(const polynomial_chebyshev_preconditioner& other){
            return *this = polynomial_chebyshev_preconditioner(other);
            }

        Field operator()(Field& r) override{
            mesh_type& mesh = r.get_mesh();
            layout_type& layout = r.getLayout();

            Field res(mesh ,layout);
            Field x(mesh , layout);
            Field x_old(mesh , layout);
            Field A(mesh , layout);
            Field z(mesh , layout);

            // Define rho if not defined yet
            if (rho_m == nullptr) {
                if(analytical_m) {
                    //Analytical eigenvalues for the 3 dimensional laplace operator
                    beta_m = 0;
                    alpha_m = 0;
                    unsigned long n;
                    double h;
                    for (unsigned int d = 0; d < Dim; ++d) {
                        n = mesh.getGridsize(d);
                        h = mesh.getMeshSpacing(d);
                        double local_min = 4/std::pow(h,2); // theoretical maximum
                        double local_max = 0;
                        double test;
                        for (unsigned int i = 1; i<n; ++i) {
                            test = 4. / std::pow(h, 2) * std::pow(std::sin(i* M_PI * h/2.), 2);
                            if (test > local_max){
                                local_max = test;
                            }
                            if (test < local_min){
                                local_min = test;
                            }
                        }
                        beta_m += local_max;
                        alpha_m += local_min;
                    }
                    std::cout << "Analytical results: alpha : " << alpha_m << std::endl;
                    std::cout << "Analytical results: beta : " << beta_m << std::endl;
                }
                else {
                    Field x_0(mesh, layout);
                    x_0 = r.deepCopy() + 0.1;
                    beta_m = powermethod(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, Field), x_0);
                    x_0 = r.deepCopy() + 0.1;
                    // Trick for computing the smallest Eigenvalue of SPD Matrix
                    alpha_m = adapted_powermethod(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, Field), x_0, beta_m);
                }
                theta_m = (beta_m + alpha_m) / 2.0 * (1.0 + zeta_m);
                delta_m = (beta_m - alpha_m) / 2.0;
                sigma_m = theta_m / delta_m;

                rho_m = new double[degree_m + 1];
                rho_m[0] = 1.0 / sigma_m;
                for (unsigned int i = 1; i < degree_m + 1; ++i) {
                    rho_m[i] = 1.0 / (2.0 * sigma_m - rho_m[i - 1]);
                }
            }
            res = r.deepCopy();

            x_old = r/theta_m;
            A = -laplace(r);
            x = 2.0*rho_m[1]/delta_m * (2.0*r - A/theta_m);

            if (degree_m == 0){
                return x_old;
            }

            if(degree_m==1){
                return x;
            }
            for (unsigned int i = 2; i<degree_m+1; ++i){
                A = -laplace(x);
                z = 2.0/delta_m * (r - A);
                res = rho_m[i] * (2*sigma_m*x - rho_m[i-1]*x_old + z);
                x_old = x.deepCopy();
                x = res.deepCopy();
            }
            return res;
        }
        std::string get_type() override {return type_m;};
    protected:
        std::string type_m;
        unsigned degree_m;
        double theta_m;
        double sigma_m;
        double delta_m;
        double beta_m;
        double alpha_m;
        double zeta_m;
        double* rho_m = nullptr;
        bool analytical_m = true;
    };

    /*!
    * Computes the largest Eigenvalue of the Functor f
    * @param f Functor
    * @param x_0 initial guess
    * @param max_iter maximum number of iterations
    * @param tol tolerance
    */
    template<typename Functor , typename Field>
    double powermethod(Functor&& f, Field& x_0 , unsigned int max_iter = 5000, double tol = 1e-3){
        unsigned int i=0;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;
        mesh_type& mesh = x_0.get_mesh();
        layout_type& layout = x_0.getLayout();
        Field x_new(mesh , layout);
        Field x_diff(mesh , layout);
        double error = 1.0;
        double lambda = 1.0;
        while (error > tol && i < max_iter){
            x_new =  f(x_0);
            lambda = norm(x_new);
            x_diff = x_new-lambda*x_0;
            error = norm(x_diff);
            x_new = x_new / lambda;
            x_0 = x_new.deepCopy();
            ++i;
        }
        if(i == max_iter) {
            std::cerr << "Powermethod did not converge, lambda_max : " << lambda << ", error : " << error << std::endl;
        }
        else {
            std::cout << "Powermethod did converge, lambda_max : " << lambda << std::endl;
        }
        return lambda;
    }
    /*!
    * Computes the smallest Eigenvalue of the Functor f (must be SPD)
    * @param f Functor
    * @param x_0 initial guess
    * * @param lambda_max largest Eigenvalue
    * @param max_iter maximum number of iterations
    * @param tol tolerance
    */
    template<typename Functor , typename Field>
    double adapted_powermethod(Functor&& f, Field& x_0 ,double lambda_max, unsigned int max_iter = 5000, double tol = 1e-3){
        unsigned int i=0;
        using mesh_type = typename Field::Mesh_t;
        using layout_type = typename Field::Layout_t;
        mesh_type& mesh = x_0.get_mesh();
        layout_type& layout = x_0.getLayout();
        Field x_new(mesh , layout);
        Field x_diff(mesh , layout);
        double error = 1.0;
        double lambda = 1.0;
        while (error > tol && i < max_iter){
            x_new = f(x_0);
            x_new = x_new - lambda_max*x_0;
            lambda = -norm(x_new); // We know that lambda < 0;
            x_diff = x_new-lambda*x_0;
            error = norm(x_diff);
            x_new = x_new / -lambda;
            x_0 = x_new.deepCopy();
            ++i;
        }
        lambda = lambda + lambda_max;
        if(i == max_iter) {
            std::cerr << "Powermethod did not converge, lambda_min : " << lambda  << ", error : " << error << std::endl;
        }
        else {
            std::cout << "Powermethod did converge, lambda_min : " << lambda << std::endl;
        }
        return lambda;
    }

} //namespace ippl

#endif  // IPPL_PRECONDITIONER_H
