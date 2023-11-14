//
// Class Poisson
//   Base class for solvers for the Poisson problem
//

#ifndef IPPL_POISSON_H
#define IPPL_POISSON_H

#include "Utility/ParameterList.h"

#include "Field/Field.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class Poisson {
        constexpr static unsigned Dim = FieldLHS::dim;
        typedef typename FieldLHS::Mesh_t Mesh;
        typedef typename FieldLHS::Centering_t Centering;
        typedef typename Mesh::matrix_type Matrix_t;
        typedef Field<Matrix_t, Dim, Mesh, Centering> MField_t;

    public:
        using lhs_type = FieldLHS;
        using rhs_type = FieldRHS;
        using Tlhs     = typename FieldLHS::value_type;
        using Trhs     = typename FieldRHS::value_type;

        using grad_type = Field<Vector<Tlhs, Dim>, Dim, Mesh, Centering>;

        /*!
         * Represents the types of fields that should
         * be output by the solver
         */
        enum OutputType {
            SOL          = 0b01,
            GRAD         = 0b10,
            SOL_AND_GRAD = 0b11
        };

        /*!
         * Default constructor for electrostatic solvers;
         * desired output type defaults to solution only
         */
        Poisson()
            : grad_mp(nullptr) {
            static_assert(std::is_floating_point<Trhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        Poisson(lhs_type& lhs, rhs_type& rhs)
            : grad_mp(nullptr) {
            setLhs(lhs);
            setRhs(rhs);

            static_assert(std::is_floating_point<Trhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        /*!
         * Update one of the solver's parameters
         * @param key The parameter key
         * @param value The new value
         * @throw IpplException Fails if there is no existing parameter with the given key
         */
        template <typename T>
        void updateParameter(const std::string& key, const T& value) {
            params_m.update<T>(key, value);
        }

        /*!
         * Updates all solver parameters based on values in another parameter set
         * @param params Parameter list with updated values
         * @throw IpplException Fails if the provided parameter list includes keys not already
         * present
         */
        void updateParameters(const ParameterList& params) { params_m.update(params); }

        /*!
         * Merges another parameter set into the solver's parameters, overwriting
         * existing parameters in case of conflict
         * @param params Parameter list with desired values
         */
        void mergeParameters(const ParameterList& params) { params_m.merge(params); }

        /*!
         * Set the problem LHS
         * @param lhs Reference to problem LHS field
         */
        void setLhs(lhs_type& lhs) { lhs_mp = &lhs; }

        /*!
         * Set the problem RHS
         * @param rhs Reference to problem RHS field
         */
        virtual void setRhs(rhs_type& rhs) { rhs_mp = &rhs; }

        /*!
         * Get the Hessian matrix of the solution
         * @return Matrix field containing the Hessian of the lhs
         */
        virtual MField_t* getHessian() { return nullptr; }

        /*!
         * Set the field in which the gradient of the computed potential
         * should be stored
         * @param grad Reference to field in which to store the gradient
         */
        void setGradient(grad_type& grad) { grad_mp = &grad; }

        /*!
         * Solve the Poisson problem described by
         * -laplace(lhs) = rhs
         */
        virtual void solve() = 0;

        virtual ~Poisson() {}

    protected:
        ParameterList params_m;

        rhs_type* rhs_mp = nullptr;
        lhs_type* lhs_mp = nullptr;

        grad_type* grad_mp;

        /*!
         * Utility function for initializing a solver's default
         * parameters (to be overridden for each base class)
         */
        virtual void setDefaultParameters() { this->params_m.add("output_type", SOL); }
    };
}  // namespace ippl

#endif
