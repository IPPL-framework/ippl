//
// Class Solver
//   Base class for all solvers.
//

#ifndef IPPL_SOLVER_H
#define IPPL_SOLVER_H

#include "Utility/ParameterList.h"

#include "Field/Field.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class Solver {
    public:
        using lhs_type = FieldLHS;
        using rhs_type = FieldRHS;

        constexpr static unsigned Dim = FieldLHS::dim;
        typedef typename FieldLHS::Mesh_t Mesh;
        typedef typename FieldLHS::Centering_t Centering;
        typedef typename Mesh::matrix_type Matrix_t;
        typedef Field<Matrix_t, Dim, Mesh, Centering> MField_t;

        /*!
         * Default constructor
         */
        Solver() {}

        /*!
         * Convenience constructor with LHS and RHS parameters
         * @param lhs The LHS for the problem to solve
         * @param rhs The RHS for the problem to solve
         */
        Solver(lhs_type& lhs, rhs_type& rhs) {
            setLhs(lhs);
            setRhs(rhs);
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

        virtual MField_t* getHessian() { return nullptr; }

    protected:
        ParameterList params_m;

        rhs_type* rhs_mp = nullptr;
        lhs_type* lhs_mp = nullptr;

        /*!
         * Utility function for initializing a solver's default
         * parameters (to be overridden for each base class)
         */
        virtual void setDefaultParameters() {}
    };
}  // namespace ippl

#endif
