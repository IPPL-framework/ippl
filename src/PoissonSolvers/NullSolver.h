//
// Class NullSolver
//   Dummy solver which can be used when we require no action
//   to be done on our LHS in the simulation while keeping the
//   software design of the PIC framework in-tact.
//

#ifndef IPPL_NULL_SOLVER_H
#define IPPL_NULL_SOLVER_H

#include "Poisson.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class NullSolver : public Poisson<FieldLHS, FieldRHS> {
    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;

        // constructors
        NullSolver()
            : Base() {}

        NullSolver(rhs_type& rhs) {
            using T = typename FieldLHS::value_type::value_type;
            static_assert(std::is_floating_point<T>::value, "Not a floating point type");

            Base::setRhs(rhs);
            this->setDefaultParameters();
        }

        NullSolver(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs) {}

        void solve() override {
            // Overwrite the RHS (source rho) with the solution (potential phi), which is 0
            *(this->rhs_mp) = 0.0;

            // The gradient of the potential, which is the E-field, is also 0
            if (this->lhs_mp != nullptr) {
                *(this->lhs_mp) = 0.0;
            }
        }
    };

}  // namespace ippl

#endif
