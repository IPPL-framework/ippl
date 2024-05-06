/**
@page custom_solver Creating Custom Solver

Here, we'll provide detailed guidelines on how to extend this base class of the 'SolverAlgorithm' to develop your own solver algorithm, 
using the PCG (Preconditioned Conjugate Gradient) solver as an example.

## Introduction to SolverAlgorithm Base Class

The SolverAlgorithm class is a template abstract class designed to serve as a
foundation for various numerical solver algorithms. 
It provides a common interface for solving problems of the form 
Op(lhs) = rhs, where Op is a differential operator and lhs and rhs are fields.

** Key Components of SolverAlgorithm **
- Template Parameters: The class template parameters FieldLHS and FieldRHS define the types for the left-hand side (LHS) and right-hand side (RHS) of the equation respectively.
- Virtual Function: The operator() function is a pure virtual function that must be implemented by derived classes. It is where the main logic of the solver is implemented.


@code
*
#include <functional>

#include "Utility/ParameterList.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class SolverAlgorithm {
    public:
        using lhs_type = FieldLHS;
        using rhs_type = FieldRHS;

        /*!
         * Solve the problem described by Op(lhs) = rhs, where Op is an unspecified
         * differential operator (handled by derived classes)
         * @param lhs The problem's LHS
         * @param rhs The problem's RHS
         * @param params A set of parameters for the solver algorithm
         * /
        virtual void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) = 0;
    };

}  // namespace ippl

@endcode 


## Steps to Create a Custom Solver
### 1. Define the Solver Class

Start by defining your solver class that inherits from SolverAlgorithm. Specify any additional data members or methods needed for your solver.

@code
#include "SolverAlgorithm.h"

template <typename FieldLHS, typename FieldRHS = FieldLHS>
class MySolver : public ippl::SolverAlgorithm<FieldLHS, FieldRHS> {
    using Base = ippl::SolverAlgorithm<FieldLHS, FieldRHS>;
public:
    // Additional methods or data members here

};
@endcode
## 2. Implement the Solver Logic

Implement the operator() function, which contains the core logic for the solver. This function should use the provided lhs, rhs, and params to compute the solution to the problem.

@code
void operator()(typename Base::lhs_type& lhs, typename Base::rhs_type& rhs, const ParameterList& params) override {
    // Initialization and setup
    // Iterative solution process
    // Post-processing and cleanup
}
@endcode


*/
