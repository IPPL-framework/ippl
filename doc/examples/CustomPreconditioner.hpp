/** @page custom_preconditioner Defining Custom Preconditioners

# Introduction to the Preconditioner Base Class

The preconditioner class in the IPPL framework serves as an abstract base class for different preconditioning strategies applied in iterative solvers. It encapsulates common functionalities and interfaces that are essential for all types of preconditioners.
Structure of the Preconditioner Base Class

- Template Parameter: The class template Field determines the type of the numerical field the preconditioner will operate on.
- Constructor: Constructors initialize the preconditioner, optionally setting a type name to identify the preconditioner strategy.
- Virtual Function ('operator()'): This is a pure virtual function in the base class that must be implemented by derived classes to define the specific preconditioning behavior.

Steps to Create a Custom Preconditioner
## Step 1: Define the Preconditioner Class

Begin by defining a new class that inherits from the 'preconditioner<Field>' provided by the IPPL framework. This new class should implement all abstract methods from the base class and can add additional members as needed for its specific strategy.

@code
template <typename Field>
struct custom_preconditioner : public preconditioner<Field> {
    // Additional member variables here

    custom_preconditioner(std::string name = "Custom") : preconditioner<Field>(name) {
        // Initialization code here
    }

    Field operator()(Field& u) override {
        // Implementation of preconditioning logic here
        return u; // Placeholder return
    }
};
@endcode

## Step 2: Implement the Constructor

Use the constructor of your custom preconditioner to initialize any data members and forward any necessary parameters to the base class constructor, which sets the type name of the preconditioner.
## Step 3: Override the Operator Function

Implement the 'operator()' function to apply your preconditioning logic to the input field. This method is crucial as it defines how the preconditioner modifies the input data, aligning with the specific algorithmic needs of your solver.
# Example Implementations

## Jacobi Preconditioner

The 'jacobi_preconditioner' provided by IPPL is a derived class of preconditioner. It implements a specific preconditioning strategy using an inverse diagonal matrix and a damping factor. Here is how it extends the base class:


@code
template <typename Field, typename InvDiagF>
struct jacobi_preconditioner : public preconditioner<Field> {
    InvDiagF inverse_diagonal_m;
    double w_m;  // Damping factor

    jacobi_preconditioner(InvDiagF&& inverse_diagonal, double w = 1.0)
        : preconditioner<Field>("Jacobi"), inverse_diagonal_m(std::move(inverse_diagonal)), w_m(w) {}

    Field operator()(Field& u) override {
        typename Field::mesh_type& mesh = u.get_mesh();
        typename Field::layout_type& layout = u.getLayout();
        Field res(mesh, layout);

        res = inverse_diagonal_m(u);  // Apply inverse diagonal matrix
        res = w_m * res;              // Apply damping factor
        return res;
    }
};
@endcode
## Scaling Preconditioner

A simple example of a custom preconditioner that scales the input field can be structured as follows:


@code
template <typename Field>
struct scaling_preconditioner : public preconditioner<Field> {
    double scale_factor;

    scaling_preconditioner(double factor = 1.0)
        : preconditioner<Field>("Scaling"), scale_factor(factor) {}

    Field operator()(Field& u) override {
        return scale_factor * u; // Scales the input field
    }
};
@endcode

# Integration with Solvers

To utilize your custom preconditioner in a solver, instantiate it and configure the solver to use it.

*/ 