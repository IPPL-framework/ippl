#include "Ippl.h"

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform out("Test LagrangeSpace1DMidpoint");

        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_elements = 10;
        const double interval_size        = 2.0;
        const ippl::UniformCartesian<double, 1> mesh(number_of_elements,
                                                     {interval_size / number_of_elements}, {0.0});

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        const ippl::EdgeElement<double, 1> edge_element;

        // Create LagrangeSpace
        const ippl::LagrangeSpace<double, 1, 2, 1> lagrange_space(mesh, edge_element,
                                                                  midpoint_quadrature);

        // Print all the elements
        for (unsigned i = 0; i < number_of_elements; ++i) {
            std::cout << "Element " << i << " vertices: ";
            for (unsigned j = 0; j < 2; ++j) {
                const auto element_indices = lagrange_space.getDimensionIndicesForElement(i);
                std::cout << lagrange_space.getGlobalVerticesForElement(element_indices)[j] << " ";
            }
            std::cout << "\n";
        }
    }
    ippl::finalize();

    return 0;
}