#include "Ippl.h"

#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_elements = 10;
        const double interval_size        = 2.0;
        const ippl::UniformCartesian<double, 1> mesh(number_of_elements,
                                                     {interval_size / number_of_elements}, {0.0});

        // Create Midpoint Quadrature
        ippl::MidpointQuadrature<double> midpoint_quadrature(1);

        // Refernce element
        const ippl::LineElement edge_element = ippl::LineElement::getInstance();

        // Create LagrangeSpace
        ippl::LagrangeSpace<double, 1> lagrange_space(mesh, edge_element, midpoint_quadrature, 1);

        // Print all the elements
        for (unsigned i = 0; i < number_of_elements; ++i) {
            std::cout << "Element " << i << " vertices: ";
            for (unsigned j = 0; j < 2; ++j) {
                std::cout << lagrange_space.getVerticesForElement(i)[j] << " ";
            }
            std::cout << std::endl;
        }
    }
}