#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform out("Test LagrangeSpace1DMidpoint");

        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_vertices = 10;
        const unsigned number_of_elements = number_of_vertices - 1;
        const double interval_size        = 2.0;
        const ippl::UniformCartesian<double, 1> mesh(number_of_vertices,
                                                     {interval_size / number_of_elements}, {0.0});

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        const ippl::EdgeElement<double, 1> edge_element;

        // Create LagrangeSpace
        const ippl::LagrangeSpace<double, 1, 2, 1> lagrange_space(mesh, edge_element,
                                                                  midpoint_quadrature);

        // Print the 1D mesh vertices for plotting
        std::ofstream vertex_out("~1D_lagrange_vertices.dat");

        vertex_out << "vertex_index,x\n";
        ippl::Vector<double, 1> vertex_coordinates;
        for (unsigned i = 0; i < number_of_vertices; i += 1) {
            vertex_out << i;
            vertex_coordinates = lagrange_space.getCoordinatesForVertex(i);
            vertex_out << "," << vertex_coordinates[0];
            vertex_out << "\n";
        }

        // Print all the elements for plotting
        std::ofstream elem_out("~1D_lagrange_elements.dat");

        elem_out << "element_index,a,b\n";
        for (unsigned i = 0; i < number_of_elements; ++i) {
            elem_out << i;
            const auto element_indices = lagrange_space.getNDIndexForElement(i);
            for (unsigned j = 0; j < 2; ++j) {
                elem_out << "," << element_indices[j];
            }
            elem_out << "\n";
        }

        // Print the basis values for plotting
        const unsigned number_of_points = 100;
        const double dx                 = interval_size / (number_of_points - 1);
        ippl::Vector<double, 1> x       = {0.0};
        std::ofstream basis_out("~1D_lagrange_basis.dat");

        basis_out << "x,\n";
        for (ippl::Vector<double, 1> x = {-1.0}; x[0] <= 1.0; x[0] += dx) {
            for (unsigned i = 0; i < number_of_vertices; ++i) {
                basis_out << lagrange_space.evaluateBasis(i, x);
                basis_out << "\n";
            }
        }
    }
    ippl::finalize();

    return 0;
}