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
                                                     {interval_size / number_of_elements}, {-1.0});

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        const ippl::EdgeElement<double> edge_element;

        // Create LagrangeSpace
        const unsigned number_of_local_vertices = 2;
        const ippl::LagrangeSpace<double, 1, 1, number_of_local_vertices, 1> lagrange_space(
            mesh, edge_element, midpoint_quadrature);

        // Print the 1D mesh vertices for plotting
        const std::string vertex_filename = "~1D_lagrange_vertices.csv";
        std::cout << "Writing vertices to " << vertex_filename << "\n";
        std::ofstream vertex_out(vertex_filename, std::ios::out);

        vertex_out << "vertex_index,x" << '\n';
        ippl::Vector<double, 1> vertex_coordinates;
        for (unsigned i = 0; i < number_of_vertices; i++) {
            vertex_out << i;
            vertex_coordinates = lagrange_space.getCoordinatesForDof(i);
            vertex_out << "," << vertex_coordinates[0];
            vertex_out << "\n";
        }
        vertex_out.close();

        // Print all the elements for plotting
        const std::string element_filename = "~1D_lagrange_elements.csv";
        std::cout << "Writing elements to " << element_filename << "\n";
        std::ofstream elem_out(element_filename, std::ios::out);

        elem_out << "element_index";
        for (unsigned vertex_i = 0; vertex_i < number_of_local_vertices; ++vertex_i) {
            elem_out << "," << vertex_i;
        }
        elem_out << "\n";

        for (unsigned i = 0; i < number_of_elements; ++i) {
            elem_out << i;
            const auto element_indices  = lagrange_space.getElementPositionFromIndex(i);
            const auto element_vertices = lagrange_space.getElementMeshVertices(element_indices);

            for (unsigned j = 0; j < element_vertices.dim; ++j) {
                elem_out << "," << element_vertices[j];
            }
            elem_out << "\n";
        }
        elem_out.close();

        // Print the global basis values for plotting
        const unsigned number_of_points = 200;
        const double dx                 = interval_size / (number_of_points - 1);

        // Print the local basis values for plotting
        const std::string local_basis_filename = "~1D_lagrange_local_basis.csv";
        std::cout << "Writing local basis functions to " << local_basis_filename << "\n";
        std::ofstream local_basis_out(local_basis_filename, std::ios::out);

        local_basis_out << "x";
        for (unsigned i = 0; i < number_of_local_vertices; ++i) {
            local_basis_out << ",v_" << i;
        }

        local_basis_out << "\n";

        for (ippl::Vector<double, 1> x = {0.0}; x[0] <= 1.0; x[0] += dx) {
            local_basis_out << x[0];
            for (unsigned i = 0; i < number_of_local_vertices; ++i) {
                local_basis_out << "," << lagrange_space.evaluateRefElementBasis(i, x);
            }
            local_basis_out << "\n";
        }
    }
    ippl::finalize();

    return 0;
}