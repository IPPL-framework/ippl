#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform out("Test LagrangeSpace1DMidpoint");

        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_vertices_per_dim = 5;
        const unsigned number_of_elements_per_dim = number_of_vertices_per_dim - 1;
        const double interval_size                = 2.0;
        const double h                            = interval_size / number_of_elements_per_dim;
        const ippl::NDIndex<2> meshIndex(number_of_vertices_per_dim, number_of_vertices_per_dim);
        const ippl::UniformCartesian<double, 2> mesh(meshIndex, {h, h}, {-1.0, -1.0});

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        const ippl::QuadrilateralElement<double, 2> quad_element;

        // Create LagrangeSpace
        const ippl::LagrangeSpace<double, 2, 4, 1> lagrange_space(mesh, quad_element,
                                                                  midpoint_quadrature);

        // Plot the basis function for a vertex
        const unsigned number_of_points_per_dim = 200;
        const unsigned vertex_index             = 12;
        const double dx                         = interval_size / (number_of_points_per_dim - 1);

        const std::string basis_filename = "~2D_lagrange_basis.csv";
        std::cout << "Writing basis function to " << basis_filename << "\n";
        std::ofstream basis_out(basis_filename, std::ios::out);

        basis_out << "x,y," << vertex_index << "\n";

        for (double x = -1.3; x <= 1.3; x += dx) {
            for (double y = -1.3; y <= 1.3; y += dx) {
                basis_out << x << "," << y << ",";
                basis_out << lagrange_space.evaluateBasis(vertex_index, {x, y});
                basis_out << "\n";
            }
        }
        basis_out.close();
    }
    ippl::finalize();

    return 0;
}