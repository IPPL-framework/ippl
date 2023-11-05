#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform out("Test LagrangeSpace1DMidpoint");

        const unsigned number_of_points_per_dim = 200;

        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_vertices_per_dim = 5;
        const unsigned number_of_elements_per_dim = number_of_vertices_per_dim - 1;
        const double interval_size                = 2.0;
        const double h                            = interval_size / (number_of_elements_per_dim);
        const double dx                           = interval_size / number_of_points_per_dim;
        const ippl::NDIndex<2> meshIndex(number_of_vertices_per_dim, number_of_vertices_per_dim);
        const ippl::UniformCartesian<double, 2> mesh(meshIndex, {h, h}, {-1.0, -1.0});

        std::cout << "mesh spacing = " << mesh.getMeshSpacing() << "\n";

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        const ippl::QuadrilateralElement<double> quad_element;

        // Create LagrangeSpace
        const ippl::LagrangeSpace<double, 2, 1, 4, 1> lagrange_space(mesh, quad_element,
                                                                     midpoint_quadrature);

        // Print the values for the local basis functions
        const std::string local_basis_filename = "~2D_lagrange_local_basis.csv";
        std::cout << "Writing local basis function to " << local_basis_filename << "\n";
        std::ofstream local_basis_out(local_basis_filename, std::ios::out);

        local_basis_out << "x,y";
        for (unsigned i = 0; i < 4; ++i) {
            local_basis_out << ",v_" << i;
        }
        local_basis_out << "\n";

        for (double x = 0.0; x <= 1.0; x += dx) {
            for (double y = 0.0; y <= 1.0; y += dx) {
                local_basis_out << x << "," << y;
                for (unsigned i = 0; i < 4; ++i) {
                    local_basis_out << "," << lagrange_space.evaluateRefElementBasis(i, {x, y});
                }
                local_basis_out << "\n";
            }
        }
        local_basis_out.close();
    }
    ippl::finalize();

    return 0;
}