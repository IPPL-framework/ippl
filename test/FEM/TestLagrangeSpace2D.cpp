#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform out("Test LagrangeSpace1DMidpoint");

        // Create a 1D uniform mesh centered at 0.0.
        const unsigned number_of_vertices_per_dim = 10;
        const unsigned number_of_elements_per_dim = number_of_vertices_per_dim - 1;
        const double interval_size                = 2.0;
        const double h                            = interval_size / number_of_elements_per_dim;
        const ippl::NDIndex<2> meshIndex(number_of_vertices_per_dim, number_of_vertices_per_dim);
        const ippl::UniformCartesian<double, 2> mesh(meshIndex, {h, h}, {-1.0, -1.0});

        // Create Midpoint Quadrature
        const ippl::MidpointQuadrature<double, 1> midpoint_quadrature;

        // Refernce element
        // const ippl::QuadrilateralElement<double, 1> quad_element;

        // Create LagrangeSpace
        // const ippl::LagrangeSpace<double, 1, 2, 1> lagrange_space(mesh, quad_element,
        //                                                           midpoint_quadrature);

        // TODO
    }
    ippl::finalize();

    return 0;
}