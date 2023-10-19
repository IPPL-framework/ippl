#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        const ippl::QuadrilateralElement<double, 2> quad_element;

        // Print the local vertices
        const auto local_quad_vertices = quad_element.getLocalVertices();
        std::cout << "local_quad_vertices = (" << local_quad_vertices[0][0] << ","
                  << local_quad_vertices[0][1] << "), (" << local_quad_vertices[1][0] << ","
                  << local_quad_vertices[1][1] << "), (" << local_quad_vertices[2][0] << ","
                  << local_quad_vertices[2][1] << "), (" << local_quad_vertices[3][0] << ","
                  << local_quad_vertices[3][1] << ")\n";

        // A parallelogram (bilinear transformations are not supported yet) thus (only) an affine
        // transfomration is presented here
        const ippl::Vector<ippl::Vector<double, 2>, 4> global_quad_vertices = {
            {1.0, 1.0}, {2.0, 1.5}, {1.2, 2.0}, {2.2, 2.5}};

        std::cout << "global_quad_vertices = (" << global_quad_vertices[0][0] << ","
                  << global_quad_vertices[0][1] << "), (" << global_quad_vertices[1][0] << ","
                  << global_quad_vertices[1][1] << "), (" << global_quad_vertices[2][0] << ","
                  << global_quad_vertices[2][1] << "), (" << global_quad_vertices[3][0] << ","
                  << global_quad_vertices[3][1] << ")\n";

        const auto inv_jac =
            quad_element.getInverseLinearTransformationJacobian(global_quad_vertices);

        std::cout << "inv_jac = "
                  << " (" << inv_jac[0][0] << "," << inv_jac[0][1] << "), (" << inv_jac[1][0] << ","
                  << inv_jac[1][1] << ")\n";

        // get the transformed vertex for these local points
        const ippl::Vector<double, 2> end_point = {1.0, 1.0};
        const ippl::Vector<double, 2> mid_point = {0.5, 0.5};

        const ippl::Vector<double, 2> end_prod = {inv_jac[0].dot(end_point),
                                                  inv_jac[1].dot(end_point)};

        const ippl::Vector<double, 2> mid_prod = {inv_jac[0].dot(mid_point),
                                                  inv_jac[1].dot(mid_point)};

        ippl::Vector<double, 2> transformed_end_point = end_prod + global_quad_vertices[0];
        ippl::Vector<double, 2> transformed_mid_point = mid_prod + global_quad_vertices[0];

        std::cout << "transformed_end_point = (" << transformed_end_point[0] << ","
                  << transformed_end_point[1] << ")\n";

        std::cout << "transformed_mid_point = (" << transformed_mid_point[0] << ","
                  << transformed_mid_point[1] << ")\n";
    }

    ippl::finalize();

    return 0;
}