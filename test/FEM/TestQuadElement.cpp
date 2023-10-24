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

        // get the transformed vertex for these local pointsd
        const ippl::Vector<double, 2> end_point = {1.0, 1.0};
        const ippl::Vector<double, 2> mid_point = {0.5, 0.5};

        ippl::Vector<double, 2> transformed_end_point =
            quad_element.localToGlobal(global_quad_vertices, end_point);

        ippl::Vector<double, 2> transformed_mid_point =
            quad_element.localToGlobal(global_quad_vertices, mid_point);

        std::cout << "transformed_end_point = (" << transformed_end_point[0] << ","
                  << transformed_end_point[1] << ")\n";

        std::cout << "transformed_mid_point = (" << transformed_mid_point[0] << ","
                  << transformed_mid_point[1] << ")\n";

        // get the local point for these global points

        // TODO This is still broken
        const auto back_jac = quad_element.getLinearTransformationJacobian(global_quad_vertices);
        std::cout << "back_jac = (" << back_jac[0][0] << ", " << back_jac[0][1] << "), ("
                  << back_jac[1][0] << ", " << back_jac[1][1] << ")\n";

        // const ippl::Vector<double, 2> local_end_point =
        //     quad_element.globalToLocal(global_quad_vertices, global_quad_vertices[3]);

        // std::cout << "local_end_point = (" << local_end_point[0] << "," << local_end_point[1]
        //           << ")\n";
    }

    ippl::finalize();

    return 0;
}