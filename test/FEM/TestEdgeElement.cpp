#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // test the transformation in 2D

        // Create an Edge in 2D
        const ippl::EdgeElement<double, 2> edge_element;

        // Print the local vertices
        const auto local_edge_vertices = edge_element.getLocalVertices();
        std::cout << "local_edge_vertices = (" << local_edge_vertices[0][0] << "), ("
                  << local_edge_vertices[1][0] << ")\n";

        const ippl::Vector<ippl::Vector<double, 2>, 2> global_edge_vertices = {{1.0, 2.0},
                                                                               {2.0, 3.0}};
        std::cout << "global_edge_vertices = (" << global_edge_vertices[0][0] << ","
                  << global_edge_vertices[0][1] << "), (" << global_edge_vertices[1][0] << ","
                  << global_edge_vertices[1][1] << ")\n";

        // The reference edge is form 0 to 1 on x and 0 on y

        // Rotate it by 90 degrees
        const auto inv_jac =
            edge_element.getInverseLinearTransformationJacobian(global_edge_vertices);

        // get the transformed vertex for this local point
        const double end_point = 1.0;
        const double mid_point = 0.5;

        const auto end_prod = inv_jac * end_point;
        const auto mid_prod = inv_jac * mid_point;

        ippl::Vector<double, 2> transformed_end_point = {end_prod[0][0], end_prod[1][0]};
        transformed_end_point += global_edge_vertices[0];

        ippl::Vector<double, 2> transformed_mid_point = {mid_prod[0][0], mid_prod[1][0]};
        transformed_mid_point += global_edge_vertices[0];

        std::cout << "transformed_end_point = (" << transformed_end_point[0] << ","
                  << transformed_end_point[1] << ")\n";
        std::cout << "transformed_mid_point = (" << transformed_mid_point[0] << ","
                  << transformed_mid_point[1] << ")\n";
    }
    ippl::finalize();

    return 0;
}