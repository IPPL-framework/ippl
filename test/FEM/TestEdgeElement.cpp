#include "Ippl.h"

#include <fstream>

// #include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        const ippl::EdgeElement<double> edge_element;

        // Get and print the reference vertices of an edge
        // The reference edge is form 0 to 1 on x and 0 on y
        const auto local_edge_vertices = edge_element.getLocalVertices();
        std::cout << "local_edge_vertices = (" << local_edge_vertices[0][0] << "), ("
                  << local_edge_vertices[1][0] << ")\n";

        // The global edge points into the opposite direction and has a length of 3
        const ippl::Vector<ippl::Vector<double, 1>, 2> global_edge_vertices = {{5.0}, {2.0}};

        std::cout << "global_edge_vertices = (" << global_edge_vertices[0][0] << ","
                  << global_edge_vertices[1][0] << ")\n";

        // get the transformed vertex for these local points
        const ippl::Vector<double, 1> start_point = {0.0};
        const ippl::Vector<double, 1> mid_point   = {0.5};
        const ippl::Vector<double, 1> end_point   = {1.0};

        ippl::Vector<double, 1> transformed_start_point =
            edge_element.localToGlobal(global_edge_vertices, start_point);
        ippl::Vector<double, 1> transformed_mid_point =
            edge_element.localToGlobal(global_edge_vertices, mid_point);
        ippl::Vector<double, 1> transformed_end_point =
            edge_element.localToGlobal(global_edge_vertices, end_point);

        std::cout << "transformed_start_point = " << transformed_start_point[0] << "\n";
        std::cout << "transformed_mid_point = " << transformed_mid_point[0] << "\n";
        std::cout << "transformed_end_point = " << transformed_end_point[0] << "\n";
    }
    ippl::finalize();

    return 0;
}