// Tests the FEM poison solver by // TODO

#include "Ippl.h"
#include "Solver/FEMPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned dim = 3;  // 3D problem
        using Mesh_t           = ippl::UniformCartesian<double, 3>;
        using Centering_t      = Mesh_t::DefaultCentering;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

        ippl::FEMPoissonSolver<field_type> solver;

        // IpplTimings::print("timings" + std::to_string(pt) + ".dat");
    }
    ippl::finalize();

    return 0;
}