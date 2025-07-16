// TestStandardFDTDSolver_convergence
// Check the README.md file in this directory for information about the test and how to run it.

#include <cstddef>
using std::size_t;
#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include "Types/Vector.h"

#include "Field/Field.h"

#include "MaxwellSolvers/StandardFDTDSolver.h"

template <typename scalar1, typename... scalar>
    requires((std::is_floating_point_v<scalar1>))
KOKKOS_INLINE_FUNCTION float gauss(scalar1 mean, scalar1 stddev, scalar... x) {
    uint32_t dim = sizeof...(scalar);
    ippl::Vector<scalar1, sizeof...(scalar)> vec{scalar1(x - mean)...};
    scalar1 vecsum(0);
    for (unsigned d = 0; d < dim; d++) {
        vecsum += vec[d] * vec[d];
    }
    return Kokkos::exp(-(vecsum) / (stddev * stddev));
}

void compute_convergence(char direction, unsigned int np, std::string fname) {
    using scalar       = float;
    const unsigned dim = 3;
    using vector_type  = ippl::Vector<scalar, 3>;
    using vector4_type = ippl::Vector<scalar, 4>;

    using SourceField = ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>,
                                    typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
    using EMField     = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>,
                                    typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;

    // Get variable for direction (0 for x, 1 for y and 2 for z)
    const int dir = (direction == 'x') ? 1 : (direction == 'y') ? 2 : 3;

    // Specifie number of gridpoints in each direction (more gridpoints in z needed for CFL
    // condition)
    ippl::Vector<uint32_t, 3> nr{np, np, np};
    ippl::NDIndex<3> owned(nr[0], nr[1], nr[2]);

    // specifies decomposition; here all dimensions are parallel
    std::array<bool, 3> isParallel;
    isParallel.fill(true);

    // unit box
    ippl::Vector<scalar, 3> hx;
    for (unsigned d = 0; d < 3; d++) {
        hx[d] = 1 / (scalar)nr[d];
    }
    ippl::Vector<scalar, 3> origin = {0.0, 0.0, 0.0};
    ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

    // Define the source and field types
    SourceField source(mesh, layout);
    EMField E(mesh, layout);
    EMField B(mesh, layout);
    source = vector4_type(0);

    // Create the StandardFDTDSolver object
    ippl::StandardFDTDSolver<EMField, SourceField, ippl::periodic> sfdsolver(source, E, B);

    // Initialize the source field with a Gaussian distribution in the given direction and zeros in
    // the other directions
    auto aview    = sfdsolver.A_n.getView();
    auto am1view  = sfdsolver.A_nm1.getView();
    auto ap1view  = sfdsolver.A_np1.getView();
    auto lDom     = layout.getLocalNDIndex();
    size_t nghost = sfdsolver.A_n.getNghost();

    // Initialize the fields and calculate the sum of the squared magnitudes for error calculation
    // later
    double sum_norm = 0.0;
    Kokkos::parallel_reduce(
        ippl::getRangePolicy(aview, 1),
        KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref) {
            // Calculate position in x, y and z
            const size_t ig = i + lDom.first()[0] - nghost;
            const size_t jg = j + lDom.first()[1] - nghost;
            const size_t kg = k + lDom.first()[2] - nghost;
            scalar x        = scalar(ig) / nr[0];
            scalar y        = scalar(jg) / nr[1];
            scalar z        = scalar(kg) / nr[2];
            (void)x;
            (void)y;
            (void)z;

            // Calculate gaussian pules in direction dir
            const scalar coord     = (dir == 1) ? x : (dir == 2) ? y : z;
            const scalar magnitude = gauss(scalar(0.5), scalar(0.05), coord);

            // Initialize fields
            aview(i, j, k)      = vector4_type{scalar(0), scalar(0), scalar(0), scalar(0)};
            aview(i, j, k)[dir] = magnitude;

            am1view(i, j, k)      = vector4_type{scalar(0), scalar(0), scalar(0), scalar(0)};
            am1view(i, j, k)[dir] = magnitude;

            // Calculate the sum of the squared magnitudes for L2 norm error calculation later
            ref += magnitude * magnitude;
        },
        sum_norm);
    Kokkos::fence();

    // Apply the boundary conditions to the initialized fields
    sfdsolver.A_n.getFieldBC().apply(sfdsolver.A_n);
    sfdsolver.A_np1.getFieldBC().apply(sfdsolver.A_np1);
    sfdsolver.A_nm1.getFieldBC().apply(sfdsolver.A_nm1);
    sfdsolver.A_n.fillHalo();
    sfdsolver.A_nm1.fillHalo();

    // Run the simulation for 1s, with periodic boundary conditions this should be the same state as
    // at time 0
    for (size_t s = 0; s < 1. / sfdsolver.getDt(); s++) {
        // Solve the FDTD equations
        sfdsolver.solve();
    }

    // Calculate the L2 norm error between the computed and expected values
    double sum_error = 0.0;
    Kokkos::parallel_reduce(
        ippl::getRangePolicy(aview, 1),
        KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref) {
            // Calculate position in x, y and z
            const size_t ig = i + lDom.first()[0] - nghost;
            const size_t jg = j + lDom.first()[1] - nghost;
            const size_t kg = k + lDom.first()[2] - nghost;
            scalar x        = scalar(ig) / nr[0];
            scalar y        = scalar(jg) / nr[1];
            scalar z        = scalar(kg) / nr[2];
            (void)x;
            (void)y;
            (void)z;

            // Get coordinate in the right direction
            const scalar coord = (dir == 1) ? x : (dir == 2) ? y : z;

            // Calculate error in given direction at this point
            double original_value = gauss(scalar(0.5), scalar(0.05), coord);

            // Calculate the difference between the computed and expected values
            ippl::Vector<scalar, 4> diff = aview(i, j, k);
            diff[dir] -= original_value;

            // Accumulate the squared differences for L2 norm and the original value for
            // normalization
            ref += ippl::dot(diff, diff).apply();
        },
        sum_error);

    // Write results to output file
    Inform csvout(NULL, fname.c_str(), Inform::APPEND);
    csvout.precision(16);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    csvout << direction << "," << np << "," << Kokkos::sqrt(sum_error / sum_norm) << endl;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // gridsizes to iterate over
        std::array<int, 7> N = {4, 8, 16, 32, 64, 128, 256};

        // directions to iterate over
        std::array<char, 3> directions = {'x', 'y', 'z'};

        // Create outputfile and plot header
        std::string fname = "StandardFDTDSolver_convergence.csv";

        Inform csvout(NULL, fname.c_str(), Inform::OVERWRITE);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        csvout << "GaussianPulseDir,NGridpoints,ConverganceError" << endl;

        for (char dir : directions) {
            for (int pt : N) {
                compute_convergence(dir, pt, fname);
            }
        }
    }
    ippl::finalize();
}