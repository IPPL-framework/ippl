// TestNonStandardFDTDSolver
// Check the README.md file in this directory for information about the test and how to run it.

#include <cstddef>
using std::size_t;
#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include "Types/Vector.h"

#include "Field/Field.h"

#include "MaxwellSolvers/NonStandardFDTDSolver.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
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

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        using scalar       = float;
        const unsigned dim = 3;
        using vector_type  = ippl::Vector<scalar, 3>;
        using vector4_type = ippl::Vector<scalar, 4>;

        using SourceField =
            ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>,
                        typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using EMField = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>,
                                    typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;

        // Direction of the Gaussian pulse (can be 'x', 'y', or 'z')
        char direction = 'z';

        // Get variable for direction (1 for x, 2 for y and 3 for z)
        const int dir = (direction == 'x') ? 1 : (direction == 'y') ? 2 : 3;

        // Specifie number of gridpoints in each direction (more gridpoints in z needed for CFL
        // condition)
        constexpr size_t n = 100;
        ippl::Vector<uint32_t, 3> nr{n / 2, n / 2, 2 * n};
        ippl::NDIndex<3> owned(nr[0], nr[1], nr[2]);

        // specifies decomposition;
        std::array<bool, 3> isParallel;
        isParallel.fill(true);

        // unit box
        ippl::Vector<scalar, 3> extents{scalar(1), scalar(1), scalar(1)};
        ippl::Vector<scalar, 3> hx;
        for (unsigned d = 0; d < 3; d++) {
            hx[d] = extents[d] / (scalar)nr[d];
        }
        ippl::Vector<scalar, 3> origin = {0, 0, 0};
        ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        // Define the source and field types
        SourceField source(mesh, layout);
        EMField E(mesh, layout);
        EMField B(mesh, layout);
        source = vector4_type(0);

        // Create the NonStandardFDTDSolver object
        ippl::NonStandardFDTDSolver<EMField, SourceField, ippl::periodic> sfdsolver(source, E, B);

        // Initialize the source field with a Gaussian distribution in the z-direction and zeros in
        // the x and y directions
        auto aview    = sfdsolver.A_n.getView();
        auto am1view  = sfdsolver.A_nm1.getView();
        auto ap1view  = sfdsolver.A_np1.getView();
        auto lDom     = layout.getLocalNDIndex();
        size_t nghost = sfdsolver.A_n.getNghost();

        // Initialize the fields and calculate the sum of the squared magnitudes for error
        // calculation later
        double sum_norm = 0.0;
        Kokkos::parallel_reduce(
            ippl::getRangePolicy(aview, 1),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref) {
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

                ref += magnitude
                       * magnitude;  // Accumulate the squared magnitude for error calculation
            },
            sum_norm);
        Kokkos::fence();

        // Apply the boundary conditions to the initialized fields
        sfdsolver.A_n.getFieldBC().apply(sfdsolver.A_n);
        sfdsolver.A_np1.getFieldBC().apply(sfdsolver.A_np1);
        sfdsolver.A_nm1.getFieldBC().apply(sfdsolver.A_nm1);
        sfdsolver.A_n.fillHalo();
        sfdsolver.A_nm1.fillHalo();

        // Create a 500x500 image with 3 channels (RGB)
        int img_width    = 500;
        int img_height   = 500;
        float* imagedata = new float[img_width * img_height * 3];

        // Initialize the image data to black
        std::fill(imagedata, imagedata + img_width * img_height * 3, 0.0f);
        uint8_t* imagedata_final = new uint8_t[img_width * img_height * 3];

        // Run the simulation for 1s, with periodic boundary conditions this should be the same
        // state as at time 0
        for (size_t s = 0; s < 1. / sfdsolver.getDt(); s++) {
            auto ebh = sfdsolver.A_n.getHostMirror();
            Kokkos::deep_copy(ebh, sfdsolver.A_n.getView());
            for (int i = 1; i < img_width; i++) {
                for (int j = 1; j < img_height; j++) {
                    // Map the pixel coordinates to the simulation domain
                    int i_remap = (double(i) / (img_width - 1)) * (nr[2] - 4) + 2;
                    int j_remap = (double(j) / (img_height - 1)) * (nr[0] - 4) + 2;

                    // Check if the remapped coordinates are within the local domain
                    if (i_remap >= lDom.first()[2] && i_remap <= lDom.last()[2]) {
                        if (j_remap >= lDom.first()[0] && j_remap <= lDom.last()[0]) {
                            // Get the corresponding field vector
                            ippl::Vector<scalar, 4> acc =
                                ebh(j_remap + 1 - lDom.first()[0], nr[1] / 2,
                                    i_remap + 1 - lDom.first()[2]);

                            // Normalize field vector and set pixel color
                            float normalized_colorscale_value = acc.Pnorm();
                            imagedata[(j * img_width + i) * 3 + 0] =
                                normalized_colorscale_value * 255.0f;
                            imagedata[(j * img_width + i) * 3 + 1] =
                                normalized_colorscale_value * 255.0f;
                            imagedata[(j * img_width + i) * 3 + 2] =
                                normalized_colorscale_value * 255.0f;
                        }
                    }
                }
            }

            // Convert the float image data to unsigned char
            std::transform(imagedata, imagedata + img_height * img_width * 3, imagedata_final,
                           [](float x) {
                               return (unsigned char)std::min(255.0f, std::max(0.0f, x));
                           });
            char output[1024] = {0};
            snprintf(output, 1023, "%soutimage%.05lu.bmp", "renderdataNonStandard/", s);

            // Save the image every 4th step
            if (s % 4 == 0)
                stbi_write_bmp(output, img_width, img_height, 3, imagedata_final);

            // Solve the FDTD equations
            sfdsolver.solve();
        }

        // Calculate the error between the computed and expected values
        double sum_error = 0.0;
        Kokkos::parallel_reduce(
            ippl::getRangePolicy(aview, 1),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref) {
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
        std::cout << "Sum after evaluating boundaryconditions: "
                  << Kokkos::sqrt(sum_error / sum_norm) << "\n";
    }
    ippl::finalize();
}