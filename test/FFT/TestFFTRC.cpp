#include "Ippl.h"

#include <array>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"
#include "Utility/TypeUtils.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        std::array<int, dim> pt = {64, 64, 64};
        ippl::Index Iinput(pt[0]);
        ippl::Index Jinput(pt[1]);
        ippl::Index Kinput(pt[2]);
        ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        ippl::FieldLayout<dim> layoutInput(MPI_COMM_WORLD, ownedInput, isParallel);

        std::array<double, dim> dx = {
            1.0 / double(pt[0]),
            1.0 / double(pt[1]),
            1.0 / double(pt[2]),
        };
        ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        ippl::UniformCartesian<double, 3> meshInput(ownedInput, hx, origin);

        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
            field_type_complex;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type_real;

        field_type_real fieldInput(meshInput, layoutInput);

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);
        fftParams.add("r2c_direction", 0);

        ippl::NDIndex<dim> ownedOutput;

        if (fftParams.get<int>("r2c_direction") == 0) {
            ownedOutput[0] = ippl::Index(pt[0] / 2 + 1);
            ownedOutput[1] = ippl::Index(pt[1]);
            ownedOutput[2] = ippl::Index(pt[2]);
        } else if (fftParams.get<int>("r2c_direction") == 1) {
            ownedOutput[0] = ippl::Index(pt[0]);
            ownedOutput[1] = ippl::Index(pt[1] / 2 + 1);
            ownedOutput[2] = ippl::Index(pt[2]);
        } else if (fftParams.get<int>("r2c_direction") == 2) {
            ownedOutput[0] = ippl::Index(pt[0]);
            ownedOutput[1] = ippl::Index(pt[1]);
            ownedOutput[2] = ippl::Index(pt[2] / 2 + 1);
        } else {
            if (ippl::Comm->rank() == 0) {
                std::cerr << "RCDirection need to be 0, 1 or 2 and it"
                          << "indicates the dimension in which data is shortened" << std::endl;
            }
            return 0;
        }
        ippl::FieldLayout<dim> layoutOutput(MPI_COMM_WORLD, ownedOutput, isParallel);

        Mesh_t meshOutput(ownedOutput, hx, origin);
        field_type_complex fieldOutput(meshOutput, layoutOutput);

        typedef ippl::FFT<ippl::RCTransform, field_type_real> FFT_type;

        std::unique_ptr<FFT_type> fft;

        fft = std::make_unique<FFT_type>(layoutInput, layoutOutput, fftParams);

        typename field_type_real::view_type& view            = fieldInput.getView();
        typename field_type_real::HostMirror fieldInput_host = fieldInput.getHostMirror();

        const int nghost = fieldInput.getNghost();
        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    fieldInput_host(i, j, k) = unif(eng);  // 1.0;
                }
            }
        }

        Kokkos::deep_copy(fieldInput.getView(), fieldInput_host);

        // Forward transform
        fft->transform(ippl::FORWARD, fieldInput, fieldOutput);
        // Reverse transform
        fft->transform(ippl::BACKWARD, fieldInput, fieldOutput);

        auto field_result =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fieldInput.getView());

        double max_error_local = 0.0;
        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    double error = std::fabs(fieldInput_host(i, j, k) - field_result(i, j, k));

                    if (error > max_error_local)
                        max_error_local = error;

                    std::cout << "Error: " << std::setprecision(16) << error << std::endl;
                }
            }
        }

        // Kokkos::complex<double> max_error(0.0, 0.0);
        // MPI_Reduce(&max_error_local, &max_error, 1,
        //            MPI_C_DOUBLE_COMPLEX, MPI_MAX, 0, ippl::Comm->getCommunicator());

        // if(ippl::Comm->rank() == 0) {
        std::cout << "Rank:" << ippl::Comm->rank() << "Max. error " << std::setprecision(16)
                  << max_error_local << std::endl;
        //}
    }
    ippl::finalize();

    return 0;
}
