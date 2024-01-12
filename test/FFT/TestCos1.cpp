#include "Ippl.h"

#include <array>
#include <iostream>
#include <random>
#include <typeinfo>

#include "Utility/ParameterList.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        std::array<int, dim> pt = {32, 32, 32};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        std::array<double, dim> dx = {
            1.0 / double(pt[0]),
            1.0 / double(pt[1]),
            1.0 / double(pt[2]),
        };
        ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

        field_type field(mesh, layout);

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", false);
        fftParams.add("use_pencils", true);
        fftParams.add("use_reorder", false);
        fftParams.add("use_gpu_aware", true);
        fftParams.add("comm", ippl::p2p_pl);

        typedef ippl::FFT<ippl::Cos1Transform, field_type> FFT_type;

        std::unique_ptr<FFT_type> fft;

        fft = std::make_unique<FFT_type>(layout, fftParams);

        typename field_type::view_type& view       = field.getView();
        typename field_type::HostMirror field_host = field.getHostMirror();

        const int nghost = field.getNghost();
        std::mt19937_64 eng(42 + ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    field_host(i, j, k) = unif(eng);  // 1.0;
                }
            }
        }

        Kokkos::deep_copy(field.getView(), field_host);

        // Forward transform
        fft->transform(ippl::FORWARD, field);
        // Reverse transform
        fft->transform(ippl::BACKWARD, field);

        auto field_result =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());

        double max_error_local = 0.0;
        for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
            for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
                for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
                    double error = std::fabs(field_host(i, j, k) - field_result(i, j, k));

                    if (error > max_error_local)
                        max_error_local = error;

                    std::cout << "Error: " << std::setprecision(16) << error << std::endl;
                }
            }
        }

        double max_error = 0.0;
        MPI_Reduce(&max_error_local, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
                   ippl::Comm->getCommunicator());

        std::cout << "Rank:" << ippl::Comm->rank() << "Max. error " << std::setprecision(16)
                  << max_error_local << std::endl;
        if (ippl::Comm->rank() == 0) {
            std::cout << "Overall Max. error " << std::setprecision(16) << max_error << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
