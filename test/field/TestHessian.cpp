//
////
//// TestHessian.cpp
////   This program tests the Hessian operator on a vector field.
////   The problem size can be given by the user (N^3), and a bool (0 or 1)
////   indicates whether the field is f=xyz or a Gaussian field.
////
//// Usage:
////   srun ./TestHessian N 0 --info 10
////
////
//

#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 1.0,
                                       double mu = 0.5) {
    double pi        = std::acos(-1.0);
    double prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt         = std::atoi(argv[1]);
        bool gauss_fct = std::atoi(argv[2]);
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        // Specifies SERIAL, PARALLEL dims
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // type definitions
        typedef ippl::Vector<double, dim> Vector_t;
        typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Vector<Vector_t, dim> Matrix_t;
        typedef ippl::Field<Matrix_t, dim, Mesh_t, Centering_t> MField_t;

        // domain [0,1]^3
        double dx       = 1.0 / double(pt);
        Vector_t hx     = {dx, dx, dx};
        Vector_t origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        Field_t field(mesh, layout, 1);
        MField_t result(mesh, layout, 1);
        MField_t exact(mesh, layout, 1);

        typename Field_t::view_type& view = field.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
        const int nghost               = field.getNghost();
        using mdrange_type             = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for(
            "Assign field",
            mdrange_type({0, 0, 0}, {view.extent(0), view.extent(1), view.extent(2)}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                if (gauss_fct) {
                    view(i, j, k) = gaussian(x, y, z);
                } else {
                    view(i, j, k) = x * y * z;
                }
            });

        typename MField_t::view_type& view_exact  = exact.getView();
        typename MField_t::view_type& view_result = result.getView();

        Kokkos::parallel_for(
            "Assign exact",
            mdrange_type({nghost, nghost, nghost},
                         {view_exact.extent(0) - nghost, view_exact.extent(1) - nghost,
                          view_exact.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                double mu = 0.5;

                if (gauss_fct) {
                    view_exact(i, j, k)[0] = {((x - mu) * (x - mu) - 1.0) * gaussian(x, y, z),
                                              (x - mu) * (y - mu) * gaussian(x, y, z),
                                              (x - mu) * (z - mu) * gaussian(x, y, z)};
                    view_exact(i, j, k)[1] = {(x - mu) * (y - mu) * gaussian(x, y, z),
                                              ((y - mu) * (y - mu) - 1.0) * gaussian(x, y, z),
                                              (y - mu) * (z - mu) * gaussian(x, y, z)};
                    view_exact(i, j, k)[2] = {(x - mu) * (z - mu) * gaussian(x, y, z),
                                              (y - mu) * (z - mu) * gaussian(x, y, z),
                                              ((z - mu) * (z - mu) - 1.0) * gaussian(x, y, z)};
                } else {
                    view_exact(i, j, k)[0] = {0.0, z, y};
                    view_exact(i, j, k)[1] = {z, 0.0, x};
                    view_exact(i, j, k)[2] = {y, x, 0.0};
                }
            });

        result     = {0.0, 0.0, 0.0};
        auto timer = IpplTimings::getTimer("Hessian");
        IpplTimings::startTimer(timer);
        for (int i = 0; i < 100; i++) {
            result = hess(field);
        }
        IpplTimings::stopTimer(timer);

        result = result - exact;

        ippl::Vector<Vector_t, 3> err_hess{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

        double avg = 0.0;

        for (size_t dim1 = 0; dim1 < dim; ++dim1) {
            for (size_t dim2 = 0; dim2 < dim; ++dim2) {
                double valN(0.0);

                Kokkos::parallel_reduce(
                    "Relative error",
                    mdrange_type({nghost, nghost, nghost},
                                 {view_result.extent(0) - nghost, view_result.extent(1) - nghost,
                                  view_result.extent(2) - nghost}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                        double myVal = pow(view_result(i, j, k)[dim1][dim2], 2);
                        val += myVal;
                    },
                    Kokkos::Sum<double>(valN));

                double globalN(0.0);
                ippl::Comm->allreduce(valN, globalN, 1, std::plus<double>());
                double errorN = std::sqrt(globalN);

                double valD(0.0);

                Kokkos::parallel_reduce(
                    "Relative error",
                    mdrange_type({nghost, nghost, nghost},
                                 {view_exact.extent(0) - nghost, view_exact.extent(1) - nghost,
                                  view_exact.extent(2) - nghost}),
                    KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                        double myVal = pow(view_exact(i, j, k)[dim1][dim2], 2);
                        val += myVal;
                    },
                    Kokkos::Sum<double>(valD));

                double globalD(0.0);
                ippl::Comm->allreduce(valD, globalD, 1, std::plus<double>());
                double errorD = std::sqrt(globalD);

                // Compute relative Error
                if ((errorD < 1e-15) && (errorN < 1e-15)) {
                    err_hess[dim1][dim2] = 0.0;
                } else {
                    err_hess[dim1][dim2] = errorN / errorD;
                }

                if (ippl::Comm->rank() == 0) {
                    std::cout << std::setprecision(16) << "Error (" << dim1 + 1 << "," << dim2 + 1
                              << "): " << err_hess[dim1][dim2] << std::endl;
                }

                avg += err_hess[dim1][dim2];
            }
        }

        // print total error (average of each matrix entry)
        avg /= 9.0;
        std::cout << std::setprecision(16) << "Average error = " << avg;

        IpplTimings::print();
    }
    ippl::finalize();

    return 0;
}
