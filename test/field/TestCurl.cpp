//
////
//// TestCurl.cpp
////   This program tests the Curl operator on a vector field.
////   The problem size can be given by the user (N^3), and a bool (0 or 1)
////   indicates whether the vector field is A=(xyz, xyz, xyz) or a Gaussian
////   field in all three dimensions.
////
//// Usage:
////   srun ./TestCurl N 0 --info 10
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

        // domain [0,1]^3
        double dx                      = 1.0 / double(pt);
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        typedef ippl::Vector<double, dim> Vector_t;
        typedef ippl::Field<Vector_t, dim, Mesh_t, Centering_t> Vfield_t;

        Vfield_t vfield(mesh, layout);
        Vfield_t result(mesh, layout);
        Vfield_t exact(mesh, layout);

        typename Vfield_t::view_type& view        = vfield.getView();
        typename Vfield_t::view_type& view_exact  = exact.getView();
        typename Vfield_t::view_type& view_result = result.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
        const int nghost               = vfield.getNghost();
        using mdrange_type             = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        for (unsigned int gd = 0; gd < dim; ++gd) {
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
                        view(i, j, k)[gd] = gaussian(x, y, z);
                    } else {
                        view(i, j, k)[gd] = x * y * z;
                    }

                    if (gd == 0) {
                        if (gauss_fct) {
                            view_exact(i, j, k)[gd] = (z - y) * gaussian(x, y, z);
                        } else {
                            view_exact(i, j, k)[gd] = x * z - x * y;
                        }
                    } else if (gd == 1) {
                        if (gauss_fct) {
                            view_exact(i, j, k)[gd] = (x - z) * gaussian(x, y, z);
                        } else {
                            view_exact(i, j, k)[gd] = x * y - y * z;
                        }
                    } else {
                        if (gauss_fct) {
                            view_exact(i, j, k)[gd] = (y - x) * gaussian(x, y, z);
                        } else {
                            view_exact(i, j, k)[gd] = y * z - x * z;
                        }
                    }
                });
        }

        result = 0.0;
        result = curl(vfield);

        Vector_t errE;
        result = result - exact;

        for (unsigned int gd = 0; gd < dim; ++gd) {
            double temp = 0.0;

            Kokkos::parallel_reduce(
                "Vector errorNr reduce",
                mdrange_type({nghost, nghost, nghost},
                             {view_result.extent(0) - nghost, view_result.extent(1) - nghost,
                              view_result.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& valL) {
                    double myVal = pow(view_result(i, j, k)[gd], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));
            double globaltemp = 0.0;
            ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<double>());
            double errorNr = std::sqrt(globaltemp);

            temp       = 0.0;
            globaltemp = 0.0;

            Kokkos::parallel_reduce(
                "Vector errorDr reduce",
                mdrange_type({nghost, nghost, nghost},
                             {view_exact.extent(0) - nghost, view_exact.extent(1) - nghost,
                              view_exact.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& valL) {
                    double myVal = pow(view_exact(i, j, k)[gd], 2);
                    valL += myVal;
                },
                Kokkos::Sum<double>(temp));
            ippl::Comm->allreduce(temp, globaltemp, 1, std::plus<double>());
            double errorDr = std::sqrt(globaltemp);

            errE[gd] = errorNr / errorDr;
        }

        if (ippl::Comm->rank() == 0) {
            std::cout << "Error: " << errE[0] << ", " << errE[1] << ", " << errE[2] << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
