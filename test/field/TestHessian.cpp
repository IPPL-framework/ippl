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
//// Copyright (c) 2022, Sonali Mayani,
//// Paul Scherrer Institut, Villigen, Switzerland
//// All rights reserved
////
//// This file is part of IPPL.
////
//// IPPL is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// You should have received a copy of the GNU General Public License
//// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
////
//

#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>

KOKKOS_INLINE_FUNCTION
double gaussian(double x, double y, double z, double sigma = 1.0, double mu = 0.5) {

    double pi = std::acos(-1.0);
    double prefactor = (1/std::sqrt(2*2*2*pi*pi*pi))*(1/(sigma*sigma*sigma));
    double r2 = (x-mu)*(x-mu) + (y-mu)*(y-mu) + (z-mu)*(z-mu);

    return -prefactor * std::exp(-r2/(2*sigma*sigma));
}

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = std::atoi(argv[1]);
    bool gauss_fct = std::atoi(argv[2]);
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    // Specifies SERIAL, PARALLEL dims
    ippl::e_dim_tag decomp[dim];
    for (unsigned int d=0; d<dim; d++)
        decomp[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned,decomp);

    // type definitions
    typedef ippl::Vector<double, dim> Vector_t;
    typedef ippl::Field<double, dim> Field_t;
    typedef ippl::Vector<Vector_t, dim> Matrix_t;
    typedef ippl::Field<Matrix_t, dim> MField_t;

    // domain [0,1]^3
    double dx = 1.0 / double(pt);
    Vector_t hx = {dx, dx, dx};
    Vector_t origin = {0.0, 0.0, 0.0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    Field_t field(mesh, layout);
    MField_t result(mesh, layout);
    MField_t exact(mesh, layout);

    typename Field_t::view_type& view = field.getView();

    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
    const int nghost = field.getNghost();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    Kokkos::parallel_for("Assign field",
                mdrange_type({0, 0, 0},
                             {view.extent(0),
                              view.extent(1),
                              view.extent(2)}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {

                //local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;
            
                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                if (gauss_fct) {
                    view(i, j, k) = gaussian(x,y,z);
                } else {
                    view(i, j, k) = x*y*z;
                }
    });

    typename MField_t::view_type& view_exact = exact.getView();
    typename MField_t::view_type& view_result = result.getView();

    std::cout << "Exact" << std::endl;
    Kokkos::parallel_for("Assign exact", mdrange_type({nghost,nghost,nghost},
                             {view_exact.extent(0) - nghost,
                              view_exact.extent(1) - nghost,
                              view_exact.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {

                    //local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;
            
                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];
                    
                    if (gauss_fct) {
                        view_exact(i, j, k)[0] = {(x*x-1)*gaussian(x,y,z), 
                                                  x*y*gaussian(x,y,z),
                                                  x*z*gaussian(x,y,z)};
                        view_exact(i, j, k)[1] = {x*y*gaussian(x,y,z), 
                                                  (y*y-1)*gaussian(x,y,z),
                                                  y*z*gaussian(x,y,z)};
                        view_exact(i, j, k)[2] = {x*z*gaussian(x,y,z), 
                                                  y*z*gaussian(x,y,z),
                                                  (z*z-1)*gaussian(x,y,z)};
                    } else {
                        view_exact(i, j, k)[0] = {0.0, z, y};
                        view_exact(i, j, k)[1] = {z, 0.0, x};
                        view_exact(i, j, k)[2] = {y, x, 0.0};
                    }

                    std::cout << "(" << ig << "," << jg << "," << kg << ") = " << 
                              view_exact(i,j,k)[0] << ", " << view_exact(i,j,k)[1]
                              << ", " << view_exact(i,j,k)[2] << std::endl;
    });

    result = {0.0, 0.0, 0.0};
    result = hess(field);

    std::cout << "Result" << std::endl;
    Kokkos::parallel_for("Assign exact", mdrange_type({nghost,nghost,nghost},
                             {view_result.extent(0) - nghost,
                              view_result.extent(1) - nghost,
                              view_result.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {

                    //local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;
            
                    std::cout << "(" << ig << "," << jg << "," << kg << ") = " << 
                              view_result(i,j,k)[0] << ", " << view_result(i,j,k)[1]
                              << ", " << view_result(i,j,k)[2] << std::endl;
    });

    result = result - exact;

    std::cout << "Diff" << std::endl;
    Kokkos::parallel_for("Assign exact", mdrange_type({nghost,nghost,nghost},
                             {view_result.extent(0) - nghost,
                              view_result.extent(1) - nghost,
                              view_result.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {

                    //local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;
            
                    std::cout << "(" << ig << "," << jg << "," << kg << ") = " << 
                              view_result(i,j,k)[0] << ", " << view_result(i,j,k)[1]
                              << ", " << view_result(i,j,k)[2] << std::endl;
    });

    ippl::Vector<Vector_t, 3> err_hess {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    for (size_t dim1 = 0; dim1 < dim; ++dim1) {
        for (size_t dim2 = 0; dim2 < dim; ++dim2) {
            
            double valN(0.0);

            Kokkos::parallel_reduce("Relative error", mdrange_type({nghost, nghost, nghost},
                                {view_result.extent(0) - nghost,
                                 view_result.extent(1) - nghost,
                                 view_result.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_result(i,j,k)[dim1][dim2], 2);
                    val += myVal;
            }, Kokkos::Sum<double>(valN));

            double globalN(0.0);
            MPI_Allreduce(&valN, &globalN, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorN = std::sqrt(globalN);

            double valD(0.0);

            Kokkos::parallel_reduce("Relative error", mdrange_type({nghost, nghost, nghost},
                                {view_exact.extent(0) - nghost,
                                 view_exact.extent(1) - nghost,
                                 view_exact.extent(2) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_exact(i,j,k)[dim1][dim2], 2);
                    val += myVal;
            }, Kokkos::Sum<double>(valD));

            double globalD(0.0);
            MPI_Allreduce(&valD, &globalD, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorD = std::sqrt(globalD);

            if ((errorD == 0) && (errorN == 0)) {
                err_hess[dim1][dim2] = 0.0;
            } else { 
                err_hess[dim1][dim2] = errorN/errorD;
            }

            if (Ippl::Comm->rank() == 0) {
                std::cout << std::setprecision(16) << "Error (" << dim1+1 << "," << dim2+1 << "): "
                    << err_hess[dim1][dim2] << std::endl;
            }

        }
    }

    return 0;
}
