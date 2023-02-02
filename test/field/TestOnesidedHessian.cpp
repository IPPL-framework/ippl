//
////
//// TestOnesidedHessian.cpp
////   This program tests the onesided Hessian operator on a vector field.
////   The problem size can be given by the user (N^3), and a bool (0 or 1)
////   indicates whether the field is f=xyz or a Gaussian field.
////
//// Usage:
////   srun ./TestOnesidedHessian N 0 --info 10
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
    Inform msg(argv[0], "TestOnesidedHessian");
    Inform msg2all(argv[0], INFORM_ALL_NODES);

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

    Field_t field(mesh, layout, 1);
    MField_t result(mesh, layout, 1);
    MField_t exact(mesh, layout, 1);

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

                    double mu = 0.5;
                    
                    if (gauss_fct) {
                        view_exact(i, j, k)[0] = {((x-mu)*(x-mu)-1.0)*gaussian(x,y,z), 
                                                  (x-mu)*(y-mu)*gaussian(x,y,z), 
                                                  (x-mu)*(z-mu)*gaussian(x,y,z)};
                        view_exact(i, j, k)[1] = {(x-mu)*(y-mu)*gaussian(x,y,z), 
                                                  ((y-mu)*(y-mu)-1.0)*gaussian(x,y,z), 
                                                  (y-mu)*(z-mu)*gaussian(x,y,z)};
                        view_exact(i, j, k)[2] = {(x-mu)*(z-mu)*gaussian(x,y,z), 
                                                  (y-mu)*(z-mu)*gaussian(x,y,z), 
                                                  ((z-mu)*(z-mu)-1.0)*gaussian(x,y,z)};
                    } else {
                        view_exact(i, j, k)[0] = {0.0, z, y};
                        view_exact(i, j, k)[1] = {z, 0.0, x};
                        view_exact(i, j, k)[2] = {y, x, 0.0};
                    }
    });

    // Test to create a subview of the Domain Boundaries
    //
    // Check if we have actually have non-periodic B.C.
    // One-sided Hessian is not well defined for periodic B.C.
    const auto &bConds = field.getFieldBC();
    if (layout.isAllPeriodic_m) {
        throw IpplException("Ippl::onesidedHess", 
                "`onesidedHess()` operator not applicable with periodic b.c.");
    }


    for(const auto &bc : bConds){
        if (bc->getBCType() == ippl::FieldBC::PERIODIC_FACE){
            throw IpplException("Ippl::onesidedHess", 
                    "`onesidedHess()` operator not applicable with periodic b.c.");
        }
        
        
        msg << "Face: " << bc->getFace() << endl;
        typename Field_t::view_type diffSlice;
    }

    // Check if on physical boundary
    const auto &domain = layout.getDomain();
    const auto &lDomains = layout.getHostLocalDomains();
    int myRank = Ippl::Comm->rank();
    const auto &faceNeighbors = layout.getFaceNeighbors();

    for (unsigned int d = 0; d < 2*dim; ++d){
        msg << "faceneighbors[" << d << "].size() = " << faceNeighbors[d].size() << endl;
        bool isBoundary = (lDomains[myRank][d].max() == domain[d].max()) ||
            (lDomains[myRank][d].min() == domain[d].min());
        msg << "Rank " << Ippl::Comm->rank() << ": " << "dim" << d << " isBoundary = " << isBoundary << endl;
    } 

    result = {0.0, 0.0, 0.0};
    msg << "Calling `onesidedHess()`" << endl;
    //msg << "(" << view.extent(0) << "," << view.extent(0) << "," << view.extent(0) << ")" << endl;
    

    result = backwardHess(field);

    result = result - exact;

    ippl::Vector<Vector_t, 3> err_hess {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    double avg = 0.0;

    for (size_t dim1 = 0; dim1 < dim; ++dim1) {
        for (size_t dim2 = 0; dim2 < dim; ++dim2) {
            
            double valN(0.0);

            Kokkos::parallel_reduce("Relative error", mdrange_type({3*nghost, 3*nghost, 3*nghost},
                                {view_result.extent(0) - 3*nghost,
                                 view_result.extent(1) - 3*nghost,
                                 view_result.extent(2) - 3*nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_result(i,j,k)[dim1][dim2], 2);
                    val += myVal;
            }, Kokkos::Sum<double>(valN));

            double globalN(0.0);
            MPI_Allreduce(&valN, &globalN, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorN = std::sqrt(globalN);

            double valD(0.0);

            Kokkos::parallel_reduce("Relative error", mdrange_type({3*nghost, 3*nghost, 3*nghost},
                                {view_exact.extent(0) - 3*nghost,
                                 view_exact.extent(1) - 3*nghost,
                                 view_exact.extent(2) - 3*nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_exact(i,j,k)[dim1][dim2], 2);
                    val += myVal;
            }, Kokkos::Sum<double>(valD));

            double globalD(0.0);
            MPI_Allreduce(&valD, &globalD, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorD = std::sqrt(globalD);

            if ((errorD < 1e-15) && (errorN < 1e-15)) {
                err_hess[dim1][dim2] = 0.0;
            } else { 
                err_hess[dim1][dim2] = errorN/errorD;
            }

            err_hess[dim1][dim2] = errorN;

            msg << std::setprecision(16) << "Error (" << dim1+1 << "," << dim2+1 << "): "
                << err_hess[dim1][dim2] << endl;

            avg += err_hess[dim1][dim2];
        }
    }

    // print total error (average of each matrix entry)
    avg /= 9.0;
    msg << std::setprecision(16) << "Average error = " << avg << endl;

    return 0;
}
