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

#include "Hessian.h"

template <typename T, unsigned Dim>
void pickHessianIdx(
    ippl::Field<T, Dim>& out_field, ippl::Field<ippl::Vector<ippl::Vector<T, Dim>, Dim>, Dim>& hessian_field,
    size_t row_idx, size_t col_idx, size_t nghost) {
    typedef ippl::Vector<ippl::Vector<T, Dim>, Dim> Matrix_t;
    typename ippl::Field<T, Dim>::view_type idx_view         = out_field.getView();
    typename ippl::Field<Matrix_t, Dim>::view_type hess_view = hessian_field.getView();
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<Dim>> mdrange_type;

    Kokkos::parallel_for(
        "Pick Index from Hessian",
        mdrange_type(
            {nghost, nghost, nghost}, {idx_view.extent(0) - nghost, idx_view.extent(1) - nghost,
                                       idx_view.extent(2) - nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            idx_view(i, j, k) = hess_view(i, j, k)[row_idx][col_idx];
        });
}

template <typename T, unsigned Dim>
void dumpVTKScalar(
    ippl::Field<T, Dim>& f, int iteration, double dx, double dy, double dz,
    std::string label = "gaussian") {
    ippl::NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
    int nx                = lDom[0].length();
    int ny                = lDom[1].length();
    int nz                = lDom[2].length();

    std::string filename;
    filename = "data/";
    filename += label;
    filename += "_nod_";
    filename += std::to_string(Ippl::Comm->rank());
    filename += "_it_";
    filename += std::to_string(iteration);
    filename += ".vtk";

    Inform vtkout(NULL, filename.c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "toyfdtd" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
    vtkout << "ORIGIN " << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "POINT_DATA " << nx * ny * nz << endl;
    vtkout << "SCALARS Scalar_Value float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = lDom[2].first(); z <= lDom[2].last(); z++) {
        for (int y = lDom[1].first(); y <= lDom[1].last(); y++) {
            for (int x = lDom[0].first(); x <= lDom[0].last(); x++) {
                vtkout << f(x, y, z) << endl;
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
double gaussian(double x, double y, double z, double sigma = 1.0, double mu = 0.5) {
    double pi        = std::acos(-1.0);
    double prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    Inform msg(argv[0], "TestOnesidedHessian");
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    constexpr unsigned int dim = 3;

    // Define often used types
    typedef ippl::Vector<double, dim> Vector_t;
    typedef ippl::Field<double, dim> Field_t;
    typedef ippl::Vector<Vector_t, dim> Matrix_t;
    typedef ippl::Field<Matrix_t, dim> MField_t;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<dim>> mdrange_type;
    typedef Field_t::view_type FView_t;
    typedef MField_t::view_type MView_t;

    int pt         = std::atoi(argv[1]);
    bool gauss_fct = std::atoi(argv[2]);
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    // Specifies SERIAL, PARALLEL dims
    ippl::e_dim_tag decomp[dim];
    for (unsigned int d = 0; d < dim; d++)
        decomp[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<dim> layout(owned, decomp);

    // domain [0,1]^3
    double dx       = 1.0 / double(pt);
    double dxInv    = double(pt);
    Vector_t hx     = {dx, dx, dx};
    Vector_t hxInv  = {dxInv, dxInv, dxInv};
    Vector_t origin = {0.0, 0.0, 0.0};

    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    const unsigned int nghost = 2;
    Field_t field(mesh, layout, nghost);
    Field_t hessReductionField(mesh, layout, nghost);
    MField_t result(mesh, layout, nghost);
    MField_t subResult(mesh, layout, nghost);
    MField_t exact(mesh, layout, nghost);

    FView_t& view = field.getView();

    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

    Kokkos::parallel_for(
        "Assign field", mdrange_type({0, 0, 0}, {view.extent(0), view.extent(1), view.extent(2)}),
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

    MView_t& view_exact  = exact.getView();
    MView_t& view_result = result.getView();

    Kokkos::parallel_for(
        "Assign exact",
        mdrange_type(
            {nghost, nghost, nghost}, {view_exact.extent(0) - nghost, view_exact.extent(1) - nghost,
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
                view_exact(i, j, k)[0] = {
                    ((x - mu) * (x - mu) - 1.0) * gaussian(x, y, z),
                    (x - mu) * (y - mu) * gaussian(x, y, z),
                    (x - mu) * (z - mu) * gaussian(x, y, z)};
                view_exact(i, j, k)[1] = {
                    (x - mu) * (y - mu) * gaussian(x, y, z),
                    ((y - mu) * (y - mu) - 1.0) * gaussian(x, y, z),
                    (y - mu) * (z - mu) * gaussian(x, y, z)};
                view_exact(i, j, k)[2] = {
                    (x - mu) * (z - mu) * gaussian(x, y, z),
                    (y - mu) * (z - mu) * gaussian(x, y, z),
                    ((z - mu) * (z - mu) - 1.0) * gaussian(x, y, z)};
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
    const auto& bConds = field.getFieldBC();
    if (layout.isAllPeriodic_m) {
        throw IpplException(
            "Ippl::onesidedHess", "`onesidedHess()` operator not applicable with periodic b.c.");
    }

    for (const auto& bc : bConds) {
        if (bc->getBCType() == ippl::FieldBC::PERIODIC_FACE) {
            throw IpplException(
                "Ippl::onesidedHess",
                "`onesidedHess()` operator not applicable with periodic b.c.");
        }
    }

    // Check if on physical boundary
    // const auto& domain        = layout.getDomain();
    // const auto& lDomains      = layout.getHostLocalDomains();
    int myRank                = Ippl::Comm->rank();
    const auto& faceNeighbors = layout.getFaceNeighbors();

    // Assign initial values to subField
    result = {0.0, 0.0, 0.0};
    // subResult = {0.0, 0.0, 0.0};

    ////////////////////////////////////////
    // Define subfield and its properties //
    ////////////////////////////////////////

    // Define properties of subField
    // int subPt     = 20;
    // int subNghost = 1;
    // ippl::Index subI(subPt);
    // ippl::NDIndex<dim> subOwned(subI, I, I);

    //// Create subLayout of desired size
    // ippl::UniformCartesian<double, dim> subMesh(subOwned, hx, origin);
    // ippl::FieldLayout<dim> subLayout(subOwned, decomp);

    // Field_t subField = field.subField(
    // subMesh, subLayout, subNghost, Kokkos::make_pair(0, 30), Kokkos::ALL, Kokkos::ALL);
    // FView_t subView = subField.getView();

    // msg2all << "(" << subView.extent(0) << "," << subView.extent(1) << "," << subView.extent(2)
    //<< ")" << endl;

    // Test to write slice of field
    // TODO There is an error in running this in parallel!
    // dumpVTKScalar(field, 0, dx, dx, dx);
    // Kokkos::fence();

    msg2all << "Extents of view: (" << view.extent(0) << "," << view.extent(1) << ","
            << view.extent(2) << ")" << endl;

    // Test backwardHess on subField only
    // result = ippl::hess(field);
    // subResult = ippl::hess(subField);
    // pickHessianIdx(hessReductionField, result, 0, 0, 3);
    // dumpVTKScalar(hessReductionField, 0, dx, dx, dx);

    /////////////////////////////
    // Kokkos loop for Hessian //
    /////////////////////////////

    // Define Opereators
    // CenteredHessOp centered_hess(field);
    // OnesidedHessOp<std::plus<size_t>> forward_hess(field);
    // OnesidedHessOp<std::minus<size_t>> backward_hess(field);
    hessOp::GeneralizedHessOp<double, hessOp::DiffType::Centered, hessOp::DiffType::Centered, hessOp::DiffType::Centered>
        centerHess(field, hxInv);

    // Check whether system boundaries are touched
    const size_t stencilWidth = 3;
    // const size_t nghostExt = nghost + stencilWidth;
    const size_t extents[dim] = {view.extent(0), view.extent(1), view.extent(2)};
    ippl::NDIndex<dim> isSystemBoundary[2 * dim];
    // ippl::NDIndex<dim> isCenterDomain = ippl::NDIndex<dim>(ippl::Index(nghostExt,
    // extents[0] - nghostExt),
    //                                                                  ippl::Index(nghostExt,
    //                                                                  extents[1] - nghostExt),
    //                                                                  ippl::Index(nghostExt,
    //                                                                  extents[2] - nghostExt));

    // Assign to each system boundary face a domain for which onesided differencing should be used
    for (size_t face = 0; face < 2 * dim; ++face) {
        if (faceNeighbors[face].size() == 0) {
            isSystemBoundary[face] = ippl::NDIndex<dim>(
                ippl::Index(nghost, extents[0] - nghost), ippl::Index(nghost, extents[1] - nghost),
                ippl::Index(nghost, extents[2] - nghost));
            size_t d = face / 2;

            // Backward difference
            if (face & 1) {
                isSystemBoundary[face][d] =
                    ippl::Index(extents[d] - nghost - stencilWidth, extents[d] - nghost);
            } else {  // Forward difference
                isSystemBoundary[face][d] = ippl::Index(nghost, nghost + stencilWidth);
            }
        }
    }

    if (myRank == 0) {
        for (const auto& idxRange : isSystemBoundary) {
            std::cout << idxRange << std::endl;
        }
    }
    // ippl::NDIndex testIndex = ippl::NDIndex<dim>(ippl::Index(4, 4+1),
    //                                                          ippl::Index(4, 4+1),
    //                                                          ippl::Index(1, 1+1));
    // std::cout << "testIndex is center: " << isCenterDomain.contains(testIndex) << std::endl;

    // for(size_t i = 1*nghost; i < view.extent(0) - 1*nghost; ++i){
    // for(size_t j = 1*nghost; j < view.extent(1) - 1*nghost; ++j){
    // for(size_t k = 1*nghost; k < view.extent(2) - 1*nghost; ++k){
    // ippl::NDIndex currNDIndex = ippl::NDIndex<dim>(ippl::Index(i, i+1),
    // ippl::Index(j, j+1),
    // ippl::Index(k, k+1));
    ////std::cout << "(i,j,k) = " << "(" << i << "," << j << "," << k << ")" << std::endl;
    // if (isSystemBoundary[0].contains(currNDIndex)){
    // std::cout << currNDIndex << " || " << std::endl;
    //}
    // printf("(%lu, %lu, %lu)\n", i, j, k);

    //// Check all faces
    // view_result(i, j, k) =
    // isCenterDomain.contains(currNDIndex) * centered_hess(i,j,k);
    // isSystemBoundary[0].contains(currNDIndex) * forward_hess(i,j,k) +
    // isSystemBoundary[1].contains(currNDIndex) * backward_hess(i,j,k) +
    // isSystemBoundary[2].contains(currNDIndex) * forward_hess(i,j,k) +
    // isSystemBoundary[3].contains(currNDIndex) * backward_hess(i,j,k) +
    // isSystemBoundary[4].contains(currNDIndex) * forward_hess(i,j,k) +
    // isSystemBoundary[5].contains(currNDIndex) * backward_hess(i,j,k);

    //}
    //}
    //}

    Kokkos::parallel_for(
        "Onesided Hessian Loop",
        mdrange_type(
            {2 * nghost, 2 * nghost, 2 * nghost},
            {view.extent(0) - 2 * nghost, view.extent(1) - 2 * nghost,
             view.extent(2) - 2 * nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // Check which type of differencing is needed
            // ippl::NDIndex currNDIndex = ippl::NDIndex<dim>(ippl::Index(i, i+1),
            //                                                          ippl::Index(j, j+1),
            //                                                          ippl::Index(k, k+1));

            // std::cout << "(i,j,k) = " << "(" << i << "," << j << "," << k << ")" << std::endl;

            // if (myRank == 0){
            //     printf("(%d, %d, %d)\n", i, j, k);
            // }
            // if (isSystemBoundary[0].contains(currNDIndex)){
            //     std::cout << currNDIndex << " || " << std::endl;
            // }

            view_result(i, j, k) = centerHess(i,j,k);

            // Check all faces
            // view_result(i, j, k) =
            //                  isCenterDomain.contains(currNDIndex) * centered_hess(i,j,k) +
            //                 isSystemBoundary[0].contains(currNDIndex) * forward_hess(i,j,k) +
            //                 isSystemBoundary[1].contains(currNDIndex) * backward_hess(i,j,k) +
            //                 isSystemBoundary[2].contains(currNDIndex) * forward_hess(i,j,k) +
            //                 isSystemBoundary[3].contains(currNDIndex) * backward_hess(i,j,k) +
            //                 isSystemBoundary[4].contains(currNDIndex) * forward_hess(i,j,k) +
            //                 isSystemBoundary[5].contains(currNDIndex) * backward_hess(i,j,k);
        });

    pickHessianIdx<double,dim>(hessReductionField, result, 0, 1, 2);
    dumpVTKScalar<double,dim>(hessReductionField, 0, dx, dx, dx);
    // dumpVTKScalar(field, 0, dx, dx, dx);

    result = result - exact;

    ippl::Vector<Vector_t, dim> err_hess{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    double avg = 0.0;

    for (size_t dim1 = 0; dim1 < dim; ++dim1) {
        for (size_t dim2 = 0; dim2 < dim; ++dim2) {
            double valN(0.0);

            Kokkos::parallel_reduce(
                "Relative error",
                mdrange_type(
                    {2* nghost, 2* nghost, 2* nghost},
                    {view_result.extent(0) - 2* nghost, view_result.extent(1) - 2* nghost,
                     view_result.extent(2) - 2* nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_result(i, j, k)[dim1][dim2], 2);
                    val += myVal;
                },
                Kokkos::Sum<double>(valN));

            double globalN(0.0);
            MPI_Allreduce(&valN, &globalN, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorN = std::sqrt(globalN);

            double valD(0.0);

            Kokkos::parallel_reduce(
                "Relative error",
                mdrange_type(
                    {2* nghost, 2* nghost, 2* nghost},
                    {view_exact.extent(0) - 2* nghost, view_exact.extent(1) - 2* nghost,
                     view_exact.extent(2) - 2* nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& val) {
                    double myVal = pow(view_exact(i, j, k)[dim1][dim2], 2);
                    val += myVal;
                },
                Kokkos::Sum<double>(valD));

            double globalD(0.0);
            MPI_Allreduce(&valD, &globalD, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
            double errorD = std::sqrt(globalD);

            if ((errorD < 1e-15) && (errorN < 1e-15)) {
                err_hess[dim1][dim2] = 0.0;
            } else {
                err_hess[dim1][dim2] = errorN / errorD;
            }

            err_hess[dim1][dim2] = errorN;

            msg << std::setprecision(16) << "Error (" << dim1 + 1 << "," << dim2 + 1
                << "): " << err_hess[dim1][dim2] << endl;

            avg += err_hess[dim1][dim2];
        }
    }

    // print total error (average of each matrix entry)
    avg /= 9.0;
    msg << std::setprecision(16) << "Average error = " << avg << endl;

    return 0;
}
