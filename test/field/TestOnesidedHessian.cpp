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
    Inform msg("TestOnesidedHessian");
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
    const auto& faceNeighbors = layout.getFaceNeighbors();

    // Assign initial values to subField
    result = {0.0, 0.0, 0.0};

    /////////////////////////////
    // Kokkos loop for Hessian //
    /////////////////////////////

    // Define Opereators
    // CenteredHessOp centered_hess(field);
    // OnesidedHessOp<std::plus<size_t>> forward_hess(field);
    // OnesidedHessOp<std::minus<size_t>> backward_hess(field);
    hessOp::GeneralizedHessOp<double, Matrix_t, hessOp::DiffType::Centered, hessOp::DiffType::Centered, hessOp::DiffType::Centered>
        centerHess(field, hxInv);

    // Check whether system boundaries are touched
    const size_t stencilWidth = 5;
    const size_t halfStencilWidth = stencilWidth / 2;
    const size_t extents[dim] = {view.extent(0), view.extent(1), view.extent(2)};
    std::vector<ippl::NDIndex<dim> > systemBoundaries;
    ippl::NDIndex<dim> centerDomain = ippl::NDIndex<dim>(ippl::Index(nghost, extents[0] - nghost),
                                                           ippl::Index(nghost, extents[1] - nghost),
                                                           ippl::Index(nghost, extents[2] - nghost));

    // Container containing operators for each face
    std::vector<std::unique_ptr<hessOp::GeneralDiffOpInterface<double, Matrix_t>> > faceDiffOps;
    faceDiffOps.reserve(14);

    // Allocate the operators manually [could be fully templated later on]
    // Also there it would not be necessary to create all operators if we run in parallel
    // Operator for faces
    { using namespace hessOp;
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Forward, DiffType::Centered, DiffType::Centered> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Backward, DiffType::Centered, DiffType::Centered> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Centered, DiffType::Forward, DiffType::Centered> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Centered, DiffType::Backward, DiffType::Centered> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Centered, DiffType::Centered, DiffType::Forward> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Centered, DiffType::Centered, DiffType::Backward> >(field, hxInv));

    // Operator for corners
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Forward, DiffType::Forward, DiffType::Forward> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Backward, DiffType::Forward, DiffType::Forward> >(field, hxInv));

    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Forward, DiffType::Backward, DiffType::Forward> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Backward, DiffType::Backward, DiffType::Forward> >(field, hxInv));

    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Forward, DiffType::Forward, DiffType::Backward> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Backward, DiffType::Forward, DiffType::Backward> >(field, hxInv));

    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Forward, DiffType::Backward, DiffType::Backward> >(field, hxInv));
    faceDiffOps.emplace_back(std::make_unique<GeneralizedHessOp<double,Matrix_t,DiffType::Backward, DiffType::Backward, DiffType::Backward> >(field, hxInv));
    } // namespace hessOp

    // Assign to each system boundary face a domain for which onesided differencing should be used
    size_t nSystemFaces = 0;
    for (size_t face = 0; face < 2 * dim; ++face) {
        size_t d = face / 2;

        // System boundary case
        if (faceNeighbors[face].size() == 0) {

            // Create Hessian Operators for face
            // hessOp::DiffType diffArr[3] = {hessOp::DiffType::Centered, hessOp::DiffType::Centered, hessOp::DiffType::Centered};
            // if (face & 1) {
            //     diffArr[d] = hessOp::DiffType::Backward;
            // } else {  // Forward difference
            //     diffArr[d] = hessOp::DiffType::Forward;
            // }
            //faceDiffOps[face] = std::make_unique<hessOp::GeneralDiffOpInterface<double, Matrix_t> >(field, hxInv);
            // faceDiffOps.push_back(std::make_unique<hessOp::GeneralizedHessOp<double,Matrix_t,hessOp::DiffType::Centered, hessOp::DiffType::Centered, hessOp::DiffType::Centered> >(field, hxInv));

            // Create Index Range to apply the face operator to
            systemBoundaries.emplace_back(ippl::NDIndex<dim>(
                ippl::Index(nghost, extents[0] - nghost),
                ippl::Index(nghost, extents[1] - nghost),
                ippl::Index(nghost, extents[2] - nghost)));

            // Backward difference
            if (face & 1) {
                systemBoundaries[nSystemFaces][d] = ippl::Index(extents[d] - nghost - halfStencilWidth,
                                                        extents[d] - nghost);
            } else {  // Forward difference
                systemBoundaries[nSystemFaces][d] = ippl::Index(nghost,
                                                        nghost + halfStencilWidth);
            }
            nSystemFaces++;

            size_t cLow = centerDomain[d].first();
            size_t cHigh = centerDomain[d].last();
            if (face % 2 == 0){
                centerDomain[d] = ippl::Index(cLow+halfStencilWidth, cHigh);
            } else {
                centerDomain[d] = ippl::Index(cLow, cHigh-halfStencilWidth);
            }
        }
        msg << "centerDomain : " << centerDomain[0] << ", " << centerDomain[1] << ", " << centerDomain[2] << endl;
    }

    msg << "IdxRanges : " << endl;
    for (const auto& idxRange : systemBoundaries) {
        msg << idxRange << endl;
    }

    msg << "centerDomain : " << centerDomain[0] << ", " << centerDomain[1] << ", " << centerDomain[2] << endl;


    Kokkos::parallel_for(
        "Onesided Hessian Loop [Center]",
        mdrange_type(
            {centerDomain[0].first(), centerDomain[1].first(), centerDomain[2].first()},
            {centerDomain[0].last(), centerDomain[1].last(), centerDomain[2].last()}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            view_result(i, j, k) = centerHess(i,j,k);
        });

    // Kokkos::parallel_for(
    //     "Onesided Hessian Loop [Faces]",
    //     mdrange_type(
    //         {systemBoundaries[0][0].first(), systemBoundaries[0][1].first(), systemBoundaries[0][2].first()},
    //         {systemBoundaries[0][0].last(), systemBoundaries[0][1].last(), systemBoundaries[0][2].last()}),
    //     KOKKOS_LAMBDA(const int i, const int j, const int k) {
    //         view_result(i, j, k) = faceDiffOps[0]->operator()(i,j,k);
    //     });

    Kokkos::parallel_for(
        "Onesided Hessian Loop [Faces]",
        mdrange_type(
            {5, 5, 5},
            {10,10,10}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            view_result(i, j, k) = faceDiffOps[0]->operator()(i,j,k);
        });

    msg << "Created " << faceDiffOps.size() << " DiffOps for faces." << endl;
    view_result(20,20,20) = faceDiffOps[0]->operator()(20,20,20);

    // pickHessianIdx<double,dim>(hessReductionField, result, 0, 1, 2);
    // dumpVTKScalar<double,dim>(hessReductionField, 0, dx, dx, dx);
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
