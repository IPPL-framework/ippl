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
//// Copyright (c) 2022,
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

template <unsigned Dim = 3>
using Matrix_t = Vector<Vector<double>>;

template <unsigned Dim = 3>
using MField_t = Field<Matrix_t<Dim>, Dim>;

constexpr unsigned int dim = 3;

using index_type   = typename Kokkos::RangePolicy<int>::index_type;
using idx_vec_t    = typename Kokkos::Array<index_type, dim>;
using range_pair_t = typename std::pair<idx_vec_t, idx_vec_t>;

template <typename T, unsigned Dim>
void pickHessianIdx(Field_t<Dim>& out_field, MField_t<Dim>& hessian_field, size_t row_idx,
                    size_t col_idx, size_t nghost) {
    typename Field_t<Dim>::view_type idx_view   = out_field.getView();
    typename MField_t<Dim>::view_type hess_view = hessian_field.getView();
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<Dim>> mdrange_type;

    Kokkos::parallel_for(
        "Pick Index from Hessian",
        mdrange_type({nghost, nghost, nghost},
                     {idx_view.extent(0) - nghost, idx_view.extent(1) - nghost,
                      idx_view.extent(2) - nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            idx_view(i, j, k) = hess_view(i, j, k)[row_idx][col_idx];
        });
}

template <typename T, unsigned Dim>
void dumpVTKScalar(Field_t<Dim>& f, int iteration, double dx, double dy, double dz,
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

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 1.0,
                                       double mu = 0.5) {
    double pi        = std::acos(-1.0);
    double prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}

std::vector<range_pair_t> genFaceIterators(typename Field_t<dim>::view_type& fieldView,
                                           int nghost) {
    int x_max = fieldView.extent(0);
    int y_max = fieldView.extent(1);
    int z_max = fieldView.extent(2);

    std::vector<range_pair_t> face_iterators(6);

    // x low
    face_iterators[0] = range_pair_t({0, 1, 1}, {1, y_max - 1, z_max - 1});
    // x high
    face_iterators[1] = range_pair_t({x_max - 1, 1, 1}, {x_max, y_max - 1, z_max - 1});
    // y low
    face_iterators[2] = range_pair_t({1, 0, 1}, {x_max - 1, 1, z_max - 1});
    // y high
    face_iterators[3] = range_pair_t({1, y_max - 1, 1}, {x_max - 1, y_max, z_max - 1});
    // z low
    face_iterators[4] = range_pair_t({1, 1, 0}, {x_max - 1, y_max - 1, 1});
    // z high
    face_iterators[5] = range_pair_t({1, 1, y_max - 1}, {x_max - 1, y_max - 1, z_max});

    return face_iterators;
}

std::vector<range_pair_t> genEdgeDim(Vector<int, dim> extents, unsigned dir) {
    std::vector<range_pair_t> face_iterators(4);

    Kokkos::Array<index_type, dim> lower_bound({0, 0, 0});
    Kokkos::Array<index_type, dim> upper_bound({0, 0, 0});

    lower_bound[dir]     = 1;
    upper_bound[dir]     = extents[dir] - 1;
    face_iterators[0]    = std::make_pair(lower_bound, upper_bound);
    index_type new_idx   = (dir + 1) % dim;
    lower_bound[new_idx] = extents[new_idx];
    upper_bound[new_idx] = lower_bound[new_idx];
    face_iterators[1]    = std::make_pair(lower_bound, upper_bound);
    new_idx              = (dir + 2) % dim;
    lower_bound[new_idx] = extents[new_idx];
    upper_bound[new_idx] = lower_bound[new_idx];
    face_iterators[2]    = std::make_pair(lower_bound, upper_bound);
    new_idx              = (dir + 1) % dim;
    lower_bound[new_idx] = 0;
    upper_bound[new_idx] = lower_bound[new_idx];
    face_iterators[3]    = std::make_pair(lower_bound, upper_bound);

    // Edges along x-dim:
    // y low, z low
    // face_iterators[0] = Kokkos::MDRangePolicy<Kokkos::Rank<dim>>({1, 0, 0}, {x_max - 1, 0, 0});
    // // y high, z low
    // face_iterators[1] =
    //     Kokkos::MDRangePolicy<Kokkos::Rank<dim>>({1, y_max, 0}, {x_max - 1, y_max, 0});
    // // y high, z high
    // face_iterators[2] =
    //     Kokkos::MDRangePolicy<Kokkos::Rank<dim>>({1, y_max, z_max}, {x_max - 1, y_max, z_max});
    // // y low, z high
    // face_iterators[3] =
    //     Kokkos::MDRangePolicy<Kokkos::Rank<dim>>({1, 0, z_max}, {x_max - 1, 0, z_max - 1});

    return face_iterators;
}

std::vector<range_pair_t> genEdgeIterators(typename Field_t<dim>::view_type& fieldView) {
    int x_max                = fieldView.extent(0);
    int y_max                = fieldView.extent(1);
    int z_max                = fieldView.extent(2);
    Vector<int, dim> extents = {x_max, y_max, z_max};

    std::vector<range_pair_t> face_iterators(12);

    for (unsigned d = 0; d < dim; d++) {
        auto dim_iterators = genEdgeDim(extents, d);
        face_iterators.insert(face_iterators.begin() + 4 * d, dim_iterators.begin(),
                              dim_iterators.end());
    }
    return face_iterators;
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    Inform msg("TestOnesidedHessian");
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    // Define often used types
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<dim>> mdrange_type;
    typedef typename Field_t<dim>::view_type FView_t;
    typedef typename MField_t<dim>::view_type MView_t;

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
    double dx            = 1.0 / double(pt);
    double dxInv         = double(pt);
    Vector_t<dim> hx     = {dx, dx, dx};
    Vector_t<dim> hxInv  = {dxInv, dxInv, dxInv};
    Vector_t<dim> origin = {0.0, 0.0, 0.0};

    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    const unsigned int nghost = 1;
    Field_t<dim> field(mesh, layout, nghost);
    Field_t<dim> hessReductionField(mesh, layout, nghost);
    MField_t<dim> result(mesh, layout, nghost);
    MField_t<dim> subResult(mesh, layout, nghost);
    MField_t<dim> exact(mesh, layout, nghost);

    FView_t& view = field.getView();

    const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

    auto faceIterators = genFaceIterators(view);
    auto edgeIterators = genEdgeIterators(view);
    auto& currIterator = edgeIterators[0];
    for (int i = currIterator.first[0]; i < currIterator.second[0]; ++i) {
        for (int j = currIterator.first[1]; j < currIterator.second[1]; ++j) {
            for (int k = currIterator.first[2]; k < currIterator.second[2]; ++k) {
                msg << view(i, j, k) << endl;
            }
        }
    }
    // Kokkos::parallel_for(
    //     "Test Boundary Iterators", edgeIterators[0],
    //     KOKKOS_LAMBDA(const int i, const int j, const int k) {
    //         std::cout << view(i, j, k) << std::endl;
    //     });
    // Kokkos::fence();

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

    // Test to create a subview of the Domain Boundaries
    //
    // Check if we have actually have non-periodic B.C.
    // One-sided Hessian is not well defined for periodic B.C.
    const auto& bConds = field.getFieldBC();
    if (layout.isAllPeriodic_m) {
        throw IpplException("Ippl::onesidedHess",
                            "`onesidedHess()` operator not applicable with periodic b.c.");
    }

    for (const auto& bc : bConds) {
        if (bc->getBCType() == ippl::FieldBC::PERIODIC_FACE) {
            throw IpplException("Ippl::onesidedHess",
                                "`onesidedHess()` operator not applicable with periodic b.c.");
        }
    }

    // Check if on physical boundary
    // TODO Check how we could check this now. Previously used `getFaceNeighbors()`
    const auto& faceNeighbors = layout.getNeighbors();

    // Assign initial values to subField
    result = {0.0, 0.0, 0.0};

    /////////////////////////////
    // Kokkos loop for Hessian //
    /////////////////////////////

    // Define Opereators
    // CenteredHessOp centered_hess(field);
    // OnesidedHessOp<std::plus<size_t>> forward_hess(field);
    // OnesidedHessOp<std::minus<size_t>> backward_hess(field);
    GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered, DiffType::Centered,
                      DiffType::Centered>
        centerHess(field, hxInv);

    // Check whether system boundaries are touched
    const size_t stencilWidth     = 5;
    const size_t halfStencilWidth = stencilWidth / 2;
    const size_t extents[dim]     = {view.extent(0), view.extent(1), view.extent(2)};
    std::vector<ippl::NDIndex<dim>> systemBoundaries;
    ippl::NDIndex<dim> centerDomain = ippl::NDIndex<dim>(ippl::Index(nghost, extents[0] - nghost),
                                                         ippl::Index(nghost, extents[1] - nghost),
                                                         ippl::Index(nghost, extents[2] - nghost));

    // Container containing operators for each face
    std::vector<std::shared_ptr<GeneralDiffOpInterface<dim, double, Matrix_t<dim>>>> faceDiffOps;
    faceDiffOps.reserve(14);

    // Allocate the operators manually [could be fully templated later on]
    // Also there it would not be necessary to create all operators if we run in parallel
    // Operator for faces
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Forward,
                                           DiffType::Centered, DiffType::Centered>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Backward,
                                           DiffType::Centered, DiffType::Centered>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered,
                                           DiffType::Forward, DiffType::Centered>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered,
                                           DiffType::Backward, DiffType::Centered>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered,
                                           DiffType::Centered, DiffType::Forward>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered,
                                           DiffType::Centered, DiffType::Backward>>(field, hxInv));

    // Operator for corners
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Forward,
                                           DiffType::Forward, DiffType::Forward>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Backward,
                                           DiffType::Forward, DiffType::Forward>>(field, hxInv));

    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Forward,
                                           DiffType::Backward, DiffType::Forward>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Backward,
                                           DiffType::Backward, DiffType::Forward>>(field, hxInv));

    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Forward,
                                           DiffType::Forward, DiffType::Backward>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Backward,
                                           DiffType::Forward, DiffType::Backward>>(field, hxInv));

    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Forward,
                                           DiffType::Backward, DiffType::Backward>>(field, hxInv));
    faceDiffOps.emplace_back(
        std::make_shared<GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Backward,
                                           DiffType::Backward, DiffType::Backward>>(field, hxInv));

    // Assign to each system boundary face a domain for which onesided differencing should be used
    size_t nSystemFaces = 0;
    for (size_t face = 0; face < 2 * dim; ++face) {
        size_t d = face / 2;

        // System boundary case
        if (faceNeighbors[face].size() == 0) {
            // Create Hessian Operators for face
            // hessOp::DiffType diffArr[3] = {hessOp::DiffType::Centered,
            // hessOp::DiffType::Centered, hessOp::DiffType::Centered}; if (face & 1) {
            //     diffArr[d] = hessOp::DiffType::Backward;
            // } else {  // Forward difference
            //     diffArr[d] = hessOp::DiffType::Forward;
            // }
            // faceDiffOps[face] = std::make_unique<hessOp::GeneralDiffOpInterface<double, Matrix_t>
            // >(field, hxInv);
            // faceDiffOps.push_back(std::make_unique<hessOp::GeneralizedHessOp<double,Matrix_t,hessOp::DiffType::Centered,
            // hessOp::DiffType::Centered, hessOp::DiffType::Centered> >(field, hxInv));

            // Create Index Range to apply the face operator to
            systemBoundaries.emplace_back(ippl::NDIndex<dim>(
                ippl::Index(nghost, extents[0] - nghost), ippl::Index(nghost, extents[1] - nghost),
                ippl::Index(nghost, extents[2] - nghost)));

            // Forward difference
            if (face % 2 == 0) {
                systemBoundaries[nSystemFaces][d] = ippl::Index(nghost, nghost + halfStencilWidth);
            } else {  // Backward difference
                systemBoundaries[nSystemFaces][d] =
                    ippl::Index(extents[d] - nghost - halfStencilWidth, extents[d] - nghost);
            }
            nSystemFaces++;

            size_t cLow  = centerDomain[d].first();
            size_t cHigh = centerDomain[d].last();
            if (face % 2 == 0) {
                centerDomain[d] = ippl::Index(cLow + halfStencilWidth, cHigh);
            } else {
                centerDomain[d] = ippl::Index(cLow, cHigh - halfStencilWidth);
            }
        }
    }

    msg << "IdxRanges : " << endl;
    for (const auto& idxRange : systemBoundaries) {
        msg << idxRange << endl;
    }

    msg << "centerDomain : " << centerDomain[0] << ", " << centerDomain[1] << ", "
        << centerDomain[2] << endl;

    Kokkos::parallel_for(
        "Onesided Hessian Loop [Center]",
        mdrange_type({centerDomain[0].first(), centerDomain[1].first(), centerDomain[2].first()},
                     {centerDomain[0].last(), centerDomain[1].last(), centerDomain[2].last()}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            view_result(i, j, k) = centerHess(i, j, k);
        });

    for (size_t i = 0; i < 2 * dim; ++i) {
        msg << "apply operator on systemBoundaries[" << i << "] : " << endl;
        Kokkos::parallel_for(
            "Onesided Hessian Loop [Faces]",
            mdrange_type({systemBoundaries[i][0].first(), systemBoundaries[i][1].first(),
                          systemBoundaries[i][2].first()},
                         {systemBoundaries[i][0].last(), systemBoundaries[i][1].last(),
                          systemBoundaries[i][2].last()}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                view_result(i, j, k) = faceDiffOps[i]->operator()(i, j, k);
            });
    }

    // Kokkos::parallel_for(
    //"Onesided Hessian Loop [Faces]",
    // mdrange_type(
    //{systemBoundaries[1][0].first(), systemBoundaries[1][1].first(),
    // systemBoundaries[1][2].first()}, {systemBoundaries[1][0].last(),
    // systemBoundaries[1][1].last(), systemBoundaries[1][2].last()}),
    ////{5, 5, 5},
    ////{10,10,10}),
    // KOKKOS_LAMBDA(const int i, const int j, const int k) {
    // printf("i,j,k = %d, %d, %d\n", i, j, k);
    // view_result(i, j, k) = faceDiffOps[1]->operator()(i,j,k);
    //});

    // pickHessianIdx<double,dim>(hessReductionField, result, 0, 1, 2);
    // dumpVTKScalar<double,dim>(hessReductionField, 0, dx, dx, dx);
    // dumpVTKScalar(field, 0, dx, dx, dx);

    result = result - exact;

    ippl::Vector<Vector_t<dim>, dim> err_hess{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    double avg = 0.0;

    for (size_t dim1 = 0; dim1 < dim; ++dim1) {
        for (size_t dim2 = 0; dim2 < dim; ++dim2) {
            double valN(0.0);

            Kokkos::parallel_reduce(
                "Relative error",
                mdrange_type(
                    {2 * nghost, 2 * nghost, 2 * nghost},
                    {view_result.extent(0) - 2 * nghost, view_result.extent(1) - 2 * nghost,
                     view_result.extent(2) - 2 * nghost}),
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
                mdrange_type({2 * nghost, 2 * nghost, 2 * nghost},
                             {view_exact.extent(0) - 2 * nghost, view_exact.extent(1) - 2 * nghost,
                              view_exact.extent(2) - 2 * nghost}),
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
