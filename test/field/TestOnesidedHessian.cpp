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

void getFaceIteratorRanges(std::vector<range_pair_t>& result, const idx_vec_t& extents,
                           const int& nghost) {
    int x_max = extents[0];
    int y_max = extents[1];
    int z_max = extents[2];

    // x low
    result.push_back(
        {{nghost, nghost + 1, nghost + 1}, {nghost + 1, y_max - nghost - 1, z_max - nghost - 1}});
    // x high
    result.push_back({{x_max - nghost - 1, nghost + 1, nghost + 1},
                      {x_max - nghost, y_max - nghost - 1, z_max - nghost - 1}});
    // y low
    result.push_back(
        {{nghost + 1, nghost, nghost + 1}, {x_max - nghost - 1, nghost + 1, z_max - nghost - 1}});
    // y high
    result.push_back({{nghost + 1, y_max - nghost - 1, nghost + 1},
                      {x_max - nghost - 1, y_max - nghost, z_max - nghost - 1}});
    // z low
    result.push_back(
        {{nghost + 1, nghost + 1, nghost}, {x_max - nghost - 1, y_max - nghost - 1, nghost + 1}});
    // z high
    result.push_back({{nghost, nghost, z_max - nghost - 1},
                      {x_max - nghost - 1, y_max - nghost - 1, z_max - nghost}});
}

std::vector<range_pair_t> getEdgeDim(const idx_vec_t& extents, const unsigned& dir,
                                     const int& nghost) {
    std::vector<range_pair_t> face_iterators(4);

    idx_vec_t lower_bound({nghost, nghost, nghost});
    idx_vec_t upper_bound({nghost + 1, nghost + 1, nghost + 1});

    // Set contant bounds in dimension given with `dir`
    lower_bound[dir] = nghost + 1;
    upper_bound[dir] = extents[dir] - nghost - 1;
    // Go through all combination of lower and upper bounds for the other two dimensions
    // 2^2 -> 4 combinations
    face_iterators[0]    = std::make_pair(lower_bound, upper_bound);
    index_type new_idx   = (dir + 1) % dim;
    lower_bound[new_idx] = extents[new_idx] - nghost;
    upper_bound[new_idx] = lower_bound[new_idx] + 1;
    face_iterators[1]    = std::make_pair(lower_bound, upper_bound);
    new_idx              = (dir + 2) % dim;
    lower_bound[new_idx] = extents[new_idx] - nghost;
    upper_bound[new_idx] = lower_bound[new_idx] + 1;
    face_iterators[2]    = std::make_pair(lower_bound, upper_bound);
    new_idx              = (dir + 1) % dim;
    lower_bound[new_idx] = nghost;
    upper_bound[new_idx] = lower_bound[new_idx] + 1;
    face_iterators[3]    = std::make_pair(lower_bound, upper_bound);

    return face_iterators;
}

void getEdgeIteratorRanges(std::vector<range_pair_t>& result, const idx_vec_t& extents,
                           int nghost) {
    for (unsigned d = 0; d < dim; d++) {
        auto dim_iterators = getEdgeDim(extents, d, nghost);
        result.insert(result.end() + 4 * d, dim_iterators.begin(), dim_iterators.end());
    }
}

void getCornerIteratorRanges(std::vector<range_pair_t>& result, const idx_vec_t& extents,
                             const int& nghost) {
    int x_max = extents[0];
    int y_max = extents[1];
    int z_max = extents[2];

    result.push_back({{nghost, nghost, nghost}, {nghost + 1, nghost + 1, nghost + 1}});
    result.push_back(
        {{x_max - nghost - 1, nghost, nghost}, {x_max - nghost, nghost + 1, nghost + 1}});

    result.push_back(
        {{nghost, y_max - nghost - 1, nghost}, {nghost + 1, y_max - nghost, nghost + 1}});
    result.push_back({{x_max - nghost - 1, y_max - nghost - 1, nghost},
                      {x_max - nghost, x_max - nghost, nghost + 1}});

    result.push_back(
        {{nghost, nghost, z_max - nghost - 1}, {nghost + 1, nghost + 1, z_max - nghost}});
    result.push_back({{x_max - nghost - 1, nghost, z_max - nghost - 1},
                      {x_max - nghost, nghost + 1, z_max - nghost}});
    result.push_back({{nghost, y_max - nghost - 1, z_max - nghost - 1},
                      {nghost + 1, y_max - nghost, z_max - nghost}});
    result.push_back({{x_max - nghost - 1, y_max - nghost - 1, z_max - nghost - 1},
                      {x_max - nghost, y_max - nghost, z_max - nghost}});
}

void getCenterIteratorRange(std::vector<range_pair_t>& result, const idx_vec_t& extents,
                            const int& nghost) {
    int x_max = extents[0];
    int y_max = extents[1];
    int z_max = extents[2];

    result.push_back({{nghost + 1, nghost + 1, nghost + 1},
                      {x_max - nghost - 2, y_max - nghost - 1, z_max - nghost - 1}});
}

int getTernaryEncoding(const range_pair_t& range, const idx_vec_t& extents, const int& nghost) {
    int opCode = 0;
    for (int d = 0; d < 3; ++d) {
        if (range.first[d] - nghost == 0) {  // Forward
            opCode += std::pow(3, d) * 1;
        } else if (range.second[d] + nghost == extents[d]) {  // Backward
            opCode += std::pow(3, d) * 2;
        } else {  // Centered
            opCode += std::pow(3, d) * 0;
        }
    }
    return opCode;
}

std::array<DiffType, 3> getOperatorTypes(int opEncoding) {
    std::array<DiffType, 3> operators;
    operators[0] = static_cast<DiffType>(opEncoding / 9);
    opEncoding -= opEncoding % 9;
    operators[1] = static_cast<DiffType>(opEncoding / 3);
    opEncoding -= opEncoding % 3;
    operators[2] = static_cast<DiffType>(opEncoding);
    return operators;
}

// Helper template to generate array of operator class instantiations
template <DiffType DiffX, DiffType DiffY, DiffType DiffZ, int Index>
struct OperatorGenerator {
    static constexpr int TotalCombinations = 27;

    static void generate(GeneralDiffOpInterface<dim, double, Matrix_t<dim>>* arr,
                         const Field_t<dim>& field, Vector_t<dim> hInvVector) {
        arr[Index] =
            GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffX, DiffY, DiffZ>(field, hInvVector);
        OperatorGenerator<(DiffType)((Index + 1) / 9), (DiffType)(((Index + 1) / 3) % 3),
                          (DiffType)((Index + 1) % 3), Index + 1>::generate(arr, field, hInvVector);
    }
};

// Base case to stop recursion
template <DiffType DiffX, DiffType DiffY, DiffType DiffZ>
struct OperatorGenerator<DiffX, DiffY, DiffZ,
                         OperatorGenerator<DiffX, DiffY, DiffZ, 27>::TotalCombinations> {
    static void generate(GeneralDiffOpInterface<dim, double, Matrix_t<dim>>* arr,
                         const Field_t<dim>& field, Vector_t<dim> hInvVector) {
        // Do nothing
    }
};

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

    index_type x_max             = view.extent(0);
    index_type y_max             = view.extent(1);
    index_type z_max             = view.extent(2);
    const idx_vec_t test_extents = {x_max, y_max, z_max};
    std::vector<range_pair_t> boundaryRanges;
    boundaryRanges.reserve(27);
    getFaceIteratorRanges(boundaryRanges, test_extents, nghost);
    getEdgeIteratorRanges(boundaryRanges, test_extents, nghost);
    getCornerIteratorRanges(boundaryRanges, test_extents, nghost);
    auto& currRange = boundaryRanges[1];
    for (int i = currRange.first[0]; i < currRange.second[0]; ++i) {
        for (int j = currRange.first[1]; j < currRange.second[1]; ++j) {
            for (int k = currRange.first[2]; k < currRange.second[2]; ++k) {
                msg << view(i, j, k) << endl;
            }
        }
    }

    GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered, DiffType::Centered,
                      DiffType::Centered>
        operators[27];
    OperatorGenerator<DiffType::Centered, DiffType::Centered, DiffType::Centered, 0>::generate(
        operators, field, hxInv);

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
    GeneralizedHessOp<dim, double, Matrix_t<dim>, DiffType::Centered, DiffType::Centered,
                      DiffType::Centered>
        centerHess(field, hxInv);

    Kokkos::parallel_for(
        "Assign Hessian",
        mdrange_type({0, 0, 0},
                     {view_result.extent(0), view_result.extent(1), view_result.extent(2)}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            view_result(i, j, k) = centerHess(i, j, k);
        });

    // pickHessianIdx<double,dim>(hessReductionField, result, 0, 1, 2);

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
