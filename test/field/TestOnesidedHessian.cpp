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

#include <array>
#include <iostream>
#include <typeinfo>

namespace hess_test {

    constexpr unsigned int dim = 3;

    // type definitions
    typedef ippl::UniformCartesian<double, 3> Mesh_t;
    typedef ippl::Vector<double, dim> Vector_t;
    typedef ippl::Field<double, dim> Field_t;
    typedef ippl::Vector<Vector_t, dim> Matrix_t;
    typedef ippl::Field<Matrix_t, dim> MField_t;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<dim>> mdrange_type;
    typedef Field_t::view_type fview_type;
    typedef MField_t::view_type mview_type;

    void pickHessianIdx(
        Field_t& out_field, MField_t& hessian_field, size_t row_idx, size_t col_idx,
        size_t nghost) {
        mview_type hess_view = hessian_field.getView();
        fview_type idx_view   = out_field.getView();

        Kokkos::parallel_for(
            "Pick Index from Hessian",
            mdrange_type(
                {nghost, nghost, nghost}, {idx_view.extent(0) - nghost, idx_view.extent(1) - nghost,
                                           idx_view.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                idx_view(i, j, k) = hess_view(i, j, k)[row_idx][col_idx];
            });
    }

	void dumpVTKScalar( Field_t & f, int iteration, double dx, double dy, double dz, std::string label="gaussian") {
			ippl::NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
			int nx =lDom[0].length() ; int ny = lDom[1].length(); int nz=lDom[2].length() ;
            std::cout << nx << " " << ny << " " << nz << std::endl;
            std::cout << lDom[0].last() << std::endl;

            std::string filename;
            filename = "data/";
            filename+= label;
            filename += "_nod_";
            filename+= std::to_string(Ippl::Comm->rank());
            filename+= "_it_";
            filename+= std::to_string(iteration);
            filename+=".vtk";

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
			vtkout << "POINT_DATA " << nx*ny*nz << endl;
			vtkout << "SCALARS Scalar_Value float" << endl;
			vtkout << "LOOKUP_TABLE default" << endl;
			for (int z=lDom[2].first(); z<=lDom[2].last(); z++) {
					for (int y=lDom[1].first(); y<=lDom[1].last(); y++) {
							for (int x=lDom[0].first(); x<=lDom[0].last(); x++) {
									vtkout << f(x,y,z) << endl;
							}
					}
			}
	}

    KOKKOS_INLINE_FUNCTION
    double gaussian(double x, double y, double z, double sigma = 1.0, double mu = 0.5) {
        double pi = std::acos(-1.0);
        double prefactor =
            (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
        double r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

        return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
    }

    struct CenteredHessOp {
        CenteredHessOp(Field_t &field) : f_m(field.getView()){
            Mesh_t& mesh = field.get_mesh();
            hvector_m = mesh.getMeshSpacing();
        }

        KOKKOS_INLINE_FUNCTION
        Mesh_t::matrix_type operator()(size_t i, size_t j, size_t k) const {
            typename Mesh_t::vector_type row_1, row_2, row_3;

            // clang-format off
            row_1 = xvector_m * ((f_m(i+1,j,k) - 2.0*f_m(i,j,k) + f_m(i-1,j,k))/(hvector_m[0]*hvector_m[0])) +
                    yvector_m * ((f_m(i+1,j+1,k) - f_m(i-1,j+1,k) - f_m(i+1,j-1,k) + f_m(i-1,j-1,k))/(4.0*hvector_m[0]*hvector_m[1])) +
                    zvector_m * ((f_m(i+1,j,k+1) - f_m(i-1,j,k+1) - f_m(i+1,j,k-1) + f_m(i-1,j,k-1))/(4.0*hvector_m[0]*hvector_m[2]));

            row_2 = xvector_m * ((f_m(i+1,j+1,k) - f_m(i+1,j-1,k) - f_m(i-1,j+1,k) + f_m(i-1,j-1,k))/(4.0*hvector_m[1]*hvector_m[0])) +
                    yvector_m * ((f_m(i,j+1,k) - 2.0*f_m(i,j,k) + f_m(i,j-1,k))/(hvector_m[1]*hvector_m[1])) +
                    zvector_m * ((f_m(i,j+1,k+1) - f_m(i,j-1,k+1) - f_m(i,j+1,k-1) + f_m(i,j-1,k-1))/(4.0*hvector_m[1]*hvector_m[2]));

            row_3 = xvector_m * ((f_m(i+1,j,k+1) - f_m(i+1,j,k-1) - f_m(i-1,j,k+1) + f_m(i-1,j,k-1))/(4.0*hvector_m[2]*hvector_m[0])) +
                    yvector_m * ((f_m(i,j+1,k+1) - f_m(i,j+1,k-1) - f_m(i,j-1,k+1) + f_m(i,j-1,k-1))/(4.0*hvector_m[2]*hvector_m[1])) +
                    zvector_m * ((f_m(i,j,k+1) - 2.0*f_m(i,j,k) + f_m(i,j,k-1))/(hvector_m[2]*hvector_m[2]));
            // clang-format on

            typename Mesh_t::matrix_type hessian = {row_1, row_2, row_3};
            return hessian; 
        }

        fview_type &f_m;
        typename Mesh_t::vector_type hvector_m;
        const typename Mesh_t::vector_type xvector_m = {1.0, 0.0, 0.0};
        const typename Mesh_t::vector_type yvector_m = {0.0, 1.0, 0.0};
        const typename Mesh_t::vector_type zvector_m = {0.0, 0.0, 1.0};
    };

    template <typename IdxOp>
    struct OnesidedHessOp {
        OnesidedHessOp(Field_t &field) : f_m(field.getView()){
            Mesh_t& mesh = field.get_mesh();
            hvector_m = mesh.getMeshSpacing();
        }

        KOKKOS_INLINE_FUNCTION
        Mesh_t::matrix_type operator()(size_t i, size_t j, size_t k) const {
            typename Mesh_t::vector_type row_1, row_2, row_3;

            row_1 = xvector_m * ((2.0*f_m(i,j,k) - 5.0*f_m(op(i,1),j,k) + 4.0*f_m(op(i,2),j,k) - f_m(op(i,3),j,k)) / (hvector_m[0]*hvector_m[0])) +

                yvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(i,op(j,1),k)       + coeffs_m[2]*f_m(i,op(j,2),k)        +
                            coeffs_m[3]*f_m(op(i,1),j,k) + coeffs_m[4]*f_m(op(i,1),op(j,1),k) + coeffs_m[5]*f_m(op(i,1),op(j,2),k)  +
                            coeffs_m[6]*f_m(op(i,2),j,k) + coeffs_m[7]*f_m(op(i,2),op(j,1),k) + coeffs_m[8]*f_m(op(i,2),op(j,2),k)) / (hvector_m[0]*hvector_m[1])) +

                zvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(i,j,op(k,1))       + coeffs_m[2]*f_m(i,j,op(k,2))        +
                            coeffs_m[3]*f_m(op(i,1),j,k) + coeffs_m[4]*f_m(op(i,1),j,op(k,1)) + coeffs_m[5]*f_m(op(i,1),j,op(k,2))  +
                            coeffs_m[6]*f_m(op(i,2),j,k) + coeffs_m[7]*f_m(op(i,2),j,op(k,1)) + coeffs_m[8]*f_m(op(i,2),j,op(k,2))) / (hvector_m[0]*hvector_m[2]));


            row_2 = xvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(op(i,1),j,k)       + coeffs_m[2]*f_m(op(i,2),j,k)        +
                        coeffs_m[3]*f_m(i,op(j,1),k) + coeffs_m[4]*f_m(op(i,1),op(j,1),k) + coeffs_m[5]*f_m(op(i,2),op(j,1),k)  +
                        coeffs_m[6]*f_m(i,op(j,2),k) + coeffs_m[7]*f_m(op(i,1),op(j,2),k) + coeffs_m[8]*f_m(op(i,2),op(j,2),k)) / (hvector_m[1]*hvector_m[0])) +

                yvector_m * ((2.0*f_m(i,j,k) - 5.0*f_m(i,op(j,1),k) + 4.0*f_m(i,op(j,2),k) - f_m(i,op(j,3),k))/(hvector_m[1]*hvector_m[1])) +

                zvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(i,j,op(k,1))       + coeffs_m[2]*f_m(i,j,op(k,2))        +
                            coeffs_m[3]*f_m(i,op(j,1),k) + coeffs_m[4]*f_m(i,op(j,1),op(k,1)) + coeffs_m[5]*f_m(i,op(j,1),op(k,2))  +
                            coeffs_m[6]*f_m(i,op(j,2),k) + coeffs_m[7]*f_m(i,op(j,2),op(k,1)) + coeffs_m[8]*f_m(i,op(j,2),op(k,2))) / (hvector_m[1]*hvector_m[2]));


            row_3 = xvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(op(i,1),j,k)       + coeffs_m[2]*f_m(op(i,2),j,k)        +
                        coeffs_m[3]*f_m(i,j,op(k,1)) + coeffs_m[4]*f_m(op(i,1),j,op(k,1)) + coeffs_m[5]*f_m(op(i,2),j,op(k,1))  +
                        coeffs_m[6]*f_m(i,j,op(k,2)) + coeffs_m[7]*f_m(op(i,1),j,op(k,2)) + coeffs_m[8]*f_m(op(i,2),j,op(k,2))) / (hvector_m[2]*hvector_m[0])) +

                yvector_m * ((coeffs_m[0]*f_m(i,j,k)     + coeffs_m[1]*f_m(i,op(j,1),k)       + coeffs_m[2]*f_m(i,op(j,2),k)        +
                            coeffs_m[3]*f_m(i,j,op(k,1)) + coeffs_m[4]*f_m(i,op(j,1),op(k,1)) + coeffs_m[5]*f_m(i,op(j,2),op(k,1))  +
                            coeffs_m[6]*f_m(i,j,op(k,2)) + coeffs_m[7]*f_m(i,op(j,1),op(k,2)) + coeffs_m[8]*f_m(i,op(j,2),op(k,2))) / (hvector_m[2]*hvector_m[1])) +

                zvector_m * ((2.0*f_m(i,j,k) - 5.0*f_m(i,j,op(k,1)) + 4.0*f_m(i,j,op(k,2)) - f_m(i,j,op(k,3))) / (hvector_m[2]*hvector_m[2]));

            typename Mesh_t::matrix_type hessian = {row_1, row_2, row_3};
            return hessian; 
        }

        fview_type &f_m;
        IdxOp op;
        const float coeffs_m[9] = {2.25, -3.0, 0.75, -3.0, 4.0, -1.0, 0.75, -1.0, 0.25};
        const typename Mesh_t::vector_type xvector_m = {1.0, 0.0, 0.0};
        const typename Mesh_t::vector_type yvector_m = {0.0, 1.0, 0.0};
        const typename Mesh_t::vector_type zvector_m = {0.0, 0.0, 1.0};
        typename Mesh_t::vector_type hvector_m;
    };

}  // namespace hess_test

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    Inform msg(argv[0], "TestOnesidedHessian");
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    int pt         = std::atoi(argv[1]);
    bool gauss_fct = std::atoi(argv[2]);
    ippl::Index I(pt);
    ippl::NDIndex<hess_test::dim> owned(I, I, I);

    // Specifies SERIAL, PARALLEL hess_test::dims
    ippl::e_dim_tag decomp[hess_test::dim];
    for (unsigned int d = 0; d < hess_test::dim; d++)
        decomp[d] = ippl::PARALLEL;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<hess_test::dim> layout(owned, decomp);

    // domain [0,1]^3
    double dx                  = 1.0 / double(pt);
    hess_test::Vector_t hx     = {dx, dx, dx};
    hess_test::Vector_t origin = {0.0, 0.0, 0.0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    const unsigned int nghost = 1;
    hess_test::Field_t field(mesh, layout, nghost);
    hess_test::Field_t hessReductionField(mesh, layout, nghost);
    hess_test::MField_t result(mesh, layout, nghost);
    hess_test::MField_t subResult(mesh, layout, nghost);
    hess_test::MField_t exact(mesh, layout, nghost);

    typename hess_test::fview_type& view = field.getView();

    const ippl::NDIndex<hess_test::dim>& lDom = layout.getLocalNDIndex();

    Kokkos::parallel_for(
        "Assign field",
        hess_test::mdrange_type({0, 0, 0}, {view.extent(0), view.extent(1), view.extent(2)}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // local to global index conversion
            const int ig = i + lDom[0].first() - nghost;
            const int jg = j + lDom[1].first() - nghost;
            const int kg = k + lDom[2].first() - nghost;

            double x = (ig + 0.5) * hx[0] + origin[0];
            double y = (jg + 0.5) * hx[1] + origin[1];
            double z = (kg + 0.5) * hx[2] + origin[2];

            if (gauss_fct) {
                view(i, j, k) = hess_test::gaussian(x, y, z);
            } else {
                view(i, j, k) = x * y * z;
            }
        });

    typename hess_test::mview_type& view_exact  = exact.getView();
    typename hess_test::mview_type& view_result = result.getView();

    Kokkos::parallel_for(
        "Assign exact",
        hess_test::mdrange_type(
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
                    ((x - mu) * (x - mu) - 1.0) * hess_test::gaussian(x, y, z),
                    (x - mu) * (y - mu) * hess_test::gaussian(x, y, z),
                    (x - mu) * (z - mu) * hess_test::gaussian(x, y, z)};
                view_exact(i, j, k)[1] = {
                    (x - mu) * (y - mu) * hess_test::gaussian(x, y, z),
                    ((y - mu) * (y - mu) - 1.0) * hess_test::gaussian(x, y, z),
                    (y - mu) * (z - mu) * hess_test::gaussian(x, y, z)};
                view_exact(i, j, k)[2] = {
                    (x - mu) * (z - mu) * hess_test::gaussian(x, y, z),
                    (y - mu) * (z - mu) * hess_test::gaussian(x, y, z),
                    ((z - mu) * (z - mu) - 1.0) * hess_test::gaussian(x, y, z)};
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
    //const auto& domain        = layout.getDomain();
    //const auto& lDomains      = layout.getHostLocalDomains();
    int myRank                = Ippl::Comm->rank();
    const auto& faceNeighbors = layout.getFaceNeighbors();

    //for (unsigned int d = 0; d < 2 * hess_test::dim; ++d) {
        //msg << "faceneighbors[" << d << "].size() = " << faceNeighbors[d].size() << endl;
        //bool isBoundary = (lDomains[myRank][d].max() == domain[d].max())
                          //|| (lDomains[myRank][d].min() == domain[d].min());
        //msg << "Rank " << Ippl::Comm->rank() << ": "
            //<< "hess_test::dim" << d << " isBoundary = " << isBoundary << endl;
    //}

    // Assign initial values to subField

    result = {0.0, 0.0, 0.0};
    //subResult = {0.0, 0.0, 0.0};

    ////////////////////////////////////////
    // Define subfield and its properties //
    ////////////////////////////////////////

    // Define properties of subField
    //int subPt     = 20;
    //int subNghost = 1;
    //ippl::Index subI(subPt);
    //ippl::NDIndex<hess_test::dim> subOwned(subI, I, I);

    //// Create subLayout of desired size
    //ippl::UniformCartesian<double, hess_test::dim> subMesh(subOwned, hx, origin);
    //ippl::FieldLayout<hess_test::dim> subLayout(subOwned, decomp);

    //hess_test::Field_t subField = field.subField(
        //subMesh, subLayout, subNghost, Kokkos::make_pair(0, 30), Kokkos::ALL, Kokkos::ALL);
    //hess_test::fview_type subView = subField.getView();

    //msg2all << "(" << subView.extent(0) << "," << subView.extent(1) << "," << subView.extent(2)
            //<< ")" << endl;

    // Test to write slice of field
    // TODO There is an error in running this in parallel!
    hess_test::dumpVTKScalar(field, 0, dx, dx, dx);
    //Kokkos::fence();
    
    msg2all << "Extents of view: (" << view.extent(0) << "," << view.extent(1) << "," << view.extent(2) << ")" << endl;

    // Test backwardHess on subField only
    //result = ippl::hess(field);
    //subResult = ippl::hess(subField);
    //hess_test::pickHessianIdx(hessReductionField, result, 0, 0, 3);
    //hess_test::dumpVTKScalar(hessReductionField, 0, dx, dx, dx);

    /////////////////////////////
    // Kokkos loop for Hessian //
    /////////////////////////////
	
	// Define Opereators
    hess_test::CenteredHessOp centered_hess(field);
    hess_test::OnesidedHessOp<std::plus<size_t>> forward_hess(field);
    hess_test::OnesidedHessOp<std::minus<size_t>> backward_hess(field);

    // Check whether system boundaries are touched
    const size_t stencilWidth = 3;
    const size_t nghostExt = nghost + stencilWidth;
    const size_t extents[hess_test::dim] = {view.extent(0), view.extent(1), view.extent(2)};
    ippl::NDIndex<hess_test::dim> isSystemBoundary[2*hess_test::dim];
    ippl::NDIndex<hess_test::dim> isCenterDomain = ippl::NDIndex<hess_test::dim>(ippl::Index(nghostExt, extents[0] - nghostExt),
                                                                     ippl::Index(nghostExt, extents[1] - nghostExt),
                                                                     ippl::Index(nghostExt, extents[2] - nghostExt));


    // Assign to each system boundary face a domain for which onesided differencing should be used
    for(size_t face = 0; face < 2*hess_test::dim; ++face){
        if (faceNeighbors[face].size() == 0) {
            isSystemBoundary[face] = ippl::NDIndex<hess_test::dim>(ippl::Index(nghost, extents[0] - nghost),
                                                                     ippl::Index(nghost, extents[1] - nghost),
                                                                     ippl::Index(nghost, extents[2] - nghost));
            size_t d = face/2;

            // Backward difference
            if (face & 1){
                isSystemBoundary[face][d] = ippl::Index(extents[d]-nghost-stencilWidth, extents[d]-nghost);
            } else { // Forward difference
                isSystemBoundary[face][d] = ippl::Index(nghost, nghost+stencilWidth);
            }

        }
    }

    if (myRank == 0){
        for(const auto& idxRange : isSystemBoundary){
            std::cout << idxRange << std::endl;
        }
    }

    ippl::NDIndex testIndex = ippl::NDIndex<hess_test::dim>(ippl::Index(4, 4+1),
                                                             ippl::Index(4, 4+1),
                                                             ippl::Index(1, 1+1));
    std::cout << "testIndex is center: " << isCenterDomain.contains(testIndex) << std::endl;

    //for(size_t i = 1*nghost; i < view.extent(0) - 1*nghost; ++i){
        //for(size_t j = 1*nghost; j < view.extent(1) - 1*nghost; ++j){
            //for(size_t k = 1*nghost; k < view.extent(2) - 1*nghost; ++k){
                //ippl::NDIndex currNDIndex = ippl::NDIndex<hess_test::dim>(ippl::Index(i, i+1),
                                                                         //ippl::Index(j, j+1),
                                                                         //ippl::Index(k, k+1));
                ////std::cout << "(i,j,k) = " << "(" << i << "," << j << "," << k << ")" << std::endl;
                //if (isSystemBoundary[0].contains(currNDIndex)){
                    //std::cout << currNDIndex << " || " << std::endl;
                //}
                //printf("(%lu, %lu, %lu)\n", i, j, k);

                //// Check all faces
                //view_result(i, j, k) = 
                                //isCenterDomain.contains(currNDIndex) * centered_hess(i,j,k);
                                //isSystemBoundary[0].contains(currNDIndex) * forward_hess(i,j,k) +
                                //isSystemBoundary[1].contains(currNDIndex) * backward_hess(i,j,k) +
                                //isSystemBoundary[2].contains(currNDIndex) * forward_hess(i,j,k) +
                                //isSystemBoundary[3].contains(currNDIndex) * backward_hess(i,j,k) +
                                //isSystemBoundary[4].contains(currNDIndex) * forward_hess(i,j,k) +
                                //isSystemBoundary[5].contains(currNDIndex) * backward_hess(i,j,k);

            //}
        //}
    //}

    Kokkos::parallel_for("Onesided Hessian Loop", hess_test::mdrange_type(
                    {3*nghost, 3*nghost, 3*nghost},
                    {view.extent(0) - 3*nghost,
                     view.extent(1) - 3*nghost,
                     view.extent(2) - 3*nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            // Check which type of differencing is needed
            ippl::NDIndex currNDIndex = ippl::NDIndex<hess_test::dim>(ippl::Index(i, i+1),
                                                                     ippl::Index(j, j+1),
                                                                     ippl::Index(k, k+1));

            //std::cout << "(i,j,k) = " << "(" << i << "," << j << "," << k << ")" << std::endl;
            
            if (myRank == 0){
                printf("(%d, %d, %d)\n", i, j, k);
            }
            //printf("asdf");
            if (isSystemBoundary[0].contains(currNDIndex)){
                std::cout << currNDIndex << " || " << std::endl;
            }

            // Check all faces
            view_result(i, j, k) = 
                             isCenterDomain.contains(currNDIndex) * centered_hess(i,j,k) +
                            isSystemBoundary[0].contains(currNDIndex) * forward_hess(i,j,k) +
                            isSystemBoundary[1].contains(currNDIndex) * backward_hess(i,j,k) +
                            isSystemBoundary[2].contains(currNDIndex) * forward_hess(i,j,k) +
                            isSystemBoundary[3].contains(currNDIndex) * backward_hess(i,j,k) +
                            isSystemBoundary[4].contains(currNDIndex) * forward_hess(i,j,k) +
                            isSystemBoundary[5].contains(currNDIndex) * backward_hess(i,j,k);

        });

    Kokkos::fence();

    hess_test::pickHessianIdx(hessReductionField, result, 0, 2, 3);
    //hess_test::dumpVTKScalar(hessReductionField, 0, dx, dx, dx);

    result = result - exact;


    ippl::Vector<hess_test::Vector_t, hess_test::dim> err_hess{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    double avg = 0.0;

    for (size_t dim1 = 0; dim1 < hess_test::dim; ++dim1) {
        for (size_t dim2 = 0; dim2 < hess_test::dim; ++dim2) {
            double valN(0.0);

            Kokkos::parallel_reduce(
                "Relative error",
                hess_test::mdrange_type(
                    {1 * nghost, 1 * nghost, 1 * nghost},
                    {view_result.extent(0) - 1 * nghost,
                     view_result.extent(1) - 1 * nghost,
                     view_result.extent(2) - 1 * nghost}),
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
                hess_test::mdrange_type(
                    {1 * nghost, 1 * nghost, 1 * nghost},
                    {view_exact.extent(0) - 1 * nghost,
                     view_exact.extent(1) - 1 * nghost,
                     view_exact.extent(2) - 1 * nghost}),
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
