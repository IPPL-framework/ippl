#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    std::array<int, dim> pt = {8, 8, 8};
    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, allParallel);

    std::array<double, dim> dx = {
        1.0 / double(pt[0]),
        1.0 / double(pt[1]),
        1.0 / double(pt[2]),
    };
    ippl::Vector<double, 3> hx = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    typedef ippl::Field<Kokkos::complex<double>, dim> field_type;

    field_type field(mesh, layout);

    ippl::HeffteParams fftParams;

    fftParams.setAllToAll( true );
    fftParams.setPencils( true );
    fftParams.setReorder( true );

    typedef ippl::FFT<ippl::CCTransform, 3, double> FFT_type;

    std::unique_ptr<FFT_type> fft;

    fft = std::make_unique<FFT_type>(layout, fftParams);

    typename field_type::view_type& view = field.getView();    
    typename field_type::HostMirror field_host = field.getHostMirror();

    const int nghost = field.getNghost();

    for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
    
                field_host(i, j, k).real() = 1.0; 
                field_host(i, j, k).imag() = 1.0; 
                              
            }
        }
    }

    Kokkos::deep_copy(field.getView(), field_host);

    //Forward transform
    fft->transform(1, field);
    //Reverse transform
    fft->transform(-1, field);
    
    auto field_result = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), field.getView() );

    //std::complex<double> max_error_local(0.0, 0.0);
    //for (int i = nghost; i < view.extent(0) - nghost; ++i) {
    //    for (int j = nghost; j < view.extent(1) - nghost; ++j) {
    //        for (int k = nghost; k < view.extent(2) - nghost; ++k) {
    //
    //            std::complex<double> error(std::fabs(field_host(i, j, k).real() - field_result(i, j, k).real()), 
    //                                       std::fabs(field_host(i, j, k).imag() - field_result(i, j, k).imag()));

    //            //if(error.real() > max_error_local.real()) max_error_local.real() = error.real();
    //            //if(error.imag() > max_error_local.imag()) max_error_local.imag() = error.imag();
    //            max_error_local.real( error.real() );             
    //            max_error_local.imag( error.imag() );             
    //        }
    //    }
    //}

    //std::complex<double> max_error(0.0, 0.0);
    //MPI_Reduce(&max_error_local, &max_error, 1, 
    //           MPI_C_DOUBLE_COMPLEX, MPI_MAX, 0, Ippl::getComm());

    //if(Ippl::Comm->rank() == 0) {
    //    std::cout << "Max. error " << std::setprecision(16) << max_error << std::endl;
    //}
    return 0;
}
