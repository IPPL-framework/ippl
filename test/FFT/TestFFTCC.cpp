#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>
#include <random>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    Inform msg("TestFFTCC ",INFORM_ALL_NODES);

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);

    static IpplTimings::TimerRef fieldInit = IpplTimings::getTimer("fieldInit");
    IpplTimings::startTimer(fieldInit);

    constexpr unsigned int dim = 3;

    std::array<int,dim> pt = {
      std::atoi(argv[1]),
      std::atoi(argv[2]),
      std::atoi(argv[3])
    };

    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    ippl::e_dim_tag domDec[dim];   
    for (unsigned int d=0; d<dim; d++)
        domDec[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, domDec);

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

    ippl::FFTParams fftParams;

    fftParams.setAllToAll( true );
    fftParams.setPencils( true );
    fftParams.setReorder( true );

    typedef ippl::FFT<ippl::CCTransform, 3, double> FFT_type;

    std::unique_ptr<FFT_type> fft;

    fft = std::make_unique<FFT_type>(layout, fftParams);

    typename field_type::view_type& view = field.getView();    
    typename field_type::HostMirror field_host = field.getHostMirror();

    const int nghost = field.getNghost();
    std::mt19937_64 engReal(42 + Ippl::Comm->rank());
    std::uniform_real_distribution<double> unifReal(0, 1);
    
    std::mt19937_64 engImag(43 + Ippl::Comm->rank());
    std::uniform_real_distribution<double> unifImag(0, 1);
    
    msg << "(" << view.extent(0) << "," << view.extent(1) << "," << view.extent(2)  << ")" << endl;

    for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
    
                field_host(i, j, k).real() = unifReal(engReal);
                field_host(i, j, k).imag() = unifImag(engImag);
                              
            }
        }
    }

    Kokkos::deep_copy(field.getView(), field_host);

    IpplTimings::stopTimer(fieldInit);

    //Forward transform
    static IpplTimings::TimerRef forwardT = IpplTimings::getTimer("forwardT");
    IpplTimings::startTimer(forwardT);
    fft->transform(1, field);
    IpplTimings::stopTimer(forwardT);

    //Reverse transform
    static IpplTimings::TimerRef backwarT = IpplTimings::getTimer("backwarT");
    IpplTimings::startTimer(backwarT);
    fft->transform(-1, field);
    IpplTimings::stopTimer(backwarT);

    static IpplTimings::TimerRef postProc = IpplTimings::getTimer("postProc");
    IpplTimings::startTimer(postProc);
    
    auto field_result = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), field.getView());

    Kokkos::complex<double> max_error_local(0.0, 0.0);
    for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
    
                Kokkos::complex<double> 
                    error(std::fabs(field_host(i, j, k).real() - 
                                    field_result(i, j, k).real()), 
                          std::fabs(field_host(i, j, k).imag() - 
                                    field_result(i, j, k).imag()));

                if(error.real() > max_error_local.real()) 
                    max_error_local.real() = error.real();
                
                if(error.imag() > max_error_local.imag()) 
                    max_error_local.imag() = error.imag();
            }
        }
    }

    msg << "Error= " << std::setprecision(16) << max_error_local << endl;
    IpplTimings::stopTimer(postProc);
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    std::string fn = std::string("TestFFTCC-")+std::string(argv[1]);
    IpplTimings::print(fn);
    return 0;
}
