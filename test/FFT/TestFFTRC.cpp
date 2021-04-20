#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <fstream>
#include <random>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    Inform msg(argv[0],INFORM_ALL_NODES);

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

    ippl::Index Iinput(pt[0]);
    ippl::Index Jinput(pt[1]);
    ippl::Index Kinput(pt[2]);
    ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

    ippl::e_dim_tag domDec[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        domDec[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layoutInput(ownedInput, domDec);

    std::array<double, dim> dx = {
        1.0 / double(pt[0]),
        1.0 / double(pt[1]),
        1.0 / double(pt[2]),
    };
    ippl::Vector<double, 3> hx = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> meshInput(ownedInput, hx, origin);

    typedef ippl::Field<Kokkos::complex<double>, dim> field_type_complex;
    typedef ippl::Field<double, dim> field_type_real;

    field_type_real fieldInput(meshInput, layoutInput);

    ippl::FFTParams fftParams;

    fftParams.setAllToAll( true );
    fftParams.setPencils( true );
    fftParams.setReorder( true );
    fftParams.setRCDirection( 0 );

    ippl::NDIndex<dim> ownedOutput;

    if(fftParams.getRCDirection() == 0) {
        ownedOutput[0] = ippl::Index(pt[0]/2 + 1);
        ownedOutput[1] = ippl::Index(pt[1]);
        ownedOutput[2] = ippl::Index(pt[2]);
    }
    else if(fftParams.getRCDirection() == 1) {
        ownedOutput[0] = ippl::Index(pt[0]);
        ownedOutput[1] = ippl::Index(pt[1]/2 + 1);
        ownedOutput[2] = ippl::Index(pt[2]);
    }
    else if(fftParams.getRCDirection() == 2) {
        ownedOutput[0] = ippl::Index(pt[0]);
        ownedOutput[1] = ippl::Index(pt[1]);
        ownedOutput[2] = ippl::Index(pt[2]/2 + 1);
    }
    else {
        if (Ippl::Comm->rank() == 0) {
            std::cerr << "RCDirection need to be 0, 1 or 2 and it" 
                      << "indicates the dimension in which data is shortened" 
                      << std::endl;
        }
        return 0;
    }
    ippl::FieldLayout<dim> layoutOutput(ownedOutput, domDec);

    ippl::UniformCartesian<double, 3> meshOutput(ownedOutput, hx, origin);
    field_type_complex fieldOutput(meshOutput, layoutOutput);

    typedef ippl::FFT<ippl::RCTransform, 3, double> FFT_type;

    std::unique_ptr<FFT_type> fft;

    fft = std::make_unique<FFT_type>(layoutInput, layoutOutput, fftParams);

    typename field_type_real::view_type& view = fieldInput.getView();    
    typename field_type_real::HostMirror fieldInput_host = 
                                         fieldInput.getHostMirror();

    const int nghost = fieldInput.getNghost();
    std::mt19937_64 eng(42 + Ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(0, 1);

    msg << "(" << view.extent(0) << "," << view.extent(1) << "," << view.extent(2)  << ")" << endl;

    for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
    
                fieldInput_host(i, j, k) = unif(eng);//1.0; 
                              
            }
        }
    }

    Kokkos::deep_copy(fieldInput.getView(), fieldInput_host);

    IpplTimings::stopTimer(fieldInit);

    //Forward transform
    static IpplTimings::TimerRef forwardT = IpplTimings::getTimer("forwardT");
    IpplTimings::startTimer(forwardT);
    fft->transform(1, fieldInput, fieldOutput);
    IpplTimings::stopTimer(forwardT);

    //Reverse transform
    static IpplTimings::TimerRef backwarT = IpplTimings::getTimer("backwarT");
    IpplTimings::startTimer(backwarT);
    fft->transform(-1, fieldInput, fieldOutput);
    IpplTimings::stopTimer(backwarT);

    static IpplTimings::TimerRef postProc = IpplTimings::getTimer("postProc");
    IpplTimings::startTimer(postProc);
    
    auto field_result = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), fieldInput.getView());

    double max_error_local = 0.0;
    for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
    
                double error = std::fabs(fieldInput_host(i, j, k) - 
                                         field_result(i, j, k));
                if(error > max_error_local) 
                    max_error_local = error;
	    }
        }
    }

    msg << "Error= " << std::setprecision(16) << max_error_local << endl;
    IpplTimings::stopTimer(postProc);
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    std::string fn = std::string("TestFFTRC-")+std::string(argv[1]);
    IpplTimings::print(fn);
    //Kokkos::complex<double> max_error(0.0, 0.0);
    //MPI_Reduce(&max_error_local, &max_error, 1, 
    //           MPI_C_DOUBLE_COMPLEX, MPI_MAX, 0, Ippl::getComm());

    return 0;
}
