/**
@page basics_fft Basics: FFT
@section fft Introduction
The FFT(Fast Fourier Transform) class performs complex-to-complex, real-to-complex and real-to-real
on IPPL Fields. Currently, we use heffte for taking the transforms and the class FFT serves as an
interface between IPPL and heffte. In making this interface, we have referred Cabana library
https://github.com/ECP-copa/Cabana.

 * @subsection FFT Transformation Types
 *
 * FFT is templated on the type of transform to be performed, the dimensionality of the Field to
transform, and the
 * floating-point precision type of the Field (float or double).
 *
 * **Types of transformation**
 * - complex-to-complex FFT: 'ippl::CCTransform'
 * - real-to-complex FFT: 'ippl::RCTransform'
 * - Sine transform: 'ippl::SineTransform'
 * - Cosine transform: 'ippl::CosTransform'
 * - Cosine type 1 transform: 'ippl::Cos1Transform'

@section Example Usage

Consider an example of a transformation from real valued Field to complex valued Field.
Given a Field fieldInput, we can perform a forward FFT as follows:
@code
// Define FFT-Parameters
ippl::ParameterList fftParams;
fftParams.add("use_heffte_defaults", true);
fftParams.add("r2c_direction", 0);


// Define layout of complex valued Field output
std::array<int, dim> pt = {64, 64, 64};
std::array<double, dim> dx = {
    1.0 / double(pt[0]),
    1.0 / double(pt[1]),
    1.0 / double(pt[2]),
};
ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
ippl::Vector<double, 3> origin = {0, 0, 0};

ippl::NDIndex<dim> ownedOutput;
if (fftParams.get<int>("r2c_direction") == 0) {
        ownedOutput[0] = ippl::Index(pt[0] / 2 + 1);
        ownedOutput[1] = ippl::Index(pt[1]);
        ownedOutput[2] = ippl::Index(pt[2]);
} else if (fftParams.get<int>("r2c_direction") == 1) {
        ownedOutput[0] = ippl::Index(pt[0]);
        ownedOutput[1] = ippl::Index(pt[1] / 2 + 1);
        ownedOutput[2] = ippl::Index(pt[2]);
} else if (fftParams.get<int>("r2c_direction") == 2) {
        ownedOutput[0] = ippl::Index(pt[0]);
        ownedOutput[1] = ippl::Index(pt[1]);
        ownedOutput[2] = ippl::Index(pt[2] / 2 + 1);
} else {
        if (ippl::Comm->rank() == 0) {
            std::cerr << "RCDirection need to be 0, 1 or 2 and it"
            << "indicates the dimension in which data is shortened" << std::endl;
            }
        return 0;
}

ippl::FieldLayout<dim> layoutOutput(MPI_COMM_WORLD, ownedOutput, isParallel);
Mesh_t meshOutput(ownedOutput, hx, origin);
field_type_complex fieldOutput(meshOutput, layoutOutput);


// Fill Field fieldInput with random numbers
typename field_type_real::view_type& view            = fieldInput.getView();
typename field_type_real::HostMirror fieldInput_host = fieldInput.getHostMirror();

const int nghost = fieldInput.getNghost();
std::mt19937_64 eng(42 + ippl::Comm->rank());
std::uniform_real_distribution<double> unif(0, 1);

for (size_t i = nghost; i < view.extent(0) - nghost; ++i) {
    for (size_t j = nghost; j < view.extent(1) - nghost; ++j) {
        for (size_t k = nghost; k < view.extent(2) - nghost; ++k) {
            fieldInput_host(i, j, k) = unif(eng);  // 1.0;
        }
    }
}
Kokkos::deep_copy(fieldInput.getView(), fieldInput_host);


// Define FFT object
typedef ippl::FFT<ippl::RCTransform, field_type_real> FFT_type;
std::unique_ptr<FFT_type> fft;
fft = std::make_unique<FFT_type>(layoutInput, layoutOutput, fftParams);


// Perform FFT
fft->transform(ippl::FORWARD, fieldInput, fieldOutput); // Forward transform
fft->transform(ippl::BACKWARD, fieldInput, fieldOutput); // Reverse transform


@endcode
*/