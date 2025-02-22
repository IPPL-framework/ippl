#include "Ippl.h"
#include <array>
#include <iostream>
#include <random>
#include <typeinfo>
#include "Utility/ParameterList.h"

//
// TestLumi.cpp: constructs a FFT object
// 
int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv); // hides heffte and Kokkos initialisation
  {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;
    
    std::array<int, dim> pt = {32, 32, 32};
    ippl::Index I(pt[0]); ippl::Index J(pt[1]);ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);
    
    std::array<bool, dim> isParallel;
    isParallel.fill(true);
    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);
    std::array<double, dim> dx = { 1.0 / double(pt[0]), 1.0 / double(pt[1]), 1.0 / double(pt[2]) };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    Mesh_t mesh(owned, hx, origin);
    typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
    field_type field(mesh, layout);
    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);
    
    typedef ippl::FFT<ippl::SineTransform, field_type> FFT_type;
    FFT_type fft(layout, fftParams);
    }
    ippl::finalize();
    return 0;
}
