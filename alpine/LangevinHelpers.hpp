#ifndef LANGEVINHELPERS_HPP
#define LANGEVINHELPERS_HPP

#include "Utility/PAssert.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

template <typename T>
using VectorD = Vector<T, Dim>;

template <typename T>
using Vector2D = Vector<T, 2*Dim>;

using VectorD_t = VectorD<double>;

using Vector2D_t = Vector2D<double>;

using MatrixD_t = VectorD<VectorD<double>>;

using Matrix2D_t = Vector2D<Vector2D<double>>;


template<typename T>
KOKKOS_INLINE_FUNCTION
typename T::value_type L2Norm(T &x) {
  return sqrt(dot(x, x).apply());
}


// Generate random numbers in Sphere given by `beamRadius`, centered at the origin
template <typename T, class GeneratorPool>
struct GenerateBoxMuller {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random positions in the sphere
  view_type r;
  const value_type beamRadius;
  const value_type oneThird = 1.0 / 3.0;
  const T origin = {0.0,0.0,0.0};

  // The GeneratorPool
  GeneratorPool pool;

  // Initialize all members
  GenerateBoxMuller(view_type r_, value_type beamRadius_, GeneratorPool pool_) :
                    r(r_), beamRadius(beamRadius_), pool(pool_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = pool.get_state();

    T X({rand_gen.normal(), rand_gen.normal(), rand_gen.normal()});
    value_type uniform(rand_gen.drand(0.0, 1.0));

    r(i) = origin + beamRadius * std::pow(uniform, oneThird) / L2Norm(X) * X;

    // Give the state back, which will allow another thread to acquire it
    pool.free_state(rand_gen);
  }
};


///////////////////////
// DUMPING FUNCTIONS //
///////////////////////

// Works only if ranks == 1
// TODO Allow dumping from multiple ranks
void dumpVTKScalar(Field_t<Dim>& F, VectorD_t cellSpacing, VectorD<size_t> nCells, VectorD_t origin,
                   int iteration, double scalingFactor, std::string out_dir, std::string label) {

  int nx = nCells[0];
  int ny = nCells[1];
  int nz = nCells[2];
  
  typename Field_t<Dim>::view_type::host_mirror_type host_view = F.getHostMirror();

  std::stringstream fname;
  fname << out_dir;
  fname << "/";
  fname << label;
  fname << "_it";
  fname << std::setw(4) << std::setfill('0') << iteration;
  fname << ".vtk";

  Kokkos::deep_copy(host_view, F.getView());

  Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
  vtkout.precision(10);
  vtkout.setf(std::ios::scientific, std::ios::floatfield);

  // start with header
  vtkout << "# vtk DataFile Version 2.0" << endl;
  vtkout << TestName << endl;
  vtkout << "ASCII" << endl;
  vtkout << "DATASET STRUCTURED_POINTS" << endl;
  vtkout << "DIMENSIONS " << nx+1 << " " << ny+1 << " " << nz+1 << endl;
  vtkout << "ORIGIN " << origin[0] << " " 
                      << origin[1] << " " 
                      << origin[2] << endl;
  vtkout << "SPACING " << cellSpacing[0] << " "
                       << cellSpacing[1] << " "
                       << cellSpacing[2] << endl;
  vtkout << "CELL_DATA " << nx * ny * nz << endl;
  vtkout << "SCALARS " << label << " float" << endl;
  for (int z=1; z<nz+1; z++) {
    for (int y=1; y<ny+1; y++) {
      for (int x=1; x<nx+1; x++) {
        vtkout << scalingFactor*host_view(x,y,z) << endl;
      }
    }
  }
}


void dumpVTKVector(VField_t<Dim>& F, VectorD_t cellSpacing, VectorD<size_t> nCells, VectorD_t origin,
                   int iteration, double scalingFactor, std::string out_dir, std::string label) {

  int nx = nCells[0];
  int ny = nCells[1];
  int nz = nCells[2];
  
  typename VField_t<Dim>::view_type::host_mirror_type host_view = F.getHostMirror();

  std::stringstream fname;
  fname << out_dir;
  fname << "/";
  fname << label;
  fname << "_it";
  fname << std::setw(4) << std::setfill('0') << iteration;
  fname << ".vtk";

  Kokkos::deep_copy(host_view, F.getView());

  Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
  vtkout.precision(10);
  vtkout.setf(std::ios::scientific, std::ios::floatfield);

  // start with header
  vtkout << "# vtk DataFile Version 2.0" << endl;
  vtkout << TestName << endl;
  vtkout << "ASCII" << endl;
  vtkout << "DATASET STRUCTURED_POINTS" << endl;
  vtkout << "DIMENSIONS " << nx+1 << " " << ny+1 << " " << nz+1 << endl;
  vtkout << "ORIGIN " << origin[0] << " " 
                      << origin[1] << " " 
                      << origin[2] << endl;
  vtkout << "SPACING " << cellSpacing[0] << " "
                       << cellSpacing[1] << " "
                       << cellSpacing[2] << endl;
  vtkout << "CELL_DATA " << nx * ny * nz << endl;
  vtkout << "VECTORS " << label << " float" << endl;
  for (int z=1; z<nz+1; z++) {
    for (int y=1; y<ny+1; y++) {
      for (int x=1; x<nx+1; x++) {
        vtkout << scalingFactor*host_view(x,y,z)[0] << "\t"
               << scalingFactor*host_view(x,y,z)[1] << "\t"
               << scalingFactor*host_view(x,y,z)[2] << endl;
      }
    }
  }
}


// Store all BeamStatistics to be gathered for file dumping
// Not in use yet (could improve readibility of dumping later on)
// TODO Write sensible comments
struct BeamStatistics{
    double fieldEnergy;
    double ExAmplitude;
    double lorentzAvg;
    double lorentzMax;

    VectorD_t temperature;

    VectorD_t vmax; 
    VectorD_t vmin; 

    VectorD_t rmax; 
    VectorD_t rmin;

    // Moments
    Vector2D_t centroid;
    Matrix2D_t moment;

    // Normalized Moments
    Vector2D_t Ncentroid;
    Matrix2D_t Nmoment;

    // mean position
    VectorD_t rmean;
    // mean momenta
    VectorD_t vmean;

    // rms beam size
    VectorD_t rrms;
    // rms momenta
    VectorD_t vrms;

    // rms emittance (not normalized)
    VectorD_t eps;
    VectorD_t eps2;

    // Normalized position & momenta
    VectorD_t Nrrms;
    VectorD_t Nvrms;

    // Normalized correlation
    VectorD_t Nrvrms;

    // rms emittance (normalized)
    VectorD_t Neps;
    VectorD_t Neps2;
};

#endif /* LANGEVINHELPERS_HPP */
