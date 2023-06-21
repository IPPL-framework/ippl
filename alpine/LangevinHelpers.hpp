#ifndef LANGEVINHELPERS_HPP
#define LANGEVINHELPERS_HPP

#include <cmath>

#include "Utility/PAssert.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

template <typename T>
using VectorD = ippl::Vector<T, Dim>;

template <typename T>
using Vector2D = ippl::Vector<T, 2 * Dim>;

using VectorD_t = VectorD<double>;

using Vector2D_t = Vector2D<double>;

using MatrixD_t = VectorD<VectorD<double>>;

using Matrix2D_t = Vector2D<Vector2D<double>>;

template <unsigned Dim = 3>
using MField_t = Field<MatrixD_t, Dim>;

// View types (of particle attributes)
typedef ParticleAttrib<double>::view_type attr_view_t;
typedef ParticleAttrib<VectorD_t>::view_type attr_Dview_t;
typedef ParticleAttrib<MatrixD_t>::view_type attr_DMatrixView_t;
typedef ParticleAttrib<double>::HostMirror attr_mirror_t;
typedef ParticleAttrib<VectorD_t>::HostMirror attr_Dmirror_t;

// View types (of Fields)
typedef typename ippl::detail::ViewType<double, Dim>::view_type Field_view_t;
typedef typename ippl::detail::ViewType<VectorD_t, Dim>::view_type VField_view_t;
typedef typename ippl::detail::ViewType<MatrixD_t, Dim>::view_type MField_view_t;

template <typename T>
KOKKOS_INLINE_FUNCTION typename T::value_type L2Norm(T& x) {
    return sqrt(dot(x, x).apply());
}

// Generate random numbers in Sphere given by `beamRadius`, centered at the origin
template <typename T, class GeneratorPool>
struct GenerateBoxMuller {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random positions in the sphere
    view_type r;
    const value_type beamRadius;
    const value_type oneThird = 1.0 / 3.0;
    const T origin            = {0.0, 0.0, 0.0};

    // The GeneratorPool
    GeneratorPool pool;

    // Initialize all members
    GenerateBoxMuller(view_type r_, value_type beamRadius_, GeneratorPool pool_)
        : r(r_)
        , beamRadius(beamRadius_)
        , pool(pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = pool.get_state();

        T X({rand_gen.normal(), rand_gen.normal(), rand_gen.normal()});
        value_type uniform(rand_gen.drand(0.0, 1.0));

        r(i) = origin + beamRadius * std::pow(uniform, oneThird) / L2Norm(X) * X;

        // Give the state back, which will allow another thread to acquire it
        pool.free_state(rand_gen);
    }
};

// Generate random numbers in Sphere given by `beamRadius`, centered at the origin
// This works only on we initialize all particles on one processor and call the `update()`
// method on the particle bunch
// If local initialization is needed, have a look at the initialization in `PenningTrap.cpp`
template <typename T, class GeneratorPool>
struct GenerateMaxwellian {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random positions in the sphere
    view_type r;
    view_type v;
    double mu;
    double sigma;
    const value_type halfBoxL_r;
    const value_type halfBoxL_v;

    // The GeneratorPool
    GeneratorPool pool;

    // Initialize all members
    GenerateMaxwellian(view_type r_, view_type v_, double mu_, double sigma_, value_type boxL_r_,
                       value_type boxL_v_, GeneratorPool pool_)
        : r(r_)
        , v(v_)
        , mu(mu_)
        , sigma(sigma_)
        , halfBoxL_r(0.5 * boxL_r_)
        , halfBoxL_v(0.5 * boxL_v_)
        , pool(pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = pool.get_state();

        r(i) = {rand_gen.drand(-halfBoxL_r, halfBoxL_r), rand_gen.drand(-halfBoxL_r, halfBoxL_r),
                rand_gen.drand(-halfBoxL_r, halfBoxL_r)};

        v(i) = {rand_gen.normal(mu, sigma), rand_gen.normal(mu, sigma), rand_gen.normal(mu, sigma)};
        // Could be that some sampled velocities are outside our velocity domain
        for (unsigned d = 0; d < Dim; ++d) {
            while (v(i)[d] < -halfBoxL_v || v(i)[d] > halfBoxL_v) {
                v(i)[d] = rand_gen.normal(mu, sigma);
            }
        }

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

    const int nghost = F.getNghost();

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
    vtkout << "DIMENSIONS " << nx + nghost << " " << ny + nghost << " " << nz + nghost << endl;
    vtkout << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << endl;
    vtkout << "SPACING " << cellSpacing[0] << " " << cellSpacing[1] << " " << cellSpacing[2]
           << endl;
    vtkout << "CELL_DATA " << nx * ny * nz << endl;
    vtkout << "SCALARS " << label << " float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = nghost; z < nz + nghost; z++) {
        for (int y = nghost; y < ny + nghost; y++) {
            for (int x = nghost; x < nx + nghost; x++) {
                vtkout << scalingFactor * host_view(x, y, z) << endl;
            }
        }
    }
}

template <typename T>
void dumpVTKVector(VField_t<T, Dim>& F, VectorD_t cellSpacing, VectorD<size_t> nCells,
                   VectorD_t origin, int iteration, double scalingFactor, std::string out_dir,
                   std::string label) {
    int nx = nCells[0];
    int ny = nCells[1];
    int nz = nCells[2];

    const int nghost = F.getNghost();

    typename VField_t<T, Dim>::view_type::host_mirror_type host_view = F.getHostMirror();

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
    vtkout << "DIMENSIONS " << nx + nghost << " " << ny + nghost << " " << nz + nghost << endl;
    vtkout << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << endl;
    vtkout << "SPACING " << cellSpacing[0] << " " << cellSpacing[1] << " " << cellSpacing[2]
           << endl;
    vtkout << "CELL_DATA " << nx * ny * nz << endl;
    vtkout << "VECTORS " << label << " float" << endl;
    for (int z = nghost; z < nz + nghost; z++) {
        for (int y = nghost; y < ny + nghost; y++) {
            for (int x = nghost; x < nx + nghost; x++) {
                vtkout << scalingFactor * host_view(x, y, z)[0] << "\t"
                       << scalingFactor * host_view(x, y, z)[1] << "\t"
                       << scalingFactor * host_view(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpCSVMatrix(MatrixD_t m, std::string matrixPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        // Write header
        csvout << "nv,";

        csvout << matrixPrefix << "_xx," << matrixPrefix << "_xy," << matrixPrefix << "_xz,"
               << matrixPrefix << "_yx," << matrixPrefix << "_yy," << matrixPrefix << "_yz,"
               << matrixPrefix << "_zx," << matrixPrefix << "_zy," << matrixPrefix << "_zz" << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    for (size_type i = 0; i < Dim; ++i) {
        for (size_type j = 0; j < Dim; ++j) {
            csvout << m[i][j];
            if (i == Dim - 1 && j == Dim - 1) {
                csvout << endl;
            } else {
                csvout << ",";
            }
        }
    }
}

void dumpCSVVector(VectorD_t v, std::string vectorPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << vectorPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        csvout << "nv,";

        // Write header
        csvout << vectorPrefix << "_x," << vectorPrefix << "_y," << vectorPrefix << "_z" << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    for (size_type i = 0; i < Dim; ++i) {
        csvout << v[i];
        if (i == Dim - 1) {
            csvout << endl;
        } else {
            csvout << ",";
        }
    }
}

void dumpCSVScalar(double val, std::string scalarPrefix, size_type iteration, bool create_header,
                   std::string folder) {
    std::stringstream fname;
    fname << folder << "/";
    fname << scalarPrefix;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (create_header) {
        csvout << "nv,";

        // Write header
        csvout << scalarPrefix << endl;
    }

    // And dump into file
    csvout << iteration << ",";
    csvout << val << endl;
}

template <typename T>
void dumpCSVMatrixField(VField_t<T, Dim>& M0, VField_t<T, Dim>& M1, VField_t<T, Dim>& M2,
                        ippl::Vector<size_type, Dim> nx, std::string matrixPrefix,
                        size_type iteration, std::string folder) {
    VField_view_t M0View = M0.getView();
    VField_view_t M1View = M1.getView();
    VField_view_t M2View = M2.getView();

    typename VField_view_t::host_mirror_type hostM0View = M0.getHostMirror();
    typename VField_view_t::host_mirror_type hostM1View = M1.getHostMirror();
    typename VField_view_t::host_mirror_type hostM2View = M2.getHostMirror();

    Kokkos::deep_copy(hostM0View, M0View);
    Kokkos::deep_copy(hostM1View, M1View);
    Kokkos::deep_copy(hostM2View, M2View);

    const int nghost = M0.getNghost();

    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix << "field_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    // Write header
    csvout << matrixPrefix << "0_x," << matrixPrefix << "0_y," << matrixPrefix << "0_z,"
           << matrixPrefix << "1_x," << matrixPrefix << "1_y," << matrixPrefix << "1_z,"
           << matrixPrefix << "2_x," << matrixPrefix << "2_y," << matrixPrefix << "2_z" << endl;

    // And dump into file
    for (unsigned x = nghost; x < nx[0] + nghost; x++) {
        for (unsigned y = nghost; y < nx[1] + nghost; y++) {
            for (unsigned z = nghost; z < nx[2] + nghost; z++) {
                csvout << hostM0View(x, y, z)[0] << "," << hostM0View(x, y, z)[1] << ","
                       << hostM0View(x, y, z)[2] << "," << hostM1View(x, y, z)[0] << ","
                       << hostM1View(x, y, z)[1] << "," << hostM1View(x, y, z)[2] << ","
                       << hostM2View(x, y, z)[0] << "," << hostM2View(x, y, z)[1] << ","
                       << hostM2View(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpCSVMatrixAttr(ParticleAttrib<VectorD_t>& M0, ParticleAttrib<VectorD_t>& M1,
                       ParticleAttrib<VectorD_t>& M2, size_type particleNum,
                       std::string matrixPrefix, size_type iteration, std::string folder) {
    attr_Dview_t M0View = M0.getView();
    attr_Dview_t M1View = M1.getView();
    attr_Dview_t M2View = M2.getView();

    typename attr_Dview_t::host_mirror_type hostM0View = M0.getHostMirror();
    typename attr_Dview_t::host_mirror_type hostM1View = M1.getHostMirror();
    typename attr_Dview_t::host_mirror_type hostM2View = M2.getHostMirror();

    Kokkos::deep_copy(hostM0View, M0View);
    Kokkos::deep_copy(hostM1View, M1View);
    Kokkos::deep_copy(hostM2View, M2View);

    std::stringstream fname;
    fname << folder << "/";
    fname << matrixPrefix << "attr_it";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".csv";

    Inform csvout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    // Write header
    csvout << matrixPrefix << "0_x," << matrixPrefix << "0_y," << matrixPrefix << "0_z,"
           << matrixPrefix << "1_x," << matrixPrefix << "1_y," << matrixPrefix << "1_z,"
           << matrixPrefix << "2_x," << matrixPrefix << "2_y," << matrixPrefix << "2_z" << endl;

    // And dump into file
    for (size_type i = 0; i < particleNum; ++i) {
        csvout << hostM0View(i)[0] << "," << hostM0View(i)[1] << "," << hostM0View(i)[2] << ","
               << hostM1View(i)[0] << "," << hostM1View(i)[1] << "," << hostM1View(i)[2] << ","
               << hostM2View(i)[0] << "," << hostM2View(i)[1] << "," << hostM2View(i)[2] << endl;
    }
}

template <typename T>
void extractScalarFieldDim(VField_t<T, Dim>& vectorField, Field_t<Dim>& scalarField,
                           size_t dimToExtract) {
    VField_view_t vectorView = vectorField.getView();
    Field_view_t scalarView  = scalarField.getView();

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "Assign initial velocity PDF and reference solution for H",
        ippl::getRangePolicy(vectorView, 0), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::apply(scalarView, args) = ippl::apply(vectorView, args)[dimToExtract];
        });
}

template <typename T>
void extractScalarField(VField_t<T, Dim>& vectorField, Field_t<Dim>& scalarField0,
                        Field_t<Dim>& scalarField1, Field_t<Dim>& scalarField2) {
    extractScalarFieldDim(vectorField, scalarField0, 0);
    extractScalarFieldDim(vectorField, scalarField1, 1);
    extractScalarFieldDim(vectorField, scalarField2, 2);
}

template <typename T>
void constructVFieldFromFields(VField_t<T, Dim>& vectorField, Field_t<Dim>& scalarField0,
                               Field_t<Dim>& scalarField1, Field_t<Dim>& scalarField2) {
    VField_view_t vectorView = vectorField.getView();
    Field_view_t scalarView0 = scalarField0.getView();
    Field_view_t scalarView1 = scalarField1.getView();
    Field_view_t scalarView2 = scalarField2.getView();

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "constructVFieldFromFields(VField&, Field&, Field&, Field&)",
        ippl::getRangePolicy(vectorView, 0), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::apply(vectorView, args)[0] = ippl::apply(scalarView0, args);
            ippl::apply(vectorView, args)[1] = ippl::apply(scalarView1, args);
            ippl::apply(vectorView, args)[2] = ippl::apply(scalarView2, args);
        });
}

void extractScalarField(MField_t<Dim>& matrixField, Field_t<Dim>& scalarField, size_t rowIdx,
                        size_t colIdx) {
    MField_view_t matrixView = matrixField.getView();
    Field_view_t scalarView  = scalarField.getView();

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "extractScalarField(MField&, Field&, rowIdx, colIdx)", ippl::getRangePolicy(matrixView, 0),
        KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::apply(scalarView, args) = ippl::apply(matrixView, args)[rowIdx][colIdx];
        });
}

template <typename T>
double L2VectorNorm(const VField_t<T, Dim>& vectorField, const int shift) {
    double sum             = 0;
    VField_view_t view     = vectorField.getView();
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_reduce(
        "L2VectorNorm(VField&, shift)", ippl::getRangePolicy(view, shift),
        KOKKOS_LAMBDA(const index_array_type& args, double& val) {
            VectorD_t& tmp_vec = apply(view, args);
            val += dot(tmp_vec, tmp_vec).apply();
        },
        Kokkos::Sum<double>(sum));
    double globalSum  = 0;
    MPI_Datatype type = get_mpi_datatype<double>(sum);
    MPI_Allreduce(&sum, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
    return Kokkos::sqrt(globalSum);
}

MatrixD_t MFieldRelError(const MField_t<Dim>& matrixFieldAppr,
                         const MField_t<Dim>& matrixFieldExact, const int shift) {
    MatrixD_t relError;
    MField_view_t viewAppr  = matrixFieldAppr.getView();
    MField_view_t viewExact = matrixFieldExact.getView();

    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
    for (size_type m = 0; m < Dim; ++m) {
        for (size_type n = 0; n < Dim; ++n) {
            double diffNorm  = 0;
            double exactNorm = 0;
            Kokkos::parallel_reduce(
                "MFieldError(MField&, MField&, shift)",
                mdrange_type({shift, shift, shift},
                             {viewAppr.extent(0) - shift, viewAppr.extent(1) - shift,
                              viewAppr.extent(2) - shift}),
                KOKKOS_LAMBDA(const size_type i, const size_type j, const size_type k,
                              double& diffVal, double& exactVal) {
                    diffVal += Kokkos::pow(viewAppr(i, j, k)[m][n] - viewExact(i, j, k)[m][n], 2);
                    exactVal += Kokkos::pow(viewExact(i, j, k)[m][n], 2);
                },
                Kokkos::Sum<double>(diffNorm), Kokkos::Sum<double>(exactNorm));
            relError[m][n] = Kokkos::sqrt(diffNorm / exactNorm);
        }
    }
    // TODO MPI Reduction
    return relError;
}

/*!
 * Computes the inner product of two fields given a margin of `nghost` in which the values are
 * ignored
 * @param f field
 * @return Result of f^T f
 */
template <typename T, unsigned Dim>
T subfieldNorm(const Field<T, Dim>& f, const int shift, const int p = 2) {
    T sum                  = 0;
    auto view              = f.getView();
    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_reduce(
        "subfieldSum(Field&, shift)", ippl::getRangePolicy(view, shift),
        KOKKOS_LAMBDA(const index_array_type& args, T& val) {
            val += Kokkos::pow(apply(view, args), p);
        },
        Kokkos::Sum<T>(sum));
    T globalSum       = 0;
    MPI_Datatype type = get_mpi_datatype<T>(sum);
    MPI_Allreduce(&sum, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
    return std::pow(globalSum, 1.0 / p);
}

// Store all BeamStatistics to be gathered for file dumping
// Not in use yet (could improve readibility of dumping later on)
// TODO Write sensible comments
struct BeamStatistics {
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

// Cholesky Decomposition for positive-definite matrices
// Simplest algorithm, might return NaN's for ill-conditioned matrices
KOKKOS_INLINE_FUNCTION MatrixD_t cholesky3x3(const MatrixD_t& M) {
    MatrixD_t L;
    L[0][0] = Kokkos::sqrt(M[0][0]);
    L[1][0] = M[1][0] / L[0][0];
    L[1][1] = Kokkos::sqrt(M[1][1] - L[1][0] * L[1][0]);
    L[2][0] = M[2][0] / L[0][0];
    L[2][1] = (M[2][1] - L[2][0] * L[1][0]) / L[1][1];
    L[2][2] = Kokkos::sqrt(M[2][2] - L[2][0] * L[2][0] - L[2][1] * L[2][1]);

    return L;
}

// Only pick the diagonal values of the input Matrix
// Avoids division by zero as seen in `cholesky3x3()`
KOKKOS_INLINE_FUNCTION MatrixD_t cholesky3x3_diagonal(const MatrixD_t& M) {
    MatrixD_t L;
    L[0][0] = Kokkos::sqrt(M[0][0]);
    L[1][1] = Kokkos::sqrt(M[1][1]);
    L[2][2] = Kokkos::sqrt(M[2][2]);
    return L;
}

// Cholesky decomposition for semi-positive definite matrices
// Avoids sqrt of negative numbers by pivoting
// Computation is inplace
KOKKOS_INLINE_FUNCTION MatrixD_t LDLtCholesky3x3(const MatrixD_t& M) {
    MatrixD_t Q;
    VectorD_t row_factors;

    // Compute first row multiplicators
    row_factors[0] = M[1][0] / M[0][0];
    row_factors[1] = M[2][0] / M[0][0];

    // Eliminate value at [1,0]
    M[1] = M[1] - row_factors[0] * M[0];

    // Eliminate value at [2,0]
    M[2] = M[2] - row_factors[1] * M[0];

    // Eliminate value at [2,1]
    row_factors[2] = M[2][1] / M[1][1];
    M[2]           = M[2] - row_factors[2] * M[1];

    // Check that the input matrix is semi-positive definite
    VectorD_t D = {M[0][0], M[1][1], M[2][2]};
    PAssert_GE(M[0][0], 0.0);
    PAssert_GE(M[1][1], 0.0);
    PAssert_GE(M[2][2], 0.0);

    // Compute Q = sqrt(D) * L^T
    // Where D is diag(M) and `row-factors` are the lower triangular values of L^T
    // Loop is unrolled as we only ever do this for 3x3 Matrices
    Q[0][0] = Kokkos::sqrt(M[0][0]);
    Q[1][0] = row_factors[0] * Kokkos::sqrt(M[1][1]);
    Q[1][1] = Kokkos::sqrt(M[1][1]);
    Q[2][0] = row_factors[1] * Kokkos::sqrt(M[2][2]);
    Q[2][1] = row_factors[2] * Kokkos::sqrt(M[2][2]);
    Q[2][2] = Kokkos::sqrt(M[2][2]);

    return Q;
}

KOKKOS_INLINE_FUNCTION VectorD_t matrixVectorMul3x3(const MatrixD_t& M, const VectorD_t& v) {
    VectorD_t res;
    res[0] = dot(M[0], v).apply();
    res[1] = dot(M[1], v).apply();
    res[2] = dot(M[2], v).apply();
    return res;
}

#endif /* LANGEVINHELPERS_HPP */
