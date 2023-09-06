#include "Ippl.h"
#include <complex>
#include <fstream>
#include <source_location>
#include <Eigen/Dense>
template<typename scalar>
using complex = std::complex<scalar>;
template<typename scalar>
using spinor = ippl::Vector<complex<scalar>, 4>;
template<typename scalar>
using eigen_spinor = Eigen::Matrix<complex<scalar>, 4, 1>;
template<typename value_type, unsigned int dim>
using ippl_matrix = ippl::Vector<ippl::Vector<value_type, dim>, dim>;

template<typename value_type, unsigned int dim>
using eigen_vector = Eigen::Matrix<value_type, dim, 1>;

template<typename value_type, unsigned int dim>
using eigen_matrix = Eigen::Matrix<value_type, dim, dim>;

template<typename value_type, unsigned int dim>
eigen_matrix<value_type, dim> to_eigen(const ippl_matrix<value_type, dim>& mat){
    eigen_matrix<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        for(unsigned int j = 0;j < dim;j++)
            ret(i, j) = mat[i][j];

    return ret;
}
template<typename value_type, unsigned int dim>
eigen_vector<value_type, dim> to_eigen(const ippl::Vector<value_type, dim>& vec){
    eigen_vector<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        ret(i) = vec[i];

    return ret;
}
template<typename value_type, unsigned int dim>
ippl::Vector<value_type, dim> to_ippl(const eigen_vector<value_type, dim>& vec){
    ippl::Vector<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        ret[i] = vec(i);

    return ret;
}

template<typename value_type, unsigned int dim>
ippl_matrix<value_type, dim> to_ippl(const eigen_matrix<value_type, dim>& mat){
    ippl_matrix<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        for(unsigned int j = 0;j < dim;j++)
            ret[i][j] = mat(i, j);

    return ret;
}

template<typename scalar>
spinor<scalar> flux_function_for_dx(const spinor<scalar>& psi){
    spinor<scalar> ret{-psi[3], -psi[2], -psi[1], -psi[0]};
    return ret;
}
template<typename scalar>
spinor<scalar> flux_function_for_dy(const spinor<scalar>& psi){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));
    spinor<scalar> ret{psi[3] * I, -psi[2] * I, psi[1] * I, -psi[0] * I};
    return ret;
}
template<typename scalar>
spinor<scalar> flux_function_for_dz(const spinor<scalar>& psi){
    spinor<scalar> ret{-psi[2], psi[3], -psi[0], psi[1]};
    return ret;
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dx(){
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor<scalar>{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][3] = -1;
    ret[1][2] = -1;
    ret[2][1] = -1;
    ret[3][0] = -1;
    return ret;
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dy(){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));;
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][3] = I;
    ret[1][2] = -I;
    ret[2][1] = I;
    ret[3][0] = -I;
    return ret;
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dz(){
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][2] = -1;
    ret[1][3] = 1;
    ret[2][0] = -1;
    ret[3][1] = 1;
    return ret;
}
template<typename scalar>
spinor<scalar> mass_term(const spinor<scalar>& psi){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));
    spinor<scalar> ret(psi);
    ret[0] *= -I;ret[1] *= -I;ret[2] *= I;ret[3] *= I;
    return ret;
}
template<typename value_type, unsigned int dim>
struct eigensystem{
    using ippl_matrix_type = ippl_matrix <value_type, dim>;
    using matrix_type      = eigen_matrix<value_type, dim>;
    using vector_type      = eigen_vector<value_type, dim>;
    vector_type eigenvalues;
    matrix_type eigenvectors;
    matrix_type eigenvectors_inverse;
    constexpr unsigned int Dim(){return dim;}
    eigensystem() = default;
    eigensystem(const matrix_type& emat){
        Eigen::ComplexEigenSolver<eigen_matrix<value_type, dim>> solver(emat);
        eigenvalues = solver.eigenvalues();
        eigenvectors = solver.eigenvectors();
        eigenvectors_inverse = eigenvectors.inverse();
    }
    eigensystem(const ippl_matrix_type& x) : eigensystem(to_eigen(x)){}
};
template<typename value_type, unsigned int dim>
eigensystem(const eigen_matrix<value_type, dim>& x) -> eigensystem<value_type, dim>;
template<typename value_type, unsigned int dim>
eigensystem(const ippl_matrix<value_type, dim>& x) -> eigensystem<value_type, dim>;

template<typename scalar, int dim>
spinor<scalar> lax_friedrichs_flux_d(const spinor<scalar>& l, const spinor<scalar>& r){
    //return l;
    eigensystem<complex<scalar>, 4U> sys;
    if constexpr(dim == 0)
        sys = eigensystem(flux_J_for_dx<scalar>());
    else if constexpr (dim == 1)
        sys = eigensystem(flux_J_for_dy<scalar>());
    else if constexpr (dim == 2)
        sys = eigensystem(flux_J_for_dz<scalar>());

    eigen_spinor<scalar> l_in_eigenbasis = sys.eigenvectors_inverse * to_eigen(l);
    eigen_spinor<scalar> r_in_eigenbasis = sys.eigenvectors_inverse * to_eigen(r);
    

    eigen_spinor<scalar> composition_in_eigenbasis;
    //composition_in_eigenbasis.fill(decltype(composition_in_eigenbasis)::Scalar(0.0));
    for(unsigned int i = 0;i < sys.Dim();i++){
        if(sys.eigenvalues(i).real() < 0.0){
            composition_in_eigenbasis(i) = l_in_eigenbasis(i);
        }
        else{
            composition_in_eigenbasis(i) = r_in_eigenbasis(i);
        }
    }
    eigen_spinor<scalar> composition = sys.eigenvectors * composition_in_eigenbasis;
    if constexpr(dim == 0)
    return flux_function_for_dx<scalar>(to_ippl<complex<scalar>, 4>(composition));
    else if constexpr (dim == 1)
    return flux_function_for_dy<scalar>(to_ippl<complex<scalar>, 4>(composition));
    else if constexpr (dim == 2)
    return flux_function_for_dz<scalar>(to_ippl<complex<scalar>, 4>(composition));
}
constexpr unsigned int Dim = 3;
template<typename scalar>
using mesh_t      = ippl::UniformCartesian<scalar, Dim>;
template<typename scalar>
using centering_t = typename mesh_t<scalar>::DefaultCentering;
template<typename scalar>
using spinor_field_template = ippl::Field<spinor<scalar>, Dim, mesh_t<scalar>, centering_t<scalar>>;

template<typename scalar>
using scalar_field_template = ippl::Field<scalar, Dim, mesh_t<scalar>, centering_t<scalar>>;

template<typename scalar>
scalar_field_template<scalar> to_scalar_field(const spinor_field_template<scalar>& spinor_field){
    scalar_field_template<scalar> scalar_field;
    mesh_t<scalar> mesh = spinor_field.get_mesh();
    ippl::FieldLayout<Dim> layout = spinor_field.getLayout();
    scalar_field.initialize(mesh, layout, 0);
    const Kokkos::View<const spinor<scalar>***> spinor_field_view = spinor_field.getView();
          Kokkos::View<             scalar ***> scalar_field_view = scalar_field.getView();
    Kokkos::parallel_for(ippl::getRangePolicy(spinor_field.getView()),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                #ifndef __CUDA_ARCH__
                using std::conj;
                #endif
                scalar_field_view(i, j, k) = 
                (spinor_field_view(i, j, k)[0] * conj(spinor_field_view(i, j, k)[0])).real() +
                (spinor_field_view(i, j, k)[1] * conj(spinor_field_view(i, j, k)[1])).real() +
                (spinor_field_view(i, j, k)[2] * conj(spinor_field_view(i, j, k)[2])).real() +
                (spinor_field_view(i, j, k)[3] * conj(spinor_field_view(i, j, k)[3])).real() ;
            }
    );
    return scalar_field;
}
template<typename scalar>
void dumpVTK(const scalar_field_template<scalar>& rho,
     /*Extents */int    nx, int    ny, int    nz,      int iteration,
     /*Spacings*/double dx, double dy, double dz) {
    using field_type = scalar_field_template<scalar>;
    typename field_type::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    std::ofstream vtkout(fname.str().c_str(), std::ios::trunc);
    vtkout.precision(6);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0\n";
    vtkout << "TestDirac\n";
    vtkout << "ASCII\n";
    vtkout << "DATASET STRUCTURED_POINTS\n";
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << '\n';
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << '\n';
    vtkout << "SPACING " << dx << " " << dy << " " << dz << '\n';
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << '\n';

    vtkout << "SCALARS Rho float\n";
    vtkout << "LOOKUP_TABLE default\n";
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                vtkout << host_view(x, y, z) << '\n';
            }
        }
    }
    vtkout << std::endl;
}
template<size_t i, typename... Ts>
auto get(Ts&&... args){
    return std::get<i>(std::tie(std::forward<Ts>(args)...));
}


int main(int argc, char* argv[]) {
    using namespace std::complex_literals;
    ippl::initialize(argc, argv);{
        using scalar = float;
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);
        
        // get the gridsize from the user
        
        //using scalar_field = ippl::Field<scalar, Dim, mesh_t, centering_t>;
        using spinor_field = spinor_field_template<scalar>;
        constexpr scalar electron_mass = 0.5110;

        constexpr size_t extents = 64; 
        ippl::Vector<size_t, Dim> nr = {extents, extents, extents};
        
        scalar dx                        = scalar(1.0) / nr[0];
        scalar dy                        = scalar(1.0) / nr[1];
        scalar dz                        = scalar(1.0) / nr[2];
        scalar dt = std::min<scalar>({dx, dy, dz}) * 0.4;
        ippl::Vector<scalar, Dim> hr     = {dx, dy, dz};
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        ippl::e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }
        
        ippl::FieldLayout<Dim> layout(owned, decomp);
        mesh_t<scalar> mesh(owned, hr, origin);

        spinor_field field;
        spinor_field field_n_plus_one;
        field.initialize(mesh, layout, 0);
        field_n_plus_one.initialize(mesh, layout, 0);
        field = spinor<scalar>{0,0,0,0};
        field_n_plus_one = spinor<scalar>{0,0,0,0};
        Kokkos::View<spinor<scalar>***> field_view = field.getView();
        Kokkos::View<spinor<scalar>***> field_n_plus_one_view = field_n_plus_one.getView();
        field_view(extents / 2, extents / 2, extents / 2) = spinor<scalar>{1, 0, 1, 0};
        unsigned int steps = 0.5 / dt;
        std::cout << steps << " steps" << std::endl;

        for(unsigned step = 0;step < steps;step++){
            Kokkos::parallel_for(ippl::getRangePolicy(field.getView()),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                spinor<scalar> value_ijk = field_view(i, j, k);
                ippl::Vector<spinor<scalar>, 3> previouses;
                ippl::Vector<spinor<scalar>, 3> nexts;
                ippl::Vector<spinor<scalar>, 3> minus_one_half_fluxes;
                ippl::Vector<spinor<scalar>, 3> plus_one_half_fluxes;

                previouses[0] = get<0>(i, j, k) > 0 ? field_view(i - 1, j, k) : spinor<scalar>{0,0,0,0};
                previouses[1] = get<1>(i, j, k) > 0 ? field_view(i, j - 1, k) : spinor<scalar>{0,0,0,0};
                previouses[2] = get<2>(i, j, k) > 0 ? field_view(i, j, k - 1) : spinor<scalar>{0,0,0,0};

                nexts[0] = (get<0>(i, j, k) + 1 < field_view.extent(0)) ? field_view(i + 1, j, k) : spinor<scalar>{0,0,0,0};
                nexts[1] = (get<1>(i, j, k) + 1 < field_view.extent(1)) ? field_view(i, j + 1, k) : spinor<scalar>{0,0,0,0};
                nexts[2] = (get<2>(i, j, k) + 1 < field_view.extent(2)) ? field_view(i, j, k + 1) : spinor<scalar>{0,0,0,0};
                
                minus_one_half_fluxes[0] = lax_friedrichs_flux_d<scalar, 0>(previouses[0], value_ijk);
                plus_one_half_fluxes [0] = lax_friedrichs_flux_d<scalar, 0>(value_ijk, nexts[0]);

                minus_one_half_fluxes[1] = lax_friedrichs_flux_d<scalar, 1>(previouses[1], value_ijk);
                plus_one_half_fluxes [1] = lax_friedrichs_flux_d<scalar, 1>(value_ijk, nexts[1]);

                minus_one_half_fluxes[2] = lax_friedrichs_flux_d<scalar, 2>(previouses[2], value_ijk);
                plus_one_half_fluxes [2] = lax_friedrichs_flux_d<scalar, 2>(value_ijk, nexts[2]);
                
                spinor<scalar> rate_of_change = (plus_one_half_fluxes[0] - minus_one_half_fluxes[0]) / hr[0] +
                                                (plus_one_half_fluxes[1] - minus_one_half_fluxes[1]) / hr[1] +
                                                (plus_one_half_fluxes[2] - minus_one_half_fluxes[2]) / hr[2] +
                                                 mass_term(field_view(i, j, k)) * electron_mass;
                
                
                field_n_plus_one_view(i, j, k) = field_view(i, j, k) + rate_of_change * dt;
            });
            scalar_field_template<scalar> scalar_field = to_scalar_field(field);
            dumpVTK(scalar_field, extents, extents, extents, step, hr[0], hr[1], hr[2]);
            Kokkos::deep_copy(field_view, field_n_plus_one_view);
        }
        
    }
    ippl::finalize();
}
