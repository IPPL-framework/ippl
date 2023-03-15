// This program tests the FDTDSolver class with a Gaussian source.
// The problem size is given by the user:
//   srun ./TestFDTD 64 64 64 --info 10

#include "Ippl.h"
#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include "FDTDSolver.h"
#include <cstdlib>

KOKKOS_INLINE_FUNCTION
double gaussian(double n, double dt, double sigma = 0.03, double mu = 0.1) {
    double r2 = (n*dt - mu)*(n*dt -mu);
    return exp(-r2/(sigma*sigma));
}

void dumpVTK(ippl::Field<ippl::Vector<double, 3>, 3>& E, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {


    typename ippl::Field<ippl::Vector<double, 3>, 3>::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << endl;
    vtkout << "ORIGIN "     << -dx  << " " << -dy  << " "  << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z)[0] << "\t"
                       << host_view(x,y,z)[1] << "\t"
                       << host_view(x,y,z)[2] << endl;
            }
        }
    }
}

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    Inform msg(argv[0]);
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    const unsigned int Dim = 3;

    // get the gridsize from the user
    ippl::Vector<int, Dim> nr = { 
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    // get the total simulation time from the user
    const unsigned int iterations = std::atof(argv[4]);

    // domain
    ippl::NDIndex<Dim> owned;
    for (unsigned i = 0; i< Dim; i++) {
        owned[i] = ippl::Index(nr[i]);
    }

    // specifies decomposition; here all dimensions are parallel
    ippl::e_dim_tag decomp[Dim];
    for (unsigned int d = 0; d < Dim; d++) {
        decomp[d] = ippl::PARALLEL;
    }    

    // unit box
    double dx = 1.0/nr[0];
    double dy = 1.0/nr[1];
    double dz = 1.0/nr[2];
    ippl::Vector<double, Dim> hr = {dx, dy, dz};
    ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
    ippl::UniformCartesian<double, Dim> mesh(owned, hr, origin);

    // CFL condition lambda = c*dt/h < 1/sqrt(d) = 0.57 for d = 3
    // we set a more conservative limit by choosing lambda = 0.5
    // we take h = minimum(dx, dy, dz)
    const double c = 1.0; //299792458.0;
    double dt = std::min({dx, dy, dz}) * 0.5 / c;

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<Dim> layout(owned, decomp);
	
    // define the R (rho) field
    typedef ippl::Field<double, Dim> Field_t;
    Field_t rho;
    rho.initialize(mesh, layout);
    rho = 0.0;

    // define the Vector field E (LHS)
    typedef ippl::Field<ippl::Vector<double, Dim>, Dim> VField_t;
    VField_t fieldE, fieldB;
    fieldE.initialize(mesh, layout);
    fieldB.initialize(mesh, layout);
    fieldE = 0.0;
    fieldB = 0.0;

    // define current = 0
    VField_t current;
    current.initialize(mesh, layout);
    current = 0.0;

    // define an FDTDSolver object
    ippl::FDTDSolver<double, Dim> solver(rho, current, fieldE, fieldB, dt);

    // add pulse at center of domain
    auto view_rho = rho.getView();
    const int nghost = rho.getNghost();
    auto ldom = layout.getLocalNDIndex();

    // time-loop
    for (unsigned int it = 0; it < iterations; ++it) {

        msg << "Timestep number = " << it << " , time = " << it*dt << endl;

        Kokkos::parallel_for("Assign gaussian source at center",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
                                                   {view_rho.extent(0) - nghost,
                                                    view_rho.extent(1) - nghost,
                                                    view_rho.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                if ((ig == nr[0]/2 - 1) && (jg == nr[1]/2 - 1) && (kg == nr[2]/2 - 1))
                    view_rho(i,j,k) = gaussian(it, dt);
        });
  
        solver.solve();

        dumpVTK(fieldE, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
    }

    return 0;
}
