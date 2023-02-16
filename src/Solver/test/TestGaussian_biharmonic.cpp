// This program tests the FFTPoissonSolver class with a Gaussian source.
// Different problem sizes are used for the purpose of convergence tests.
// The algorithm used is chosen by the user:
//     srun ./TestGaussian_convergence HOCKNEY --info 10 
// OR  srun ./TestGaussian_convergence VICO --info 10

#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "FFTPoissonSolver.h"

KOKKOS_INLINE_FUNCTION
double gaussian(double x, double y, double z, double sigma = 0.05, double mu = 0.5) {

    double pi = std::acos(-1.0);
    double prefactor = (1/std::sqrt(2*2*2*pi*pi*pi))*(1/(sigma*sigma*sigma));
    double r2 = (x-mu)*(x-mu) + (y-mu)*(y-mu) + (z-mu)*(z-mu);

    return -prefactor * exp(-r2/(2*sigma*sigma));
}

KOKKOS_INLINE_FUNCTION
double exact_fct(double x, double y, double z, double sigma = 0.05, double mu = 0.5) {

    double pi = std::acos(-1.0);
    double r = std::sqrt((x-mu)*(x-mu) + (y-mu)*(y-mu) + (z-mu)*(z-mu));
    double r2 = (x-mu)*(x-mu) + (y-mu)*(y-mu) + (z-mu)*(z-mu);

    return (1/(8.0*pi)) * (sigma*std::sqrt(2.0/pi)*exp(-r2/(2*sigma*sigma)) + std::erf(r/(std::sqrt(2.0)*sigma))*(r + (sigma*sigma/r)));
}

// Define vtk dump function for plotting the fields
void dumpVTK(std::string path, ippl::Field<double,3>& rho, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {
    
    typename ippl::Field<double,3>::view_type::host_mirror_type host_view = rho.getHostMirror();
    Kokkos::deep_copy(host_view, rho.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << path;
    fname << "/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    if (!vtkout)
    {
        std::cout <<"couldn't open" << std::endl;
    }
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "GaussianSource" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+1 << " " << ny+1 << " " << nz+1 << std::endl;
    vtkout << "ORIGIN " << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx)*(ny)*(nz) << std::endl;
    
    vtkout << "SCALARS Phi float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=1; z<nz+1; z++) {
        for (int y=1; y<ny+1; y++) {
            for (int x=1; x<nx+1; x++) {
                vtkout << host_view(x,y,z) << std::endl;
            }
        }
    }
    
    // close the output file for this iteration:
    vtkout.close();
}

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);
    Inform msg("");
    Inform msg2all("", INFORM_ALL_NODES);
    
    std::string algorithm = "BIHARMONIC";

    // start a timer to time the FFT Poisson solver
    static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
    IpplTimings::startTimer(allTimer);

    // number of interations
    const int n = 6;

    // number of gridpoints to iterate over   
    //std::array<int, n> N = {std::atoi(argv[1])};
    std::array<int, n> N = {4,8,16,32,64,128};

    msg << "Spacing Error" << endl;

    for (int p = 0; p < n; ++p) {

    // domain
	int pt = N[p];
	ippl::Index I(pt);
	ippl::NDIndex<3> owned(I, I, I);

	// specifies decomposition; here all dimensions are parallel
	ippl::e_dim_tag decomp[3];
	for (unsigned int d = 0; d < 3; d++)
	    decomp[d] = ippl::PARALLEL;

	// unit box
	double dx = 1.0/pt;
	ippl::Vector<double, 3> hx = {dx, dx, dx};
	ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
	ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

	// all parallel layout, standard domain, normal axis order
	ippl::FieldLayout<3> layout(owned, decomp);
		
	// define the R (rho) field
	typedef ippl::Field<double, 3> field;
	field rho;
	rho.initialize(mesh, layout);
    
    // define the exact solution field
    field exact;
    exact.initialize(mesh, layout);

	// assign the rho field with a gaussian
	typename field::view_type view_rho = rho.getView();
	const int nghost = rho.getNghost();
	const auto& ldom = layout.getLocalNDIndex();

	Kokkos::parallel_for("Assign rho field",
	                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
	                                                            {view_rho.extent(0) - nghost,
								                                 view_rho.extent(1) - nghost,
								                                 view_rho.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                // go from local to global indices
			    const int ig = i + ldom[0].first() - nghost;
				const int jg = j + ldom[1].first() - nghost;
				const int kg = k + ldom[2].first() - nghost;
								
				// define the physical points (cell-centered)
				double x = (ig + 0.5) * hx[0] + origin[0];
				double y = (jg + 0.5) * hx[1] + origin[1];
				double z = (kg + 0.5) * hx[2] + origin[2];

				view_rho(i, j, k) = gaussian(x, y, z);
	});

    // debug
    //std::cout << "exact source:" << std::endl;
    //rho.write();

    // assign the exact field with its values (erf function)
    typename field::view_type view_exact = exact.getView();
        
    Kokkos::parallel_for("Assign exact field",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
                                                                {view_exact.extent(0) - nghost,
                                                                     view_exact.extent(1) - nghost,
                                                                     view_exact.extent(2) - nghost}),
                 KOKKOS_LAMBDA(const int i, const int j, const int k){
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];

                    view_exact(i, j, k) = exact_fct(x,y,z);
        });

    // debug
    //std::cout << "exact potential:" << std::endl;
    //exact.write();
        
    // set the FFT parameters	
    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", false);  
    fftParams.add("use_pencils", true);  
    fftParams.add("use_gpu_aware", true);  
    fftParams.add("comm", ippl::a2av);  
    fftParams.add("r2c_direction", 0); 

	// define an FFTPoissonSolver object
	ippl::FFTPoissonSolver<ippl::Vector<double,3>, double, 3> FFTsolver(rho, fftParams, algorithm);
	
	// solve the Poisson equation -> rho contains the solution (phi) now
	FFTsolver.solve();

    // debug
    //std::cout << "solution:" << std::endl;
    //rho.write();

    // compute relative error norm for potential
	rho = rho - exact;
	double err = norm(rho)/norm(exact);
        
    msg << std::setprecision(16) << dx << " " << err << endl;

    }
    
    // stop the timer   
    IpplTimings::stopTimer(allTimer);
    IpplTimings::print(std::string("timing.dat"));
    
    return 0;
}
