// This program tests the FFTPoissonSolver class Biharmonic solver.
// We use a Maxwellian distribution and solve for the Rosenbluth potentials,
// and compare with the analytical solution to obtain the accuracy.
//
// The test runs for various gridsizes in order to study the convergence.
//
// Usage:
//   srun ./TestBiharmonic --info 10
// where the 3 numbers are the gridsize in each direction.

#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "FFTPoissonSolver.h"

int main(int argc, char *argv[]){

    Ippl ippl(argc, argv);
     
    Inform msg(argv[0]);
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    // number of interations and gridpoints to iterate over
    const int number = 6;
    std::array<int, number> N = {4,8,16,32,64,128};

    msg << "Spacing Error" << endl;

    // constants for the Maxwellian
    const double n = 1;
    const double vth = 2.0;
    const double vth2 = vth*vth;
    const double w2 = sqrt(2.0);
    double max = 5.0*vth;

    // algorithm = biharmonic solver
    std::string algorithm = "BIHARMONIC";

    for (int p = 0; p < number; ++p) {

        // domain
	    int pt = N[p];
	    ippl::Index I(pt);
	    ippl::NDIndex<3> owned(I, I, I);

	    // specifies decomposition; here all dimensions are parallel
	    ippl::e_dim_tag decomp[3];
	    for (unsigned int d = 0; d < 3; d++)
	        decomp[d] = ippl::PARALLEL;

	    // unit box
	    double dv = 2.0*max/pt;
	    ippl::Vector<double, 3> hv = {dv, dv, dv};
	    ippl::Vector<double, 3> vmin = {-max, -max, -max};
	    ippl::Vector<double, 3> vmax = {max, max, max};
	    ippl::Vector<double, 3> zero = {0.0, 0.0, 0.0};
	    ippl::UniformCartesian<double, 3> mesh(owned, hv, vmin);

	    // all parallel layout, standard domain, normal axis order
	    ippl::FieldLayout<3> layout(owned, decomp);

	    // define the R (rho) field
	    typedef ippl::Field<double, 3> field;
	    field fv, G_exact;
	    fv.initialize(mesh, layout);
	    G_exact.initialize(mesh, layout);
        
        // assign the exact field
        typename field::view_type view_fv = fv.getView();
        typename field::view_type view_G = G_exact.getView();
        const int nghost = fv.getNghost();
        const auto& lDom = layout.getLocalNDIndex();

        Kokkos::parallel_for("Assign fv and G_exact",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
	                                               {view_fv.extent(0) - nghost,
                                                    view_fv.extent(1) - nghost,
                                                    view_fv.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                
                //local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                const double vx = vmin[0] + (ig + 0.5)*hv[0];
                const double vy = vmin[1] + (jg + 0.5)*hv[1];
                const double vz = vmin[2] + (kg + 0.5)*hv[2];
                        
                const double vnorm2  = vx*vx+vy*vy+vz*vz;
                const double vnorm = sqrt(vnorm2); 
                const double ERF = std::erf(vnorm/(w2*vth));
                const double EXP = exp(-vnorm2/(2*vth2));
            
                view_fv(i,j,k) = n/pow(2*M_PI*vth2, 1.5)*EXP;
                view_G(i,j,k) = w2*n*vth*(EXP/sqrt(M_PI) + ERF*(vth/(w2*vnorm)+vnorm/(w2*vth)));
        });

        Kokkos::fence();

        // scale
        fv = -8.0 * M_PI * fv;
        
        // set the FFT parameters	
        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", false);  
        fftParams.add("use_pencils", true);  
        fftParams.add("use_gpu_aware", true);  
        fftParams.add("comm", ippl::a2av);  
        fftParams.add("r2c_direction", 0);  
	    
        // define an FFTPoissonSolver object
        mesh.setOrigin({0, 0, 0});
	    ippl::FFTPoissonSolver<ippl::Vector<double,3>, double, 3> FFTsolver(fv, fftParams, algorithm);
	
    	// solve the Poisson equation -> rho contains the solution (phi) now
	    FFTsolver.solve();
        mesh.setOrigin(vmin);

        // compute relative error norm for potential
	    fv = fv - G_exact;
	    double err = norm(fv)/norm(G_exact);
        
        msg << std::setprecision(16) << dv << " " << err << endl;
    }

    return 0;
}
