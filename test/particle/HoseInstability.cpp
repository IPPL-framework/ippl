//////////////////////////////////////////////////////////////////////////////////
//
// HoseInstability.cpp
//
// This program seeks to simulate the Hose Instability for a coasting beam 
// (electrostatic case). The Poisson solver employed is an all open BCs 
// FFT-based Poisson solver which employs the Hockney trick.
//
// Usage:
//     srun ./TestHose 64 64 64 10000 300 Hockney 0 --info 10
//
// FFT parameters used: Pencils, point2point, no reordering
//
//////////////////////////////////////////////////////////////////////////////////

#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>
#include <chrono>

#include <random>
#include "Utility/IpplTimings.h"
#include "Solver/FFTPoissonSolver.h"

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::ParticleSpatialLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;

template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;
typedef ippl::FFTPoissonSolver<ippl::Vector<double,3>, double, Dim> Solver_t;

// some useful constants
double c = 299792458;

// function that allows to create a vtk file for Paraview to visualize rho
void dumpVTK(Field_t& rho, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();
    Kokkos::deep_copy(host_view, rho.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open new data file and write
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "PenningTrap" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN " << -2*dx << " " << -2*dy << " " << -2*dz << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;

    vtkout << "SCALARS Rho float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z) << std::endl;
            }
        }
    }

    // close file
    vtkout.close();
}

// define a class for charged particle bunch
template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    VField_t E_m;
    VField_t B_m;
    Field_t rho_m; // charge density
    Field_t temp_field; // charge density

    ippl::e_dim_tag decomp_m[Dim];

    Vector<int, Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::shared_ptr<Solver_t> solver_mp;
    std::string stype_m;

    double time_m;
   
    double rhoNorm_m;

    double gammaz;

    bool vtk;

    // add a particle ID to distinguish ions from electrons
    typename ippl::ParticleAttrib<double> id;

    // particle mass
    typename ippl::ParticleAttrib<double> m;

    // gamma
    typename ippl::ParticleAttrib<double> gamma;
    
    // charge
    typename ippl::ParticleAttrib<double> q;

    // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type V;

    // electric & magnetic fields at particle position    
    typename ippl::ParticleBase<PLayout>::particle_position_type E;
    typename ippl::ParticleBase<PLayout>::particle_position_type B;

    // constructors
    ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        this->addAttribute(q);
        this->addAttribute(V);
        this->addAttribute(E);
        this->addAttribute(B);
        this->addAttribute(id);
        this->addAttribute(m);
        this->addAttribute(gamma);
    }

    ChargedParticles(PLayout& pl, Vector_t nr, Vector_t hr, Vector_t rmin, Vector_t rmax,
                     ippl::e_dim_tag decomp[Dim], double Q, std::string type, double gammaz, bool vtk)
    : ippl::ParticleBase<PLayout>(pl), nr_m(nr), hr_m(hr), rmin_m(rmin), rmax_m(rmax), Q_m(Q), gammaz(gammaz), vtk(vtk)
    {
        this->addAttribute(q);
        this->addAttribute(V);
        this->addAttribute(E);
        this->addAttribute(B);
        this->addAttribute(id);
        this->addAttribute(m);
        this->addAttribute(gamma);

        setupBCs();

        stype_m = type;

        for (unsigned int i = 0; i < Dim; ++i)
            decomp_m[i] = decomp[i];
    }

    ~ChargedParticles(){ }

    void setupBCs() {
        // open BCs
        this->setParticleBC(ippl::BC::NO);
    }

    void initSolver() {
        Inform m("solver ");

        ippl::FFTParams fftParams;

        fftParams.setAllToAll( false );
        fftParams.setPencils( true );
        fftParams.setReorder( false );
        fftParams.setRCDirection( 0 );
   
        solver_mp = std::make_shared<Solver_t>(E_m, rho_m, fftParams, stype_m);

        solver_mp->setGradFD();
    }

    void gatherCIC() {

        // transform back the E-fields (Ez remains same)
        Vector_t scale = {gammaz, gammaz, 1.0/gammaz};

        auto view_fieldE = E_m.getView();
        auto view_fieldB = B_m.getView();
        auto nghost = E_m.getNghost();

        Kokkos::parallel_for("Scale",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {nghost, nghost, nghost},
                                 {view_fieldE.extent(0) - nghost,
                                  view_fieldE.extent(1) - nghost, 
                                  view_fieldE.extent(2) - nghost}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){
                view_fieldE(i,j,k)[0] *= scale[0];
                view_fieldE(i,j,k)[1] *= scale[1];
                view_fieldE(i,j,k)[2] *= scale[2];
        });

        // compute beta/c
        double betaC = std::sqrt(gammaz * gammaz - 1.0) / gammaz / c;

        Kokkos::parallel_for("Compute B-field",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {nghost, nghost, nghost},
                                 {view_fieldB.extent(0) - nghost,
                                  view_fieldB.extent(1) - nghost, 
                                  view_fieldB.extent(2) - nghost}),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){
                view_fieldB(i,j,k)[0] -= betaC * view_fieldE(i,j,k)[1];
                view_fieldB(i,j,k)[1] += betaC * view_fieldE(i,j,k)[0];
                view_fieldB(i,j,k)[2] = 0.0;
        });

        // gather
        gather(this->E, E_m, this->R);
        gather(this->B, B_m, this->R);
    }

    void scatterCIC(unsigned int totalP, int iteration) {     
         Inform m("scatter ");

         E_m = 0.0;
         rho_m = 0.0;

         scatter(q, rho_m, this->R);

         // check - charge ///////////////////////////////////////////////////////////////
         double Q_grid = rho_m.sum();
         unsigned int Total_particles = 0;
         unsigned int local_particles = this->getLocalNum();

         MPI_Reduce(&local_particles, &Total_particles, 1, 
                       MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());

         if(Ippl::Comm->rank() == 0) {
             if(Total_particles != totalP) {
                 std::cout << "Total particles in the sim. " << totalP 
                           << " " << "after update: " 
                           << Total_particles << std::endl;
                 std::cout << "Total particles not matched in iteration: " 
                           << iteration << std::endl;
                 exit(1);
             }
         }

         m << "Rel. error in charge conservation = " << std::fabs((Q_m-Q_grid)/Q_m) << endl;

         //////////////////////////////////////////////////////////////////////////////////

         temp_field = rho_m;

         // charge density in real units 
         rho_m = rho_m / (hr_m[0] * hr_m[1] * hr_m[2]);

         if (vtk == true)
             dumpVTK(rho_m, nr_m[0], nr_m[1], nr_m[2], iteration, hr_m[0], hr_m[1], hr_m[2]);

         // Lorentz transformation in longitudinal direction
         Vector_t hr_scaled = {hr_m[0], hr_m[1], hr_m[2]*gammaz};
         (rho_m.get_mesh()).setMeshSpacing(hr_scaled);
         rho_m = rho_m / gammaz;

         solver_mp->solve();

         // transform back mesh spacing
         (rho_m.get_mesh()).setMeshSpacing(hr_m);

         double ke = 8.988e9; // coupling constant         
         rho_m = rho_m * ke;
         E_m = E_m * ke;
    }

    void dumpData(unsigned int totalP) {        

        // Kinetic energy
        double c_device = c;
        auto mView = this->m.getView();
        auto Vview = V.getView();
        double v = 0.0;
        Kokkos::parallel_reduce("Particle Energy", Vview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(Vview(i), Vview(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(v));
        double v2 = 0.0;
        MPI_Reduce(&v, &v2, 1, 
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        v2 = v2/totalP;
        double g = 1.0/sqrt(1.0 - (v2/pow(c_device,2)));
        v2 = 0.0;
        Kokkos::parallel_reduce("Particle Energy", Vview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = mView(i) * c_device * c_device * (g-1.0);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(v2));
        double kin = 0.0;
        MPI_Reduce(&v2, &kin, 1, 
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        // Potential energy
        temp_field = rho_m * temp_field;
        double pot = 0.5 * temp_field.sum();


        // Norm of E-field
        const int nghostE = E_m.getNghost();
        auto Eview = E_m.getView();
        Vector_t normE;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        for (unsigned d=0; d<Dim; ++d) {

            double temp = 0.0;                                                                                        
            Kokkos::parallel_reduce("Vector E reduce",                                                                       
                                mdrange_type({nghostE, nghostE, nghostE},                 
                                             {Eview.extent(0) - nghostE,            
                                              Eview.extent(1) - nghostE,            
                                              Eview.extent(2) - nghostE}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j,                           
                                              const size_t k, double& valL) 
                                {                                
                                    double myVal = pow(Eview(i, j, k)[d], 2);                                              
                                    valL += myVal;                                                                      
                                }, Kokkos::Sum<double>(temp));                                                     
            double globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());                                 
            normE[d] = sqrt(globaltemp);
        }


        // Norm of B-field
        const int nghostB = B_m.getNghost();
        auto Bview = B_m.getView();
        Vector_t normB;

        for (unsigned d=0; d<Dim; ++d) {

            double temp = 0.0;                                                                                        
            Kokkos::parallel_reduce("Vector B reduce",                                                                       
                                mdrange_type({nghostB, nghostB, nghostB},                 
                                             {Bview.extent(0) - nghostB,            
                                              Bview.extent(1) - nghostB,            
                                              Bview.extent(2) - nghostB}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j,                           
                                              const size_t k, double& valL) 
                                {                                
                                    double myVal = pow(Bview(i, j, k)[d], 2);                                              
                                    valL += myVal;                                                                      
                                }, Kokkos::Sum<double>(temp));                                                     
            double globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            normB[d] = sqrt(globaltemp);
        }

        // centroid
        double centroid_xi = 0.0;
        double centroid_yi = 0.0;
        double centroid_xb = 0.0;
        double centroid_yb = 0.0;
        double center_x = (rmax_m[0] - rmin_m[0])/2.0;
        double center_y = (rmax_m[1] - rmin_m[1])/2.0;

        auto Rview = this->R.getView();
        auto idView = this->id.getView();
        double temp = 0.0;

        Kokkos::parallel_reduce("Particle positions", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    if ((idView(i) == 1.0)) {
                                        double myVal = Rview(i)[0] - center_x;
                                        valL += myVal;
                                    }
                                }, Kokkos::Sum<double>(temp));                                                                                  
        MPI_Reduce(&temp, &centroid_xi, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        temp = 0.0;
        Kokkos::parallel_reduce("Particle positions", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    if ((idView(i) == 1.0)) {
                                        double myVal = Rview(i)[1] - center_y;
                                        valL += myVal;
                                    }
                                }, Kokkos::Sum<double>(temp));
        MPI_Reduce(&temp, &centroid_yi, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        centroid_xi = centroid_xi / (0.25 * totalP);
        centroid_yi = centroid_yi / (0.25 * totalP);

        temp = 0.0;
        Kokkos::parallel_reduce("Particle positions", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    if ((idView(i) == -1.0)) {
                                        double myVal = Rview(i)[0] - center_x;
                                        valL += myVal;
                                    }
                                }, Kokkos::Sum<double>(temp));
        MPI_Reduce(&temp, &centroid_xb, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        temp = 0.0;
        Kokkos::parallel_reduce("Particle positions", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    if ((idView(i) == -1.0)) { 
                                        double myVal = Rview(i)[1] - center_y;
                                        valL += myVal;
                                    }
                                }, Kokkos::Sum<double>(temp));
        MPI_Reduce(&temp, &centroid_yb, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        centroid_xb = centroid_xb / (0.75*totalP);
        centroid_yb = centroid_yb / (0.75*totalP);

        // rms x,y,z
        Vector_t rms_r; // positions
        Vector_t rms_p; // momentum
        Vector_t av_r; // average position
        Vector_t av_p; // average momentum

        for (unsigned int d=0; d<Dim; ++d) {

            // average position
            double temp = 0.0;
            Kokkos::parallel_reduce("average", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = Rview(i)[d];
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
            double globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            av_r[d] = globaltemp/totalP;

            // average momentum
            temp = 0.0;
            Kokkos::parallel_reduce("average", Vview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = Vview(i)[d] * mView(i) * gammaz;
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
            globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            av_p[d] = globaltemp/totalP;


            // RMS
            temp = 0.0;
            Kokkos::parallel_reduce("RMS", Rview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                        double x = Rview(i)[d] - av_r[d];
                                        double myVal = pow(x,2);
                                        valL += myVal;
                                }, Kokkos::Sum<double>(temp));
            globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            rms_r[d] = sqrt(globaltemp/totalP);


            temp = 0.0;
            Kokkos::parallel_reduce("RMS", Vview.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                        double p = (Vview(i)[d] * mView(i) * gammaz) - av_p[d];
                                        double myVal = pow(p,2);
                                        valL += myVal;
                                }, Kokkos::Sum<double>(temp));
            globaltemp = 0.0;                                                                                  
            MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
            rms_p[d] = sqrt(globaltemp/totalP);
        }

        // print position of 1st particle;
        double x = Rview(0)[0];
        double y = Rview(0)[1];
        double z = Rview(0)[2];

        // print
        if(Ippl::Comm->rank() == 0) {
            std::ofstream csvout;
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << Ippl::Comm->size();
            fname << ".csv";
            csvout.open(fname.str().c_str(), std::ios::out | std::ofstream::app);

            if(time_m == 0.0) {
                csvout << "time E_kin E_pot Ex_norm Ey_norm Ez_norm Bx_norm By_norm Bz_norm "
                       << "Ion_centroid_x Ion_centroid_y Beam_centroid_x Beam_centroid_y "
                       << "Rms_x Rms_y Rms_z Rms_px Rms_py Rms_pz x y z" << std::endl;
            }

            csvout << time_m << " "
                   << kin << " "
                   << pot << " "
                   << normE[0] << " "
                   << normE[1] << " "
                   << normE[2] << " "
                   << normB[0] << " "
                   << normB[1] << " "
                   << normB[2] << " "
                   << centroid_xi << " "
                   << centroid_yi << " "
                   << centroid_xb << " "
                   << centroid_yb << " "
                   << rms_r[0] << " "
                   << rms_r[1] << " "
                   << rms_r[2] << " "
                   << rms_p[0] << " "
                   << rms_p[1] << " "
                   << rms_p[2] << " "
                   << x << " "
                   << y << " "
                   << z <<
                   std::endl;

            csvout.close();
        }
        Ippl::Comm->barrier();
    }
};


int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);
    
    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
    IpplTimings::startTimer(mainTimer);                                                    
    
    // get the problem size
    ippl::Vector<int,Dim> nr = {std::atoi(argv[1]),
                                std::atoi(argv[2]),
                                std::atoi(argv[3])};
    
    // get the total number of particles in the beam and the timesteps
    const unsigned long long int totalP = std::atoi(argv[4]);
    const unsigned int nt = std::atoi(argv[5]);

    // get the algorithm (Hockney or Vico)
    std::string type = argv[6];

    // print VTK or not
    bool vtk;
    if (std::atoi(argv[7]) == 0){
        vtk = false;
    } else {
        vtk = true;
    }

    // print out info                                  
    msg << "Hose Instability " << "Nt = " << nt << " Np = " << totalP << " Grid = " << nr << " Method = " << type << endl;

    // create domain and set parallel decomposition
    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }
    
    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }

    // gammaz (longitudinal velocity of coasting beam
    double gammaz = 25.0;
    // compute vz = beta*c
    double vz = c * std::sqrt(gammaz * gammaz - 1.0) / gammaz;

    // create particle mesh, field layout, and particle layout
    Vector_t rmin(0.0);
    Vector_t rmax = {0.1, 0.1, 0.4}; // m
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp);
    PLayout_t PL(FL, mesh);

    // get smallest mesh spacing
    double deltaX = dx;
    for (size_t i = 0; i < Dim; ++i) {
        if (hr[i] < deltaX)
            deltaX = hr[i];
    } 
    
    // set timestep according to CFL condition
    const double dt = 4.0e-10;
    msg << "size of timestep = " << dt << endl;

    // create particle bunch
    using bunch_type = ChargedParticles<PLayout_t>;
    std::unique_ptr<bunch_type> P;

    // set the total charge (two species) and the external B-field
    double Q_i = (+1.0e-6); // C - ion channel
    double Q_b = (-4.0e-3); // C - electron beam
    double Bext = 0.083; //T

    // set the elementary charge
    double q_e = 1.602176634e-19; // C

    // set the real masses
    double m_i = 2.988e-26; // kg - ion mass (H2O)
    double m_e = 9.1e-31; // kg - electron mass

    // compute mass for correct charge to mass ratio
    m_i = Q_i/(0.25*totalP)/(q_e/m_i);
    m_e = -Q_b/(0.75*totalP)/(q_e/m_e);

    // construct the particle bunch with the parameters
    P = std::make_unique<bunch_type>(PL, nr, hr, rmin, rmax, decomp, Q_i+Q_b, type, gammaz, vtk);

    // check if particles can be divided equally between ranks
    unsigned long long int nloc = totalP / Ippl::Comm->size();

    int rest = (int) (totalP - nloc * Ippl::Comm->size());    
    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    // start of particle creation
    P->create(nloc);

    // set the initial distribution for the bunches
    Vector_t length = rmax - rmin;
    
    std::mt19937_64 eng[4*Dim];
    for (unsigned i = 0; i < 4*Dim; ++i) {
        eng[i].seed(42 + i * Dim);
        eng[i].discard( nloc * Ippl::Comm->rank());
    }
    
    std::vector<double> states(2*Dim);   
    double u1,u2; //,r,theta;
    
    std::uniform_real_distribution<double> dist_uniform_beam (hr[2] + rmin[2], hr[2] + (5e-2));
    std::uniform_real_distribution<double> dist_uniform (hr[2] + rmin[2], rmax[2] - hr[2]);
    std::uniform_real_distribution<double> dist_uniform_norm (0.0, 1.0);

    // get positions (R) and momenta (P)
    typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
    typename bunch_type::particle_position_type::HostMirror V_host = P->V.getHostMirror();
    typename ParticleAttrib<double>::HostMirror   q_host = P->q.getHostMirror();
    typename ParticleAttrib<double>::HostMirror   id_host = P->id.getHostMirror();
    typename ParticleAttrib<double>::HostMirror   m_host = P->m.getHostMirror();
    typename ParticleAttrib<double>::HostMirror   gamma_host = P->gamma.getHostMirror();
 
    double sum_coord = 0.0;
    
    // ion channel (background plasma)
    for (unsigned long long int i = 0; i < nloc/4; i++) {
       
        // x,y gaussian 
        u1 = dist_uniform_norm(eng[0]);
        u2 = dist_uniform_norm(eng[1]);
        
        states[0] = (1e-3) * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + (0.5*length[0]);

        u1 = dist_uniform_norm(eng[2]);
        u2 = dist_uniform_norm(eng[3]);
        states[1] = (1e-3) * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + (0.5*length[1]);

        // uniform distribution [0, L_z] in z-direction
        states[2] =  dist_uniform(eng[3]);

        // all velocities are 0 for the plasma (ions)
        states[3] = 0.0;
        states[4] = 0.0;
        states[5] = 0.0;

        for (unsigned d = 0; d<Dim; d++) {
        
            // set position
            R_host(i)[d] = states[d];
            sum_coord += R_host(i)[d];

            // set velocity
            V_host(i)[d] = states[Dim + d];

        }
        // set the charge
        q_host(i) = Q_i/(0.25*totalP);

        // set the id
        id_host(i) = 1.0; // 1 for ions

        // set the mass
        m_host(i) = m_i;

        // set gamma factor for ions (1 because at rest)
        gamma_host(i) = 1.0;
    }
    

    // electron beam
    for (unsigned long long int i = nloc/4; i < nloc; i++) {

        // gaussian in x,y,z
        u1 = dist_uniform_norm(eng[0]);
        u2 = dist_uniform_norm(eng[1]);
        states[0] = (1e-3) * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + (0.5*length[0]);

        u1 = dist_uniform_norm(eng[2]);
        u2 = dist_uniform_norm(eng[3]);
        states[1] = (1e-3) * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + (0.5*length[1]);
        
        // add sinusoidal perturbation to both x and y
        double f = 21.6e6; // frequency
        double A = 1e-5; // amplitude
        states[0] = states[0] * (1 + A*sin(2*pi*f*states[0]));
        states[1] = states[1] * (1 + A*cos(2*pi*f*states[1]));

        // z
        states[2] =  dist_uniform_beam(eng[4]);

        // gaussian in vx, vy; mu = 0, std = 1
        u1 = dist_uniform_norm(eng[6]);
        u2 = dist_uniform_norm(eng[7]);
        
        states[3] = 1.0 * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + 0.0; 
        
        u1 = dist_uniform_norm(eng[8]);
        u2 = dist_uniform_norm(eng[9]);

        states[4] = 1.0 * (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2)) + 0.0; 

        // constant vz
        states[5] = vz;

        for (unsigned d = 0; d<Dim; d++) {

            // set position
            R_host(i)[d] = states[d];
            sum_coord += R_host(i)[d];

            // set velocity
            V_host(i)[d] = states[Dim + d];

        }
        // set the charge
        q_host(i) = Q_b/(0.75*totalP);

        // set the id
        id_host(i) = -1.0; // -1 for electrons

        // set the mass
        m_host(i) = m_e;

        // set gamma factor for electrons
        gamma_host(i) = gammaz;
    }


    // check that particles are created at same position each time
    double global_sum_coord = 0.0;
    MPI_Reduce(&sum_coord, &global_sum_coord, 1, 
               MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

    if(Ippl::Comm->rank() == 0) {
        std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
    }

    // copy back initialized position and momenta
    Kokkos::deep_copy(P->R.getView(), R_host);
    Kokkos::deep_copy(P->V.getView(), V_host);
    Kokkos::deep_copy(P->q.getView(), q_host);
    Kokkos::deep_copy(P->id.getView(), id_host);
    Kokkos::deep_copy(P->m.getView(), m_host);
    Kokkos::deep_copy(P->gamma.getView(), gamma_host);

    // end of particle creation

    // initialize fields
    P->E_m.initialize(mesh, FL);
    P->B_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);
    P->temp_field.initialize(mesh, FL);

    // first update
    
    bunch_type bunchBuffer(PL);
    bunchBuffer.create(100);
    PL.update(*P, bunchBuffer);

    msg << "particles created and initial conditions assigned " << endl;

    // initialize solver + first scatter, solve, gather
    P->initSolver();
    P->time_m = 0.0;
    P->scatterCIC(totalP, 0);
    P->gatherCIC();
    P->dumpData(totalP);

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration

        // kick
        auto Rview = P->R.getView();
        auto Vview = P->V.getView();
        auto Eview = P->E.getView();
        auto Bview = P->B.getView();
        auto q = P->q.getView();
        auto mass = P->m.getView();
        auto gamma = P->gamma.getView();

        Kokkos::parallel_for("Kick1", Rview.extent(0),
                              KOKKOS_LAMBDA(const size_t j){

                Vector_t Bfield = Bview(j);
                Bfield[2] = Bfield[2] + Bext;

                Vview(j)[0] += q(j) * 0.5 * dt * ((Eview(j)[0]) + (Vview(j)[1] * Bfield[2]) 
                                        - (Vview(j)[2] * Bfield[1])) / (mass(j) * gamma(j));
                Vview(j)[1] += q(j) * 0.5 * dt * ((Eview(j)[1]) + (Vview(j)[2] * Bfield[0])
                                        - (Vview(j)[0] * Bfield[2])) / (mass(j) * gamma(j));
                Vview(j)[2] += q(j) * 0.5 * dt * ((Eview(j)[2]) + (Vview(j)[0] * Bfield[1])
                                        - (Vview(j)[1] * Bfield[0])) / (mass(j) * gamma(j));
                
        });


        // drift
        P->R = P->R + dt * P->V;

        // update
        PL.update(*P, bunchBuffer);

        // scatter
        P->scatterCIC(totalP, it+1);

        // gather
        P->gatherCIC();

        // kick
        auto R2view = P->R.getView();
        auto V2view = P->V.getView();
        auto E2view = P->E.getView();
        auto B2view = P->B.getView();
        auto q2 = P->q.getView();
        auto mass2 = P->m.getView();
        auto gamma2 = P->gamma.getView();

        Kokkos::parallel_for("Kick2", R2view.extent(0),
                              KOKKOS_LAMBDA(const size_t j){

                Vector_t Bfield = B2view(j);
                Bfield[2] = Bfield[2] + Bext;

                V2view(j)[0] += q2(j) * 0.5 * dt * ((E2view(j)[0]) + (V2view(j)[1] * Bfield[2]) 
                                         - (V2view(j)[2] * Bfield[1])) / (mass2(j) * gamma2(j));
                V2view(j)[1] += q2(j) * 0.5 * dt * ((E2view(j)[1]) + (V2view(j)[2] * Bfield[0])
                                         - (V2view(j)[0] * Bfield[2])) / (mass2(j) * gamma2(j));
                V2view(j)[2] += q2(j) * 0.5 * dt * ((E2view(j)[2]) + (V2view(j)[0] * Bfield[1])
                                         - (V2view(j)[1] * Bfield[0])) / (mass2(j) * gamma2(j));
                
        });

        // increment time
        P->time_m += dt;
        P->dumpData(totalP);

        msg << "Finished iteration: " << it << " time: " << P->time_m << endl;
    }

    // end
    msg << "Hose Instability: End." << endl;

    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}

 
