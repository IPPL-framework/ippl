// Penning Trap
//
//   Usage:
//     srun ./PenningTrap 128 128 128 10000 300 FFT Gaussian --info 10
//     srun ./PenningTrap 128 128 128 10000 300 FFT Uniform --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan, 
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>
#include <chrono>

#include <random>
#include "Utility/IpplTimings.h"
#include "Solver/FFTPeriodicPoissonSolver.h"

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
typedef ippl::FFTPeriodicPoissonSolver<VField_t, Field_t,double,Dim> Solver_t;

double pi = acos(-1.0);

void dumpVTK(VField_t& E, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {


    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    Kokkos::deep_copy(host_view, E.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "PenningTrap" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN "     << -2*dx  << " " << -2*dy  << " "  << -2*dz << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;

    vtkout << "VECTORS E-Field float" << std::endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z)[0] << "\t"
                       << host_view(x,y,z)[1] << "\t"
                       << host_view(x,y,z)[2] << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}


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

    //double vol = dx*dy*dz;

    // open a new data file for this iteration
    // and start with header
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


    // close the output file for this iteration:
    vtkout.close();
}



template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    VField_t E_m;
    Field_t rho_m;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::string stype_m;

    std::shared_ptr<Solver_t> solver_mp;

    double time_m;

    double rhoNorm_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position


    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the update function invokes this
    */
    ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }
    
    ChargedParticles(PLayout& pl,
                     Vector_t hr, 
                     Vector_t rmin, 
                     Vector_t rmax, 
                     ippl::e_dim_tag decomp[Dim],
                     double Q)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    ~ChargedParticles(){ }

    void setupBCs() {
        setBCAllPeriodic();
    }


    void gatherStatistics(unsigned int totalP, int iteration) {
        
        std::cout << "Rank " << Ippl::Comm->rank() << " has " 
                  << (double)this->getLocalNum()/totalP*100.0 
                  << "percent of the total particles " << std::endl;
    }
    
    void gatherCIC() {

        gather(this->E, E_m, this->R);

    }

    void scatterCIC(unsigned int totalP, int iteration, Vector_t& hrField) {
         
         
         Inform m("scatter ");
         
         rho_m = 0.0;
         scatter(q, rho_m, this->R);
         
         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");           
         IpplTimings::startTimer(sumTimer);                                                    
         double Q_grid = rho_m.sum();
        
         unsigned int Total_particles = 0;
         unsigned int local_particles = this->getLocalNum();

         MPI_Reduce(&local_particles, &Total_particles, 1, 
                       MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());

         double rel_error = std::fabs((Q_m-Q_grid)/Q_m);
         m << "Rel. error in charge conservation = " << rel_error << endl;

         if(Ippl::Comm->rank() == 0) {
             if((Total_particles != totalP) || (rel_error > 1e-10)) {
             //if((Total_particles != totalP)) {
                 std::cout << "Total particles in the sim. " << totalP 
                           << " " << "after update: " 
                           << Total_particles << std::endl;
                 std::cout << "Total particles not matched in iteration: " 
                           << iteration << std::endl;
                 std::cout << "Rel. error in charge conservation: " 
                           << rel_error << std::endl;
                 exit(1);
             }
         }

         rho_m = rho_m / (hrField[0] * hrField[1] * hrField[2]);

         const int nghostRho = rho_m.getNghost();
         auto Rhoview = rho_m.getView();
         using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

         double temp = 0.0;                                                                                        
         Kokkos::parallel_reduce("Rho reduce",                                                                       
                                mdrange_type({nghostRho, nghostRho, nghostRho},                 
                                             {Rhoview.extent(0) - nghostRho,            
                                              Rhoview.extent(1) - nghostRho,            
                                              Rhoview.extent(2) - nghostRho}),          
                                KOKKOS_LAMBDA(const size_t i, const size_t j,                           
                                              const size_t k, double& valL) 
                                {                                
                                    double myVal = pow(Rhoview(i, j, k), 2);                                              
                                    valL += myVal;                                                                      
                                }, Kokkos::Sum<double>(temp));                                                     
         double globaltemp = 0.0;                                                                                  
         MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());                                 
         rhoNorm_m = sqrt(globaltemp);
         IpplTimings::stopTimer(sumTimer);
         
         //dumpVTK(rho_m,nr_m[0],nr_m[1],nr_m[2],iteration,hrField[0],hrField[1],hrField[2]);

         //rho = rho_e - rho_i
         rho_m = rho_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    }
    
    void initSolver() {

        Inform m("solver ");
        if(stype_m == "FFT")
            initFFTSolver();
        else
            m << "No solver matches the argument" << endl;

    }

    void initFFTSolver() {
        ippl::SolverParams sp;
        sp.add<int>("output_type",1);
        
        ippl::FFTParams fftParams;

        fftParams.setAllToAll( false );
        fftParams.setPencils( true );
        fftParams.setReorder( false );
        fftParams.setRCDirection( 0 );

        solver_mp = std::make_shared<Solver_t>(fftParams);

        solver_mp->setParameters(sp);
        
        solver_mp->setRhs(&rho_m);
        
        solver_mp->setLhs(&E_m);
    }



     void dumpData() {
        
        auto Pview = P.getView();
        
        double Energy = 0.0;

        Kokkos::parallel_reduce("Particle Energy", this->getLocalNum(),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(Pview(i), Pview(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;
        
        MPI_Reduce(&Energy, &gEnergy, 1, 
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());


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
                csvout << "time, Kinetic energy, Rho_norm2, Ex_norm2, Ey_norm2, Ez_norm2" << std::endl;
            }

            csvout << time_m << " "
                   << gEnergy << " "
                   << rhoNorm_m << " "
                   << normE[0] << " "
                   << normE[1] << " "
                   << normE[2] << std::endl;

            csvout.close();
        }
        
        Ippl::Comm->barrier();

     }

private:
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }

};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    //Inform msg(argv[0]);
    Inform msg("PenningTrap");
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(2);


    auto start = std::chrono::high_resolution_clock::now();
    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
    IpplTimings::startTimer(mainTimer);                                                    
    const unsigned long long int totalP = std::atol(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    
    msg << "Penning Trap "
        << endl
        << "nt " << nt << " Np= "
        << totalP << " grid = " << nr
        << endl;


    using bunch_type = ChargedParticles<PLayout_t>;

    std::unique_ptr<bunch_type>  P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }
    
    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t rmin(0.0);
    Vector_t rmax(20.0);
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};
    const double dt = 0.05;//size of timestep

    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, true);
    PLayout_t PL(FL, mesh);


    double Q = -1562.5;
    double Bext = 5.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);

    std::string dist;
    dist = argv[7];

    P->nr_m = nr;
    unsigned long long int nloc = totalP / Ippl::Comm->size();

    int rest = (int) (totalP - nloc * Ippl::Comm->size());
    
    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    //if ( rest > 0 ) {
    //    msg << "Total particles are not an exact multiple of ranks" << endl;
    //    exit(1);
    //}


    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");           
    IpplTimings::startTimer(particleCreation);                                                    

    Vector_t length = rmax - rmin;
    
    std::vector<double> mu(2*Dim);
    std::vector<double> sd(2*Dim);
    std::vector<double> states(2*Dim);
   

    for (unsigned d = 0; d<Dim; d++) {
        mu[d] = length[d]/2;
        mu[Dim + d] = 0.0;
        sd[Dim + d] = 1.0;
    }
    sd[0] = 0.15*length[0];
    sd[1] = 0.05*length[1];
    sd[2] = 0.20*length[2];

    unsigned int Total_particles = 0;
    if(dist == "Uniform") {

        std::function<double(double& x,
                             double& y,
                             double& z)> func;
        
        func = [&](double& x,
                   double& y,
                   double& z)
        {
            double val_conf =  std::pow((x - mu[0])/sd[0], 2) +
                               std::pow((y - mu[1])/sd[1], 2) +
                               std::pow((z - mu[2])/sd[2], 2);
            
            return std::exp(-0.5 * val_conf);
        };
        
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        std::vector<double> Rmin(Dim), Rmax(Dim);
        for (unsigned d = 0; d <Dim; ++d) {
            Rmin[d] = origin[d] + lDom[d].first() * hr[d];
            Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
        }
        std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
        std::uniform_real_distribution<double> distribution_x(Rmin[0], Rmax[0]);
        std::uniform_real_distribution<double> distribution_y(Rmin[1], Rmax[1]);
        std::uniform_real_distribution<double> distribution_z(Rmin[2], Rmax[2]);
        
        std::mt19937_64 eng[3*Dim];
        for (unsigned i = 0; i < 3*Dim; ++i) {
            eng[i].seed(42 + i * Dim);
            eng[i].discard( nloc * Ippl::Comm->rank());
        }
        

        double sum_f = 0.0;
        std::size_t ip = 0;
        double sum_coord=0.0;
        P->create(nloc);
        typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
        typename bunch_type::particle_position_type::HostMirror P_host = P->P.getHostMirror();
        typename bunch_type::particle_charge_type::HostMirror q_host = P->q.getHostMirror();
        for (unsigned long long int i = 0; i< nloc; i++) {

            states[0] = distribution_x(eng[0]);
            states[1] = distribution_y(eng[1]);
            states[2] = distribution_z(eng[2]);
            
            for (unsigned istate = 0; istate < Dim; ++istate) {
                double u1 = dist_uniform(eng[istate*2 + Dim]);
                double u2 = dist_uniform(eng[istate*2+1 + Dim]);
                states[istate + Dim] = fabs(sd[istate + Dim] * (std::sqrt(-2.0 * std::log(u1)) * 
                                 std::cos(2.0 * pi * u2)) + mu[istate + Dim]); 
            }

            double f = func(states[0], states[1], states[2]);
            f = f * states[3] * states[4] * states[5];

            //if( f > 1e-9 ) {
                for (unsigned d = 0; d<Dim; d++) {
                    R_host(ip)[d] =  states[d];
                    P_host(ip)[d] =  states[Dim + d];
                    sum_coord += R_host(ip)[d];
                }
                sum_f += f; 
                q_host(ip) = f;                    
                ++ip;
            //}
        }
        double Total_sum = 0.0;
        MPI_Allreduce(&sum_f, &Total_sum, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        double global_sum_coord = 0.0;
        MPI_Reduce(&sum_coord, &global_sum_coord, 1, 
                   MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        if(Ippl::Comm->rank() == 0) {
            std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        }
        //P->setLocalNum(ip);
        Kokkos::deep_copy(P->R.getView(), R_host);
        Kokkos::deep_copy(P->P.getView(), P_host);
        Kokkos::deep_copy(P->q.getView(), q_host);

        P->q = P->q * (P->Q_m/Total_sum);


        
        unsigned int local_particles = P->getLocalNum();
        MPI_Reduce(&local_particles, &Total_particles, 1, 
                   MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());
        msg << "#particles: " << Total_particles << endl;
        double PPC = Total_particles/((double)(nr[0] * nr[1] * nr[2]));
        msg << "#PPC: " << PPC << endl;

    }
    else if(dist == "Gaussian") {
    
        P->create(nloc);
        std::mt19937_64 eng[4*Dim];
        for (unsigned i = 0; i < 4*Dim; ++i) {
            eng[i].seed(42 + i * Dim);
            eng[i].discard( nloc * Ippl::Comm->rank());
        }
        
        
        std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

        typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
        typename bunch_type::particle_position_type::HostMirror P_host = P->P.getHostMirror();

        double sum_coord=0.0;
        for (unsigned long long int i = 0; i< nloc; i++) {
            for (unsigned istate = 0; istate < 2*Dim; ++istate) {
                double u1 = dist_uniform(eng[istate*2]);
                double u2 = dist_uniform(eng[istate*2+1]);
                states[istate] = sd[istate] * (std::sqrt(-2.0 * std::log(u1)) * 
                                 std::cos(2.0 * pi * u2)) + mu[istate]; 
            }    
            for (unsigned d = 0; d<Dim; d++) {
                R_host(i)[d] =  std::fabs(std::fmod(states[d],length[d]));
                sum_coord += R_host(i)[d];
                P_host(i)[d] = states[Dim + d];
            }
        }
        ///Just to check are we getting the same particle distribution for 
        //different no. of processors
        double global_sum_coord = 0.0;
        MPI_Reduce(&sum_coord, &global_sum_coord, 1, 
                   MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        if(Ippl::Comm->rank() == 0) {
            std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        }

        Kokkos::deep_copy(P->R.getView(), R_host);
        Kokkos::deep_copy(P->P.getView(), P_host);
        P->q = P->Q_m/totalP;

        Total_particles = totalP;
    }
    IpplTimings::stopTimer(particleCreation);                                                    

    
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);


    bunch_type bunchBuffer(PL);
    static IpplTimings::TimerRef FirstUpdateTimer = IpplTimings::getTimer("FirstUpdate");           
    IpplTimings::startTimer(FirstUpdateTimer);                                               
    PL.update(*P, bunchBuffer);
    IpplTimings::stopTimer(FirstUpdateTimer);                                                    

    msg << "particles created and initial conditions assigned " << endl;

    P->stype_m = argv[6];
    P->initSolver();

    P->time_m = 0.0;
    
    P->scatterCIC(Total_particles, 0, hr);
    
   
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("Solve");           
    IpplTimings::startTimer(SolveTimer);                                               
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();


    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");           
    IpplTimings::startTimer(dumpDataTimer);                                               
    P->dumpData();
    IpplTimings::stopTimer(dumpDataTimer);                                               

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
   
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for 
        // all the particles hence eliminating the need to store mass as 
        // an attribute
        // kick
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityPush");           
        IpplTimings::startTimer(PTimer);                                                    
        auto Rview = P->R.getView();
        auto Pview = P->P.getView();
        auto Eview = P->E.getView();
        double V0 = 30*rmax[2];
        Kokkos::parallel_for("Kick1", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
        
        double Eext_x = -(Rview(j)[0] - (rmax[0]/2)) * (V0/(2*pow(rmax[2],2)));
        double Eext_y = -(Rview(j)[1] - (rmax[1]/2)) * (V0/(2*pow(rmax[2],2)));
        double Eext_z =  (Rview(j)[2] - (rmax[2]/2)) * (V0/(pow(rmax[2],2)));
        
        Pview(j)[0] -= 0.5 * dt * ((Eview(j)[0] + Eext_x) + Pview(j)[1] * Bext);
        Pview(j)[1] -= 0.5 * dt * ((Eview(j)[1] + Eext_y) - Pview(j)[0] * Bext);
        Pview(j)[2] -= 0.5 * dt *  (Eview(j)[2] + Eext_z);
        
        });
        IpplTimings::stopTimer(PTimer);                                                    
        
        //drift
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionPush");           
        IpplTimings::startTimer(RTimer);                                                    
        P->R = P->R + dt * P->P;
        Ippl::Comm->barrier();
        IpplTimings::stopTimer(RTimer);                                                    

        //Since the particles have moved spatially update them to correct processors 
        PL.update(*P, bunchBuffer);

        
        //scatter the charge onto the underlying grid
        P->scatterCIC(Total_particles, it+1, hr);
        
        //Field solve
        IpplTimings::startTimer(SolveTimer);                                               
        P->solver_mp->solve();
        IpplTimings::stopTimer(SolveTimer);                                               
        
        // gather E field
        P->gatherCIC();

        //kick
        IpplTimings::startTimer(PTimer);
        auto R2view = P->R.getView();
        auto P2view = P->P.getView();
        auto E2view = P->E.getView();
        Kokkos::parallel_for("Kick2", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
        
        double Eext_x = -(R2view(j)[0] - (rmax[0]/2)) * (V0/(2*pow(rmax[2],2)));
        double Eext_y = -(R2view(j)[1] - (rmax[1]/2)) * (V0/(2*pow(rmax[2],2)));
        double Eext_z =  (R2view(j)[2] - (rmax[2]/2)) * (V0/(pow(rmax[2],2)));
        
        P2view(j)[0] -= 0.5 * dt * ((E2view(j)[0] + Eext_x) + P2view(j)[1] * Bext);
        P2view(j)[1] -= 0.5 * dt * ((E2view(j)[1] + Eext_y) - P2view(j)[0] * Bext);
        P2view(j)[2] -= 0.5 * dt *  (E2view(j)[2] + Eext_z);
        
        });
        IpplTimings::stopTimer(PTimer);                                                    

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);                                               
        P->dumpData();
        IpplTimings::stopTimer(dumpDataTimer);                                               
        msg << "Finished iteration: " << it << " time: " << P->time_m << endl;
    }
    
    msg << "Penning Trap: End." << endl;
    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
