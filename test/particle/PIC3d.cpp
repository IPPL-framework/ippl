// Test PIC3d
//   This test program sets up a simple sine-wave electric field in 3D,
//   creates a population of particles with random q/m values (charge-to-mass
//   ratio) and velocities, and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation and periodic BCs.
//
//   Usage:
//     srun ./PIC3d 128 128 128 10000 10 --info 10
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <random>
#include "Utility/IpplTimings.h"

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

double pi = acos(-1.0);

//TODO Make dumpVTK work for multi-procs efficiently!
//void dumpVTK(VField_t& EFD, int nx, int ny, int nz, int iteration,
//             double dx, double dy, double dz) {
//
//
//    typename VField_t::view_type::host_mirror_type host_view = EFD.getHostMirror();
//
//    Kokkos::deep_copy(host_view, EFD.getView());
//    std::ofstream vtkout;
//    vtkout.precision(10);
//    vtkout.setf(std::ios::scientific, std::ios::floatfield);
//
//    std::stringstream fname;
//    fname << "data/ef_";
//    fname << std::setw(4) << std::setfill('0') << iteration;
//    fname << ".vtk";
//
//    // open a new data file for this iteration
//    // and start with header
//    vtkout.open(fname.str().c_str(), std::ios::out);
//    vtkout << "# vtk DataFile Version 2.0" << std::endl;
//    vtkout << "pic3d" << std::endl;
//    vtkout << "ASCII" << std::endl;
//    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
//    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
//    vtkout << "ORIGIN "     << -dx  << " " << -dy  << " "  << -dz << std::endl;
//    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
//    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;
//
//    vtkout << "VECTORS E-Field float" << std::endl;
//    for (int z=0; z<nz+2; z++) {
//        for (int y=0; y<ny+2; y++) {
//            for (int x=0; x<nx+2; x++) {
//                
//                vtkout << host_view(x,y,z)[0] << "\t"
//                       << host_view(x,y,z)[1] << "\t"
//                       << host_view(x,y,z)[2] << std::endl;
//            }
//        }
//    }
//
//    // close the output file for this iteration:
//    vtkout.close();
//}
//
//
//void dumpVTK(Field_t& EFD, int nx, int ny, int nz, int iteration,
//             double dx, double dy, double dz) {
//
//    typename Field_t::view_type::host_mirror_type host_view = EFD.getHostMirror();
//    Kokkos::deep_copy(host_view, EFD.getView());
//    std::ofstream vtkout;
//    vtkout.precision(10);
//    vtkout.setf(std::ios::scientific, std::ios::floatfield);
//
//    std::stringstream fname;
//    fname << "data/scalar_";
//    fname << std::setw(4) << std::setfill('0') << iteration;
//    fname << ".vtk";
//
//    double vol = dx*dy*dz;
//
//    // open a new data file for this iteration
//    // and start with header
//    vtkout.open(fname.str().c_str(), std::ios::out);
//    vtkout << "# vtk DataFile Version 2.0" << std::endl;
//    vtkout << "toyfdtd" << std::endl;
//    vtkout << "ASCII" << std::endl;
//    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
//    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
//    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << std::endl;
//    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
//    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;
//
//    vtkout << "SCALARS Rho float" << std::endl;
//    vtkout << "LOOKUP_TABLE default" << std::endl;
//    for (int z=0; z<nz+2; z++) {
//        for (int y=0; y<ny+2; y++) {
//            for (int x=0; x<nx+2; x++) {
//                
//                vtkout << host_view(x,y,z)/vol << std::endl;
//            }
//        }
//    }
//
//
//    // close the output file for this iteration:
//    vtkout.close();
//}

template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    Field<Vector<double, Dim>, Dim> EFD_m;
    Field<double,Dim> EFDMag_m;


    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;


public:
    ParticleAttrib<double>     qm; // charge-to-mass ratio
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
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
    }
    
    ChargedParticles(PLayout& pl,
                     Vector_t hr, Vector_t rmin, Vector_t rmax, ippl::e_dim_tag decomp[Dim],
                     double Q)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    {
        // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    void setupBCs() {
        setBCAllPeriodic();
    }

    void update() {
        PLayout& layout = this->getLayout();
        layout.update(*this);
    }


    void gatherStatistics(unsigned int totalP, int iteration) {

        unsigned int Total_particles = 0;
        unsigned int local_particles = this->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, 
                      MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());

        if(Ippl::Comm->rank() == 0) {
            if(Total_particles != totalP) {
                std::cout << "Total particles in the sim. " << totalP 
                          << " " << "after update: " 
                          << Total_particles << std::endl;
                std::cout << "Total particles not matched after update in iteration:" 
                          << " " << iteration << std::endl;
            }
            exit(1);
        }

        Ippl::Comm->barrier();
        
        std::cout << "Rank " << Ippl::Comm->rank() << " has " 
                  << (double)local_particles/Total_particles*100.0 
                  << "percent of the total particles " << std::endl;
    }
    
    void gatherCIC(int iteration) {

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");           
        IpplTimings::startTimer(gatherTimer);                                                    
        gather(this->E, EFD_m, this->R);
        IpplTimings::stopTimer(gatherTimer);                                                    

        iteration *= 1;
        scatterCIC();
        //if(iteration % 1 == 0) {
        //    static IpplTimings::TimerRef vtkTimer = IpplTimings::getTimer("dumpVTKscalar");           
        //    IpplTimings::startTimer(vtkTimer);                                                    
        //    dumpVTK(EFDMag_m,nr_m[0],nr_m[1],nr_m[2],iteration,hr_m[0],hr_m[1],hr_m[2]);
        //    IpplTimings::stopTimer(vtkTimer);                                                    
        //}
    }

    void scatterCIC() {
         
         static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");           
         IpplTimings::startTimer(scatterTimer);                                                    
         Inform m("scatter ");
         EFDMag_m = 0.0;
         scatter(qm, EFDMag_m, this->R);
         IpplTimings::stopTimer(scatterTimer);                                                    
         
         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("CheckCharge");           
         IpplTimings::startTimer(sumTimer);                                                    
         double Q_grid = EFDMag_m.sum();
         
         m << "Rel. error in charge conservation = " << std::fabs((Q_m-Q_grid)/Q_m) << endl;
         m << "Charge in grid = " << Q_grid << endl;
         //m << "Error in charge conservation = " << std::fabs(qm.sum()-Q_grid) << endl;
         IpplTimings::stopTimer(sumTimer);                                                    
    }
     
     void initFields() {
         static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");           
         IpplTimings::startTimer(initFieldsTimer);                                                    
         Inform m("initFields ");

         ippl::NDIndex<Dim> domain = EFD_m.getDomain();

         for (unsigned int i=0; i<Dim; i++)
             nr_m[i] = domain[i].length();


         double phi0 = 0.1;
         double pi = acos(-1.0);
         double scale_fact = 1e6;

         Vector_t hr = hr_m;

         m << "rmin= " << rmin_m << " rmax= " << rmax_m << " h= " 
           << hr_m << " n= " << nr_m << endl;


         typename VField_t::view_type& view = EFD_m.getView();
         const FieldLayout_t& layout = EFD_m.getLayout(); 
         const ippl::NDIndex<Dim>& lDom = layout.getLocalNDIndex();
         const int nghost = EFD_m.getNghost();
         using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

         Kokkos::parallel_for("Assign EFD_m[0]", 
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){

                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() + nghost;
                                const size_t jg = j + lDom[1].first() + nghost;
                                const size_t kg = k + lDom[2].first() + nghost;
                                
                                view(i, j, k)[0] = -scale_fact*2.0*pi*phi0 * 
                                                    cos(2.0*pi*(ig+0.5)*hr[0]) *
                                                    cos(4.0*pi*(jg+0.5)*hr[1]) * 
                                                    cos(pi*(kg+0.5)*hr[2]);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[1]", 
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){
                                
                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() + nghost;
                                const size_t jg = j + lDom[1].first() + nghost;

                                view(i, j, k)[1] = scale_fact*4.0*pi*phi0 * 
                                                   sin(2.0*pi*(ig+0.5)*hr[0]) * 
                                                   sin(4.0*pi*(jg+0.5)*hr[1]);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[2]", 
                              mdrange_type({nghost, nghost, nghost},
                                            {view.extent(0) - nghost,
                                             view.extent(1) - nghost,
                                             view.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){
                                
                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() + nghost;
                                const size_t jg = j + lDom[1].first() + nghost;

                                view(i, j, k)[2] = scale_fact*4.0*pi*phi0 * 
                                                   sin(2.0*pi*(ig+0.5)*hr[0]) * 
                                                   sin(4.0*pi*(jg+0.5)*hr[1]);
                              
                              });

         EFDMag_m = dot(EFD_m, EFD_m);
         EFDMag_m = sqrt(EFDMag_m);
         IpplTimings::stopTimer(initFieldsTimer);

         //static IpplTimings::TimerRef vtkTimervec = IpplTimings::getTimer("dumpVTKvector");           
         //IpplTimings::startTimer(vtkTimervec);                                                    
         //dumpVTK(EFD_m,nr_m[0],nr_m[1],nr_m[2],0,hr_m[0],hr_m[1],hr_m[2]);
         //IpplTimings::stopTimer(vtkTimervec);                                                    
     }

     //Vector_t getRMin() { return rmin_m;}
     //Vector_t getRMax() { return rmax_m;}
     //Vector_t getHr() { return hr_m;}

     void dumpParticleData(int iteration) {
        
        ParticleAttrib<Vector_t>::view_type& view = P.getView();
        
        double Energy = 0.0;

        Kokkos::parallel_reduce("Particle Energy", view.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(view(i), view(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;
        
        MPI_Reduce(&Energy, &gEnergy, 1, 
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

        if(Ippl::Comm->rank() == 0) {
            std::ofstream csvout;
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname;
            fname << "data/energy.csv";
            csvout.open(fname.str().c_str(), std::ios::out | std::ofstream::app);

            csvout << iteration << " "
                   << gEnergy << std::endl;

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
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);


    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
    IpplTimings::startTimer(mainTimer);                                                    
    const unsigned int totalP = std::atoi(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    
    msg << "Particle test PIC3d "
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
    Vector_t rmax(1.0);
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};
    const double dt = 0.5 * dx; // size of timestep

    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp);
    PLayout_t PL(FL, mesh);


    double Q=1.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);


    unsigned long int nloc = totalP / Ippl::Comm->size();

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");           
    IpplTimings::startTimer(particleCreation);                                                    
    P->create(nloc);
    
    std::mt19937_64 eng[Dim];
    for (unsigned i = 0; i < Dim; ++i) {
        eng[i].seed(42 + i * Dim);
        eng[i].discard( nloc * Ippl::Comm->rank());
    }
    double dx_ref = 1.0/64;
    std::uniform_real_distribution<double> unif(rmin[0]+(dx_ref/2), rmax[0]-(dx_ref/2));

    typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();

    double sum_coord=0.0;
    for (unsigned long int i = 0; i< nloc; i++) {
        for (int d = 0; d<3; d++) {
            R_host(i)[d] =  unif(eng[d]);
            sum_coord += R_host(i)[d];
        }
    }
    double global_sum_coord = 0.0;
    MPI_Reduce(&sum_coord, &global_sum_coord, 1, 
               MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

    if(Ippl::Comm->rank() == 0) {
        std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
    }

    ////For Gaussian distribution
    //std::mt19937_64 eng[2*Dim];
   
    ////There is no reason for picking 42 or multiplying by 
    ////Dim with i, just want the initial seeds to be
    ////farther apart.
    //for (int i = 0; i < 2*3; ++i) {
    //    eng[i].seed(42 + Dim * i);
    //}

    //std::vector<double> mu(Dim);
    //std::vector<double> sd(Dim);
    //std::vector<double> states(Dim);
   

    //mu[0] = 1.0/2;
    //mu[1] = 1.0/2;
    //mu[2] = 1.0/2;
    //sd[0] = 0.15;
    //sd[1] = 0.05;
    //sd[2] = 0.20;


    //std::uniform_real_distribution<double> dist_uniform (0.0, 1.0);

    //for (unsigned long int i = 0; i< nloc; i++) {
    //    
    //    for (int istate = 0; istate < 3; ++istate) {
    //        double u1 = dist_uniform(eng[istate*2]);
    //        double u2 = dist_uniform(eng[istate*2+1]);
    //        states[istate] = sd[istate] * (std::sqrt(-2.0 * std::log(u1)) 
    //                         * std::cos(2.0 * pi * u2)) + mu[istate]; 
    //    }    
    //    for (int d = 0; d<3; d++)
    //        R_host(i)[d] = std::fabs(std::fmod(states[d],1.0));
    //    
    //    Q_host(i) = q;
    //}

    Kokkos::deep_copy(P->R.getView(), R_host);
    P->qm = P->Q_m/totalP;
    P->P = 0.0;
    IpplTimings::stopTimer(particleCreation);                                                    

    static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");           
    IpplTimings::startTimer(UpdateTimer);                                               
    P->update();
    IpplTimings::stopTimer(UpdateTimer);                                                    
    
    //ippl::PRegion<double> region0(rmin[0] + (dx/2), rmax[0] - (dx/2));
    //ippl::PRegion<double> region1(rmin[1] + (dy/2), rmax[1] - (dy/2));
    //ippl::PRegion<double> region2(rmin[2] + (dz/2), rmax[2] - (dz/2));

    //ippl::NDRegion<double, Dim>  pr;
    //pr = ippl::NDRegion<double, Dim>(region0, region1, region2);

    msg << "particles created and initial conditions assigned " << endl;
    P->EFD_m.initialize(mesh, FL);
    P->EFDMag_m.initialize(mesh, FL);
    
    msg << "scatter test" << endl;
    P->scatterCIC();
    
    P->initFields();
    msg << "P->initField() done " << endl;
    
    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
    
        //static IpplTimings::TimerRef gatherStat = IpplTimings::getTimer("gatherStatistics");           
        //IpplTimings::startTimer(gatherStat);                                                    
        //P->gatherStatistics(totalP, it);
        //IpplTimings::stopTimer(gatherStat);                                                    
        
        
        // advance the particle positions
        // basic leapfrogging timestep scheme.  velocities are offset
        // by half a timestep from the positions.
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionUpdate");           
        IpplTimings::startTimer(RTimer);                                                    
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);                                                    

        //Apply particle BCs
        //static IpplTimings::TimerRef BCTimer = IpplTimings::getTimer("applyParticleBC");           
        //IpplTimings::startTimer(BCTimer);                                                    
        //P->getLayout().applyBC(P->R, pr);
        //IpplTimings::stopTimer(BCTimer);

        IpplTimings::startTimer(UpdateTimer);
        P->update();
        IpplTimings::stopTimer(UpdateTimer);                                                    
        
        // gather the local value of the E field and also scatter the charge
        P->gatherCIC(it);

        //static IpplTimings::TimerRef EnergyTimer = IpplTimings::getTimer("dump Energy");           
        //IpplTimings::startTimer(EnergyTimer);                                                    
        //P->dumpParticleData(it);
        //IpplTimings::stopTimer(EnergyTimer);                                                    

        // advance the particle velocities
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityUpdate");           
        IpplTimings::startTimer(PTimer);                                                    
        P->P = P->P + dt * P->qm * P->E;
        IpplTimings::stopTimer(PTimer);                                                    
        msg << "Finished iteration " << it << endl;
    }
    
    msg << "Particle test PIC3d: End." << endl;
    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
