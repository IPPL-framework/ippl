// Test PIC3d
//   This test program sets up a simple sine-wave electric field in 3D,
//   creates a population of particles with random q/m values (charge-to-mass
//   ratio) and velocities, and then tracks their motions in the static
//   electric field using cloud-in-cell interpolation.
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
#include <chrono>
#include "Utility/Timer.h"
#include "Utility/IpplCounter.h"
#include "Utility/IpplStats.h"
#include "Utility/IpplTimings.h"

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::detail::ParticleLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef Cell                                       Center_t;
typedef FieldLayout<Dim> FieldLayout_t;


template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;


//enum BC_t {OOO,OOP,PPP};

double pi = acos(-1.0);

void dumpVTK(VField_t& EFD, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {


    typename VField_t::view_type::host_mirror_type host_view = EFD.getHostMirror();

    Kokkos::deep_copy(host_view, EFD.getView());
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
    vtkout << "pic3d" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN "     << -dx  << " " << -dy  << " "  << -dz << std::endl;
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


void dumpVTK(Field_t& EFD, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    typename Field_t::view_type::host_mirror_type host_view = EFD.getHostMirror();
    Kokkos::deep_copy(host_view, EFD.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    double vol = dx*dy*dz;

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "toyfdtd" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;

    vtkout << "SCALARS Rho float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {
                
                vtkout << host_view(x,y,z)/vol << std::endl;
            }
        }
    }


    // close the output file for this iteration:
    vtkout.close();
}

template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    Field<Vector<double, Dim>, Dim> EFD_m;
    Field<double,Dim> EFDMag_m;


    Vector<int, Dim> nr_m;

    e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;


public:
    ParticleAttrib<double>     qm; // charge-to-mass ratio
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position
    typename ippl::ParticleBase<PLayout>::particle_position_type B;  // magnetic field at particle position

//     /*
//       In case we have OOP or PPP boundary conditions
//       we must define the domain, i.e can not be deduced from the
//       particles as in the OOO case.
//     */
//
    ChargedParticles(PLayout& pl,
//                   BC_t bc,
                     Vector_t hr, Vector_t rmin, Vector_t rmax, e_dim_tag decomp[Dim], 
                     double Q)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    {
//         // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(B);
        setupBCs();
        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i]=decomp[i];
    }

    void setupBCs() {
//         if (bco_m == OOO)
//             setBCAllOpen();
//         else if (bco_m == PPP)
            setBCAllPeriodic();
//         else
//             setBCOOP();
    }

     void gatherCIC(int iteration) {

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");           
        IpplTimings::startTimer(gatherTimer);                                                    
        gather(this->E, EFD_m, this->R);
        Kokkos::fence();
        IpplTimings::stopTimer(gatherTimer);                                                    

        iteration *= 1;
        scatterCIC();
        //if(iteration % 1 == 0) {
        //    static IpplTimings::TimerRef vtkTimer = IpplTimings::getTimer("dumpVTKscalar");           
        //    IpplTimings::startTimer(vtkTimer);                                                    
        //    dumpVTK(EFDMag_m,nr_m[0],nr_m[1],nr_m[2],iteration,hr_m[0],hr_m[1],hr_m[2]);
        //    Kokkos::fence();
        //    IpplTimings::stopTimer(vtkTimer);                                                    
        //}
     }

     void scatterCIC() {
         static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");           
         IpplTimings::startTimer(scatterTimer);                                                    
         Inform m("scatter ");
         EFDMag_m = 0.0;
         scatter(qm, EFDMag_m, this->R);
         Kokkos::fence();
         IpplTimings::stopTimer(scatterTimer);                                                    
         
         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("CheckCharge");           
         IpplTimings::startTimer(sumTimer);                                                    
         double Q_grid = EFDMag_m.sum(1);
         
         m << "Q grid = " << Q_grid << endl;
         m << "Error = " << Q_m-Q_grid << endl;
         Kokkos::fence();
         IpplTimings::stopTimer(sumTimer);                                                    
     }
//
//     void myUpdate() {
//
//         double hz   = hr_m[2];
//         double zmin = rmin_m[2];
//         double zmax = rmax_m[2];
//
//         if (bco_m != PPP) {
//             bounds(this->R, rmin_m, rmax_m);
//
//             NDIndex<Dim> domain = this->getFieldLayout().getDomain();
//
//             for (unsigned int i=0; i<Dim; i++)
//                 nr_m[i] = domain[i].length();
//
//             for (unsigned int i=0; i<Dim; i++)
//                 hr_m[i] = (rmax_m[i] - rmin_m[i]) / (nr_m[i] - 1.0);
//
//             if (bco_m == OOP) {
//                 rmin_m[2] = zmin;
//                 rmax_m[2] = zmax;
//                 hr_m[2] = hz;
//             }
//
//             getMesh().set_meshSpacing(&(hr_m[0]));
//             getMesh().set_origin(rmin_m);
//
//             if(withGuardCells_m) {
//                 EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
//                 EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
//             }
//             else {
//                 EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
//                 EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
//             }
//         }
//         else {
//             if(fieldNotInitialized_m) {
//                 fieldNotInitialized_m=false;
//                 getMesh().set_meshSpacing(&(hr_m[0]));
//                 getMesh().set_origin(rmin_m);
//                 if(withGuardCells_m) {
//                     EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
//                     EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
//                 }
//                 else {
//                     EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
//                     EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
//                 }
//             }
//         }
//         this->update();
//     }
//
     
     void initFields() {
         static IpplTimings::TimerRef initFieldsTimer = IpplTimings::getTimer("initFields");           
         IpplTimings::startTimer(initFieldsTimer);                                                    
         Inform m("initFields ");

         NDIndex<Dim> domain = EFD_m.getDomain();

         for (unsigned int i=0; i<Dim; i++)
             nr_m[i] = domain[i].length();


         double phi0 = 0.1;
         double pi = acos(-1.0);

         m << "rmin= " << rmin_m << " rmax= " << rmax_m << " h= " 
           << hr_m << " n= " << nr_m << endl;

         Vector_t hr = hr_m;

         typename VField_t::view_type& view = EFD_m.getView();

         Kokkos::parallel_for("Assign EFD_m[0]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[0] = -2.0*pi*phi0 * 
                                                    cos(2.0*pi*(i+0.5)*hr[0]) *
                                                    cos(4.0*pi*(j+0.5)*hr[1]) * 
                                                    cos(pi*(k+0.5)*hr[2]);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[1]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[1] = 4.0*pi*phi0 * 
                                                   sin(2.0*pi*(i+0.5)*hr[0]) * 
                                                   sin(4.0*pi*(j+0.5)*hr[1]);
                              
                              });
         
         Kokkos::parallel_for("Assign EFD_m[2]", 
                              Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1),
                                                                      view.extent(2)}),
                              KOKKOS_LAMBDA(const int i, const int j, const int k){

                                view(i, j, k)[2] = 4.0*pi*phi0 * 
                                                   sin(2.0*pi*(i+0.5)*hr[0]) * 
                                                   sin(4.0*pi*(j+0.5)*hr[1]);
                              
                              });

         EFDMag_m = dot(EFD_m, EFD_m);
         Kokkos::fence();
         IpplTimings::stopTimer(initFieldsTimer);

         //static IpplTimings::TimerRef vtkTimervec = IpplTimings::getTimer("dumpVTKvector");           
         //IpplTimings::startTimer(vtkTimervec);                                                    
         //dumpVTK(EFD_m,nr_m[0],nr_m[1],nr_m[2],0,hr_m[0],hr_m[1],hr_m[2]);
         //Kokkos::fence();
         //IpplTimings::stopTimer(vtkTimervec);                                                    
     }

     Vector_t getRMin() { return rmin_m;}
     Vector_t getRMax() { return rmax_m;}
     Vector_t getHr() { return hr_m;}

     void dumpParticleData(int iteration) {
        
        ParticleAttrib<Vector_t>::view_type& view = P.getView();
        std::ofstream csvout;
        csvout.precision(10);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        std::stringstream fname;
        fname << "data/energy.csv";
        double Energy = 0.0;

        csvout.open(fname.str().c_str(), std::ios::out | std::ofstream::app);

        Kokkos::parallel_reduce("Particle Energy", view.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(view(i), view(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        csvout << iteration << " "
               << Energy << std::endl;

        csvout.close();

     }

private:
//
//     inline void setBCAllOpen() {
//         for (unsigned i=0; i < 2*Dim; i++) {
//             this->getBConds()[i] = ParticleNoBCond;
//             bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//         }
//     }
//
    void setBCAllPeriodic() {

        this->setParticleBC(ippl::BC::PERIODIC);
    }
//
//     inline void setBCOOP() {
//         for (unsigned i=0; i < 2*Dim - 2; i++) {
//             bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//             this->getBConds()[i] = ParticleNoBCond;
//         }
//         for (unsigned i= 2*Dim - 2; i < 2*Dim; i++) {
//             bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
//             vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
//             this->getBConds()[i] = ParticlePeriodicBCond;
//         }
//     }
//

};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    if (argc != 6) {
        msg << "PIC3d [mx] [mx] [my] [#particles] [#time steps]"
            << endl;
        return -1;
    }

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");           
    IpplTimings::startTimer(mainTimer);                                                    
    auto start = std::chrono::high_resolution_clock::now();    
    const unsigned int totalP = std::atoi(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    
    msg << "Particle test PIC3d "
        << endl
        << "nt " << nt << " Np= "
        << totalP << " grid = " << nr
        << endl;
//     BC_t myBC;
//     if (std::string(argv[7])==std::string("OOO")) {
//         myBC = OOO; // open boundary
//         msg << "BC == OOO" << endl;
//     }
//     else if (std::string(argv[7])==std::string("OOP")) {
//         myBC = OOP; // open boundary in x and y, periodic in z
//         msg << "BC == OOP" << endl;
//     }
//     else {
//         myBC = PPP; // all periodic
//         msg << "BC == PPP" << endl;
//     }
//

    std::unique_ptr<Mesh_t> mesh;
    std::unique_ptr<FieldLayout_t> FL;

    using bunch_type = ChargedParticles<PLayout_t>;

    std::unique_ptr<bunch_type>  P;
    std::unique_ptr<PLayout_t> PL;

    NDIndex<Dim> domain;
    
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = Index(nr[i]);
    }
    
    e_dim_tag decomp[Dim];    
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = SERIAL;
    }

    // create mesh and layout objects for this problem domain
    double dx = 1.0 / double(nr[0]);
    double dy = 1.0 / double(nr[1]);
    double dz = 1.0 / double(nr[2]);
    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {0, 0, 0};
    const double dt = 0.5 * dx; // size of timestep
    mesh = std::make_unique<Mesh_t>(domain, hr, origin);
    FL   = std::make_unique<FieldLayout_t>(domain, decomp, 1);
    PL   = std::make_unique<PLayout_t>();


    /*
     * In case of periodic BC's define
     * the domain with hr and rmin
     */
    Vector_t rmin(0.0);
    Vector_t rmax(1.0);

    double Q=1e6;
    P = std::make_unique<bunch_type>(*PL,/*myBC,*/hr,rmin,rmax,decomp,Q);

    // initialize the particle object: do all initialization on one node,
    // and distribute to others

    unsigned long int nloc = totalP / Ippl::getNodes();

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");           
    IpplTimings::startTimer(particleCreation);                                                    
    P->create(nloc);

    std::mt19937_64 eng;//(42);
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
    typename ParticleAttrib<double>::HostMirror Q_host = P->qm.getHostMirror();

    double q = P->Q_m/totalP;

    for (unsigned long int i = 0; i< nloc; i++) {
        for (int d = 0; d<3; d++) {
            R_host(i)[d] =  unif(eng);
        }
        Q_host(i) = q;
    }
    ////For generating same distribution always
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
    Kokkos::deep_copy(P->qm.getView(), Q_host);
    P->P = 0.0;
    Kokkos::fence();
    IpplTimings::stopTimer(particleCreation);                                                    

    ippl::PRegion<double> region0(0.0, 1.0);
    ippl::PRegion<double> region1(0.0, 1.0);
    ippl::PRegion<double> region2(0.0, 1.0);

    ippl::NDRegion<double, Dim>  pr;
    pr = ippl::NDRegion<double, Dim>(region0, region1, region2);

    msg << "particles created and initial conditions assigned " << endl;
    P->EFD_m.initialize(*mesh, *FL);
    P->EFDMag_m.initialize(*mesh, *FL);
    
    // redistribute particles based on spatial layout
    // P->myUpdate();
    
    msg << "scatter test" << endl;
    P->scatterCIC();
    
    P->initFields();
    msg << "P->initField() done " << endl;
    
    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
        // advance the particle positions
        // basic leapfrogging timestep scheme.  velocities are offset
        // by half a timestep from the positions.
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionUpdate");           
        IpplTimings::startTimer(RTimer);                                                    
        P->R = P->R + dt * P->P;
        Kokkos::fence();
        IpplTimings::stopTimer(RTimer);                                                    

        //Apply particle BCs
        static IpplTimings::TimerRef BCTimer = IpplTimings::getTimer("applyParticleBC");           
        IpplTimings::startTimer(BCTimer);                                                    
        P->getLayout().applyBC(P->R, pr);
        Kokkos::fence();
        IpplTimings::stopTimer(BCTimer);                                                    

        // update particle distribution across processors
        //P->myUpdate();

        // gather the local value of the E field
        P->gatherCIC(it);


        //static IpplTimings::TimerRef EnergyTimer = IpplTimings::getTimer("dump Energy");           
        //IpplTimings::startTimer(EnergyTimer);                                                    
        //P->dumpParticleData(it);
        //Kokkos::fence();
        //IpplTimings::stopTimer(EnergyTimer);                                                    

        // advance the particle velocities
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityUpdate");           
        IpplTimings::startTimer(PTimer);                                                    
        P->P = P->P + dt * P->qm * P->E;
        Kokkos::fence();
        IpplTimings::stopTimer(PTimer);                                                    
        msg << "Finished iteration " << it << " - min/max r and h " << P->getRMin()
            << P->getRMax() << P->getHr() << endl;
    }
    
    msg << "Particle test PIC3d: End." << endl;
    Kokkos::fence();
    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_elapsed = 
                                  std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_elapsed.count() << std::endl;

    return 0;
}
