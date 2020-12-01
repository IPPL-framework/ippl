//   Usage:
//     srun ./benchmarkParticleUpdate 128 128 128 10000 10 --info 10
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
#include "Utility/IpplTimings.h"

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::ParticleSpatialLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;


template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;


template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:

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
      For PPP boundary conditions
      we must define the domain.
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
        this->addAttribute(qm);
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

    void update() {
        PLayout& layout = this->getLayout();
        layout.update(*this);
    }

    void gatherStatistics(unsigned int totalP, int iteration) {

        unsigned int Total_particles = 0;
        unsigned int local_particles = this->getLocalNum();

        MPI_Allreduce(&local_particles, &Total_particles, 1, 
                      MPI_UNSIGNED, MPI_SUM, Ippl::getComm());

        if(Total_particles != totalP) {
            if(Ippl::Comm->rank() == 0) {
            std::cout << "Total particles in the sim. " << totalP 
                      << " " << "after update: " 
                      << Total_particles << std::endl;
            std::cout << "Total particles not matched after update in iteration:" 
                      << " " << iteration << std::endl;
            }
            exit(1);
        }

        
        std::cout << "Rank " << Ippl::Comm->rank() << " has " 
                  << (double)local_particles/Total_particles*100.0 
                  << "percent of the total particles " << std::endl;
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
    auto start = std::chrono::high_resolution_clock::now();    
    const unsigned int totalP = std::atoi(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    
    msg << "benchmarkUpdate"
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
    double dx = 1.0 / double(nr[0]);
    double dy = 1.0 / double(nr[1]);
    double dz = 1.0 / double(nr[2]);
    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {0, 0, 0};
    double hr_min = std::min({dx, dy, dz});
    const double dt = 1.0; // size of timestep
    

    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp);
    PLayout_t PL(FL, mesh);

    /*
     * In case of periodic BC's define
     * the domain with hr and rmin
     */
    Vector_t rmin(0.0);
    Vector_t rmax(1.0);

    double Q=1e6;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);


    unsigned long int nloc = totalP / Ippl::Comm->size();

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");           
    IpplTimings::startTimer(particleCreation);                                                    
    P->create(nloc);

    std::mt19937_64 eng;
    eng.seed(42);
    eng.discard( nloc * Ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();


    for (unsigned long int i = 0; i< nloc; i++) {
        for (int d = 0; d<3; d++) {
            R_host(i)[d] =  unif(eng);
        }
    }


    Kokkos::deep_copy(P->R.getView(), R_host);
    P->qm = P->Q_m/totalP;
    IpplTimings::stopTimer(particleCreation);                                                    
    P->E = 0.0;

    static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");           
    IpplTimings::startTimer(UpdateTimer);                                               
    P->update();
    IpplTimings::stopTimer(UpdateTimer);                                                    


    msg << "particles created and initial conditions assigned " << endl;
    
    std::uniform_real_distribution<double> unifP(0, hr_min);
    typename bunch_type::particle_position_type::HostMirror P_host = P->P.getHostMirror();
    
    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
       
        static IpplTimings::TimerRef gatherStat = IpplTimings::getTimer("gatherStatistics");           
        IpplTimings::startTimer(gatherStat);                                                    
        P->gatherStatistics(totalP, it);
        IpplTimings::stopTimer(gatherStat);                                                    

        static IpplTimings::TimerRef RandPTimer = IpplTimings::getTimer("RandomP");           
        IpplTimings::startTimer(RandPTimer);                                                    
        std::mt19937_64 engP;
        engP.seed(42 + 10*it + 100*Ippl::Comm->rank());
        Kokkos::resize(P_host, P->P.size());
        for (unsigned long int i = 0; i<P->getLocalNum(); i++) {
            for (int d = 0; d<3; d++) {
                P_host(i)[d] =  unifP(engP);
            }
        }
        Kokkos::deep_copy(P->P.getView(), P_host);
        IpplTimings::stopTimer(RandPTimer);                                                    
        Ippl::Comm->barrier();
        
        
        // advance the particle positions
        // basic leapfrogging timestep scheme.  velocities are offset
        // by half a timestep from the positions.
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionUpdate");           
        IpplTimings::startTimer(RTimer);                                                    
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);                                                    
        
        IpplTimings::startTimer(UpdateTimer);
        P->update();
        IpplTimings::stopTimer(UpdateTimer);                                                    


        // advance the particle velocities
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityUpdate");           
        IpplTimings::startTimer(PTimer);                                                    
        P->P = P->P + dt * P->qm * P->E;
        IpplTimings::stopTimer(PTimer);                                                    
        msg << "Finished iteration " << it << " - min/max r and h " << P->getRMin()
            << P->getRMax() << P->getHr() << endl;
    }
    
    msg << "Particle update test: End." << endl;
    IpplTimings::stopTimer(mainTimer);                                                    
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_elapsed = 
                                  std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_elapsed.count() << std::endl;

    return 0;
}
