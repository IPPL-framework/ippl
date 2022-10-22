// Langevin Collision Operator Test
// Usage:
// srun ./Langevin <nx> <ny> <nz> <bR> <boxL> <nP> <rc> <alpha> \ 
//                 <dt> <nt> <pCh> <pMass> <focus> <prIntvl> \ 
//                 <stype> <lbthres> <ovfactor> --info 10

//     nx       = No. cell-centered points in the x-direction
//     ny       = No. cell-centered points in the y-direction
//     nz       = No. cell-centered points in the z-direction
//     bR       = Beam Radius
//     boxL     = Box sidelength of cubic box simulated
//     Np       = Total no. of macro-particles in the simulation
//     rc       = cutoff radius, max collision interaction distance
//     alpha    = ??????
//     dt       = timestep (time period of one iteration)
//     Nt       = Number of time steps
//     pCh      = particle charge
//     pMa      = prtile Mass
//     focus    = ????? factor for constant focusing force
//     prIntvl  = printing Intervall of Data to the data document
//     stype    = Field solver type e.g., FFT
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical 
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     
// Example:
// srun ./Langevin 16 16 16 0.001774 0.01 10000 0.0 1e6
//                 2.15623e-13 50 0.01 1 1 1.5 1 
//                 FFT 0.1 2.0 --info 10
//
// /////////////////////////////////////////////////////////////////////
// Copyright (c) 2022, Sriramkrishnan Muralikrishnan, Severin Klapproth
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
// /////////////////////////////////////////////////////////////////////


#include "ChargedParticles.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <set>
#include <chrono>

#include<Kokkos_Random.hpp>

#include <random>
#include "Utility/IpplTimings.h"

template <typename T>
struct Newton1D {

  double tol = 1e-12;
  int max_iter = 20;
  double pi = std::acos(-1.0);
  
  T k, alpha, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& k_, const T& alpha_, 
           const T& u_) 
  : k(k_), alpha(alpha_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = x  + (alpha  * (std::sin(k * x) / k)) - u;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = 1  + (alpha  * std::cos(k * x));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while (iterations < max_iter && std::fabs(f(x)) > tol) {
          x = x - (f(x)/fprime(x));
          iterations += 1;
      }
  }
};


template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random numbers
  view_type x, v;

  // The GeneratorPool
  GeneratorPool rand_pool;

  value_type alpha;

  T k, minU, maxU;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  value_type& alpha_, T& k_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        alpha(alpha_), k(k_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {

        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = u / (1 + alpha);
        Newton1D<value_type> solver(k[d], alpha, u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& alpha, const double& k) {
   double cdf = x + (alpha / k) * std::sin(k * x);
   return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t& xvec, const double& alpha, 
             const Vector_t& kw, const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * std::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

const char* TestName = "LangevinCollsion";


//==============================   ==============================  ============================== 

//PRE: beam radiues >= 0; NParticle ... 
//POST: return the nuch_type paramter with particles initialize on a cold sphere
template<typename bunch>
void createParticleDistributionColdSphere(  bunch P,
                                            double beamRadius, 
                                            unsigned Nparticle,
                                            double qi//, 
                                            // double mi
                                            ) {
    
    Inform m("Initializing Cold Sphere");
        std::default_random_engine generator(0);
        std::normal_distribution<double> normdistribution(0,1.0);
        auto normal = std::bind(normdistribution, generator);
        std::uniform_real_distribution<double> unidistribution(0,1);
        auto uni = std::bind(unidistribution, generator);
        P->Q_m=0;
        Vector_t source({0,0,0});

        if (P->singleInitNode()) { //takes care s.t only one node creates particles
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        Vector_t X(normal(),normal(),normal());
                        double U = uni();
                        Vector_t pos = source + beamRadius*pow(U,1./3.)/sqrt(dot(X,X))*X;
                        Vector_t mom({0,0,0});

                        P->q[i] = -qi;
                        P->P[i] = mom;  //zero momentum intial condition
                        P->R[i] = pos;  //positionattribute given in baseclass
                }
        }
        P->update();
}

// PRE:
// POST:
// template<typename bunch>
// void getapplyConstantFocusing(    bunch P,
//                                 double f,
//                                 double beamRadius
//                                 ) {  
//     //Inform m("computeAvgSpaceChargeForces ");
//     //	m << "apply constant focusing"<< endl;

//     const double N =  static_cast<double>(P->getTotalNum());
// 	Vector_t avgEF;
// 	double locEFsum[Dim];
// 	double globEFsum[Dim];//={0.0,0.0,0.0};

// 	for(unsigned d = 0; d<Dim; ++d){
// 		locEFsum[d]=0.0;
// 		Kokkos::parallel_reduce("get local EField sum", 
// 					 P->getLocalNum(),
// 					 KOKKOS_LAMBDA(const int i, double& valL){
//                                    		double myVal = P->E[i](d);
//                                     	valL += myVal;
//                                 	 },                    			
// 					 Kokkos::Sum<double>(locEFsum[d])
// 					);
// 	}
// 	MPI_Allreduce(locEFsum, globEFsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
	
//     for(int d=0; d<Dim; ++d) avgEF[d] =  globEFsum[d]/N;   
//     double focusingForce=sqrt(dot(avgEF,avgEF));

// 	Kokkos::parallel_for("Apply Constant Focusing",
// 				P->getLocalNum(),
// 				KOKKOS_LAMBDA(const int i){
//          				P->E[i]+=P->R[i]/beamRadius*f*focusingForce;
// 				}	
// 	);
// }

template<typename bunch>
Vector_t compAvgSCForce(    bunch P
                        ) {  
    //Inform m("computeAvgSpaceChargeForces ");
    //	m << "apply constant focusing"<< endl;

    const double N =  static_cast<double>(P->getTotalNum());
	Vector_t avgEF;
	double locEFsum[Dim];//={0.0,0.0,0.0};
	double globEFsum[Dim];

	for(unsigned d = 0; d<Dim; ++d){
		locEFsum[d]=0.0;
		Kokkos::parallel_reduce("get local EField sum", 
					 P->getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& valL){
                                   		double myVal = P->E[i](d);
                                    	valL += myVal;
                                	 },                    			
					 Kokkos::Sum<double>(locEFsum[d])
					);
	}
	MPI_Allreduce(locEFsum, globEFsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
	
    for(int d=0; d<Dim; ++d) avgEF[d] =  globEFsum[d]/N; 

    return avgEF;
}

template<typename bunch>
void applyConstantFocusing( bunch P,
                            double f,
                            double beamRadius,
                            Vector_t avgEF
                                ) {  
    double focusingForce=sqrt(dot(avgEF,avgEF));
	Kokkos::parallel_for("Apply Constant Focusing",
				P->getLocalNum(),
				KOKKOS_LAMBDA(const int i){
         				P->E[i]+=P->R[i]/beamRadius*f*focusingForce;
				}	
	);
}




//If this is only need for the temperature printing into the file
// it could be written directly into the dumplangevin function in the 
// chargedParticles.hpp file
// PRE
// POST
Vector_t compute_temperature(bunch P) {
        Inform m("compute_temperature ");

        const double N =  static_cast<double>(P->getTotalNum());

        double locVELsum[Dim]={0.0,0.0,0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];

        double locT[Dim]={0.0,0.0,0.0};
        double globT[Dim];       
	    Vector_t temperature;

        // GET AVERAGE VELOCITY GLOBALLY
        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 P->getLocalNum(), 
		    			 KOKKOS_LAMBDA(const int i, double& valL){
                                       		double myVal = P->v[i](d);
                                        	valL += myVal;
                                    	 },                    			
		    			 Kokkos::Sum<double>(locVELsum[d])
		    			);
	    }
    	MPI_Allreduce(locVELsum, globVELsum, 3, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	

        for(int d=0; d<Dim; ++d) avgVEL[d]=globVELsum[d]/N;

        // m << "avgVEL[0]= " << avgVEL[0] << " avgVEL[1]= " << avgVEL[1] << " avgVEL[2]= " << avgVEL[2] <<  endl;

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 P->getLocalNum(), 
		    			 KOKKOS_LAMBDA(const int i, double& valL){
                                       		double myVal = (P->v[i](d)-avgVEL[d])*(P->v[i](d)-avgVEL[d]);
                                        	valL += myVal;
                                    	 },                    			
		    			 Kokkos::Sum<double>(locT[d])
		    			);
	    }
    	MPI_Allreduce(locT, globT, 3, MPI_DOUBLE, MPI_SUM,Ippl::getComm());	
        // since we assume for our reduction that this function is called from 
        // multiple nodes, each node, needs to have a return
        // Allreduce shouldnt be slower than reduce, so its ok to give each
        // node the correct return even if it might ot be necessary...
        // we now can print from any node -> for nromal reduce, the other nodes could just return garbage
        // but afterwards the temperature information would only be stored on the root.

        for(int d=0; d<Dim; ++d)    temperature[d]=globT[d]/N;

        return temperature;
}
//======================================================================================== 
// MAIN
int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("Langevin");
    Inform msg2all("Langevin ",INFORM_ALL_NODES);
    Ippl::Comm->setDefaultOverallocation(std::atof(argv[17]));

    auto start = std::chrono::high_resolution_clock::now();

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };
    const double beamRadius         = std::atof(argv[4]);
    const double boxLentgh          = std::atof(argv[5]);
    const size_type nP              = std::atoll(argv[6]);
    const double interactionRadius  = std::atof(argv[7]);
    const double alpha              = std::atof(argv[8]);
    const double dt                 = std::atof(argv[9]);
    const size_type nt              = std::atoll(argv[10]); // iterations
    const double particleCharge     = std::atof(argv[11]);
    const double particleMass       = std::atof(argv[12]);
    const double focusForce         = std::atof(argv[13]);
    const int printInterval         = std::atoi(argv[14]);
    //15 -> solvertype = FFT...
    //16 -> loadbalancethreshold
    //17 -> default overallocation

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("kick");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef DummySolveTimer = IpplTimings::getTimer("solveWarmup");
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("domainDecomp");

    IpplTimings::startTimer(mainTimer);


    msg << "Start test: LANGEVIN COLLISION OPERATOR" << endl
        << "Total Timesteps = " << std::setw(20) << nt << endl
        << "Total Particles = " << std::setw(20) << nP << endl
        << "Griddimensions  = " << std::setw(20) << nr << endl
        << "Beamradius      = " << std::setw(20) << beamRadius << endl
        << "focusing force  = " << std::setw(20) << focusForce << endl;

//================================================================================== 
// MESH & DOMAIN_DECOMPOSITION

    using bunch_type = ChargedParticles<PLayout_t>;
    std::unique_ptr<bunch_type>  P;

    //initializing number of cells in mesh/domain
    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }
    //initializinh boundary conditions
    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }
    Vector_t kw = {0.5, 0.5, 0.5}; //irregularities sth inside the plasma
    const double L = Lbox*0.5;
    Vector_t box_boundaries_lower({-L, -L, -L});
    Vector_t box_boundaries_upper({L, L, L});  
    Vector_t origin({0.0, 0.0, 0.0});
    Vector_t hr = {Lbox/nr[0], Lbox/nr[1], Lbox/nr[2]}; //spacing
    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);
    double Q = nP * pCharge;
    P = std::make_unique<bunch_type>(PL,hr,
                                        box_boundaries_lower,
                                        box_boundaries_upper,
                                        decomp,Q);
    P->nr_m = nr;
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);
    bunch_type bunchBuffer(PL);
    P->stype_m = argv[15];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[16]);
    bool isFirstRepartition;

    //INITIAL LOADBALANCING
    if ((P->loadbalancethreshold_m != 1.0) && (Ippl::Comm->size() > 1)) {
        msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);
        isFirstRepartition = true;
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        const int nghost = P->rho_m.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto rhoview = P->rho_m.getView();

        Kokkos::parallel_for("Assign initial rho based on PDF",
                              mdrange_type({nghost, nghost, nghost},
                                           {rhoview.extent(0) - nghost,
                                            rhoview.extent(1) - nghost,
                                            rhoview.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k)
                              {
                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() - nghost;
                                const size_t jg = j + lDom[1].first() - nghost;
                                const size_t kg = k + lDom[2].first() - nghost;
                                double x = (ig + 0.5) * hr[0] + origin[0];
                                double y = (jg + 0.5) * hr[1] + origin[1];
                                double z = (kg + 0.5) * hr[2] + origin[2];
                                Vector_t xvec = {x, y, z};
                                rhoview(i, j, k) = PDF(xvec, alpha, kw, Dim);
                              });
        Kokkos::fence();
        P->initializeORB(FL, mesh);
        P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
        IpplTimings::stopTimer(domainDecomposition);
    }
    isFirstRepartition = false;
    msg << "First domain decomposition done" << endl;

// MESH & DOMAIN_DECOMPOSITION
//=======================================================================================
// PARTICLE CREATION
    
    createParticleDistributionColdSphere(P, beamRadius, nP, pCharge);    
    Kokkos::fence(); //??
    Ippl::Comm->barrier();  //??
    IpplTimings::stopTimer(particleCreation);    
    msg << "particles created and initial conditions assigned " << endl;

// PARTICLE CREATION
//====================================================================================== 
// TEST TIMERS

    IpplTimings::startTimer(DummySolveTimer);
    P->rho_m = 0.0;
    P->solver_mp->solve();
    IpplTimings::stopTimer(DummySolveTimer);

    P->scatterCIC(nP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLandau();
    P->gatherStatistics(nP);
    //P->dumpLocalDomains(FL, 0);
    IpplTimings::stopTimer(dumpDataTimer);

//TEST TIMERS
//====================================================================================== 
// TIMELOOP

    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        //TODO: work in charge and mass for other ratio than 1?


        // kick

        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        //drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        //Since the particles have moved spatially update them to correct processors
	    IpplTimings::startTimer(updateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(updateTimer);

        // Domain Decomposition
        if (P->balance(nP, it+1)) {
           msg << "Starting repartition" << endl;
           IpplTimings::startTimer(domainDecomposition);
           P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
           IpplTimings::stopTimer(domainDecomposition);
           //IpplTimings::startTimer(dumpDataTimer);
           //P->dumpLocalDomains(FL, it+1);
           //IpplTimings::stopTimer(dumpDataTimer);
        }


        //scatter the charge onto the underlying grid
        P->scatterCIC(nP, it+1, hr);

        //Field solve
        IpplTimings::startTimer(SolveTimer);
        P->solver_mp->solve();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        P->gatherCIC();

    // =================MYSTUFF==================================================================
        
        Vecotor_t avgEF(compAvgSCForce(P));
        applyConstantFocusing(P, focusForce, beamRadius, avgEF);


        //LANGEVIN COLLISIONS<-


    // =================MYSTUFF==================================================================
        
        //kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpLangevin(it);
        P->gatherStatistics(nP);
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

// TIMELOOP
//====================================================================================== 

    msg << "Langevin: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
