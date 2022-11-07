// Langevin Collision Operator Test
// Usage:
// srun ./Langevin <nx> <ny> <nz> <bR> <boxL> <nP> <rc> <alpha> 
//                 <dt> <nt> <pCh> <pMass> <focus> <prIntvl> 
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
void createParticleDistributionColdSphere( 	bunch& P,
						const double& beamRadius, 
                                        	const unsigned& Nparticle,
                                        	const double& qi//,
                                            // double mi
                                            ) {
    
   	Inform m("Cold Sphere Initialization ");

  	m << "creating Particles" << endl;
        P.create(Nparticle);
	
	auto mydotpr = [](Vector_t a, Vector_t b)   {
		return a(0)*b(0) + a(1)*b(1) + a(2)*b(2); 
	};
        std::default_random_engine generator(0);
        std::normal_distribution<double> normdistribution(0,1.0);
        auto normal = std::bind(normdistribution, generator);
        std::uniform_real_distribution<double> unidistribution(0,1);
        auto uni = std::bind(unidistribution, generator);
        Vector_t source({0,0,0});

   	     
   	m << "start Initializing" << endl;
	typename bunch::particle_position_type::HostMirror pRHost = P.R.getHostMirror();

	for (unsigned i = 0; i<Nparticle; ++i) {
         		Vector_t X({normal(),normal(),normal()});
        		double U = uni();
         		Vector_t pos = source + beamRadius*pow(U,1./3.)/sqrt(mydotpr(X,X))*X;
       	 		pRHost(i) = pos; // () or [] is indifferent
	}
	Vector_t mom({0,0,0});
        P.q = -qi;
        P.P = mom;  //zero momentum intial conditiov
	//	m << "()"<< pRHost(0)(0) << pRHost(0)(1) << pRHost(0)(2) << endl;
	//	m << "[]"<< pRHost[0](0) << pRHost[0](1) << pRHost[0](2) << endl;
   	m << "finished Initializing" << endl;
}

template<typename bunch>
Vector_t compAvgSCForce(bunch& P, const size_type N ) {  
    Inform m("computeAvgSpaceChargeForces ");
    	m << "start Calculation" << endl;

	Vector_t avgEF;
	double locEFsum[Dim];//={0.0,0.0,0.0};
	double globEFsum[Dim];
	
	//auto pEView = P.E.getView();
	auto pEMirror = P.E.getHostMirror();
	for(unsigned d = 0; d<Dim; ++d){
		locEFsum[d]=0.0;
		Kokkos::parallel_reduce("get local EField sum", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& valL){
                                   		double myVal =pEMirror[i](d);    //  P.E[i](d);
              		                      	valL += myVal;
                                	 },                    			
					 Kokkos::Sum<double>(locEFsum[d])
					);
	}
	MPI_Allreduce(locEFsum, globEFsum, Dim , MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
	
    for(unsigned d=0; d<Dim; ++d) avgEF[d] =  globEFsum[d]/N; 

   m << "finished Calculation; Dim = "<< Dim  <<  endl;
    return avgEF;
}

template<typename bunch>
void applyConstantFocusing( bunch& P,
                            const double f,
                            const double beamRadius,
                            const Vector_t avgEF
                                ) {  
	//Inform m("applyConstantFocusing");
	//m << "start" << endl;
	auto mydotpr = [](Vector_t a, Vector_t b)   {
		return a(0)*b(0) + a(1)*b(1) + a(2)*b(2); 
	};

	auto pEMirror = P.E.getHostMirror();
	auto pRMirror = P.R.getHostMirror();
//	auto pEView = P.E.getView();
//	auto pRView = P.R.getView();
	double tmp = sqrt(mydotpr(avgEF, avgEF))*f/beamRadius;

	Kokkos::parallel_for("Apply Constant Focusing",
				P.getLocalNum(),
				KOKKOS_LAMBDA(const int i){
					pEMirror[i] += pRMirror[i]*tmp;
				}	
	);
	//m << "finished" << endl;

}



//Is directly integrated in dumpLandau and currently unused
// PRE
// POST
template<typename bunch>
Vector_t compute_temperature(const bunch& P, const double mass, const size_type N) {
        Inform m("compute_temperature ");

        double locVELsum[Dim]={0.0,0.0,0.0};
        double globVELsum[Dim];
        double avgVEL[Dim];

        double locT[Dim]={0.0,0.0,0.0};
        double globT[Dim];       
	Vector_t temperature;
	
	auto pPMirror = P.P.getHostMirror();
	//auto pPView = P.P.getView();
        // GET AVERAGE VELOCITY GLOBALLY
        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 P.getLocalNum(), 
		    			 KOKKOS_LAMBDA(const int i, double& valL){
                                       		double myVal = pPMirror[i](d)/mass;
                                        	valL += myVal;
                                    	 },                    			
		    			 Kokkos::Sum<double>(locVELsum[d])
		    );
	    }
    	MPI_Allreduce(locVELsum, globVELsum, Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());	

        for(unsigned d=0; d<Dim; ++d) avgVEL[d]=globVELsum[d]/N;

        m << "avgVEL[0]= " << avgVEL[0] << " avgVEL[1]= " << avgVEL[1] << " avgVEL[2]= " << avgVEL[2] <<  endl;

        for(unsigned d = 0; d<Dim; ++d){
		    Kokkos::parallel_reduce("get local velocity sum", 
		    			 P.getLocalNum(), 
		    			 KOKKOS_LAMBDA(const int i, double& valL){
                                       		double myVal = (pPMirror[i](d)/mass-avgVEL[d])*(pPMirror[i](d)/mass-avgVEL[d]);
                                        	valL += myVal;
                                    	 },                    			
		    			 Kokkos::Sum<double>(locT[d])
		   );
	    }
    	MPI_Allreduce(locT, globT, Dim, MPI_DOUBLE, MPI_SUM,Ippl::getComm());	

        for(unsigned d=0; d<Dim; ++d)    temperature[d]=globT[d]/N;

        return temperature;
}


// is directly integratied into the dumpLandau function (particle Header)
template<typename bunch>
void writeBeamStatistics(const bunch& P, const size_t N, const int rank, const size_t iteration){
	//prep
	const size_t locNp = P.getLocalNum();

   //calculate Moments===========================================================
	auto pPMirror = P.P.getHostMirror();
	auto pRMirror = P.R.getHostMirror();
	double     centroid[2 * Dim];
	double       moment[2 * Dim][2 * Dim];//={};

	double loc_centroid[2 * Dim];//={};
	double   loc_moment[2 * Dim][2 * Dim];//={};
        
	for(unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for(unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = 0.0;
            }
   	 }

	for(unsigned i = 0; i< 2*Dim; ++i){

		Kokkos::parallel_reduce("write Emittance 1 redcution",
				locNp,
				KOKKOS_LAMBDA(const int k,
						double& cent,
						double& mom0,
						double& mom1,
						double& mom2,
						double& mom3,
						double& mom4,
						double& mom5
						){ 
					double    part[2 * Dim];
	            			part[1] = pPMirror[k](0);
	            			part[3] = pPMirror[k](1);
	            			part[5] = pPMirror[k](2);
	            			part[0] = pRMirror[k](0);
	            			part[2] = pRMirror[k](1);
	            			part[4] = pRMirror[k](2);
					
					cent = loc_centroid[i];
					mom0 = loc_moment[i][0];
					mom1 = loc_moment[i][1];
					mom2 = loc_moment[i][2];
					mom3 = loc_moment[i][3];
					mom4 = loc_moment[i][4];
					mom5 = loc_moment[i][5];
	            			
					cent += part[i];
					mom0 += part[i]*part[0];
					mom1 += part[i]*part[1];
					mom2 += part[i]*part[2];
					mom3 += part[i]*part[3];
					mom4 += part[i]*part[4];
					mom5 += part[i]*part[5];
				},
				Kokkos::Sum<double>(loc_centroid[i]),
				Kokkos::Sum<double>(loc_moment[i][0]),
				Kokkos::Sum<double>(loc_moment[i][1]),
				Kokkos::Sum<double>(loc_moment[i][2]),
				Kokkos::Sum<double>(loc_moment[i][3]),
				Kokkos::Sum<double>(loc_moment[i][4]),
				Kokkos::Sum<double>(loc_moment[i][5])
		);	
	}
	
    	for(unsigned i = 0; i < 2 * Dim; i++) {
    	    for(unsigned j = 0; j < i; j++) {
    	        loc_moment[j][i] = loc_moment[i][j];
    	    }
    	}

    MPI_Allreduce(loc_moment, moment, 2 * Dim * 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

    MPI_Allreduce(loc_centroid, centroid, 2 * Dim, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    
    const double zero = 0.0;
    Vector_t eps2, fac, rsqsum, vsqsum, rvsum;
	Vector_t rmean, vmean, rrms, vrms, eps, rvrms;

    	for(unsigned int i = 0 ; i < Dim; i++) {
    	    rmean(i) = centroid[2 * i] / N;
    	    vmean(i) = centroid[(2 * i) + 1] / N;
    	    rsqsum(i) = moment[2 * i][2 * i] - N * rmean(i) * rmean(i);
    	    vsqsum(i) = moment[(2 * i) + 1][(2 * i) + 1] - N * vmean(i) * vmean(i);
    	    if(vsqsum(i) < 0)
    	        vsqsum(i) = 0;
    	    rvsum(i) = moment[(2 * i)][(2 * i) + 1] - N * rmean(i) * vmean(i);
    	}

    eps2 = (rsqsum * vsqsum - rvsum * rvsum) / (N * N);
    rvsum = rvsum/double(N);

    	for(unsigned int i = 0 ; i < Dim; i++) {
   		     rrms(i) = sqrt(rsqsum(i) / N);
   		     vrms(i) = sqrt(vsqsum(i) / N);
   		     eps(i)  =  std::sqrt(std::max(eps2(i), zero));
   		     double tmp = rrms(i) * vrms(i);
   		     fac(i) = (tmp == 0) ? zero : 1.0 / tmp;
   		 }
    rvrms = rvsum * fac;

   ////=====writeBeamStatisticsVelocity ======================
  //if(Ippl::myNode()==0) {
  if(rank ==0) {

    std::stringstream fname;
    fname << "data/BeamStatistics";
    fname << ".csv";

    // open a new data file for this iteration
    // and start with header
    Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    if (iteration==0){
    	csvout << "it,rrmsX, rrmsY, rrmsZ, vrmsX,vrmsY,vrmsZ,rmeanX,rmeanY,rmeanZ,vmeanX,vmeanY,vmeanZ,epsX,epsY,epsZ,rvrmsX,rvrmsY,rvrmsZ" << endl;
    }//header
    	csvout <<iteration<<" "<<rrms<<" "<<vrms<<" "<<rmean<<" "<<vmean<<" "<<eps<<" "<<rvrms<< endl;
  }//output
}//function



//======================================================================================== 
// MAIN				UNIQUE POINTERS CANT BE COPIED!!!!
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
    const double boxL 		    = std::atof(argv[5]);
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
 	
    int rank;
    MPI_Comm_rank(Ippl::getComm(),&rank);
   
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
        
        << "Griddimensions 	= " << std::setw(20) << nr << endl
        << "Beamradius     	= " << std::setw(20) << beamRadius << endl
	<< "Box Length	   	= " << std::setw(20) << boxL << endl
        << "Total Particles 	= " << std::setw(20) << nP << endl 
	<< "Interaction Radius 	= " << std::setw(20) << interactionRadius << endl
        << "Alpha 		= " << std::setw(20) << alpha << endl
        << "Time Step  		= " << std::setw(20) << dt << endl      
	<< "total Timesteps 	= " << std::setw(20) << nt << endl
	<< "Particlecharge 	= " << std::setw(20) << particleCharge << endl
	<< "Particlemass       	= " << std::setw(20) << particleMass << endl
        << "focusing force  	= " << std::setw(20) << focusForce << endl
	<< "printing Intervall 	= " << std::setw(20) << printInterval << endl;


//================================================================================== 
// MESH & DOMAIN_DECOMPOSITION

   	 using bunch_type = ChargedParticles<PLayout_t>;



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
    const double L = boxL*0.5;
    Vector_t box_boundaries_lower({-L, -L, -L});
    Vector_t box_boundaries_upper({L, L, L});  
    Vector_t origin({0.0, 0.0, 0.0});
    Vector_t hr = {boxL/nr[0], boxL/nr[1], boxL/nr[2]}; //spacing
    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);
    const double Q = nP * (-particleCharge);
    
 
    	std::unique_ptr<bunch_type>  P;
   	P = std::make_unique<bunch_type>(PL,hr,box_boundaries_lower, box_boundaries_upper,decomp,Q);
//	bunch_type *P;
//   	P = new bunch_type(PL,hr, box_boundaries_lower, box_boundaries_upper, decomp,Q);
    
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


    IpplTimings::startTimer(particleCreation); 

    if(rank == 0) 
	    createParticleDistributionColdSphere(*P, beamRadius, nP, particleCharge);    
    

    Kokkos::fence(); //??
    Ippl::Comm->barrier();  //??
    
    msg << "Cold Sphere created" << endl;
    
    PL.update(*P, bunchBuffer);
    msg << "Layout updated"  << endl;
   

    IpplTimings::stopTimer(particleCreation);    
    msg << "particles created and initial conditions assigned " << endl;

    Vector_t avgEF(compAvgSCForce(*P, nP));

// PARTICLE CREATION
//====================================================================================== 
// TEST TIMERS

    IpplTimings::startTimer(DummySolveTimer);
    P->rho_m = 0.0;
    P->solver_mp->solve();
    IpplTimings::stopTimer(DummySolveTimer);

    P->scatterCIC(nP, 0, hr);
	msg << "scatter tested" << endl;

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);
	msg << "solve() tested" << endl;	
    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLangevin(0, particleMass, nP);;
    P->gatherStatistics(nP);
    //P->dumpLocalDomains(FL, 0);
    IpplTimings::stopTimer(dumpDataTimer);

//TEST TIMERS
//====================================================================================== 
// TIMELOOP

    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        //  'kick-drift-kick' form;
        //TODO: work in charge and mass for other ratio than 1?

        // kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E * particleCharge;
        IpplTimings::stopTimer(PTimer);

        //drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P / particleMass;
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
        
        //avgEF = compAvgSCForce(*P, nP);
        applyConstantFocusing(*P, focusForce, beamRadius, avgEF);
	
        //LANGEVIN COLLISIONS<-
	//compute_temperature(*P, particleMass, nP);
	
    
   	//error if variable not used..
	double tmp = interactionRadius;
	tmp += 1;

	// =================MYSTUFF==================================================================
        
        //kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;

        if (it%printInterval==0){
            IpplTimings::startTimer(dumpDataTimer);
            P->dumpLangevin(it+1, particleMass, nP);
            // P->gatherStatistics(nP);
            IpplTimings::stopTimer(dumpDataTimer);
        }
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
// Langevin Collision Operator Test
// Usage:

