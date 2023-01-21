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


//========================================================================================= 


double mydotpr(Vector_t a, Vector_t b){return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
double mynorm(Vector_t a){return sqrt(mydotpr(a,a)); }
	// auto mydotpr = [](Vector_t a, Vector_t b)   {
	// 	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; 
	// };
    // auto mynorm = [](Vector_t a){
	// 	return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); 
	// };
///////////////////////////////////////////////////////////////////////////////////////////

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
   	m << "start" << endl;

    P.create(Nparticle);
	

    // std::mt19937 generator(0);
    std::default_random_engine generator(0);

    std::normal_distribution<double> normdistribution(0,1.0);
    std::uniform_real_distribution<double> unidistribution(0,1);

    auto normal = std::bind(normdistribution, generator);
    auto uni = std::bind(unidistribution, generator);

    Vector_t source({0,0,0});
	Vector_t mom({0,0,0});

	typename bunch::particle_position_type::HostMirror pRHost = P.R.getHostMirror();
	Kokkos::deep_copy(pRHost, P.R.getView()); //not necessary??

	for (unsigned i = 0; i<Nparticle; ++i) {
         		Vector_t X({normal(),normal(),normal()});
                Vector_t mullerBall = pow(uni(),1.0/3.0)*X/mynorm(X);
         		Vector_t pos = source + beamRadius*mullerBall;
       	 		pRHost(i) = pos; 
	}

	Kokkos::deep_copy(P.R.getView(), pRHost);
	P.q = -qi;
    P.P = mom;
    P.fv = 1.0;
   	m << "finished" << endl;
}


    // () or [] is indifferent
	//	m << "()"<< pRHost(0)(0) << pRHost(0)(1) << pRHost(0)(2) << endl;
	//	m << "[]"<< pRHost[0](0) << pRHost[0](1) << pRHost[0](2) << endl;
///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename bunch>
Vector_t compAvgSCForce(bunch& P, const size_type N, double beamRadius ) {  
    Inform m("computeAvgSpaceChargeForces ");

	Vector_t avgEF;
	double locEFsum[Dim]={};
	double globEFsum[Dim];
    double locQ, globQ;
    double locCheck, globCheck;

	auto pEView = P.E.getView();
    auto pqView = P.q.getView();
    auto pRView = P.R.getView();

	for(unsigned d = 0; d<Dim; ++d){
		Kokkos::parallel_reduce("get local EField sum", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& lefsum){   
              		                    lefsum += fabs(pEView(i)[d]);
                                	 },                    			
					  Kokkos::Sum<double>(locEFsum[d])
					);
	}
        Kokkos::parallel_reduce("check charge", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& qsum){
                                        qsum += pqView[i];
                                	 },                    			
					  Kokkos::Sum<double>(locQ)
					);
        Kokkos::parallel_reduce("check  positioning", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& check){
                                        check += int(mynorm(pRView[i]) <= beamRadius);
                                	 },                    			
					  Kokkos::Sum<double>(locCheck)
					);

	Kokkos::fence();
	MPI_Allreduce(locEFsum, globEFsum, Dim , MPI_DOUBLE, MPI_SUM, Ippl::getComm());
	MPI_Allreduce(&locQ, &globQ, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());	
	MPI_Allreduce(&locCheck, &globCheck, 1 , MPI_DOUBLE, MPI_SUM, Ippl::getComm());	

    for(unsigned d=0; d<Dim; ++d) avgEF[d] =  globEFsum[d]/N; 

    m << "Position Check = " <<  globCheck << endl;
    m << "globQ = " <<  globQ << endl;
    m << "globSumEF = " <<  globEFsum[0] << " " << globEFsum[1] << " " << globEFsum[2] << endl;
    m << "AVG Electric SC Force = " << avgEF << endl;

    return avgEF;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename bunch>
void applyConstantFocusing( bunch& P,
                            const double f,
                            const double beamRadius,
                            const Vector_t avgEF
                                ) {
	auto pEView = P.E.getView();
	auto pRView = P.R.getView();

	double tmp = mynorm(avgEF)*f/beamRadius;
	Kokkos::parallel_for("Apply Constant Focusing",
				P.getLocalNum(),
				KOKKOS_LAMBDA(const int i){
					pEView(i) += pRView(i)*tmp;
				}	
	);
	Kokkos::fence();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
 //add a buffer zone??

template<typename bunch>
bool setVmaxmin( bunch& P, double& ext, const double& relb){

    double vmax_loc[Dim];
    double vmin_loc[Dim];
    double vmax[Dim];
    double vmin[Dim];

	auto pPView = P.P.getView();
	for(unsigned d = 0; d<Dim; ++d){

		Kokkos::parallel_reduce("vel max", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::max(pPView(i)[d], mm);
                                	 },                    			
					  Kokkos::Max<double>(vmax_loc[d])
					);

		Kokkos::parallel_reduce("vel min", 
					 P.getLocalNum(),
					 KOKKOS_LAMBDA(const int i, double& mm){   
              		                    mm = std::min(pPView(i)[d], mm);

                                	 },                    			
					  Kokkos::Min<double>(vmin_loc[d])
					);
	}
	Kokkos::fence();
	MPI_Allreduce(vmax_loc, vmax, Dim , MPI_DOUBLE, MPI_MAX, Ippl::getComm());
	MPI_Allreduce(vmin_loc, vmin, Dim , MPI_DOUBLE, MPI_MIN, Ippl::getComm());
    Ippl::Comm->barrier();

    bool change_vgrid = false;
    double max, min, next;
    max = std::max(std::max(vmax[0], vmax[1]), vmax[2]);
    min = std::min(std::min(vmin[0], vmin[1]), vmin[2]);
    next = std::max(fabs(min), fabs(max));

    if(next>ext){
        ext = relb*next;
        for(unsigned d = 0; d<Dim; ++d){
            P.vmax_mv[d]= next;
            P.vmin_mv[d]=-next;
        }
        change_vgrid=true;    
    }

    return change_vgrid;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename bunch>
// void setBC(bunch& P){
//         auto pDCView     = P.diffusionCoeff_mv.getView();
//         const int nghost = P.fv_mv.getNghost(); // ???

//     Kokkos::parallel_for("write over diff coeff from matrix to vector array",
//                     Kokkos::MDRangePolicy<Kokkos::Rank<1>>({nghost},
//                                                             {   pDCView.extent(0) - nghost,
// 							                                    pDCView.extent(1) - nghost,
// 							                                    pDCView.extent(2) - nghost}),
// 		            KOKKOS_LAMBDA(const int i, const int j, const int k){
            
//                         pDCA0View(i,j,k) = pDCView(i,j,k)[0];
//                         pDCA1View(i,j,k) = pDCView(i,j,k)[1];
//                         pDCA2View(i,j,k) = pDCView(i,j,k)[2];
//                     }
//     );
//     Kokkos::fence();
// }


template<typename bunch>
void prepareDiffCoeff(bunch& P){
        auto pDCView     = P.diffusionCoeff_mv.getView();
        auto pDCA0View   = P.diffCoeffArr_mv[0].getView();
        auto pDCA1View   = P.diffCoeffArr_mv[1].getView();
        auto pDCA2View   = P.diffCoeffArr_mv[2].getView();
        const int nghost = P.fv_mv.getNghost(); // ???

    Kokkos::parallel_for("write over diff coeff from matrix to vector array",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
                                                            {   pDCView.extent(0) - nghost,
							                                    pDCView.extent(1) - nghost,
							                                    pDCView.extent(2) - nghost}),
		            KOKKOS_LAMBDA(const int i, const int j, const int k){
            
                        pDCA0View(i,j,k) = pDCView(i,j,k)[0];
                        pDCA1View(i,j,k) = pDCView(i,j,k)[1];
                        pDCA2View(i,j,k) = pDCView(i,j,k)[2];
                    }
    );
    Kokkos::fence();
}
////////////////////////////////////////////////////////////////////////////////////////////////

//rowmajor...
template<typename V>
Matrix_t cholesky( V& d0, V& d1, V& d2){
        
        Matrix_t LL;
        V* D[] = {&d0, &d1, &d2}; 
        double epszero = DBL_EPSILON; //1e-5; //DBL_EPSILON   or -10


        //make symmetric ... is the gathering of symmetric matrices symmetric..
        //might not achieve anything here....
        int da[]={0,1,2};
        std::pair<int, int> PAIRs[6] = {{0,1},{1,0},{1,2},{2,1},{0,2},{2,0}};
        for( std::pair<int, int> pa : PAIRs){
            if(std::isnan( (*D[pa.first])(pa.second)) ){
                if(std::isnan( (*D[pa.second])(pa.first)) )
                    assert(false && "both sym element are nan");
                else    (*D[pa.first])(pa.second) = (*D[pa.second])(pa.first);
            }
        }
        
        for(int ii : da){
            if (std::isnan((*D[ii])(ii))){
                if(std::isnan((*D[(ii+1)%3])((ii+1)%3))){
                    if(std::isnan((*D[(ii)+2%3])((ii+2)%3))){
                        assert( false && "all diags are nan");
                    }
                    else (*D[ii])(ii) =  (*D[(ii+2)%3])((ii+2)%3);
                }
                else (*D[ii])(ii) =  (*D[(ii+1)%3])((ii+1)%3);
            }
        }

        // (*D[0])(1) = (*D[1])(0) = ((*D[0])(1)+(*D[1])(0)) *0.5;
        // (*D[0])(2) = (*D[2])(0) = ((*D[0])(2)+(*D[2])(0)) *0.5;
        // (*D[1])(2) = (*D[2])(1) = ((*D[1])(2)+(*D[2])(1)) *0.5; 
        // assert(     (fabs((*D[0])(1)-(*D[1])(0))) <= epszero &&
        //             (fabs((*D[0])(2)-(*D[2])(0))) <= epszero &&
        //             (fabs((*D[1])(2)-(*D[2])(1))) <= epszero 
        // );

        auto finish_LL = [&](const unsigned i0, const unsigned i1, const unsigned i2){
                // LL(0)(0) = sqrt(*D[i0])(i0));
                // LL(0)(1) = *D[i0])(i1)/LL(i0)(i0);
                LL(i0)(i2) = (*D[i0])(i2)/LL(i0)(i0);
                LL(i1)(i0) = 0.0;
                // LL(1)(1) = sqrt(*D[i1])(i1)- pow(LL(i0)(i1), 2));
                LL(i1)(i2) = ((*D[i1])(i2) - LL(i0)(i1)*LL(i0)(i2))/LL(i1)(i1);
                LL(i2)(i0) = 0.0;
                LL(i2)(i1) = 0.0;
                LL(i2)(i2) = sqrt( (*D[i2])(i2) - pow(LL(i0)(i2), 2) - pow(LL(i1)(i2), 2)) ;
        };

        auto get_2_diag = [&](const unsigned i0, const unsigned i1){//, const unsigned i2){
            return sqrt((*D[i1])(i1)- pow( LL(i0)(i1)=(*D[i0])(i1)/LL(i0)(i0), 2));
        };

            bool alternative = false;

            if     (( epszero <=fabs(LL(0)(0)=sqrt(d0(0)))      )&&(    epszero <= fabs( LL(1)(1) = get_2_diag(0,1/*,2*/) )       ))finish_LL(0, 1, 2);
            // else if(( epszero <=fabs(LL(0)(0))                  )&&(    epszero <= fabs( LL(2)(2) = get_2_diag(0,2/*,1*/) )       ))finish_LL(0, 2, 1);
            // else if(( epszero <=fabs(LL(1)(1)=sqrt(d1(1)))      )&&(    epszero <= fabs( LL(0)(0) = get_2_diag(1,0/*,2*/) )       ))finish_LL(1, 0, 2);
            // else if(( epszero <=fabs(LL(1)(1))                  )&&(    epszero <= fabs( LL(2)(2) = get_2_diag(1,2/*,0*/) )       ))finish_LL(1, 2, 0);
            // else if(( epszero <=fabs(LL(2)(2)=sqrt(d2(2)))      )&&(    epszero <= fabs( LL(1)(1) = get_2_diag(2,1/*,0*/) )       ))finish_LL(2, 1, 0);
            // else if(( epszero <=fabs(LL(2)(2))                  )&&(    epszero <= fabs( LL(0)(0) = get_2_diag(2,0/*,1*/) )       ))finish_LL(2, 0, 1);
            else{
                // how often assoon as they leave their orig cell...
                // std::cout << "no cholesky for: " << std::endl;
                // for(int id : da){
                //     for(int jd : da)
                //     std::cout << (*D[id])(jd) << " ";
                //     std::cout << std::endl;
                // }
                // assert(false && "no cholesky decomposition possible for at least one particle");

                alternative = true;

            }


            for(unsigned di = 0; di<Dim; ++di){
                for(unsigned dj = 0; dj<Dim; ++dj){
                    const double LLL = LL[di][dj];
                    if(std::isnan(LLL)){
                        // LL[di][dj]=333.333;
                        alternative = true;
                    }
                }
            }

            if(alternative){
                for(unsigned di = 0; di<Dim; ++di){
                for(unsigned dj = 0; dj<Dim; ++dj){
                    LL(di)(dj)=0.0;
                }}
                for(int di : da)   LL(di)(di) = sqrt(fabs((*D[di])(di)));




            }

        return LL;
}



        // // //make symmetric ... is the gathering of symmetric matrices symmetric..
        // (*D[0])(1) = (*D[1])(0) = ((*D[0])(1)+(*D[1])(0)) *0.5;
        // (*D[0])(2) = (*D[2])(0) = ((*D[0])(2)+(*D[2])(0)) *0.5;
        // (*D[1])(2) = (*D[2])(1) = ((*D[1])(2)+(*D[2])(1)) *0.5; 

                    // if( 
            // for(unsigned di = 0; di<Dim; ++di){
            //     for(unsigned dj = 0; dj<Dim; ++dj){
            //         // const double LLL = LL(di)(dj);
            //         if(std::isinf(LLL))
            //         LL(di)(dj)=0.0;
            //     }
            // }
                        // std::isnan(LLL)   
                        // || std::isinf(LLL)
                        // || (LLL>1e20 ||LLL<-1e20)
                        // || (LLL<1e-15&&LLL>-1e-15)
                    // )                                 

//rowmajor
// template<typename M, typename V>
// void cholesky(M& LL, const V& d0, const V& d1, const V& d2){
    
//         LL[0][0] = sqrt(d0[0]);
//         LL[0][1] = d0[1]/LL[0][0];
//         LL[0][2] = d0[2]/LL[0][0];
//         LL[1][0] = 0.0;
//         LL[1][1] = sqrt(d1[1]- pow(LL[0][1], 2));
//         LL[1][2] = (d1[2] - LL[0][1]*LL[0][2])/LL[1][1];
//         LL[2][0] = 0.0;
//         LL[2][1] = 0.0;
//         LL[2][2]= sqrt( d2[2] - pow(LL[0][1], 2) - pow(LL[0][2], 2));
// }

Vector_t  GeMV_t(const Matrix_t& M, const Vector_t V){return V[0]*M[0]+V[1]*M[1]+V[2]*M[2];}


    // P->P = P->P + GeMV_t(cholesky(P->D0, P->D1, P->D2), Gaussian3d());
template<typename bunch>
void applyLangevin(bunch& P, std::function<Vector_t()> Gaussian3d){

    auto pPView =  P.P.getView();
	auto pD0View = P.D0.getView();
	auto pD1View = P.D1.getView();
	auto pD2View = P.D2.getView();

	Kokkos::parallel_for("Apply Langevin",
				P.getLocalNum(),
				KOKKOS_LAMBDA(const int i){
                     pPView(i) += GeMV_t(cholesky(pD0View(i), pD1View(i), pD2View(i)), Gaussian3d());
	 			}	
	 );
	 Kokkos::fence();
}


//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
// MAIN	

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
   
    int rank;
    // MPI_Comm_rank(Ippl::getComm(),&rank);
    rank = Ippl::Comm->rank();
 
    Inform msg("Langevin");
    Inform msg2all("Langevin ",INFORM_ALL_NODES);
    Ippl::Comm->setDefaultOverallocation(std::atof(argv[3]));
    auto start = std::chrono::high_resolution_clock::now();

    // make unsymmetric?

    //1 -> solvertype = FFT...1
    //2 -> loadbalancethreshold2
    //3 -> default overallocation3
    const int nr                    = std::atoi(argv[4]);
    const double beamRadius         = std::atof(argv[5]);
    const double boxL 		        = std::atof(argv[6]); 
    const size_type nP              = std::atoll(argv[7]);
    const double interactionRadius  = std::atof(argv[8]);
    const double alpha              = std::atof(argv[9]);
    const double dt                 = std::atof(argv[10]);
    const size_type nt              = std::atoll(argv[11]);
    const double particleCharge     = std::atof(argv[12]);
    const double particleMass       = std::atof(argv[13]);
    const double focusForce         = std::atof(argv[14]);
    const int printInterval         = std::atoi(argv[15]);
    const double eps_inv            = std::atof(argv[16]);
    const int NV                    = std::atoi(argv[17]);
    const double VMAX               = std::atof(argv[18]);
    const double rel_buff           = std::atof(argv[19]);

    const bool FFF                  = std::atoi(argv[20]);
    const bool EEE                  = std::atoi(argv[21]);
    const bool DDD                  = std::atoi(argv[22]);
    const bool CCC                  = std::atoi(argv[23]);
    const double BBB                  = std::atoi(argv[24]);
    const bool DRAG                 = std::atoi(argv[25]);
    const bool DIFFUSION            = std::atoi(argv[26]);
    const bool PRINT                = std::atoi(argv[27]);
    const bool COLLISION            = std::atoi(argv[28]);
    std::string folder              = argv[29];
 	
     
    //cm annahme, elektronen charge mass, / nicht milisekunde...
    // const double ke=2.532638e8;
    // const double ke   =1./(4.*M_PI*8.8541878128e-14);  
    //                  = 8.9875517923e11;
    // const double ke=2.532638e9;
    // const double ke=3e9;
    // const double ke=9e9;
    //SI // const double ke = 8.9875517923e9;// =1./(4.*M_PI*8.8541878128e-12); 

   
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
    msg << "Start test: LANGEVIN COLLISION OPERATOR" << endl;


//================================================================================== 
// SPATIAL MESH & DOMAIN_DECOMPOSITION

    //box grösse mit 3 Längen parametrisieren TODO

    using bunch_type = ChargedParticles<PLayout_t>;
    //initializing number of cells in mesh/domain
    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr);
    }
    //initializing boundary conditions
    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }
    Vector_t kw = {0.5, 0.5, 0.5}; //irregularities inside plasma
    const double L = boxL*0.5;
    Vector_t box_boundaries_lower({-L, -L, -L});
    Vector_t box_boundaries_upper({L, L, L}); 
    Vector_t origin({-L, -L, -L});
    Vector_t hr = {boxL/nr, boxL/nr, boxL/nr}; //spacing

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);
    const double Q = nP * (-particleCharge);
    
    std::unique_ptr<bunch_type>  P;
   	P = std::make_unique<bunch_type>(PL,hr,box_boundaries_lower,box_boundaries_upper,decomp,Q);
    
    P->nr_m = ippl::Vector<int,Dim>({nr, nr, nr});
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);
    bunch_type bunchBuffer(PL);
    P->stype_m = argv[1];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[2]);
    bool isFirstRepartition = false;

// ========================================================================================
// VEOCITY SPACE MESH& DOMAIN_DECOMPOSITION
    //ln(LAMBDA) = 10;
    P->pMass = particleMass;
    P->GAMMA = 10.0* pow(particleCharge, 4) * eps_inv*eps_inv / ( 4*M_PI * pow(particleMass, 2));

    P->nv_mv = {NV,NV,NV};
    P->hv_mv = { 2.0*VMAX/NV,  2.0*VMAX/NV,  2.0*VMAX/NV};
    ippl::NDIndex<Dim> domain_v;
    for (unsigned i = 0; i< Dim; i++) {
        // domain_v[i] = ippl::Index(P->nv_mv[i]);
        domain_v[i] = ippl::Index(NV);
    }
    ippl::e_dim_tag decomp_v[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        // decomp[d] = ippl::SERIAL;//PARALLEL
        decomp_v[d] = ippl::PARALLEL;
    }


    bool isVallPeriodic = false; 
    // false; nooo?

    Mesh_t          mesh_v(domain_v, P->hv_mv, Vector_t({0.0, 0.0, 0.0}));
    FieldLayout_t   FL_v(domain_v, decomp_v, isVallPeriodic);
    PLayout_t       PL_v(FL_v, mesh_v);

    P->fr_m.initialize(mesh_v, FL_v);
    P->fv_mv.initialize(mesh_v, FL_v);
    P->gradRBH_mv.initialize(mesh_v, FL_v);
    P->diffusionCoeff_mv.initialize(mesh_v, FL_v);
    P->diffCoeffArr_mv[0].initialize(mesh_v, FL_v);
    P->diffCoeffArr_mv[1].initialize(mesh_v, FL_v);
    P->diffCoeffArr_mv[2].initialize(mesh_v, FL_v);

    P->initRosenbluthSolver();
    P->vmax_mv = { VMAX,  VMAX,  VMAX};
    P->vmin_mv = {-VMAX, -VMAX, -VMAX};
    double ext = VMAX;
    for(unsigned int d = 0; d>Dim; ++d) P->hv_mv[d] = (P->vmax_mv[d]-P->vmin_mv[d])/P->nv_mv[d];
    mesh_v.setOrigin(P->vmin_mv);
    mesh_v.setMeshSpacing(P->hv_mv);



// ========================================================================================

    msg
        << "Dim                 = " << std::setw(20) << Dim <<  endl
        << "Griddimensions NR   = " << std::setw(20) << nr << endl
        << "Beamradius          = " << std::setw(20) << beamRadius << endl
        << "Box Length          = " << std::setw(20) << boxL << endl
        << "Total Particles     = " << std::setw(20) << nP << endl 
        << "Interaction Radius  = " << std::setw(20) << interactionRadius << endl
        << "Alpha               = " << std::setw(20) << alpha << endl
        << "Time Step           = " << std::setw(20) << dt << endl      
        << "total Timesteps     = " << std::setw(20) << nt << endl
        << "Particlecharge      = " << std::setw(20) << particleCharge << endl
        << "Particlemass        = " << std::setw(20) << particleMass << endl
        << "focusing force      = " << std::setw(20) << focusForce << endl
        << "printing Intervall  = " << std::setw(20) << printInterval << endl
        << "Origin              = " << std::setw(20) << origin << endl
        << "MeshSpacing         = " << std::setw(20) << hr << endl
        << "total Charge        = " << std::setw(20) << Q << endl
        << "LBT                 = " << std::setw(20) << P->loadbalancethreshold_m << endl
        << "eps_inv             = " << std::setw(20) << eps_inv << endl
        << "GridDim Vel_Mesh    = " << std::setw(20) << NV << endl
        << "FFF                 = " << std::setw(20) <<  FFF   << endl
        << "EEE                 = " << std::setw(20) <<  EEE   << endl
        << "DDD                 = " << std::setw(20) <<  DDD   << endl
        << "CCC                 = " << std::setw(20) <<  CCC   << endl
        << "BBB                 = " << std::setw(20) <<  BBB   << endl
        << "DRAG                = " << std::setw(20) <<  DRAG   << endl
        << "DIFFUSION           = " << std::setw(20) <<  DIFFUSION   << endl
        << "VMAX                = " << std::setw(20) <<  P->vmax_mv   << endl
        << "VMIN                = " << std::setw(20) <<  P->vmin_mv   << endl
        << "HV                  = " << std::setw(20) <<  P->hv_mv   << endl
        << "PRINT               = " << std::setw(20) <<  PRINT << endl
        << "COLLISIONS          = " << std::setw(20) <<  COLLISION << endl;

    

    //how to perform ORB without PDF etc
    if ((P->loadbalancethreshold_m != 1.0) && (Ippl::Comm->size() > 1)) {
        // msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);//===========================
        // isFirstRepartition = true;

        // const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        // const int nghost = P->rho_m.getNghost();
        // using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        // auto rhoview = P->rho_m.getView();

        // Kokkos::parallel_for("Assign initial rho based on PDF",
        //                       mdrange_type({nghost, nghost, nghost},
        //                                    {rhoview.extent(0) - nghost,
        //                                     rhoview.extent(1) - nghost,
        //                                     rhoview.extent(2) - nghost}),
        //                       KOKKOS_LAMBDA(const int i,
        //                                     const int j,
        //                                     const int k)
        //                       {
        //                         //local to global index conversion
        //                         const size_t ig = i + lDom[0].first() - nghost;
        //                         const size_t jg = j + lDom[1].first() - nghost;
        //                         const size_t kg = k + lDom[2].first() - nghost;
        //                         double x = (ig + 0.5) * hr[0] + origin[0];
        //                         double y = (jg + 0.5) * hr[1] + origin[1];
        //                         double z = (kg + 0.5) * hr[2] + origin[2];
        //                         Vector_t xvec = {x, y, z};
        //                         rhoview(i, j, k) = PDF(xvec, alpha, kw, Dim);
        //                       });
        // Kokkos::fence();
        // P->initializeORB(FL, mesh);
        // P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
        //=======================================================================
        IpplTimings::stopTimer(domainDecomposition);//===========================
    }
    isFirstRepartition = false;

// MESH & DOMAIN_DECOMPOSITION
//=======================================================================================
// PARTICLE CREATION


    IpplTimings::startTimer(particleCreation); 
    if(rank == 0) 
	    createParticleDistributionColdSphere(*P, beamRadius, nP, particleCharge);    
    Kokkos::fence();  Ippl::Comm->barrier();  //??//


	auto pfrHost = P->fr.getHostMirror();
	auto pD0Host = P->D0.getHostMirror();
	auto pD1Host = P->D1.getHostMirror();
	auto pD2Host = P->D2.getHostMirror();

    
    //multiple node runs stop here
    PL.update(*P, bunchBuffer);
    PL_v.update(*P, bunchBuffer);
    IpplTimings::stopTimer(particleCreation);


// PARTICLE CREATION
//====================================================================================== 
// TEST TIMERS

    IpplTimings::startTimer(DummySolveTimer);
    P->rho_m = 0.0;
    P->solver_mp->solve();
    IpplTimings::stopTimer(DummySolveTimer);

    ///////////////////////////////////////////
    
    P->scatterCIC(nP, 0, hr);
    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->E_m = P->E_m * eps_inv;

    P->gatherCIC();
	msg << "scatter()solved()gather()" << endl;	

    ///////////////////////////////////////////

    Vector_t avgEF(compAvgSCForce(*P, nP, beamRadius));



    // std::mt19937 generator(0);
    std::default_random_engine generator(0);
    std::normal_distribution<double> normdistribution(0.0, dt);
    auto gauss = std::bind(normdistribution, generator);
    auto Gaussian3d = [&gauss](){return Vector_t({gauss(), gauss(), gauss()}); };


    ///////////////////////////////////////////


    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLangevin(0,  nP, folder);
	// P->dumpParticleData();
    // P->gatherStatistics(nP);
    //P->dumpLocalDomains(FL, 0);
    IpplTimings::stopTimer(dumpDataTimer);


//TEST TIMERS
//====================================================================================== ==== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
//////////////////////////////////////////////////////////////////////////////////////////
//======================================================================================== 
// TIMELOOP START

        msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        msg << "Start time step: " << it+1 << endl;

        //Vorzeichen?? checken
        msg << "kick" <<endl;
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E * particleCharge;
        IpplTimings::stopTimer(PTimer);

        msg << "drift" << endl;
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        msg << "update" << endl;
        //Since the particles have moved spatially update them to correct processors
	    IpplTimings::startTimer(updateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(updateTimer);

        // Domain Decomposition for Positions
        if (P->balance(nP, it+1)) {
           msg << "Starting repartition" << endl;
        //    IpplTimings::startTimer(domainDecomposition);
        //    P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
        //    IpplTimings::stopTimer(domainDecomposition);
        //    //IpplTimings::startTimer(dumpDataTimer);
        //    //P->dumpLocalDomains(FL, it+1);
        //    //IpplTimings::stopTimer(dumpDataTimer);
        //     msg << "LB"<<endl; 
        }


        P->dumpLangevin(it+1,  nP, folder);

        msg << "scatter" << endl;
        P->scatterCIC(nP, it+1, hr); 
        
        P->rho_m = P->rho_m * eps_inv;

        IpplTimings::startTimer(SolveTimer);
        P->solver_mp->solve();
        IpplTimings::stopTimer(SolveTimer);


        msg << "gather" << endl;
        P->gatherCIC();


// =================MYSTUFF::CONSTANT_FOCUSING======================

        msg << "constant Focusing" << endl;
        applyConstantFocusing(*P, focusForce, beamRadius, avgEF);
// =================================================================

        msg << "kick" << endl;
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);


// =================MYSTUFF::_LANGEVIN_COLLISION======================

    if(COLLISION){msg << "COLLISION"<<endl;

        //lassen wir momentan weg und haben fixes grid welches gross genug ist
        if(FFF){msg << "FFF"<<endl;
            if(setVmaxmin(*P, ext, rel_buff)){
                for(unsigned int d = 0; d>Dim; ++d) P->hv_mv[d] = (P->vmax_mv[d]-P->vmin_mv[d])/P->nv_mv[d]; 
                mesh_v.setOrigin(P->vmin_mv);
                mesh_v.setMeshSpacing(P->hv_mv);
                msg << "VELGRID_change"<<endl; 
            }
        }



        PL_v.update(*P, bunchBuffer);
        // P->dumpLangevin(0,  nP);
        // Domain Decomposition for Positions

        P->dumpLangevin(it+1,  nP, folder);

        if(EEE){msg << "EEE"<<endl;
        
            P->scatterPhaseSpaceDist(hr); 
            P->fv_mv = -8.0*M_PI*P->fv_mv;
        }

        //origin shift
        mesh_v.setOrigin({0, 0, 0});

        if(DDD){msg << "DDD" << endl;

            P->solver_mvRB->solve();
            P->gradRBH_mv = grad(P->fv_mv);
            P->gradRBH_mv = P->GAMMA * P->gradRBH_mv;

            P->solver_mvRB->solve();
            P->diffusionCoeff_mv = hess(P->fv_mv);
            P->diffusionCoeff_mv = P->GAMMA *  P->diffusionCoeff_mv; 

            //here we have to set all values of hess and grad at the bounaries to zero because they carry big errors!!!!!
            //change of plans we set them to value bordering them!!!!!
            // setBC(*P);
        }

        //undo origin shift
        mesh_v.setOrigin(P->vmin_mv);


        if(CCC){msg << "CCC" << endl;

            prepareDiffCoeff(*P);   //does:: for(unsigned d = 0; d<Dim; ++d) P->diffCoeffArr_mv[d] = P->diffusionCoeff_mv[d];
            P->gather_Fd_D_Density();
        
        }


        Kokkos::deep_copy(pfrHost, P->fr.getView());
        msg << pfrHost(9) << " " 
            << pfrHost(1000) << " " 
            << pfrHost(5000) << " " 
            << pfrHost(10000) << " " 
            << pfrHost(50000) << " " 
            << pfrHost(100000) << " " 
            <<endl;

        if(BBB){ msg << "BBB" << endl;


            
            P->Fd = P->Fd*BBB;
            P->D0 = P->D0*BBB*90000;
            P->D1 = P->D1*BBB*90000;
            P->D2 = P->D2*BBB*90000;

            // msg << 
            
        }
        else{
            msg << "spacedensity factor P->fr" << endl;
            P->Fd = P->Fd*P->fr;
            P->D0 = P->D0*P->fr;
            P->D1 = P->D1*P->fr;
            P->D2 = P->D2*P->fr;
        }

        if(DRAG){ msg << "DRAG" << endl;
            P->P = P->P + dt*P->Fd;
        }

        if(DIFFUSION){msg << "DIFFUSION" << endl;


                // Kokkos::deep_copy(pD0Host, P->D0.getView());
                // Kokkos::deep_copy(pD1Host, P->D1.getView());
                // Kokkos::deep_copy(pD2Host, P->D2.getView());

                // Kokkos::fence();

                // // diffusion coefficient does seem to be symmetric; which makes senes maybe because the derivates at border regions dont work...

                //     int ddd[] = { 10, 134, 2345, 10555, 100000};
                //     int xyz[] = {0,1,2};
                    
                //     for(int di : ddd){
                //         for(int x : xyz){
                //             // std::cout << DView(di)[x][y] << " ";//cant access particle Views directly for printing
                //             msg << pD0Host(di)[x] << " " << pD1Host(di)[x] << " " << pD2Host(di)[x] << endl;
                //         }msg << endl;
                        
                //         auto cholll = cholesky( pD0Host(di), pD1Host(di),pD2Host(di));
                //         for(int x : xyz){
                //         for(int y : xyz){
                //             msg << cholll[x][y] << " ";
                //         } msg << endl;
                //         }msg << endl;
                //     }
                    
                // Kokkos::fence();
                // // MPI_Waitall();

            applyLangevin(*P, Gaussian3d);   //does:: // P->P = P->P + GeMV_t(cholesky(P->D0, P->D1, P->D2), Gaussian3d()); //DEAD END
        }

            PL_v.update(*P, bunchBuffer);

       	        //error if variable not used..   DIFFUSIONnd we dont use it?? ever???
                if(false){
    	        double tmp = interactionRadius;
    	        tmp += 1;
                tmp += Gaussian3d()[1];
                tmp += isFirstRepartition;
                }
    }

// =================MYSTUFF==================================================================
        
        P->time_m += dt;

        msg << "Finished time step: " << it+1 << endl;
        
        if(PRINT){ msg << "PRINT" << endl;
            if (it%printInterval==0){
            IpplTimings::startTimer(dumpDataTimer);
            P->dumpLangevin(it+1,  nP, folder);
            // P->gatherStatistics(nP);
            IpplTimings::stopTimer(dumpDataTimer);
            }
        }
    

    }

// TIMELOOP END
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






    // Vector_t zerovec({1e-3, 1e-3, 1e-3});
    // msg << "J"<<endl; 
    // P->P = P->P+zerovec*P->fv;
    // P->P = P->P+zerovec*P->fv;
    // illegal syntax both ways...