// Parallel-in-time (PinT) method Parareal combined with Particle-in-cell
// and Particle-in-Fourier schemes. The example is electrostatic Landau 
// damping. The implementation of Parareal follows the open source implementation
// https://github.com/Parallel-in-Time/PararealF90 by Daniel Ruprecht. The corresponding
// publication is Ruprecht, Daniel. "Shared memory pipelined parareal." 
// European Conference on Parallel Processing. Springer, Cham, 2017.
// 
//  Usage:
//  Usage:
//     srun ./BumponTailInstabilityPinT <nmx> <nmy> <nmz> <nx> <ny> <nz> <Np> <Tend> <dtfine> <dtcoarse> <tolParareal> 
//          <nCycles> <ShapeType> <degree> <No. of space procs> <No. of time procs> 
//          <coarseTol> <fineTol> <coarseType> --info 5
//     nmx       = No. of Fourier modes in the x-direction
//     nmy       = No. of Fourier modes in the y-direction
//     nmz       = No. of Fourier modes in the z-direction
//     nx       = No. of grid points in the x-direction (not used if PIF is also used as coarse propagator)
//     ny       = No. of grid points in the y-direction (not used if PIF is also used as coarse propagator)
//     nz       = No. of grid points in the z-direction (not used if PIF is also used as coarse propagator)
//     Np       = Total no. of macro-particles in the simulation
//     tolParareal      = Parareal tolerance
//     nCycles = No. of Parareal blocks/cycles
//     ShapeType = Shape function type B-spline only for the moment
//     degree = B-spline degree (-1 for delta function)
//     No. of space procs = Number of MPI ranks to be used in the spatial parallelization
//     No. of time procs = Number of MPI ranks to be used in the time parallelization
//     coarseTol = Coarse tolerance for PIF if we use PIF as a coarse propagator (will not be used when PIC is used)
//     fineTol = fine tolerance for PIF
//     coarseType = Type of coarse propagator (PIF or PIC)
//     Example:
//     srun ./BumponTailInstabilityPinT 32 32 32 16 16 16 655360 19.2 0.05 0.05 1e-5 1 B-spline 1 4 16 1e-2 1e-4 PIC --info 5
//
// Copyright (c) 2022, Sriramkrishnan Muralikrishnan,
// Jülich Supercomputing Centre, Jülich, Germany.
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

#include "ChargedParticlesPinT.hpp"
#include "StatesBeginSlice.hpp"
#include "StatesEndSlice.hpp"
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
  
  T k, delta, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& k_, const T& delta_, 
           const T& u_) 
  : k(k_), delta(delta_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = x  + (delta  * (std::sin(k * x) / k)) - u;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = 1  + (delta  * std::cos(k * x));
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

  value_type delta, sigma, muBulk, muBeam;
  size_type nlocBulk; 

  T k, minU, maxU;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  value_type& delta_, T& k_, value_type& sigma_, 
                  value_type& muBulk_, value_type& muBeam_, 
                  size_type& nlocBulk_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        delta(delta_), sigma(sigma_), muBulk(muBulk_), muBeam(muBeam_),
        nlocBulk(nlocBulk_), k(k_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    bool isBeam = (i >= nlocBulk);
    
    value_type muZ = (value_type)(((!isBeam) * muBulk) + (isBeam * muBeam));
    
    for (unsigned d = 0; d < Dim-1; ++d) {
        
        x(i)[d] = rand_gen.drand(minU[d], maxU[d]); 
        v(i)[d] = rand_gen.normal(0.0, sigma);
    }
    v(i)[Dim-1] = rand_gen.normal(muZ, sigma);
    
    value_type u = rand_gen.drand(minU[Dim-1], maxU[Dim-1]);
    x(i)[Dim-1] = u / (1 + delta);
    Newton1D<value_type> solver(k[Dim-1], delta, u);
    solver.solve(x(i)[Dim-1]);
    

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& delta, const double& k,
           const unsigned& dim) {

   bool isDimZ = (dim == (Dim-1)); 
   double cdf = x + (double)(isDimZ * ((delta / k) * std::sin(k * x)));
   return cdf;
}

double computeRL2Error(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, 
                      Vector_t& length, MPI_Comm& spaceComm) {
    
    auto Qview = Q.getView();
    auto QprevIterView = QprevIter.getView();
    double localError = 0.0;
    double localNorm = 0.0;

    Kokkos::parallel_reduce("Abs. error and norm", Q.size(),
                            KOKKOS_LAMBDA(const int i, double& valLError, double& valLnorm){
                                Vector_t diff = Qview(i) - QprevIterView(i);

                                //This is just to undo the effect of periodic BCs during the 
                                //error calculation. Otherwise even though the actual error is 
                                //small the computed error might be very large. 
                                //The values (e.g. 10) mentioned here are just an adhoc
                                //value depending on the domain length. 
                                for (unsigned d = 0; d < 3; ++d) {
                                    bool isLeft = (diff[d] <= -10.0);
                                    bool isRight = (diff[d] >= 10.0);
                                    bool isInside = ((diff[d] > -10.0) && (diff[d] < 10.0));
                                    diff[d] = (isInside * diff[d]) + (isLeft * (diff[d] + length[d]))
                                              +(isRight * (diff[d] - length[d]));
                                }

                                double myValError = dot(diff, diff).apply();
                                valLError += myValError;
                                double myValnorm = dot(Qview(i), Qview(i)).apply();
                                valLnorm += myValnorm;
                            }, Kokkos::Sum<double>(localError), Kokkos::Sum<double>(localNorm));

    Kokkos::fence();
    double globalError = 0.0;
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    double globalNorm = 0.0;
    MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, spaceComm);

    double relError = std::sqrt(globalError) / std::sqrt(globalNorm);
    
    return relError;

}

double computePL2Error(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, MPI_Comm& spaceComm) {
    
    auto Qview = Q.getView();
    auto QprevIterView = QprevIter.getView();
    double localError = 0.0;
    double localNorm = 0.0;

    Kokkos::parallel_reduce("Abs. error and norm", Q.size(),
                            KOKKOS_LAMBDA(const int i, double& valLError, double& valLnorm){
                                Vector_t diff = Qview(i) - QprevIterView(i);
                                double myValError = dot(diff, diff).apply();
                                valLError += myValError;
                                double myValnorm = dot(Qview(i), Qview(i)).apply();
                                valLnorm += myValnorm;
                            }, Kokkos::Sum<double>(localError), Kokkos::Sum<double>(localNorm));

    Kokkos::fence();
    double globalError = 0.0;
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    double globalNorm = 0.0;
    MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, spaceComm);

    double relError = std::sqrt(globalError) / std::sqrt(globalNorm);
    
    return relError;

}

const char* TestName = "TwoStreamInstability";
//const char* TestName = "BumponTailInstability";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);

    int spaceColor, timeColor;
    MPI_Comm spaceComm, timeComm;

    int spaceProcs = std::atoi(argv[15]);
    int timeProcs = std::atoi(argv[16]);
    spaceColor = Ippl::Comm->rank() / spaceProcs; 
    timeColor = Ippl::Comm->rank() % spaceProcs;

    MPI_Comm_split(Ippl::getComm(), spaceColor, Ippl::Comm->rank(), &spaceComm);
    MPI_Comm_split(Ippl::getComm(), timeColor, Ippl::Comm->rank(), &timeComm);

    int rankSpace, sizeSpace, rankTime, sizeTime;
    MPI_Comm_rank(spaceComm, &rankSpace);
    MPI_Comm_size(spaceComm, &sizeSpace);

    MPI_Comm_rank(timeComm, &rankTime);
    MPI_Comm_size(timeComm, &sizeTime);


    Inform msg(TestName, Ippl::Comm->size()-1);
    Inform msg2all(TestName,INFORM_ALL_NODES);

    ippl::Vector<int,Dim> nmPIF = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    ippl::Vector<int,Dim> nrPIC = {
        std::atoi(argv[4]),
        std::atoi(argv[5]),
        std::atoi(argv[6])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef timeCommunication = IpplTimings::getTimer("timeCommunication");
    static IpplTimings::TimerRef deepCopy = IpplTimings::getTimer("deepCopy");
    static IpplTimings::TimerRef finePropagator = IpplTimings::getTimer("finePropagator");
    static IpplTimings::TimerRef coarsePropagator = IpplTimings::getTimer("coarsePropagator");
    static IpplTimings::TimerRef dumpData = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef computeErrors = IpplTimings::getTimer("computeErrors");
    static IpplTimings::TimerRef initializeShapeFunctionPIF = IpplTimings::getTimer("initializeShapeFunctionPIF");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[7]);
    const double tEnd = std::atof(argv[8]);
    const unsigned int nCycles = std::atoi(argv[12]);
    double tEndCycle = tEnd / nCycles;
    const double dtSlice = tEndCycle / sizeTime;
    const double dtFine = std::atof(argv[9]);
    const double dtCoarse = std::atof(argv[10]);
    const unsigned int ntFine = std::ceil(dtSlice / dtFine);
    const unsigned int ntCoarse = std::ceil(dtSlice / dtCoarse);
    const double tol = std::atof(argv[11]);

    using bunch_type = ChargedParticlesPinT<PLayout_t>;
    using states_type = StatesSlice<PLayout_t>;

    std::unique_ptr<bunch_type>  Pcoarse;
    std::unique_ptr<states_type>  Pbegin;
    std::unique_ptr<states_type>  Pend;

    ippl::NDIndex<Dim> domainPIC;
    ippl::NDIndex<Dim> domainPIF;
    for (unsigned i = 0; i< Dim; i++) {
        domainPIC[i] = ippl::Index(nrPIC[i]);
        domainPIF[i] = ippl::Index(nmPIF[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::SERIAL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t kw;
    double sigma, muBulk, muBeam, epsilon, delta;
    
   
    if(std::strcmp(TestName,"TwoStreamInstability") == 0) {
        // Parameters for two stream instability as in 
        //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
        kw = {0.5, 0.5, 0.5};
        sigma = 0.1;
        epsilon = 0.5;
        muBulk = -pi / 2.0;
        muBeam = pi / 2.0;
        delta = 0.01;
    }
    else if(std::strcmp(TestName,"BumponTailInstability") == 0) {
        kw = {0.21, 0.21, 0.21};
        sigma = 1.0 / std::sqrt(2.0);
        epsilon = 0.1;
        muBulk = 0.0;
        muBeam = 4.0;
        delta = 0.01;
    }
    else {
        //Default value is two stream instability
        kw = {0.5, 0.5, 0.5};
        sigma = 0.1;
        epsilon = 0.5;
        muBulk = -pi / 2.0;
        muBeam = pi / 2.0;
        delta = 0.01;
    }
    Vector_t rmin(0.0);
    Vector_t rmax = (2 * pi / kw);
    Vector_t length = rmax - rmin;
    double dxPIC = length[0] / nrPIC[0];
    double dyPIC = length[1] / nrPIC[1];
    double dzPIC = length[2] / nrPIC[2];


    double dxPIF = length[0] / nmPIF[0];
    double dyPIF = length[1] / nmPIF[1];
    double dzPIF = length[2] / nmPIF[2];
    Vector_t hrPIC = {dxPIC, dyPIC, dzPIC};
    Vector_t hrPIF = {dxPIF, dyPIF, dzPIF};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    const bool isAllPeriodic=true;
    Mesh_t meshPIC(domainPIC, hrPIC, origin);
    Mesh_t meshPIF(domainPIF, hrPIF, origin);
    FieldLayout_t FLPIC(domainPIC, decomp, isAllPeriodic);
    FieldLayout_t FLPIF(domainPIF, decomp, isAllPeriodic);
    PLayout_t PL(FLPIC, meshPIC);


    double factorVelBulk = 1.0 - epsilon;
    double factorVelBeam = 1.0 - factorVelBulk;
    double factorConf = 1.0 / sizeSpace;
    size_type nlocBulk = (size_type)(factorConf * factorVelBulk * totalP);
    size_type nlocBeam = (size_type)(factorConf * factorVelBeam * totalP);
    size_type nloc = nlocBulk + nlocBeam;
   
    size_type Total_particles = 0;

    MPI_Allreduce(&nloc, &Total_particles, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, spaceComm);
    
    //Q = -\int\int f dx dv
    double Q = -length[0] * length[1] * length[2];
    Pcoarse = std::make_unique<bunch_type>(PL,hrPIC,rmin,rmax,decomp,Q,Total_particles);
    Pbegin = std::make_unique<states_type>(PL);
    Pend = std::make_unique<states_type>(PL);

    Pcoarse->nr_m = nrPIC;
    Pcoarse->nm_m = nmPIF;

    Pcoarse->rhoPIF_m.initialize(meshPIF, FLPIF);
    Pcoarse->Sk_m.initialize(meshPIF, FLPIF);

    if(Pcoarse->coarsetype_m == "PIC") {
        Pcoarse->rhoPIC_m.initialize(meshPIC, FLPIC);
        Pcoarse->EfieldPIC_m.initialize(meshPIC, FLPIC);
        Pcoarse->initFFTSolver();
    }
   
    ////////////////////////////////////////////////////////////
    //Initialize an FFT object for getting rho in real space and 
    //doing charge conservation check
    
    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", false);  
    fftParams.add("use_pencils", true);  
    fftParams.add("use_reorder", false);  
    fftParams.add("use_gpu_aware", true);  
    fftParams.add("comm", ippl::p2p_pl);  
    fftParams.add("r2c_direction", 0);  

    ippl::NDIndex<Dim> domainPIFhalf;

    for(unsigned d = 0; d < Dim; ++d) {
        if(fftParams.template get<int>("r2c_direction") == (int)d)
            domainPIFhalf[d] = ippl::Index(domainPIF[d].length()/2 + 1);
        else
            domainPIFhalf[d] = ippl::Index(domainPIF[d].length());
    }
    

    FieldLayout_t FLPIFhalf(domainPIFhalf, decomp);

    ippl::Vector<double, 3> hDummy = {1.0, 1.0, 1.0};
    ippl::Vector<double, 3> originDummy = {0.0, 0.0, 0.0};
    Mesh_t meshPIFhalf(domainPIFhalf, hDummy, originDummy);

    Pcoarse->rhoPIFreal_m.initialize(meshPIF, FLPIF);
    Pcoarse->rhoPIFhalf_m.initialize(meshPIFhalf, FLPIFhalf);

    Pcoarse->fft_mp = std::make_shared<FFT_t>(FLPIF, FLPIFhalf, fftParams);
   
    ////////////////////////////////////////////////////////////

    Vector_t minU, maxU;
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], delta, kw[d], d);
        maxU[d]   = CDF(rmax[d], delta, kw[d], d);
    }

    using buffer_type = ippl::Communicate::buffer_type;
    int tag;

    Pcoarse->shapetype_m = argv[13];
    Pcoarse->shapedegree_m = std::atoi(argv[14]); 
    IpplTimings::startTimer(initializeShapeFunctionPIF);
    Pcoarse->initializeShapeFunctionPIF();
    IpplTimings::stopTimer(initializeShapeFunctionPIF);
    
    IpplTimings::startTimer(particleCreation);

    Pcoarse->create(nloc);
    Pbegin->create(nloc);
    Pend->create(nloc);

    Pcoarse->q = Pcoarse->Q_m/Total_particles;
   
    IpplTimings::stopTimer(particleCreation);

    
    double coarseTol = std::atof(argv[17]);
    double fineTol   = std::atof(argv[18]);
    Pcoarse->initNUFFTs(FLPIF, coarseTol, fineTol);
    std::string coarse = "Coarse";
    std::string fine = "Fine";
    
    IpplTimings::startTimer(particleCreation);

#ifdef KOKKOS_ENABLE_CUDA

    //If we don't do the following even with the same seed the initial 
    //condition is not the same on different GPUs

    IpplTimings::startTimer(timeCommunication);
    //For some reason using the next_tag with multiple cycles is not 
    //working so we use static tags here
    tag = 500;//Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);

    if(rankTime == 0) {
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*rankSpace));
        Kokkos::parallel_for(nloc,
                             generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                             Pbegin->R.getView(), Pbegin->P.getView(), rand_pool64, delta, kw, 
                             sigma, muBulk, muBeam, nlocBulk, minU, maxU));


        Kokkos::fence();
    }
    else {
        size_type bufSize = Pbegin->packedSize(nloc);
        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
        Ippl::Comm->recv(rankTime-1, tag, *Pbegin, *buf, bufSize, nloc, timeComm);
        buf->resetReadPos();
    }
    IpplTimings::stopTimer(timeCommunication);

    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pend->R.getView(), Pbegin->R.getView());
    Kokkos::deep_copy(Pend->P.getView(), Pbegin->P.getView());
    Kokkos::deep_copy(Pcoarse->R0.getView(), Pbegin->R.getView());
    Kokkos::deep_copy(Pcoarse->P0.getView(), Pbegin->P.getView());
    IpplTimings::stopTimer(deepCopy);

    IpplTimings::startTimer(coarsePropagator);
    if(Pcoarse->coarsetype_m == "PIC") {
        Pcoarse->LeapFrogPIC(Pend->R, Pend->P, ntCoarse, dtCoarse, rankTime * dtSlice, spaceComm);
    }
    else {
        //PIF with coarse tolerance as coarse propagator
        Pcoarse->LeapFrogPIF(Pend->R, Pend->P, ntCoarse, dtCoarse, rankTime * dtSlice, 0, 0, 0, 0, coarse, spaceComm);
    }
    IpplTimings::stopTimer(coarsePropagator);

    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pcoarse->R.getView(), Pend->R.getView());
    Kokkos::deep_copy(Pcoarse->P.getView(), Pend->P.getView());
    IpplTimings::stopTimer(deepCopy);

    IpplTimings::startTimer(timeCommunication);
    if(rankTime < sizeTime-1) {
        size_type bufSize = Pend->packedSize(nloc);
        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
        MPI_Request request;
        Ippl::Comm->isend(rankTime+1, tag, *Pend, *buf, request, nloc, timeComm);
        buf->resetWritePos();
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    IpplTimings::stopTimer(timeCommunication);

#else
    //Note the CPU version has not been tested.
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(0));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         Pcoarse->R.getView(), Pcoarse->P.getView(), rand_pool64, delta, kw, 
                         sigma, muBulk, muBeam, nlocBulk, minU, maxU));


    Kokkos::fence();
    Ippl::Comm->barrier();
#endif

    msg << "Parareal "
        << TestName
        << endl
        << "Slice dT: " << dtSlice
        << endl
        << "No. of fine time steps: " << ntFine 
        << endl
        << "No. of coarse time steps: " << ntCoarse
        << endl
        << "Tolerance: " << tol
        << " No. of cycles: " << nCycles
        << endl
        << "Np= " << Total_particles 
        << " Fourier modes = " << nmPIF
        << " Grid points = " << nrPIC
        << endl;
    
    IpplTimings::stopTimer(particleCreation);                                                    
    
    msg << "particles created and initial conditions assigned " << endl;

    int sign = 1;
    for (unsigned int nc=0; nc < nCycles; nc++) {
        
        double tStartMySlice; 
        bool sendCriteria, recvCriteria;
        bool isConverged = false;
        bool isPreviousDomainConverged = false;
        
        //even cycles
        if(nc % 2 == 0) {
            sendCriteria = (rankTime < (sizeTime-1));
            recvCriteria = (rankTime > 0);
            if(rankTime == 0) {
                isPreviousDomainConverged = true;
            }
            tStartMySlice = (nc * tEndCycle) + (rankTime * dtSlice);
            msg.setPrintNode(Ippl::Comm->size()-1);
        }
        //odd cycles
        else {
            recvCriteria = (rankTime < (sizeTime-1));
            sendCriteria = (rankTime > 0);
            if(rankTime == (sizeTime - 1)) {
                isPreviousDomainConverged = true;
            }
            tStartMySlice = (nc * tEndCycle) + (((sizeTime - 1) - rankTime) * dtSlice);
            msg.setPrintNode(0);
        }
        
        unsigned int it = 0;
        
        while (!isConverged) { 
            //Run fine integrator in parallel
            IpplTimings::startTimer(finePropagator);
            Pcoarse->LeapFrogPIF(Pbegin->R, Pbegin->P, ntFine, dtFine, tStartMySlice, nc+1, it+1, rankTime, rankSpace, fine, spaceComm);
            IpplTimings::stopTimer(finePropagator);
    

            //Difference = Fine - Coarse
            Pend->R = Pbegin->R - Pcoarse->R;
            Pend->P = Pbegin->P - Pcoarse->P;

            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pcoarse->RprevIter.getView(), Pcoarse->R.getView());
            Kokkos::deep_copy(Pcoarse->PprevIter.getView(), Pcoarse->P.getView());
            IpplTimings::stopTimer(deepCopy);
            
            IpplTimings::startTimer(timeCommunication);
            tag = 1100;//Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
            int tagbool = 1300;//Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
            
            if(recvCriteria && (!isPreviousDomainConverged)) {
                size_type bufSize = Pbegin->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
                Ippl::Comm->recv(rankTime-sign, tag, *Pbegin, *buf, bufSize, nloc, timeComm);
                buf->resetReadPos();
                MPI_Recv(&isPreviousDomainConverged, 1, MPI_C_BOOL, rankTime-sign, tagbool, 
                        timeComm, MPI_STATUS_IGNORE);
                IpplTimings::startTimer(deepCopy);
                Kokkos::deep_copy(Pcoarse->R0.getView(), Pbegin->R.getView());
                Kokkos::deep_copy(Pcoarse->P0.getView(), Pbegin->P.getView());
                IpplTimings::stopTimer(deepCopy);
            }
            IpplTimings::stopTimer(timeCommunication);

            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pbegin->R.getView(), Pcoarse->R0.getView());
            Kokkos::deep_copy(Pbegin->P.getView(), Pcoarse->P0.getView());
            Kokkos::deep_copy(Pcoarse->R.getView(), Pbegin->R.getView());
            Kokkos::deep_copy(Pcoarse->P.getView(), Pbegin->P.getView());
            IpplTimings::stopTimer(deepCopy);

            IpplTimings::startTimer(coarsePropagator);
            if(Pcoarse->coarsetype_m == "PIC") {
                Pcoarse->LeapFrogPIC(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice, spaceComm);
            }
            else {
                Pcoarse->LeapFrogPIF(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice, 0, 0, 0, 0, coarse, spaceComm);
            }
            IpplTimings::stopTimer(coarsePropagator);

            Pend->R = Pend->R + Pcoarse->R;
            Pend->P = Pend->P + Pcoarse->P;

            PL.applyBC(Pend->R, PL.getRegionLayout().getDomain());
            
            IpplTimings::startTimer(computeErrors);
            double Rerror = computeRL2Error(Pcoarse->R, Pcoarse->RprevIter, length, spaceComm);
            double Perror = computePL2Error(Pcoarse->P, Pcoarse->PprevIter, spaceComm);
            IpplTimings::stopTimer(computeErrors);

            if((Rerror <= tol) && (Perror <= tol) && isPreviousDomainConverged) {
                isConverged = true;
            }

            IpplTimings::startTimer(timeCommunication);
            if(sendCriteria) {
                size_type bufSize = Pend->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
                MPI_Request request;
                Ippl::Comm->isend(rankTime+sign, tag, *Pend, *buf, request, nloc, timeComm);
                buf->resetWritePos();
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                MPI_Send(&isConverged, 1, MPI_C_BOOL, rankTime+sign, tagbool, timeComm);
            }
            IpplTimings::stopTimer(timeCommunication);

            
            msg << "Finished iteration: " << it+1
                << " in cycle: " << nc+1
                << " Rerror: " << Rerror 
                << " Perror: " << Perror
                << endl;

            IpplTimings::startTimer(dumpData);
            Pcoarse->writelocalError(Rerror, Perror, nc+1, it+1, rankTime, rankSpace);
            IpplTimings::stopTimer(dumpData);

            MPI_Barrier(spaceComm);
            
            it += 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if((nCycles > 1) && (nc < (nCycles - 1))) {  
            tag = 1000;//Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
           
            //send, receive criteria and tStartMySlice are reversed at the end of the cycle
            if(nc % 2 == 0) {
                recvCriteria = (rankTime < (sizeTime-1));
                sendCriteria = (rankTime > 0);
                tStartMySlice = (nc * tEndCycle) + (((sizeTime - 1) - rankTime) * dtSlice);
            }
            //odd cycles
            else {
                sendCriteria = (rankTime < (sizeTime-1));
                recvCriteria = (rankTime > 0);
                tStartMySlice = (nc * tEndCycle) + (rankTime * dtSlice);
            }
            
            
            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pbegin->R.getView(), Pend->R.getView());
            Kokkos::deep_copy(Pbegin->P.getView(), Pend->P.getView());
            IpplTimings::stopTimer(deepCopy);
            
            IpplTimings::startTimer(timeCommunication);
            if(recvCriteria) {
                size_type bufSize = Pbegin->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
                Ippl::Comm->recv(rankTime+sign, tag, *Pbegin, *buf, bufSize, nloc, timeComm);
                buf->resetReadPos();
            }
            IpplTimings::stopTimer(timeCommunication);

            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pend->R.getView(), Pbegin->R.getView());
            Kokkos::deep_copy(Pend->P.getView(), Pbegin->P.getView());
            Kokkos::deep_copy(Pcoarse->R0.getView(), Pbegin->R.getView());
            Kokkos::deep_copy(Pcoarse->P0.getView(), Pbegin->P.getView());
            IpplTimings::stopTimer(deepCopy);
            
            IpplTimings::startTimer(coarsePropagator);
            if(Pcoarse->coarsetype_m == "PIC") {
                Pcoarse->LeapFrogPIC(Pend->R, Pend->P, ntCoarse, dtCoarse, tStartMySlice, spaceComm);
            }
            else {
                Pcoarse->LeapFrogPIF(Pend->R, Pend->P, ntCoarse, dtCoarse, tStartMySlice, 0, 0, 0, 0, coarse, spaceComm); 
            }
            IpplTimings::stopTimer(coarsePropagator);
            
            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pcoarse->R.getView(), Pend->R.getView());
            Kokkos::deep_copy(Pcoarse->P.getView(), Pend->P.getView());
            IpplTimings::stopTimer(deepCopy);

            IpplTimings::startTimer(timeCommunication);
            if(sendCriteria) {
                size_type bufSize = Pend->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
                MPI_Request request;
                Ippl::Comm->isend(rankTime-sign, tag, *Pend, *buf, request, nloc, timeComm);
                buf->resetWritePos();
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            IpplTimings::stopTimer(timeCommunication);
            sign *= -1;
        }
    }
    msg << TestName << " Parareal: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    MPI_Comm_free(&spaceComm);
    MPI_Comm_free(&timeComm);


    return 0;
}
