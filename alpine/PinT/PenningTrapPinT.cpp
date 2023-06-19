// Parallel-in-time (PinT) method Parareal combined with Particle-in-cell
// and Particle-in-Fourier schemes. The example is electrostatic Landau 
// damping. The implementation of Parareal follows the open source implementation
// https://github.com/Parallel-in-Time/PararealF90 by Daniel Ruprecht. The corresponding
// publication is Ruprecht, Daniel. "Shared memory pipelined parareal." 
// European Conference on Parallel Processing. Springer, Cham, 2017.
// 
//  Usage:
//     srun ./PenningTrapPinT <nmx> <nmy> <nmz> <nx> <ny> <nz> <Np> <Tend> <dtfine> <dtcoarse> <tol> <nCycles> 
//     <ShapeType> <degree> --info 5
//     nmx       = No. of Fourier modes in the x-direction
//     nmy       = No. of Fourier modes in the y-direction
//     nmz       = No. of Fourier modes in the z-direction
//     nx       = No. of grid points in the x-direction
//     ny       = No. of grid points in the y-direction
//     nz       = No. of grid points in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     nCycles = No. of Parareal blocks/cycles
//     ShapeType = Shape function type B-spline only for the moment
//     degree = B-spline degree (-1 for delta function)
//     Example:
//     srun ./PenningTrapPinT 32 32 32 32 32 32 655360 20.0 0.05 0.05 1e-5 4 B-spline 1 --info 5
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
//#include "LeapFrogPIC.cpp"
//#include "LeapFrogPIF.cpp"
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
  
  T mu, sigma, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& mu_, const T& sigma_, 
           const T& u_) 
  : mu(mu_), sigma(sigma_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = std::erf((x - mu)/(sigma * std::sqrt(2.0))) 
          - 2 * u + 1;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = (1 / sigma) * std::sqrt(2 / pi) * 
               std::exp(-0.5 * (std::pow(((x - mu) / sigma),2)));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while ((iterations < max_iter) && (std::fabs(f(x)) > tol)) {
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

  T mu, sigma, minU, maxU;

  double pi = std::acos(-1.0);

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_,
                  T& mu_, T& sigma_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        mu(mu_), sigma(sigma_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {
        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = (std::sqrt(pi / 2) * (2 * u - 1)) * 
                  sigma[d] + mu[d];
        Newton1D<value_type> solver(mu[d], sigma[d], u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& mu, const double& sigma) {
   double cdf = 0.5 * (1.0 + std::erf((x - mu)/(sigma * std::sqrt(2))));
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

                                for (unsigned d = 0; d < 3; ++d) {
                                    bool isLeft = (diff[d] <= -22.0);
                                    bool isRight = (diff[d] >= 22.0);
                                    bool isInside = ((diff[d] > -22.0) && (diff[d] < 22.0));
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
    //MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    double globalNorm = 0.0;
    //MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    //lError = std::sqrt(localError)/std::sqrt(localNorm);

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
    //MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    double globalNorm = 0.0;
    //MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
    //lError = std::sqrt(localError)/std::sqrt(localNorm);

    double relError = std::sqrt(globalError) / std::sqrt(globalNorm);
    
    return relError;

}

double computeRLinfError(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, 
                      const unsigned int& /*iter*/, const int& /*myrank*/, double& lError, 
                      Vector_t& length) {
    
    auto Qview = Q.getView();
    auto QprevIterView = QprevIter.getView();
    double localError = 0.0;
    double localNorm = 0.0;

    Kokkos::parallel_reduce("Abs. max error and norm", Q.size(),
                            KOKKOS_LAMBDA(const int i, double& valLError, double& valLnorm){
                                Vector_t diff = Qview(i) - QprevIterView(i);
                                
                                for (unsigned d = 0; d < 3; ++d) {
                                    bool isLeft = (diff[d] <= -22.0);
                                    bool isRight = (diff[d] >= 22.0);
                                    bool isInside = ((diff[d] > -22.0) && (diff[d] < 22.0));
                                    diff[d] = (isInside * diff[d]) + (isLeft * (diff[d] + length[d]))
                                              +(isRight * (diff[d] - length[d]));
                                }
                                
                                double myValError = dot(diff, diff).apply();

                                myValError = std::sqrt(myValError);

                                //bool isIncluded = (myValError < 10.0);

                                //myValError *= isIncluded;
                                
                                if(myValError > valLError) valLError = myValError;
                                
                                double myValnorm = dot(Qview(i), Qview(i)).apply();
                                myValnorm = std::sqrt(myValnorm);

                                //myValnorm *= isIncluded;
                                
                                if(myValnorm > valLnorm) valLnorm = myValnorm;
                                
                                //excluded += (!isIncluded);
                            }, Kokkos::Max<double>(localError), Kokkos::Max<double>(localNorm));

    Kokkos::fence();
    lError = localError/localNorm;
    
    double relError = lError;
    
    return relError;

}

double computePLinfError(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, 
                      const unsigned int& /*iter*/, const int& /*myrank*/, double& lError) {
    
    auto Qview = Q.getView();
    auto QprevIterView = QprevIter.getView();
    double localError = 0.0;
    double localNorm = 0.0;

    Kokkos::parallel_reduce("Abs. max error and norm", Q.size(),
                            KOKKOS_LAMBDA(const int i, double& valLError, double& valLnorm){
                                Vector_t diff = Qview(i) - QprevIterView(i);
                                double myValError = dot(diff, diff).apply();
                                myValError = std::sqrt(myValError);
                                
                                if(myValError > valLError) valLError = myValError;
                                
                                double myValnorm = dot(Qview(i), Qview(i)).apply();
                                myValnorm = std::sqrt(myValnorm);
                                
                                if(myValnorm > valLnorm) valLnorm = myValnorm;
                            }, Kokkos::Max<double>(localError), Kokkos::Max<double>(localNorm));

    Kokkos::fence();
    lError = localError/localNorm;
    
    double relError = lError;
    
    return relError;

}


double computeFieldError(CxField_t& rhoPIF, CxField_t& rhoPIFprevIter) {

    auto rhoview = rhoPIF.getView();
    auto rhoprevview = rhoPIFprevIter.getView();
    const int nghost = rhoPIF.getNghost();
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
    
    const FieldLayout_t& layout = rhoPIF.getLayout(); 
    const Mesh_t& mesh = rhoPIF.get_mesh();
    const Vector<double, Dim>& dx = mesh.getMeshSpacing();
    const auto& domain = layout.getDomain();
    Vector<double, Dim> Len;
    Vector<int, Dim> N;

    for (unsigned d=0; d < Dim; ++d) {
        N[d] = domain[d].length();
        Len[d] = dx[d] * N[d];
    }

    double AbsError = 0.0;
    double Enorm = 0.0;
    Kokkos::complex<double> imag = {0.0, 1.0};
    double pi = std::acos(-1.0);
    Kokkos::parallel_reduce("Ex field error",
                          mdrange_type({0, 0, 0},
                                       {N[0],
                                        N[1],
                                        N[2]}),
                          KOKKOS_LAMBDA(const int i,
                                        const int j,
                                        const int k,
                                        double& errorSum,
                                        double& fieldSum)
    {
    
        Vector<int, 3> iVec = {i, j, k};
        Vector<double, 3> kVec;
        double Dr = 0.0;
        for(size_t d = 0; d < Dim; ++d) {
            bool shift = (iVec[d] > (N[d]/2));
            kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
            Dr += kVec[d] * kVec[d];
        }

        double myError = 0.0;
        double myField = 0.0;
        Kokkos::complex<double> Ek = {0.0, 0.0};
        Kokkos::complex<double> Ekprev = {0.0, 0.0};
        for(size_t d = 0; d < Dim; ++d) {
            if(Dr != 0.0) {
                Ek = -(imag * kVec[d] * rhoview(i+nghost,j+nghost,k+nghost) / Dr);
                Ekprev = -(imag * kVec[d] * rhoprevview(i+nghost,j+nghost,k+nghost) / Dr);
            }
            Ekprev = Ekprev - Ek;
            myError += Ekprev.real() * Ekprev.real() + Ekprev.imag() * Ekprev.imag();
            myField += Ek.real() * Ek.real() + Ek.imag() * Ek.imag();
        }
        errorSum += myError;
        fieldSum += myField;
        //Kokkos::complex<double> rhok = rhoview(i+nghost,j+nghost,k+nghost);
        //Kokkos::complex<double> rhokprev = rhoprevview(i+nghost,j+nghost,k+nghost);
        //rhokprev = rhokprev - rhok;
        //myError = rhokprev.real() * rhokprev.real() + rhokprev.imag() * rhokprev.imag();
        //errorSum += myError;
        //myField = rhok.real() * rhok.real() + rhok.imag() * rhok.imag();
        //fieldSum += myField;

    }, Kokkos::Sum<double>(AbsError), Kokkos::Sum<double>(Enorm));
    
    Kokkos::fence();
    double globalError = 0.0;
    MPI_Allreduce(&AbsError, &globalError, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    double globalNorm = 0.0;
    MPI_Allreduce(&Enorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());
    //double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
    //fieldEnergy *= volume;

    double relError = std::sqrt(globalError)/std::sqrt(globalNorm);

    return relError;
}


const char* TestName = "PenningTrapPinT";

int main(int argc, char *argv[]){
   
    Ippl ippl(argc, argv);
    
    //int rankWorld, sizeWorld;
    //MPI_Init(&argc, &argv);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    //MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    int spaceColor, timeColor;
    MPI_Comm spaceComm, timeComm;

    int spaceProcs = std::atoi(argv[15]);
    int timeProcs = std::atoi(argv[16]);
    //spaceColor = rankWorld / spaceProcs; 
    //timeColor = rankWorld % spaceProcs;
    spaceColor = Ippl::Comm->rank() / spaceProcs; 
    timeColor = Ippl::Comm->rank() % spaceProcs;

    //MPI_Comm_split(MPI_COMM_WORLD, spaceColor, rankWorld, &spaceComm);
    //MPI_Comm_split(MPI_COMM_WORLD, timeColor, rankWorld, &timeComm);
    MPI_Comm_split(Ippl::getComm(), spaceColor, Ippl::Comm->rank(), &spaceComm);
    MPI_Comm_split(Ippl::getComm(), timeColor, Ippl::Comm->rank(), &timeComm);

    int rankSpace, sizeSpace, rankTime, sizeTime;
    MPI_Comm_rank(spaceComm, &rankSpace);
    MPI_Comm_size(spaceComm, &sizeSpace);

    MPI_Comm_rank(timeComm, &rankTime);
    MPI_Comm_size(timeComm, &sizeTime);

    //Ippl ippl(argc, argv, spaceComm);
    
    //Inform msg(TestName, sizeSpace-1);
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
    static IpplTimings::TimerRef initializeCycles = IpplTimings::getTimer("initializeCycles");

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
    //const unsigned int maxIter = std::atoi(argv[12]);

    //const double tEndMySlice = (Ippl::Comm->rank() + 1) * dtSlice; 


    using bunch_type = ChargedParticlesPinT<PLayout_t>;
    using states_begin_type = StatesBeginSlice<PLayout_t>;
    using states_end_type = StatesEndSlice<PLayout_t>;

    std::unique_ptr<bunch_type>  Pcoarse;
    std::unique_ptr<states_begin_type>  Pbegin;
    std::unique_ptr<states_end_type>  Pend;

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
    Vector_t rmin(0.0);
    Vector_t rmax(25.0);
    //Vector_t rmax(20.0);
    Vector_t length = rmax - rmin;
    double dxPIC = length[0] / nrPIC[0];
    double dyPIC = length[1] / nrPIC[1];
    double dzPIC = length[2] / nrPIC[2];


    Vector_t mu, sd;

    for (unsigned d = 0; d<Dim; d++) {
        mu[d] = 0.5 * length[d];
    }
    //sd[0] = 0.15*length[0];
    //sd[1] = 0.05*length[1];
    //sd[2] = 0.20*length[2];
    sd[0] = 0.10*20.0;//length[0];
    sd[1] = 0.05*20.0;//length[1];
    sd[2] = 0.15*20.0;//length[2];


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

    size_type nloc = (size_type)(totalP / sizeSpace);

    size_type Total_particles = 0;

    //MPI_Allreduce(&nloc, &Total_particles, 1,
    //            MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());
    MPI_Allreduce(&nloc, &Total_particles, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, spaceComm);

    //int rest = (int) (totalP - Total_particles);

    //if ( (rankTime == 0) && (rankSpace < rest) ) {
    //    ++nloc;
    //}

    double Q = -1562.5;
    double Bext = 5.0;
    Pcoarse = std::make_unique<bunch_type>(PL,hrPIC,rmin,rmax,decomp,Q,Total_particles);
    Pbegin = std::make_unique<states_begin_type>(PL);
    Pend = std::make_unique<states_end_type>(PL);

    Pcoarse->nr_m = nrPIC;
    Pcoarse->nm_m = nmPIF;

    Pcoarse->rhoPIF_m.initialize(meshPIF, FLPIF);
    Pcoarse->Sk_m.initialize(meshPIF, FLPIF);
    //Pcoarse->rhoPIFprevIter_m.initialize(meshPIF, FLPIF);
    Pcoarse->rhoPIC_m.initialize(meshPIC, FLPIC);
    Pcoarse->EfieldPIC_m.initialize(meshPIC, FLPIC);
    //Pcoarse->EfieldPICprevIter_m.initialize(meshPIC, FLPIC);

    Pcoarse->initFFTSolver();

    IpplTimings::startTimer(particleCreation);

    Vector_t minU, maxU;
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], mu[d], sd[d]);
        maxU[d] = CDF(rmax[d], mu[d], sd[d]);
    }


    Pcoarse->create(nloc);
    Pbegin->create(nloc);
    Pend->create(nloc);

    Pcoarse->q = Pcoarse->Q_m/Total_particles;
    using buffer_type = ippl::Communicate::buffer_type;
    int tag;

    Pcoarse->shapetype_m = argv[13];
    Pcoarse->shapedegree_m = std::atoi(argv[14]); 
    IpplTimings::startTimer(initializeShapeFunctionPIF);
    Pcoarse->initializeShapeFunctionPIF();
    IpplTimings::stopTimer(initializeShapeFunctionPIF);
    
    
    Pcoarse->initNUFFT(FLPIF);

#ifdef KOKKOS_ENABLE_CUDA
    //If we don't do the following even with the same seed the initial 
    //condition is not the same on different GPUs
    //tag = Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
    //if(Ippl::Comm->rank() == 0) {
    //    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
    //    Kokkos::parallel_for(nloc,
    //                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
    //                         Pbegin->R.getView(), Pbegin->P.getView(), rand_pool64, mu, sd, 
    //                         minU, maxU));


    //    Kokkos::fence();
    //    size_type bufSize = Pbegin->packedSize(nloc);
    //    std::vector<MPI_Request> requests(0);
    //    int sends = 0;
    //    for(int rank = 1; rank < Ippl::Comm->size(); ++rank) {
    //        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND + sends, bufSize);
    //        requests.resize(requests.size() + 1);
    //        Ippl::Comm->isend(rank, tag, *Pbegin, *buf, requests.back(), nloc);
    //        buf->resetWritePos();
    //        ++sends;
    //    }
    //    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    //}
    //else {
    //    size_type bufSize = Pbegin->packedSize(nloc);
    //    buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
    //    Ippl::Comm->recv(0, tag, *Pbegin, *buf, bufSize, nloc);
    //    buf->resetReadPos();
    //}

    //If we don't do the following even with the same seed the initial 
    //condition is not the same on different GPUs
    tag = Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);

    if(rankTime == 0) {
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*rankSpace));
        Kokkos::parallel_for(nloc,
                             generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                             Pbegin->R.getView(), Pbegin->P.getView(), rand_pool64, mu, sd, 
                             minU, maxU));


        Kokkos::fence();
    }
    else {
        size_type bufSize = Pbegin->packedSize(nloc);
        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
        Ippl::Comm->recv(rankTime-1, tag, *Pbegin, *buf, bufSize, nloc, timeComm);
        buf->resetReadPos();
    }

    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pend->R.getView(), Pbegin->R.getView());
    Kokkos::deep_copy(Pend->P.getView(), Pbegin->P.getView());
    Kokkos::deep_copy(Pcoarse->R0.getView(), Pbegin->R.getView());
    Kokkos::deep_copy(Pcoarse->P0.getView(), Pbegin->P.getView());
    IpplTimings::stopTimer(deepCopy);

    Pcoarse->BorisPIC(Pend->R, Pend->P, ntCoarse, dtCoarse, rankTime * dtSlice, Bext, spaceComm); 

    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pcoarse->R.getView(), Pend->R.getView());
    Kokkos::deep_copy(Pcoarse->P.getView(), Pend->P.getView());
    IpplTimings::stopTimer(deepCopy);
    
    if(rankTime < sizeTime-1) {
        size_type bufSize = Pend->packedSize(nloc);
        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
        MPI_Request request;
        Ippl::Comm->isend(rankTime+1, tag, *Pend, *buf, request, nloc, timeComm);
        buf->resetWritePos();
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
#else
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(0));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         Pcoarse->R.getView(), Pcoarse->P.getView(), rand_pool64, mu, sd, 
                         minU, maxU));

    Kokkos::fence();
    //Ippl::Comm->barrier();
#endif


    msg << "Parareal Penning trap"
        << endl
        << "Slice dT: " << dtSlice
        << endl
        << "No. of fine time steps: " << ntFine 
        << endl
        << "No. of coarse time steps: " << ntCoarse
        << endl
        << "Tolerance: " << tol
        //<< " Max. iterations: " << maxIter
        << " No. of cycles: " << nCycles
        << endl
        << "Np= " << Total_particles 
        << " Fourier modes = " << nmPIF
        << " Grid points = " << nrPIC
        << endl;
    
    IpplTimings::stopTimer(particleCreation);                                                    
    
    msg << "particles created and initial conditions assigned " << endl;

    //Copy initial conditions as they are needed later
    //IpplTimings::startTimer(deepCopy);
    //Kokkos::deep_copy(Pcoarse->R0.getView(), Pcoarse->R.getView());
    //Kokkos::deep_copy(Pcoarse->P0.getView(), Pcoarse->P.getView());
    //IpplTimings::stopTimer(deepCopy);

    ////Get initial guess for ranks other than 0 by propagating the coarse solver
    //IpplTimings::startTimer(coarsePropagator);
    //if (Ippl::Comm->rank() > 0) {
    //    Pcoarse->BorisPIC(Pcoarse->R, Pcoarse->P, Ippl::Comm->rank()*ntCoarse, dtCoarse, tStartMySlice, Bext); 
    //}
    //
    //Ippl::Comm->barrier();
    //
    //IpplTimings::stopTimer(coarsePropagator);

    //msg << "First Boris PIC done " << endl;

    //
    //IpplTimings::startTimer(deepCopy);
    //Kokkos::deep_copy(Pbegin->R.getView(), Pcoarse->R.getView());
    //Kokkos::deep_copy(Pbegin->P.getView(), Pcoarse->P.getView());
    //IpplTimings::stopTimer(deepCopy);


    ////Run the coarse integrator to get the values at the end of the time slice 
    //IpplTimings::startTimer(coarsePropagator);
    //Pcoarse->BorisPIC(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice, Bext); 
    //IpplTimings::stopTimer(coarsePropagator);
    //msg << "Second Boris PIC done " << endl;

    ////Kokkos::deep_copy(Pcoarse->EfieldPICprevIter_m.getView(), Pcoarse->EfieldPIC_m.getView());

    ////The following might not be needed
    //IpplTimings::startTimer(deepCopy);
    //Kokkos::deep_copy(Pend->R.getView(), Pcoarse->R.getView());
    //Kokkos::deep_copy(Pend->P.getView(), Pcoarse->P.getView());
    //IpplTimings::stopTimer(deepCopy);


    //msg << "Starting parareal iterations ..." << endl;
    //bool isConverged = false;
    //bool isPreviousDomainConverged;
    //if(Ippl::Comm->rank() == 0) {
    //    isPreviousDomainConverged = true;
    //}
    //else {
    //    isPreviousDomainConverged = false;
    //}


    
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
        //Pcoarse->time_m = tStartMySlice;
        
        unsigned int it = 0;
        while (!isConverged) { 
        //while ((!isPreviousDomainConverged) || (!isConverged)) { 
        //for (unsigned int it=0; it < maxIter; it++) {

            //Run fine integrator in parallel
            IpplTimings::startTimer(finePropagator);
            Pcoarse->BorisPIF(Pbegin->R, Pbegin->P, ntFine, dtFine, tStartMySlice, nc+1, it+1, 
                              Bext, rankTime, rankSpace, spaceComm);
            IpplTimings::stopTimer(finePropagator);
        

            //Difference = Fine - Coarse
            Pend->R = Pbegin->R - Pcoarse->R;
            Pend->P = Pbegin->P - Pcoarse->P;

            //Pcoarse->dumpParticleData(it+1, Pcoarse->R, Pcoarse->P, "Gk");
            //Pcoarse->dumpParticleData(it+1, Pbegin->R, Pbegin->P, "Fk");


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
            Pcoarse->BorisPIC(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice, Bext, spaceComm); 
            IpplTimings::stopTimer(coarsePropagator);

            Pend->R = Pend->R + Pcoarse->R;
            Pend->P = Pend->P + Pcoarse->P;

            //Pcoarse->dumpParticleData(it+1, Pcoarse->R, Pcoarse->P, "Gkp1");

            PL.applyBC(Pend->R, PL.getRegionLayout().getDomain());
            IpplTimings::startTimer(computeErrors);
            //double localRerror, localPerror;
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
            //Pcoarse->writeError(Rerror, Perror, it+1);
            Pcoarse->writelocalError(Rerror, Perror, nc+1, it+1, rankTime, rankSpace);
            //Pcoarse->dumpParticleData(it+1, Pend->R, Pend->P, "Parareal");
            IpplTimings::stopTimer(dumpData);

            MPI_Barrier(spaceComm);

            it += 1;
            //if(isConverged && isPreviousDomainConverged) {
            //    break;
            //}
        }
    
        //std::cout << "Before barrier in cycle: " << nc+1 << "for rank: " << Ippl::Comm->rank() << std::endl;
        //Ippl::Comm->barrier();
        MPI_Barrier(MPI_COMM_WORLD);
        //msg << "Communication started in cycle: " << nc+1 << endl;
        //std::cout << "Communication started in cycle: " << nc+1 << "for rank: " << Ippl::Comm->rank() << std::endl;
        if((nCycles > 1) && (nc < (nCycles - 1))) {  
            IpplTimings::startTimer(timeCommunication);
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

            if(recvCriteria) {
                size_type bufSize = Pbegin->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
                Ippl::Comm->recv(rankTime+sign, tag, *Pbegin, *buf, bufSize, nloc, timeComm);
                buf->resetReadPos();
            }

            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pend->R.getView(), Pbegin->R.getView());
            Kokkos::deep_copy(Pend->P.getView(), Pbegin->P.getView());
            Kokkos::deep_copy(Pcoarse->R0.getView(), Pbegin->R.getView());
            Kokkos::deep_copy(Pcoarse->P0.getView(), Pbegin->P.getView());
            IpplTimings::stopTimer(deepCopy);
            
            Pcoarse->BorisPIC(Pend->R, Pend->P, ntCoarse, dtCoarse, tStartMySlice, Bext, spaceComm); 
            
            IpplTimings::startTimer(deepCopy);
            Kokkos::deep_copy(Pcoarse->R.getView(), Pend->R.getView());
            Kokkos::deep_copy(Pcoarse->P.getView(), Pend->P.getView());
            IpplTimings::stopTimer(deepCopy);


            if(sendCriteria) {
                size_type bufSize = Pend->packedSize(nloc);
                buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
                MPI_Request request;
                Ippl::Comm->isend(rankTime-sign, tag, *Pend, *buf, request, nloc, timeComm);
                buf->resetWritePos();
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            IpplTimings::stopTimer(timeCommunication);
            //std::cout << "Communication finished in cycle: " << nc+1 << "for rank: " << Ippl::Comm->rank() << std::endl;
            //Ippl::Comm->barrier();

            //msg << "Communication finished in cycle: " << nc+1 << endl;
            //IpplTimings::startTimer(deepCopy);
            //Kokkos::deep_copy(Pcoarse->R.getView(), Pend->R.getView());
            //Kokkos::deep_copy(Pcoarse->P.getView(), Pend->P.getView());
            //IpplTimings::stopTimer(deepCopy);
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
