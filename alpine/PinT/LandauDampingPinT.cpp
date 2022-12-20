// Parallel-in-time (PinT) method Parareal combined with Particle-in-cell
// and Particle-in-Fourier schemes. The example is electrostatic Landau 
// damping. The implementation of Parareal follows the open source implementation
// https://github.com/Parallel-in-Time/PararealF90 by Daniel Ruprecht. The corresponding
// publication is Ruprecht, Daniel. "Shared memory pipelined parareal." 
// European Conference on Parallel Processing. Springer, Cham, 2017.
// 
//  Usage:
//     srun ./LandauDampingPinT <nmx> <nmy> <nmz> <nx> <ny> <nz> <Np> <Tend> <dtfine> <dtcoarse> <tol> <Niter> --info 5
//     nmx       = No. of Fourier modes in the x-direction
//     nmy       = No. of Fourier modes in the y-direction
//     nmz       = No. of Fourier modes in the z-direction
//     nx       = No. of grid points in the x-direction
//     ny       = No. of grid points in the y-direction
//     nz       = No. of grid points in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Example:
//     srun ./LandauDampingPinT 16 16 16 32 32 32 655360 20.0 0.05 0.05 1e-5 100 --info 5
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

double computeL2Error(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, 
                      const unsigned int& /*iter*/, const int& /*myrank*/, double& lError) {
    
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
    lError = std::sqrt(localError)/std::sqrt(localNorm);
    //std::cout << "Rank: " << myrank << " Iter: " << iter << " Local. Error: " << lError << std::endl;


    double globaltemp = 0.0;
    MPI_Allreduce(&localError, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

    double absError = std::sqrt(globaltemp);

    //temp = 0.0;
    //Kokkos::parallel_reduce("Q norm", Q.size(),
    //                        KOKKOS_LAMBDA(const int i, double& valL){
    //                            double myVal = dot(Qview(i), Qview(i)).apply();
    //                            valL += myVal;
    //                        }, Kokkos::Sum<double>(temp));


    globaltemp = 0.0;
    MPI_Allreduce(&localNorm, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, Ippl::getComm());

    double relError = absError / std::sqrt(globaltemp);
    
    return relError;

}

double computeLinfError(ParticleAttrib<Vector_t>& Q, ParticleAttrib<Vector_t>& QprevIter, 
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
    //std::cout << "Rank: " << myrank << " Iter: " << iter << " Local. Error: " << lError << std::endl;


    double globaltemp = 0.0;
    MPI_Allreduce(&localError, &globaltemp, 1, MPI_DOUBLE, MPI_MAX, Ippl::getComm());

    double absError = globaltemp;

    globaltemp = 0.0;
    MPI_Allreduce(&localNorm, &globaltemp, 1, MPI_DOUBLE, MPI_MAX, Ippl::getComm());

    double relError = absError / globaltemp;
    
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


const char* TestName = "LandauDampingPinT";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    
    Inform msg("LandauDampingPinT");
    Inform msg2all("LandauDampingPinT",INFORM_ALL_NODES);

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

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[7]);
    const double tEnd = std::atof(argv[8]);
    const double dtSlice = tEnd / Ippl::Comm->size();
    const double dtFine = std::atof(argv[9]);
    const double dtCoarse = std::atof(argv[10]);
    const unsigned int ntFine = (unsigned int)(dtSlice / dtFine);
    const unsigned int ntCoarse = (unsigned int)(dtSlice / dtCoarse);
    const double tol = std::atof(argv[11]);
    const unsigned int maxIter = std::atoi(argv[12]);

    const double tStartMySlice = Ippl::Comm->rank() * dtSlice; 
    //const double tEndMySlice = (Ippl::Comm->rank() + 1) * dtSlice; 

    msg << "Parareal Landau damping"
        << endl
        << "Slice dT: " << dtSlice
        << endl
        << "No. of fine time steps: " << ntFine 
        << endl
        << "No. of coarse time steps: " << ntCoarse
        << endl
        << "Tolerance: " << tol
        << " Max. iterations: " << maxIter
        << endl
        << "Np= " << totalP 
        << " Fourier modes = " << nmPIF
        << " Grid points = " << nrPIC
        << endl;

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
    Vector_t kw = {0.5, 0.5, 0.5};
    double alpha = 0.05;
    Vector_t rmin(0.0);
    Vector_t rmax = 2 * pi / kw ;
    double dxPIC = rmax[0] / nrPIC[0];
    double dyPIC = rmax[1] / nrPIC[1];
    double dzPIC = rmax[2] / nrPIC[2];


    double dxPIF = rmax[0] / nmPIF[0];
    double dyPIF = rmax[1] / nmPIF[1];
    double dzPIF = rmax[2] / nmPIF[2];
    Vector_t hrPIC = {dxPIC, dyPIC, dzPIC};
    Vector_t hrPIF = {dxPIF, dyPIF, dzPIF};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    const bool isAllPeriodic=true;
    Mesh_t meshPIC(domainPIC, hrPIC, origin);
    Mesh_t meshPIF(domainPIF, hrPIF, origin);
    FieldLayout_t FLPIC(domainPIC, decomp, isAllPeriodic);
    FieldLayout_t FLPIF(domainPIF, decomp, isAllPeriodic);
    PLayout_t PL(FLPIC, meshPIC);

    //Q = -\int\int f dx dv
    double Q = -rmax[0] * rmax[1] * rmax[2];
    Pcoarse = std::make_unique<bunch_type>(PL,hrPIC,rmin,rmax,decomp,Q);
    Pbegin = std::make_unique<states_begin_type>(PL);
    Pend = std::make_unique<states_end_type>(PL);

    Pcoarse->nr_m = nrPIC;

    Pcoarse->rhoPIF_m.initialize(meshPIF, FLPIF);
    Pcoarse->rhoPIFprevIter_m.initialize(meshPIF, FLPIF);
    Pcoarse->rhoPIC_m.initialize(meshPIC, FLPIC);
    Pcoarse->EfieldPIC_m.initialize(meshPIC, FLPIC);
    Pcoarse->EfieldPIC_m.initialize(meshPIC, FLPIC);
    //Pcoarse->EfieldPICprevIter_m.initialize(meshPIC, FLPIC);

    Pcoarse->initFFTSolver();
    Pcoarse->time_m = tStartMySlice;

    IpplTimings::startTimer(particleCreation);

    Vector_t minU, maxU;
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], alpha, kw[d]);
        maxU[d]   = CDF(rmax[d], alpha, kw[d]);
        //minU[d] = rmin[d];
        //maxU[d] = rmax[d];
    }

    size_type nloc = totalP;

    Pcoarse->create(nloc);
    Pbegin->create(nloc);
    Pend->create(nloc);

    using buffer_type = ippl::Communicate::buffer_type;
    int tag;
#ifdef KOKKOS_ENABLE_CUDA
    //If we don't do the following even with the same seed the initial 
    //condition is not the same on different GPUs
    tag = Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
    if(Ippl::Comm->rank() == 0) {
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
        Kokkos::parallel_for(nloc,
                             generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                             Pbegin->R.getView(), Pbegin->P.getView(), rand_pool64, alpha, kw, minU, maxU));

        Kokkos::fence();
        size_type bufSize = Pbegin->packedSize(nloc);
        std::vector<MPI_Request> requests(0);
        int sends = 0;
        for(int rank = 1; rank < Ippl::Comm->size(); ++rank) {
            buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND + sends, bufSize);
            requests.resize(requests.size() + 1);
            Ippl::Comm->isend(rank, tag, *Pbegin, *buf, requests.back(), nloc);
            buf->resetWritePos();
            ++sends;
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    else {
        size_type bufSize = Pbegin->packedSize(nloc);
        buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
        Ippl::Comm->recv(0, tag, *Pbegin, *buf, bufSize, nloc);
        buf->resetReadPos();
    }
    Ippl::Comm->barrier();
    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pcoarse->R.getView(), Pbegin->R.getView());
    Kokkos::deep_copy(Pcoarse->P.getView(), Pbegin->P.getView());
    IpplTimings::stopTimer(deepCopy);
#else
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(0));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         Pcoarse->R.getView(), Pcoarse->P.getView(), rand_pool64, alpha, kw, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
#endif


    Pcoarse->q = Pcoarse->Q_m/totalP;
    IpplTimings::stopTimer(particleCreation);                                                    
    
    msg << "particles created and initial conditions assigned " << endl;

    //Copy initial conditions as they are needed later
    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pcoarse->R0.getView(), Pcoarse->R.getView());
    Kokkos::deep_copy(Pcoarse->P0.getView(), Pcoarse->P.getView());
    IpplTimings::stopTimer(deepCopy);

    //Get initial guess for ranks other than 0 by propagating the coarse solver
    IpplTimings::startTimer(coarsePropagator);
    if (Ippl::Comm->rank() > 0) {
        Pcoarse->LeapFrogPIC(Pcoarse->R, Pcoarse->P, Ippl::Comm->rank()*ntCoarse, dtCoarse, tStartMySlice); 
    }
    
    Ippl::Comm->barrier();
    
    IpplTimings::stopTimer(coarsePropagator);

    msg << "First Leap frog PIC done " << endl;

    
    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pbegin->R.getView(), Pcoarse->R.getView());
    Kokkos::deep_copy(Pbegin->P.getView(), Pcoarse->P.getView());
    IpplTimings::stopTimer(deepCopy);


    //Run the coarse integrator to get the values at the end of the time slice 
    IpplTimings::startTimer(coarsePropagator);
    Pcoarse->LeapFrogPIC(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice); 
    IpplTimings::stopTimer(coarsePropagator);
    msg << "Second Leap frog PIC done " << endl;

    //Kokkos::deep_copy(Pcoarse->EfieldPICprevIter_m.getView(), Pcoarse->EfieldPIC_m.getView());

    //The following might not be needed
    IpplTimings::startTimer(deepCopy);
    Kokkos::deep_copy(Pend->R.getView(), Pcoarse->R.getView());
    Kokkos::deep_copy(Pend->P.getView(), Pcoarse->P.getView());
    IpplTimings::stopTimer(deepCopy);


    msg << "Starting parareal iterations ..." << endl;
    bool isConverged = false;
    //Kokkos::deep_copy(Pcoarse->RprevIter.getView(), Pcoarse->R0.getView());
    //Kokkos::deep_copy(Pcoarse->PprevIter.getView(), Pcoarse->P0.getView());
    //Pcoarse->LeapFrogPIF(Pcoarse->RprevIter, Pcoarse->PprevIter, (Ippl::Comm->rank()+1)*ntFine, 
    //                     dtFine, isConverged, tStartMySlice, 0);
    //Ippl::Comm->barrier();
    for (unsigned int it=0; it<maxIter; it++) {

        //Run fine integrator in parallel
        IpplTimings::startTimer(finePropagator);
        Pcoarse->LeapFrogPIF(Pbegin->R, Pbegin->P, ntFine, dtFine, isConverged, tStartMySlice, it+1);
        IpplTimings::stopTimer(finePropagator);
    

        //if(isConverged) {

            //test with the serial solution
            //Pcoarse->LeapFrogPIF(Pcoarse->R0, Pcoarse->P0, (Ippl::Comm->rank()+1)*ntFine, 
            //                     dtFine, isConverged, tStartMySlice, it+1);
            //Ippl::Comm->barrier();
            //double Rerror = computeL2Error(Pcoarse->R0, Pbegin->R, it+1, Ippl::Comm->rank());
            //double Perror = computeL2Error(Pcoarse->P0, Pbegin->P, it+1, Ippl::Comm->rank());
            //msg << "Finished iteration: " << it+1 
            //<< " Rerror: " << Rerror 
            //<< " Perror: " << Perror
            //<< endl;
        //    break;
        //}

        //Difference = Fine - Coarse
        Pend->R = Pbegin->R - Pcoarse->R;
        Pend->P = Pbegin->P - Pcoarse->P;

        IpplTimings::startTimer(deepCopy);
        Kokkos::deep_copy(Pcoarse->RprevIter.getView(), Pcoarse->R.getView());
        Kokkos::deep_copy(Pcoarse->PprevIter.getView(), Pcoarse->P.getView());
        IpplTimings::stopTimer(deepCopy);
        
        IpplTimings::startTimer(timeCommunication);
        tag = Ippl::Comm->next_tag(IPPL_PARAREAL_APP, IPPL_APP_CYCLE);
        
        if(Ippl::Comm->rank() > 0) {
            size_type bufSize = Pbegin->packedSize(nloc);
            buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_RECV, bufSize);
            Ippl::Comm->recv(Ippl::Comm->rank()-1, tag, *Pbegin, *buf, bufSize, nloc);
            buf->resetReadPos();
        }
        else {
            Kokkos::deep_copy(Pbegin->R.getView(), Pcoarse->R0.getView());
            Kokkos::deep_copy(Pbegin->P.getView(), Pcoarse->P0.getView());
        }
        IpplTimings::stopTimer(timeCommunication);

        IpplTimings::startTimer(deepCopy);
        Kokkos::deep_copy(Pcoarse->R.getView(), Pbegin->R.getView());
        Kokkos::deep_copy(Pcoarse->P.getView(), Pbegin->P.getView());
        IpplTimings::stopTimer(deepCopy);

        IpplTimings::startTimer(coarsePropagator);
        Pcoarse->LeapFrogPIC(Pcoarse->R, Pcoarse->P, ntCoarse, dtCoarse, tStartMySlice); 
        IpplTimings::stopTimer(coarsePropagator);

        Pend->R = Pend->R + Pcoarse->R;
        Pend->P = Pend->P + Pcoarse->P;

        IpplTimings::startTimer(timeCommunication);
        if(Ippl::Comm->rank() < Ippl::Comm->size()-1) {
            size_type bufSize = Pend->packedSize(nloc);
            buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize);
            MPI_Request request;
            Ippl::Comm->isend(Ippl::Comm->rank()+1, tag, *Pend, *buf, request, nloc);
            buf->resetWritePos();
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        IpplTimings::stopTimer(timeCommunication);

        //Pcoarse->EfieldPICprevIter_m = Pcoarse->EfieldPICprevIter_m - Pcoarse->EfieldPIC_m;
        //Pcoarse->rhoPIC_m = dot(Pcoarse->EfieldPICprevIter_m, Pcoarse->EfieldPICprevIter_m);
        //double absFieldError = std::sqrt(Pcoarse->rhoPIC_m.sum());
        //Pcoarse->rhoPIC_m = dot(Pcoarse->EfieldPIC_m, Pcoarse->EfieldPIC_m);
        //double EfieldNorm = std::sqrt(Pcoarse->rhoPIC_m.sum());
        //double EfieldError = absFieldError / EfieldNorm;

        IpplTimings::startTimer(computeErrors);
        double localRerror, localPerror;
        double Rerror = computeLinfError(Pcoarse->R, Pcoarse->RprevIter, it+1, Ippl::Comm->rank(), localRerror);
        double Perror = computeLinfError(Pcoarse->P, Pcoarse->PprevIter, it+1, Ippl::Comm->rank(), localPerror);
        //double Rerror = computeLinfError(Pend->R, Pcoarse->RprevIter, it+1, Ippl::Comm->rank(), localRerror);
        //double Perror = computeLinfError(Pend->P, Pcoarse->PprevIter, it+1, Ippl::Comm->rank(), localPerror);
    
        double EfieldError = 0;
        if(it > 0) {
            EfieldError = computeFieldError(Pcoarse->rhoPIF_m, Pcoarse->rhoPIFprevIter_m);
        }
        IpplTimings::stopTimer(computeErrors);

        IpplTimings::startTimer(deepCopy);
        Kokkos::deep_copy(Pcoarse->rhoPIFprevIter_m.getView(), Pcoarse->rhoPIF_m.getView());
        IpplTimings::stopTimer(deepCopy);
        
        msg << "Finished iteration: " << it+1 
            << " Rerror: " << Rerror 
            << " Perror: " << Perror
            << " Efield error: " << EfieldError
            //<< " Rhofield error: " << EfieldError
            << endl;

        IpplTimings::startTimer(dumpData);
        Pcoarse->writeError(Rerror, Perror, it+1);
        Pcoarse->writelocalError(localRerror, localPerror, it+1);
        IpplTimings::stopTimer(dumpData);
        //Kokkos::deep_copy(Pcoarse->EfieldPICprevIter_m.getView(), Pcoarse->EfieldPIC_m.getView());

        if((Rerror <= tol) && (Perror <= tol)) {
        //if(Perror <= tol) {
            break;
        }
    }

    msg << "LandauDamping Parareal: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
