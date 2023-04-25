// Electrostatic Penning trap test with Particle-in-Fourier schemes
//   Usage:
//     srun ./PenningTrapPIF <nx> <ny> <nz> <Np> <Nt> <dt> <ShapeType> <degree> --info 5
//     nx       = No. of Fourier modes in the x-direction
//     ny       = No. of Fourier modes in the y-direction
//     nz       = No. of Fourier modes in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     dt       = Time stepsize
//     ShapeType = Shape function type B-spline only for the moment
//     degree = B-spline degree (-1 for delta function)
//     Example:
//     srun ./PenningTrapPIF 32 32 32 655360 20 0.05 B-spline 1 --info 5
//
// Copyright (c) 2023, Sriramkrishnan Muralikrishnan,
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

#include "ChargedParticlesPIF.hpp"
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

const char* TestName = "PenningTrapPIF";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    
    Inform msg(TestName);
    Inform msg2all(TestName,INFORM_ALL_NODES);

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("kick");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef BCTimer = IpplTimings::getTimer("particleBC");
    static IpplTimings::TimerRef initializeShapeFunctionPIF = IpplTimings::getTimer("initializeShapeFunctionPIF");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    const double dt = std::atof(argv[6]);

    msg << TestName
        << endl
        << "nt " << nt << " Np= "
        << totalP << " Fourier modes = " << nr
        << endl;

    using bunch_type = ChargedParticlesPIF<PLayout_t>;

    std::unique_ptr<bunch_type>  P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::SERIAL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t rmin(0.0);
    Vector_t rmax(20.0);
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t length = rmax - rmin;

    Vector_t mu, sd;

    for (unsigned d = 0; d<Dim; d++) {
        mu[d] = 0.5 * length[d];
    }
    sd[0] = 0.15*length[0];
    sd[1] = 0.05*length[1];
    sd[2] = 0.20*length[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    double Q = -1562.5;
    double Bext = 5.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q,totalP);

    P->nr_m = nr;

    P->rho_m.initialize(mesh, FL);
    P->Sk_m.initialize(mesh, FL);

    P->time_m = 0.0;

    P->shapetype_m = argv[7]; 
    P->shapedegree_m = std::atoi(argv[8]); 

    IpplTimings::startTimer(particleCreation);

    //typedef ippl::detail::RegionLayout<double, Dim, Mesh_t> RegionLayout_t;
    //const RegionLayout_t& RLayout = PL.getRegionLayout();
    //const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();
    Vector_t minU, maxU;
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], mu[d], sd[d]);
        maxU[d] = CDF(rmax[d], mu[d], sd[d]);
    }

    double factor = 1.0/Ippl::Comm->size();
    size_type nloc = (size_type)(factor * totalP);
    size_type Total_particles = 0;

    MPI_Allreduce(&nloc, &Total_particles, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());

    int rest = (int) (totalP - Total_particles);

    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    P->create(nloc);
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         P->R.getView(), P->P.getView(), rand_pool64, mu, sd, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);                                                    
    
    P->q = P->Q_m/totalP;
    msg << "particles created and initial conditions assigned " << endl;

    IpplTimings::startTimer(initializeShapeFunctionPIF);
    P->initializeShapeFunctionPIF();
    IpplTimings::stopTimer(initializeShapeFunctionPIF);

    ippl::ParameterList fftParams;

    fftParams.add("gpu_method", 1);
    fftParams.add("gpu_sort", 1);
    fftParams.add("gpu_kerevalmeth", 1);
    fftParams.add("tolerance", 1e-4);

    fftParams.add("use_cufinufft_defaults", false);


    P->fft = std::make_shared<FFT_type>(FL, 1, fftParams);

    P->q.initializeNUFFT(FL, 1, fftParams);
    P->E.initializeNUFFT(FL, 2, fftParams);

    P->scatter();

    P->gather();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpEnergy();
    IpplTimings::stopTimer(dumpDataTimer);

    double alpha = -0.5 * dt;
    double DrInv = 1.0 / (1 + (std::pow((alpha * Bext), 2))); 
    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // Staggered Leap frog or Boris algorithm as per 
        // https://www.sciencedirect.com/science/article/pii/S2590055219300526
        // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
        // the analytical matrix inversion which is not complex in this case.
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        // kick
        IpplTimings::startTimer(PTimer);
        auto Rview = P->R.getView();
        auto Pview = P->P.getView();
        auto Eview = P->E.getView();
        double V0 = 30*rmax[2];
        Kokkos::parallel_for("Kick1", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
            double Eext_x = -(Rview(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_y = -(Rview(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_z =  (Rview(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

            Eview(j)[0] += Eext_x;
            Eview(j)[1] += Eext_y;
            Eview(j)[2] += Eext_z;
            
            Pview(j)[0] += alpha * (Eview(j)[0]  + Pview(j)[1] * Bext);
            Pview(j)[1] += alpha * (Eview(j)[1]  - Pview(j)[0] * Bext);
            Pview(j)[2] += alpha * Eview(j)[2];
        });
        IpplTimings::stopTimer(PTimer);

        //drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        //Apply particle BC
	    IpplTimings::startTimer(BCTimer);
        PL.applyBC(P->R, PL.getRegionLayout().getDomain());
        IpplTimings::stopTimer(BCTimer);

        //scatter the charge onto the underlying grid
        P->scatter();

        // Solve for and gather E field
        P->gather();

        //kick
        IpplTimings::startTimer(PTimer);
        auto R2view = P->R.getView();
        auto P2view = P->P.getView();
        auto E2view = P->E.getView();
        Kokkos::parallel_for("Kick2", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
            double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

            E2view(j)[0] += Eext_x;
            E2view(j)[1] += Eext_y;
            E2view(j)[2] += Eext_z;
            P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (E2view(j)[0] 
                            + P2view(j)[1] * Bext + alpha * Bext * E2view(j)[1]) );
            P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (E2view(j)[1] 
                            - P2view(j)[0] * Bext - alpha * Bext * E2view(j)[0]) );
            P2view(j)[2] += alpha * E2view(j)[2];
        });
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpEnergy();
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

    msg << TestName << " End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
