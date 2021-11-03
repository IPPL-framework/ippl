// Landau Damping Test
//
//   Usage:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 1.0 1.0 --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
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
//

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
  //KOKKOS_INLINE_FUNCTION
  //void initialize(const T& k_, 
  //                const T& alpha_, 
  //                const T& u_) {
  //    k = k_;
  //    alpha = alpha_;
  //    u = u_;
  //}

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

  //Newton1D<value_type> solver;

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
        //solver.initialize(k[d], alpha, u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(double& x, double& alpha, double& k) {
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

const char* TestName = "LandauDamping";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("LandauDamping");
    Inform msg2all("LandauDamping",INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(std::atof(argv[8]));

    auto start = std::chrono::high_resolution_clock::now();
    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef FirstUpdateTimer = IpplTimings::getTimer("initialisation");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("kick");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("domainDecomp");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);

    msg << "Landau damping"
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
    Vector_t kw = {0.5, 0.5, 0.5};
    double alpha = 0.5;
    Vector_t rmin(0.0);
    Vector_t rmax = 2 * pi / kw ;
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};
    const double dt = 0.5*dx;

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    //Q = -\int\int f dx dv
    double Q = -rmax[0] * rmax[1] * rmax[2];
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);

    P->nr_m = nr;

    IpplTimings::startTimer(particleCreation);

    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);

    bunch_type bunchBuffer(PL);

    P->stype_m = argv[6];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[7]);

    bool isFirstRepartition;

    if (P->loadbalancethreshold_m != 1.0) {
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
    
    msg << "First domain decomposition done" << endl;
    IpplTimings::startTimer(particleCreation);

    typedef ippl::detail::RegionLayout<double, Dim, Mesh_t> RegionLayout_t;
    const RegionLayout_t& RLayout = PL.getRegionLayout();
    const typename RegionLayout_t::host_mirror_type& Regions = RLayout.gethLocalRegions();
    Vector_t Nr, Dr, minU, maxU;
    int myRank = Ippl::Comm->rank();
    for (unsigned d = 0; d <Dim; ++d) {
        Nr[d] = CDF(Regions(myRank)[d].max(), alpha, kw[d]) - 
                CDF(Regions(myRank)[d].min(), alpha, kw[d]);  
        Dr[d] = CDF(rmax[d], alpha, kw[d]) - CDF(rmin[d], alpha, kw[d]);
        minU[d] = CDF(Regions(myRank)[d].min(), alpha, kw[d]);
        maxU[d]   = CDF(Regions(myRank)[d].max(), alpha, kw[d]);
    }

    double factor = (Nr[0] * Nr[1] * Nr[2]) / (Dr[0] * Dr[1] * Dr[2]);
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
                         P->R.getView(), P->P.getView(), rand_pool64, alpha, kw, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);                                                    
    
    P->q = P->Q_m/totalP;
    msg << "particles created and initial conditions assigned " << endl;
    isFirstRepartition = false;
    //The following update is not needed as the particles are all generated locally
	//IpplTimings::startTimer(updateTimer);
    //PL.update(*P, bunchBuffer);
	//IpplTimings::stopTimer(updateTimer);

    P->scatterCIC(totalP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLandau();
    P->gatherStatistics(totalP);
    IpplTimings::stopTimer(dumpDataTimer);

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
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
        if (P->balance(totalP, it+1)) {
           msg << "Starting repartition" << endl;
           IpplTimings::startTimer(domainDecomposition);
           P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
           IpplTimings::stopTimer(domainDecomposition);
        }


        //scatter the charge onto the underlying grid
        P->scatterCIC(totalP, it+1, hr);

        //Field solve
        IpplTimings::startTimer(SolveTimer);
        P->solver_mp->solve();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        P->gatherCIC();

        //kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpLandau();
        P->gatherStatistics(totalP);
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

    msg << "LandauDamping: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
