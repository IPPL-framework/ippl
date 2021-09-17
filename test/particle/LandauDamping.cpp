// Uniform Plasma Test
//
//   Usage:
//     srun ./LandauDamping 128 128 128 10000 10 FFT --info 10
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
#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>
#include <chrono>

#include<Kokkos_Random.hpp>

#include <random>
#include "Utility/IpplTimings.h"
#include "Solver/FFTPeriodicPoissonSolver.h"

// dimension of our positions
constexpr unsigned Dim = 3;

// some typedefs
typedef ippl::ParticleSpatialLayout<double,Dim>   PLayout_t;
typedef ippl::UniformCartesian<double, Dim>        Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;

template<typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template<typename T, unsigned Dim>
using Field = ippl::Field<T, Dim>;

template<typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim>  Vector_t;
typedef Field<double, Dim>   Field_t;
typedef Field<Vector_t, Dim> VField_t;
typedef ippl::FFTPeriodicPoissonSolver<Vector_t, double, Dim> Solver_t;

double pi = acos(-1.0);

void dumpVTK(VField_t& E, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    Kokkos::deep_copy(host_view, E.getView());
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
    vtkout << "UniformPlasmaTest" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN "     << -2*dx  << " " << -2*dy  << " "  << -2*dz << std::endl;
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


void dumpVTK(Field_t& rho, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();
    Kokkos::deep_copy(host_view, rho.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    //double vol = dx*dy*dz;

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "UniformPlasmaTest" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx+3 << " " << ny+3 << " " << nz+3 << std::endl;
    vtkout << "ORIGIN " << -2*dx << " " << -2*dy << " " << -2*dz << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "CELL_DATA " << (nx+2)*(ny+2)*(nz+2) << std::endl;

    vtkout << "SCALARS Rho float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=0; z<nz+2; z++) {
        for (int y=0; y<ny+2; y++) {
            for (int x=0; x<nx+2; x++) {

                vtkout << host_view(x,y,z) << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}

template <typename T>
struct Newton1D {

  double tol = 1e-12;
  int max_iter = 20;
  double pi = acos(-1.0);
  
  T k, alpha, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(T k_, T alpha_, T u_) 
       : k(k_), alpha(alpha_),
         u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T x) {
      T F;
      F = (x  + (alpha  * (sin(k * x) / k))) 
          - ((2 * pi / k) * u);
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T x) {
      T Fprime;
      Fprime = 1  + (alpha  * cos(k * x));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T xp, T x) {
      int iterations = 0;
      while (iterations < max_iter && abs(f(xp)) > tol) {
          x = xp - (f(xp)/fprime(xp));
          xp = x;
          iterations += 1;
      }
  }
};


template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  //using Nsolver_t = typename Newton1D<value_type>;
  // Output View for the random numbers
  view_type x, v;

  // The GeneratorPool
  GeneratorPool rand_pool;

  //Nsolver_t solver;
  //Newton1D<value_type> solver;

  value_type alpha;

  T len, k;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                 T len_, value_type alpha_, T k_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        len(len_), alpha(alpha_),
        k(k_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u, xp;
    for (unsigned d = 0; d < Dim; ++d) {
        u = rand_gen.drand(0, 1);
        xp = (len[d] * u) / (1 + alpha);
        Newton1D<value_type> solver(k[d], alpha, u);
        solver.solve(xp, x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

//template <typename T, class GeneratorPool, unsigned Dim>
//struct generate_random_vel {
//
//  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
//  // Output View for the random numbers
//  view_type vals;
//
//  // The GeneratorPool
//  GeneratorPool rand_pool;
//
//  // Initialize all members
//  generate_random(view_type vals_, GeneratorPool rand_pool_)
//      : vals(vals_), rand_pool(rand_pool_) {}
//
//  KOKKOS_INLINE_FUNCTION
//  void operator()(int i) const {
//    // Get a random number state from the pool for the active thread
//    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
//
//    for (unsigned d = 0; d <Dim; ++d) {
//      vals(i)[d] = rand_gen.normal(0.0, 1.0);
//    }
//
//    // Give the state back, which will allow another thread to acquire it
//    rand_pool.free_state(rand_gen);
//  }
//};

template<class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:
    VField_t E_m;
    Field_t rho_m;

    Vector<int, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    double Q_m;

    std::string stype_m;

    std::shared_ptr<Solver_t> solver_mp;

    double time_m;

    double rhoNorm_m;


public:
    ParticleAttrib<double>     q; // charge
    typename ippl::ParticleBase<PLayout>::particle_position_type P;  // particle velocity
    typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position


    /*
      This constructor is mandatory for all derived classes from
      ParticleBase as the update function invokes this
    */
    ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl)
    {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }

    ChargedParticles(PLayout& pl,
                     Vector_t hr,
                     Vector_t rmin,
                     Vector_t rmax,
                     ippl::e_dim_tag decomp[Dim],
                     double Q)
    : ippl::ParticleBase<PLayout>(pl)
    , hr_m(hr)
    , rmin_m(rmin)
    , rmax_m(rmax)
    , Q_m(Q)
    {
        // register the particle attributes
        this->addAttribute(q);
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

    void gatherStatistics(uint64_t totalP, int iteration) {
        std::cout << "Rank " << Ippl::Comm->rank() << " has "
                  << (double)this->getLocalNum()/totalP*100.0
                  << "percent of the total particles " << std::endl;
    }

    void gatherCIC() {
        gather(this->E, E_m, this->R);
    }

    void scatterCIC(uint64_t totalP, int iteration, Vector_t& hrField) {
         Inform m("scatter ");

         rho_m = 0.0;
         scatter(q, rho_m, this->R);

         static IpplTimings::TimerRef sumTimer = IpplTimings::getTimer("Check");
         IpplTimings::startTimer(sumTimer);
         double Q_grid = rho_m.sum();

         uint64_t Total_particles = 0;
         uint64_t local_particles = this->getLocalNum();

         MPI_Reduce(&local_particles, &Total_particles, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, Ippl::getComm());

         double rel_error = std::fabs((Q_m-Q_grid)/Q_m);
         m << "Rel. error in charge conservation = " << rel_error << endl;

         if(Ippl::Comm->rank() == 0) {
             if((Total_particles != totalP) || (rel_error > 1e-10)) {
                 std::cout << "Total particles in the sim. " << totalP 
                           << " " << "after update: " 
                           << Total_particles << std::endl;
                 std::cout << "Total particles not matched in iteration: "
                           << iteration << std::endl;
                 std::cout << "Rel. error in charge conservation: "
                           << rel_error << std::endl;
                 exit(1);
             }
         }

         rho_m = rho_m / (hrField[0] * hrField[1] * hrField[2]);

         rhoNorm_m = norm(rho_m);
         IpplTimings::stopTimer(sumTimer);

         //dumpVTK(rho_m,nr_m[0],nr_m[1],nr_m[2],iteration,hrField[0],hrField[1],hrField[2]);

         //rho = rho_e - rho_i
         rho_m = rho_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    }

    void initSolver() {
        Inform m("solver ");
        if(stype_m == "FFT")
            initFFTSolver();
        else
            m << "No solver matches the argument" << endl;
    }

    void initFFTSolver() {
        ippl::ParameterList sp;

        sp.add("output_type", Solver_t::GRAD);

        ippl::FFTParams fftParams;

#ifdef Heffte_ENABLE_CUDA
        fftParams.setAllToAll( false );
#else
        fftParams.setAllToAll( true );
#endif
        fftParams.setPencils( true );
        fftParams.setReorder( false );
        fftParams.setRCDirection( 0 );

        solver_mp = std::make_shared<Solver_t>(fftParams);

        solver_mp->mergeParameters(sp);
        
        solver_mp->setRhs(rho_m);

        solver_mp->setLhs(E_m);
    }



     void dumpData() {
        auto Pview = P.getView();

        double Energy = 0.0;

        Kokkos::parallel_reduce("Particle Energy", this->getLocalNum(),
                                KOKKOS_LAMBDA(const int i, double& valL){
                                    double myVal = dot(Pview(i), Pview(i)).apply();
                                    valL += myVal;
                                }, Kokkos::Sum<double>(Energy));

        Energy *= 0.5;
        double gEnergy = 0.0;

        MPI_Reduce(&Energy, &gEnergy, 1,
                    MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());


        const int nghostE = E_m.getNghost();
        auto Eview = E_m.getView();
        double fieldEnergy, ExAmp;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        double temp = 0.0;
        Kokkos::parallel_reduce("Ex inner product",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = pow(Eview(i, j, k)[0], 2);
                                    valL += myVal;
                                }, Kokkos::Sum<double>(temp));
        double globaltemp = 0.0;
        MPI_Reduce(&temp, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
        fieldEnergy = 0.5 * globaltemp * hr_m[0] * hr_m[1] * hr_m[2];

        double tempMax = 0.0;
        Kokkos::parallel_reduce("Ex max norm",
                                mdrange_type({nghostE, nghostE, nghostE},
                                             {Eview.extent(0) - nghostE,
                                              Eview.extent(1) - nghostE,
                                              Eview.extent(2) - nghostE}),
                                KOKKOS_LAMBDA(const size_t i, const size_t j,
                                              const size_t k, double& valL)
                                {
                                    double myVal = abs(Eview(i, j, k)[0]);
                                    if(myVal > valL) valL = myVal;
                                }, Kokkos::Max<double>(tempMax));
        ExAmp = 0.0;
        MPI_Reduce(&tempMax, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, Ippl::getComm());


        if(Ippl::Comm->rank() == 0) {
            std::ofstream csvout;
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << Ippl::Comm->size();
            fname << ".csv";
            csvout.open(fname.str().c_str(), std::ios::out | std::ofstream::app);

            if(time_m == 0.0) {
                csvout << "time, Kinetic energy, Ex_field_energy, Ex_max_norm" << std::endl;
            }

            csvout << time_m << " "
                   << gEnergy << " "
                   << fieldEnergy << " "
                   << ExAmp << std::endl;

            csvout.close();
        }

        Ippl::Comm->barrier();

     }

private:
    void setBCAllPeriodic() {
        this->setParticleBC(ippl::BC::PERIODIC);
    }

};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("LandauDamping");
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(2);

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
    static IpplTimings::TimerRef temp = IpplTimings::getTimer("randomMove");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");

    IpplTimings::startTimer(mainTimer);

    const uint64_t totalP = std::atoll(argv[4]);
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
    Vector_t k = {0.5, 0.5, 0,5};
    double alpha = 0.05;
    Vector_t rmin(0.0);
    Vector_t rmax = 2 * pi / k ;
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
    uint64_t nloc = totalP / Ippl::Comm->size();

    int rest = (int) (totalP - nloc * Ippl::Comm->size());

    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    IpplTimings::startTimer(particleCreation);
    P->create(nloc);

    const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
    Vector_t Rmin, Rmax;
    for (unsigned d = 0; d <Dim; ++d) {
        Rmin[d] = origin[d] + lDom[d].first() * hr[d];
        Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
    }

    Kokkos::Random_XorShift64_Pool<> rand_pool64((uint64_t)(42 + 100*Ippl::Comm->rank()));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         P->R.getView(), P->P.getView(), rand_pool64, rmax, alpha, k));
    Kokkos::fence();
    P->q = P->Q_m/totalP;
    IpplTimings::stopTimer(particleCreation);

    IpplTimings::startTimer(FirstUpdateTimer);
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);

    bunch_type bunchBuffer(PL);

	IpplTimings::startTimer(updateTimer);
    PL.update(*P, bunchBuffer);
    IpplTimings::stopTimer(updateTimer);

    msg << "particles created and initial conditions assigned " << endl;

    P->stype_m = argv[6];
    P->initSolver();
    P->time_m = 0.0;

    P->scatterCIC(totalP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpData();
    IpplTimings::stopTimer(dumpDataTimer);

    IpplTimings::stopTimer(FirstUpdateTimer);

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
        PL.update(*P, bunchBuffer);  //P->update();
        IpplTimings::stopTimer(updateTimer);

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
        P->dumpData();
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished iteration: " << it << " time: " << P->time_m << endl;
    }

    msg << "Uniform Plasma Test: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
