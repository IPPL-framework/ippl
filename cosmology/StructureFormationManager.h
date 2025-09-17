#ifndef IPPL_STRUCTURE_FORMATION_MANAGER_H
#define IPPL_STRUCTURE_FORMATION_MANAGER_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "GravityManager.h"
#include "GravityParticleContainer.hpp"
#include "Manager/BaseManager.h"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

#include "mc-4-Initializer/InputParser.h"
#include "mc-4-Initializer/DataBase.h"
#include "mc-4-Initializer/Cosmology.h"


//#define KOKKOS_PRINT    // Kokkos::printf of interesting quantities. Does not work multirank

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

typedef ippl::Field<Kokkos::complex<double>, Dim, Mesh_t<Dim>, Mesh_t<Dim>::DefaultCentering> field_type;
typedef ippl::FFT<ippl::CCTransform, field_type> CFFT_type;
typedef Field<Kokkos::complex<double>, Dim> CField_t;
typedef Field<double, Dim> RField_t; 

struct HermitianPkg {
  int    kx, ky, kz;
  double re, im;
};

/**
 * @brief Construct a new StructureFormationManager object.
 *
 * @param totalP_ Total number of particles.
 * @param nt_ Number of time steps.
 * @param nr_ Number of gridpoints in each dimension
 * @param lbt_ Lookback time.
 * @param solver_ Solver method.
 * @param stepMethod_ Time stepping method.
 * @param par_ the parser to read the input file
 * @param tfname_ filename for transfer function
 * @param readICs_ read or create initial conditions
 */
template <typename T, unsigned Dim>
class StructureFormationManager : public GravityManager<T, Dim> {

  /// all for the initializer
  std::unique_ptr<CFFT_type> Cfft_m;
  CField_t cfield_m;
  RField_t Pk_m;
  bool readICs_m;

  initializer::CosmoClass cosmo_m;
  
public:
  using ParticleContainer_t = ParticleContainer<T, Dim>;
  using FieldContainer_t    = FieldContainer<T, Dim>;
  using FieldSolver_t       = FieldSolver<T, Dim>;
  using LoadBalancer_t      = LoadBalancer<T, Dim>;

  using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
  using index_type       = typename ippl::RangePolicy<Dim>::index_type;
  
  StructureFormationManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
			    std::string& solver_, std::string& stepMethod_,	
			    initializer::InputParser par_, std::string tfname_, bool readICs_)
    : GravityManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_, par_),
      readICs_m(readICs_)
  {
    Inform m ("StructureFormationManager ");
    m << "totalP = " << totalP_ << endl;
    cosmo_m.SetParameters(initializer::GlobalStuff::instance(), tfname_.c_str()); 
  }

    /**
     * @brief Destructor for StructureFormationManager.
     */
    ~StructureFormationManager() {}

    /**
     * @brief Pre-run setup for the simulation.
     */
    void pre_run() override {
        Inform msg("Pre Run");

        if (this->solver_m == "OPEN") {
            throw IpplException("StructureFormation",
                                "Open boundaries solver incompatible with this simulation!");
        }

        // Grid
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->Hubble0 = 0.1;         // h * km/sec/kpc  (h = 0.7, H = 0.07)
        this->G       = 4.30071e04;  // kpc km^2 /s^2 / M_Sun e10

	float zm;
	float zf;
	this->parser_m.getByName("z_in", zm);     // initial z
	this->parser_m.getByName("z_fi", zf);     // final z
	this->z_m = zm;
	this->z_f = zf;
	
        this->InitialiseTime();

	float box_size;
	this->parser_m.getByName("box_size", box_size);
        this->rmin_m = 0.0;	
	this->rmax_m = box_size*1000.0; // kpc/h

        double Vol =
            std::reduce(this->rmax_m.begin(), this->rmax_m.end(), 1., std::multiplies<double>());
        this->M_m = this->rho_crit0 * Vol * this->O_m;  // 1e10 M_Sun
        msg << "total mass: " << this->M_m << endl;
        msg << "mass of a single particle " << this->M_m / this->totalP_m << endl;

        this->hr_m     = this->rmax_m / this->nr_m;
        this->origin_m = this->rmin_m;
        this->it_m     = 0;

        msg << "Discretization:" << endl
            << "nt " << this->nt_m << ", Np = " << this->totalP_m << ", grid = " << this->nr_m
            << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver(std::make_shared<FieldSolver_t>(
            this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getF(),
            &this->fcontainer_m->getPhi()));

        this->fsolver_m->initSolver();

        this->setLoadBalancer(std::make_shared<LoadBalancer_t>(
            this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m));

	if (readICs_m) {
	  msg << "Read in particles ..." << endl;
	  readParticlesDomain();  // defines particle positions, velocities
	  msg << "Read particles done" << endl;
	  static IpplTimings::TimerRef DummySolveTimer = IpplTimings::getTimer("solveWarmup");
	  IpplTimings::startTimer(DummySolveTimer);

	  this->fcontainer_m->getRho() = 0.0;
	  
	  this->fsolver_m->runSolver();

	  IpplTimings::stopTimer(DummySolveTimer);
	  this->par2grid();

	  static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");

	  IpplTimings::startTimer(SolveTimer);
	  this->fsolver_m->runSolver();
	  IpplTimings::stopTimer(SolveTimer);
	  
	  this->grid2par();
	  this->dump();
	  msg << "Done reading initial conditions";
	} else {
	  msg << "Create Particles" << endl;
	  ippl::ParameterList fftParams;
	  fftParams.add("use_heffte_defaults", true);
	  Cfft_m = std::make_unique<CFFT_type>(this->fcontainer_m->getFL(), fftParams);
	  cfield_m.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());
	  Pk_m.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());
	  msg << "FFT and Pk structures initialized";
	  createParticles();
	}
    }


  void initPwrSpec() {
    Inform msg("initspec ");

    constexpr auto tpi = 2*Kokkos::numbers::pi_v<double>;
    const double n_s = initializer::GlobalStuff::instance().n_s;
    const double sigma8=initializer::GlobalStuff::instance().Sigma_8;
    const double f_NL=initializer::GlobalStuff::instance().f_NL;
    const int ngrid=initializer::GlobalStuff::instance().ngrid;
    const int nq=ngrid/2;
    const double tpiL=tpi/initializer::GlobalStuff::instance().box_size; // k0, physical units

/* TFFlag == 2) Hu-Sugiyama transfer function
    const double Omega_m = initializer::GlobalStuff::instance().Omega_m;
    const double Omega_bar = initializer::GlobalStuff::instance().Omega_bar;
    const double h = initializer::GlobalStuff::instance().Hubble;
    const double akh1=pow(46.9*Omega_m*h*h, 0.670)*(1.0+pow(32.1*Omega_m*h*h, -0.532));
    const double akh2=pow(12.0*Omega_m*h*h, 0.424)*(1.0+pow(45.0*Omega_m*h*h, -0.582));
    const double alpha=pow(akh1, -1.0*Omega_bar/Omega_m)*pow(akh2, pow(-1.0*Omega_bar/Omega_m, 3.0));
    const double kh_tmp = Omega_m*h*Kokkos::sqrt(alpha);
*/
    // TFFlag == 4 BBKS  transfer function

    const float Omega_m = initializer::GlobalStuff::instance().Omega_m;
    const float h       = initializer::GlobalStuff::instance().Hubble;    
    const float kh_tmp   = Omega_m*h;
    
    /* Set P(k)=T^2*k^n array.
       Linear array, taking care of the mirror symmetry
       (reality condition for the density field). 
    */
    
    msg << "Pulled all needed physics quantities" << endl;
    msg << "kh_tmp= " << kh_tmp << " n_s= " << n_s << " tpiL= " << tpiL << endl;
    
    auto pkview                    = Pk_m.getView();
    auto* FL                       = &this->fcontainer_m->getFL();
    const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();  // local processor domain coordinates    
    const int nghost               = Pk_m.getNghost();

    ippl::parallel_for("Compute Pk_m", ippl::getRangePolicy(pkview, nghost),
                       KOKKOS_LAMBDA(const index_array_type& idx){
			   int i = idx[0] - nghost + ldom[0].first();
			   int j = idx[1] - nghost + ldom[1].first();
			   int k = idx[2] - nghost + ldom[2].first();

			   int k_i = (i >= nq) ? -(nq-i % nq) : i;
			   int k_j = (j >= nq) ? -(nq-j % nq) : j;
			   int k_k = (k >= nq) ? -(nq-k % nq) : k;
			   
			   int   k2 = k_i * k_i + k_j * k_j + k_k * k_k;
			   float kk = tpiL*Kokkos::sqrt(k2);
			   double trans_f = StructureFormationManager<double,3>::bbks(kk, kh_tmp);
			   // double trans_f = StructureFormationManager<double,3>::hu_sugiyama(kk, kh_tmp);
			   double val = trans_f*trans_f*Kokkos::pow(kk, n_s);
			   ippl::apply(pkview, idx) = val;
			 
#ifdef KOKKOS_PRINT
			   int index = (k_i)
			     + (k_j) * ldom[0].length()
			     + (k_k) * ldom[0].length() * ldom[1].length();

			   if ((k_i==0) && (k_j==0) && (k_k==0))
			     Kokkos::printf("Pk: i \t j \t k \t index \t kk \t transf \t Pk \n");
			   Kokkos::printf("Pk: %d \t %d \t %d \t %d \t %f \t %f \t %f \n",k_i,k_j,k_k,index,kk,trans_f,val);
#endif		
		       });

    msg << "Pk created using BBKS TFFlag ==4" << endl;
    
    double s8 = cosmo_m.Sigma_r(8.0, 1.0);                                                    
    const double norm = sigma8*sigma8/(s8*s8);                                                     
    s8 = cosmo_m.Sigma_r(8.0, norm);                                                
    msg << "sigma_8=" << s8 << ", target was " << sigma8 << endl;
    msg << "norm= " << norm << endl;

    // For non-Gaussian initial conditions: P(k)=A*k^n, trasfer function will come later
    if (f_NL != 0.0){
      msg << "Non-Gaussian initial conditions, f_NL=" << f_NL << endl;
      ippl::parallel_for("Norm Pk_m (non-Gaussian)", ippl::getRangePolicy(pkview, nghost),
			 KOKKOS_LAMBDA (const index_array_type& args) {
			   int k_i    = args[0] - nghost + ldom[0].first();
			   int k_j    = args[1] - nghost + ldom[1].first();
			   int k_k    = args[2] - nghost + ldom[2].first();
			   if (k_k >= nq) {
			     k_k = -fmod(ngrid-k_k,ngrid);
			   }
			   double kk = tpiL*Kokkos::sqrt(k_i*k_i+k_j*k_j+k_k*k_k);
			   ippl::apply(pkview, args) *= norm*pow(kk, n_s);
			 });
    }
    else {
    // For Gaussian initial conditions: P(k)=A*T^2*k^n
      msg << "Gaussian initial conditions, f_NL=" << f_NL << endl;
      ippl::parallel_for("Norm Pk_m (Gaussian)", ippl::getRangePolicy(pkview, nghost),
			 KOKKOS_LAMBDA (const index_array_type& args) {
			   ippl::apply(pkview, args) *= norm;
			 });
    }
  }

  /**
     @brief bbks  hardcoded TFFLAG==4

   */

  KOKKOS_FUNCTION static float bbks(float k, float kh_tmp) {
    float qkh, t_f;
   
    if (k == 0.0) return(0.0);
    qkh = k/kh_tmp;
    t_f = Kokkos::log(1.0+2.34*qkh)/(2.34*qkh) * Kokkos::pow(1.0+3.89*qkh+
	  Kokkos::pow(16.1*qkh, 2.0) + Kokkos::pow(5.46*qkh, 3.0) + Kokkos::pow(6.71*qkh, 4.0), -0.25);
   return(t_f);

  }

  void LinearZeldoInitMP() {
    // After creating the field layout (cfield_m) and determining global grid sizes Nx, Ny, Nz:
    Inform msg("LinearZeldoInitMP ");
    Inform m2a("LinearZeldoInitMP ", INFORM_ALL_NODES);
    
    typename CField_t::view_type& cfview = cfield_m.getView();
    typename RField_t::view_type& pkview = Pk_m.getView();
      
    auto rView = this->pcontainer_m->R.getView();
    auto vView = this->pcontainer_m->V.getView();

    const int ngh = cfield_m.getNghost();
    const ippl::NDIndex<Dim>& lDom = this->fcontainer_m->getFL().getLocalNDIndex();

    index_type lgridsize = 1;
    for (unsigned d = 0; d < Dim; d++) {
      lgridsize *= lDom[d].length();
    }
    const uint64_t global_seed = 12345ULL;  // Shared global seed for reproducibility

    const int Nx = this->nr_m[0];
    const int Ny = this->nr_m[1];
    const int Nz = this->nr_m[2];
    const Vector_t<double,3> hr = this->hr_m;
    const double Lx = this->rmax_m[0];
    const double Ly = this->rmax_m[1];
    const double Lz = this->rmax_m[2];

    msg << "Lx= " << Lx << " Ly= " << Ly << " Lz= " << Lz << endl;
    msg << "Nx= " << Nx << " Ny= " << Ny << " Nz= " << Nz << endl;
    msg << "h = " << this->hr_m << endl;

    float d_z;
    float ddot;
    float z_in = initializer::GlobalStuff::instance().z_in;
    cosmo_m.GrowthFactor(z_in, &d_z, &ddot);
    msg << "z_in= " << z_in << " d_z= " << d_z << " ddot= " << ddot << endl;

    Kokkos::Random_XorShift64_Pool<> rand_pool(global_seed);
    
    // Initialize the Fourier density field with Gaussian random modes (Hermitian symmetric)
    ippl::parallel_for("InitDeltaField", ippl::getRangePolicy(cfview, ngh),
		       KOKKOS_LAMBDA(const index_array_type& idx) {
			 const double pi = Kokkos::numbers::pi_v<double>;
			 // Compute global coordinates (i,j,k) for this local index
			 int i = idx[0] - ngh + lDom[0].first();
			 int j = idx[1] - ngh + lDom[1].first();
			 int k = idx[2] - ngh + lDom[2].first();

			 // DC mode (k=0 vector) set to 0 (no DC offset)
			 if (i == 0 && j == 0 && k == 0) {
			   ippl::apply(cfview, idx) = Kokkos::complex<double>(0.0, 0.0);
			 } else {
			   // Compute the global “negative” indices for Hermitian pair
			   int i_neg = (i == 0 ? 0 : Nx - i);
			   int j_neg = (j == 0 ? 0 : Ny - j);
			   int k_neg = (k == 0 ? 0 : Nz - k);
			   
			   // Determine if this index is its own conjugate (self-Hermitian case)
			   bool self = (i_neg == i && j_neg == j && k_neg == k);
			   // Determine lexicographically which of (i,j,k) and its negative is smaller
			   bool is_conjugate = (!self && 
						(i_neg < i || (i_neg == i && j_neg < j) || 
						 (i_neg == i && j_neg == j && k_neg < k)));

			   auto rand_gen = rand_pool.get_state();
			   // Generate a uniform random number in [0,1)
			   double u1 = rand_gen.drand();
			   double u2 = rand_gen.drand();
			   // Return the state to the pool
			   rand_pool.free_state(rand_gen);
			   
			   // Convert uniforms to Gaussian via Box-Muller
			   double R     = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
			   double theta = 2.0 * pi * u2;
			   double gauss_re = R * Kokkos::cos(theta);   // Gaussian(0,1) for real part
			   double gauss_im = R * Kokkos::sin(theta);   // Gaussian(0,1) for imaginary part
                           double Pk = ippl::apply(pkview, idx);       // power spectrum P(k) 
			   // Set amplitude: for self-conjugate modes use sqrt(Pk), otherwise sqrt(Pk/2)
			   double amp = self ? Kokkos::sqrt(Pk) : Kokkos::sqrt(Pk / 2.0);
			   double val_re = amp * gauss_re;
			   double val_im = amp * gauss_im;
			   if (self) {
			     // For self-conjugate (Nyquist) modes, enforce the mode is real
			     val_im = 0.0;
			   } else if (is_conjugate) {
			     // If this index is the "conjugate partner" (lexicographically larger), flip the imaginary sign
			     val_im = -val_im;
			   }
#ifdef KOKKOS_PRINT		   			   
			   Kokkos::printf("rho: %g %g \n",val_re,val_im);
#endif
			   // Assign the complex value to this local mode
			   ippl::apply(cfview, idx) = Kokkos::complex<double>(val_re, val_im);
			 }
		       });

    // Store delta(k) for reuse 
    auto tmpcfield = cfield_m.deepCopy();

    /*
      Now we can delete Pk and allocate the particles
    */

    // 2–4. Loop over displacement components x(0), y(1), z(2)
    double xx,yy,zz;
    int nq = Nx/2;

    for (int dim = 0; dim < 3; ++dim) {
      switch (dim) {
      case 0:
	xx = 1.0;
	yy = 0.0;
	zz = 0.0;
	break;
      case 1:
	xx = 0.0;
	yy = 1.0;
	zz = 0.0;
	break;
      case 2:
	xx = 0.0;
	yy = 0.0;
	zz = 1.0;
	break;
      }
      
      // Compute displacement component in k-space
      ippl::parallel_for("force_in_k_space", ippl::getRangePolicy(cfview, ngh),
                         KOKKOS_LAMBDA(const index_array_type& idx){
			   const double pi = Kokkos::numbers::pi_v<double>;
			   int i = idx[0] - ngh + lDom[0].first();
			   int j = idx[1] - ngh + lDom[1].first();
			   int k = idx[2] - ngh + lDom[2].first();

			   int k_i = (i >= nq) ? -(nq-i % nq) : i;
			   int k_j = (j >= nq) ? -(nq-j % nq) : j;
			   int k_k = (k >= nq) ? -(nq-k % nq) : k;
			   
			   int k2 = k_i * k_i + k_j * k_j + k_k * k_k;
			   double Grad = -2*pi/Nx*(k_i*xx+k_j*yy+k_k*zz);
			   double kk = 2*pi/Nx*sqrt(k2);
			   double Green = (kk!=0.0) ? -1.0/(kk*kk) : 0.0;
			   
			   Kokkos::complex<double> tmp = ippl::apply(tmpcfield, idx);
			   Kokkos::complex<double> res (-Grad * Green * tmp.imag(), Grad * Green * tmp.real());
			   ippl::apply(cfview, idx) = res;
#ifdef KOKKOS_PRINT
                           Kokkos::printf("feval: %i %g %g %g %g %g  %g \n", dim, Grad, Green, tmp.imag(), tmp.real(), res.real(), res.imag());
#endif
			 });

      // Inverse FFT to real space
      Cfft_m->transform(ippl::BACKWARD, cfield_m);

      const double scale = Kokkos::pow(initializer::GlobalStuff::instance().ngrid, 1.5); 
      ippl::parallel_for("scale_rho_inv", ippl::getRangePolicy(cfview, ngh),
			 KOKKOS_LAMBDA(const index_array_type& idx){
			   const Kokkos::complex<double> tmp = ippl::apply(cfview, idx);
                           ippl::apply(cfview, idx) = Kokkos::complex<double>(tmp.real()/scale, tmp.imag()/scale);
#ifdef KOKKOS_PRINT
                           Kokkos::printf("rhoinvfft: %i %g %g \n", dim, tmp.real()/scale, tmp.imag()/scale);
#endif
			 });

      ippl::parallel_for("set_particle", ippl::getRangePolicy(cfview, ngh),
                         KOKKOS_LAMBDA(const index_array_type& idx){
			   const unsigned int i = idx[0] - ngh + lDom[0].first();
			   const unsigned int j = idx[1] - ngh + lDom[1].first();
			   const unsigned int k = idx[2] - ngh + lDom[2].first();

			   const double       x = ippl::apply(cfview, idx).real();
			   const unsigned int d = (dim == 0) ? i : (dim == 1) ? j : k;

			   const unsigned int il = idx[0] - ngh;
			   const unsigned int jl = idx[1] - ngh;
			   const unsigned int kl = idx[2] - ngh;
			   const index_type    n = il + lDom[0].length()*(jl + kl*lDom[1].length());

			   rView(n)[dim] = ((d + 0.5) - d_z*x);
			   vView(n)[dim] = -ddot*x;
			   
			   //Kokkos::printf("set_particle: rank= %i \t dim= %i \t d=%i \t n=%i  \n", myrank, dim, d, n);
			   
#ifdef KOKKOS_PRINT
			   if (n==0)
			     Kokkos::printf("zeldo: dim, pos0, x, v, re: d_z= %g ddot= %g \n", d_z, ddot);
			   Kokkos::printf("zeldo: %i %g %g %g %g \n", dim, d+0.5,-d_z*x, -ddot*x, x);
#endif											 
			 });

      cfield_m = tmpcfield;
    }
  }

  /**
     * @brief Create particles using Zarijas initializer method
     */

  void createParticles() {
    
    Inform m2a("createParticles ",INFORM_ALL_NODES);
    Inform msg("createParticles ");

    static IpplTimings::TimerRef creaIpplParts = IpplTimings::getTimer("creaIpplParts");
    IpplTimings::startTimer(creaIpplParts);
    size_type nloc = this->totalP_m / ippl::Comm->size();
    std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
    pc->create(nloc);
    pc->m = this->M_m / this->totalP_m;
    IpplTimings::stopTimer(creaIpplParts);
    
    msg << "pc->create(nloc) done" << endl;
    
    /** 
	construct Pk_m 
     */
    static IpplTimings::TimerRef iniPwrSpec = IpplTimings::getTimer("initPwrSpec");
    IpplTimings::startTimer(iniPwrSpec);
    initPwrSpec();
    IpplTimings::stopTimer(iniPwrSpec);

    /**
       the following code can be found
       as standalone test in test/particles/zeldo-test-mp1.cpp
     */

    static IpplTimings::TimerRef linZeldo = IpplTimings::getTimer("LinearZeldoInitMP");
    IpplTimings::startTimer(linZeldo);
    LinearZeldoInitMP();
    IpplTimings::stopTimer(linZeldo);
    
    // Load Balancer Initialisation
    /*
    auto* mesh = &this->fcontainer_m->getMesh();
    auto* FL   = &this->fcontainer_m->getFL();
    if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
      this->isFirstRepartition_m = true;
      this->loadbalancer_m->initializeORB(FL, mesh);
      this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
    }

    pc->update();
    m2a << "local number of galaxies after initializer " << pc->getLocalNum() << endl;
    this->dumpParticles();
    */
}


  

  
    /**
     * @brief Read particle data from a file.
     */
    void readParticles() {
        Inform msg("Reading Particles");

        size_type nloc = this->totalP_m / ippl::Comm->size();
        msg << "Local number of particles: " << nloc << endl;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        pc->create(nloc);
        pc->m = this->M_m / this->totalP_m;

        this->fcontainer_m->getRho() = 0.0;

        // Load Balancer Initialisation
        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            this->isFirstRepartition_m = true;
            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
        }

        static IpplTimings::TimerRef ReadingTimer = IpplTimings::getTimer("readData");
        IpplTimings::startTimer(ReadingTimer);

        std::ifstream file(this->folder + "Data.csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening IC file!" << std::endl;
        }

        // Vector to store data read from the CSV file
        std::vector<std::vector<double>> ParticlePositions;
        std::vector<std::vector<double>> ParticleVelocities;
        double MaxPos;
        double MinPos;

        // Read the file line by line
        std::string line;
        int i = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            if (i % ippl::Comm->size() == ippl::Comm->rank()) {
                // Read each comma-separated value into the row vector
                std::string cell;
                int j = 0;
                std::vector<double> PosRow;
                std::vector<double> VelRow;
                while (j < 6 && std::getline(ss, cell, ',')) {
                    if (j < 3) {
                        double Pos = std::stod(cell);
                        PosRow.push_back(Pos);
                        ++j;
                        // Find Boundaries (x, y, z)
                        if (i + j > ippl::Comm->rank()) {
                            MaxPos = std::max(Pos, MaxPos);
                            MinPos = std::min(Pos, MinPos);
                        } else {  // very first input
                            MaxPos = Pos;
                            MinPos = Pos;
                        }
                    } else {
                        double Vel = std::stod(cell);
                        VelRow.push_back(Vel);
                        ++j;
                    }
                }
                ParticlePositions.push_back(PosRow);
                ParticleVelocities.push_back(VelRow);
            }
            ++i;
        }

        // Boundaries of Particle Positions
        msg << "Minimum Position: " << MinPos << endl;
        msg << "Maximum Position: " << MaxPos << endl;
        msg << "Defined maximum:  " << this->rmax_m << endl;

        // Number of Particles
        if (nloc != ParticlePositions.size()) {
            std::cerr << "Error: Simulation number of particles does not match input!" << std::endl;
            std::cerr << "Input N = " << ParticlePositions.size() << ", Local N = " << nloc
                      << std::endl;
        } else
            // Particle positions and velocities, which are read in above from the initial
            // conditions file, are assigned to the particle attributes R and V in the particle
            // container.
            msg << "successfully done." << endl;

        auto R_host = pc->R.getHostMirror();
        auto V_host = pc->V.getHostMirror();

        double a = this->a_m;
        for (unsigned int i = 0; i < nloc; ++i) {
            R_host(i)[0] = ParticlePositions[i][0];
            R_host(i)[1] = ParticlePositions[i][1];
            R_host(i)[2] = ParticlePositions[i][2];
            V_host(i)[0] = ParticleVelocities[i][0] * pow(a, 1.5);
            V_host(i)[1] = ParticleVelocities[i][1] * pow(a, 1.5);
            V_host(i)[2] = ParticleVelocities[i][2] * pow(a, 1.5);
        }

        Kokkos::fence();
        ippl::Comm->barrier();
        Kokkos::deep_copy(pc->R.getView(), R_host);
        Kokkos::deep_copy(pc->V.getView(), V_host);
        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(ReadingTimer);

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        bool isFirstRepartition              = false;
        std::shared_ptr<FieldContainer_t> fc = this->fcontainer_m;
        if (this->loadbalancer_m->balance(this->totalP_m, this->it_m)) {
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            printf("first repartition works \n");
        }

        msg << "Assignment of positions and velocities done." << endl;
    }

    /**
     * @brief Read particle data from a file and assign to the domain.
     */
    void readParticlesDomain() {
        Inform msg("Reading Particles");

        this->fcontainer_m->getRho() = 0.0;

        // Load Balancer Initialisation
        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            this->isFirstRepartition_m = true;
            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
        }

        static IpplTimings::TimerRef ReadingTimer = IpplTimings::getTimer("readData");
        IpplTimings::startTimer(ReadingTimer);

        // Check if the file is opened successfully
        std::ifstream file(this->folder + "Data.csv");
        if (!file.is_open()) {
            std::cerr << "Error opening IC file!" << std::endl;
        }

        // Vector to store data read from the CSV file
        std::vector<std::vector<double>> ParticlePositions;
        std::vector<std::vector<double>> ParticleVelocities;

        // Boundaries of Particle Positions
        const ippl::NDIndex<Dim>& ldom =
            FL->getLocalNDIndex();  // local processor domain coordinates
        Vector_t<double, Dim> Min;
        Vector_t<double, Dim> Max;
        for (unsigned int i = 0; i < Dim; ++i) {
            Min[i] = this->rmax_m[i] * ldom[i].first() / this->nr_m[i];
            Max[i] = this->rmax_m[i] * (ldom[i].last() + 1) / this->nr_m[i];
        }

        // Read the file line by line
        std::string line;
        while (std::getline(file, line)) {
            // New Line has begun
            std::stringstream ss(line);
            std::string cell;
            int j = 0;  // column number
            std::vector<double> PosRow;
            std::vector<double> VelRow;
            bool inDomain = true;
            while (inDomain == true && j < 6 && std::getline(ss, cell, ',')) {
                if (j < 3) {
                    double Pos = std::stod(cell);
                    // Special case where particle lies on the edge
                    // To prevent instability in the sending process when a particle is exactly at
                    // the boundary, a small perturbation (0.01%) is applied to the particle
                    // positions. This avoids double-counting of particles and ensures the total
                    // number of particles is conserved.
                    if (Pos == Max[j]) {
                        msg << "Particle was on edge. Shift position from " << Pos << " to "
                            << Pos * 0.9999 << endl;
                        Pos = 0.9999 * Pos;
                    }
                    if (Pos == 0) {
                        msg << "Particle was on edge. Shift position from " << Pos << " to "
                            << 0.0001 * Max[j] << endl;
                        Pos = 0.0001 * Max[j];
                    }
                    if (Pos > Min[j] && Pos <= Max[j])
                        PosRow.push_back(Pos);  // particle is actually in domain -> add
                    else
                        inDomain = false;  // particle is not in  domain -> leave while loop
                    ++j;
                } else {
                    double Vel = std::stod(cell);
                    VelRow.push_back(Vel);
                    ++j;
                }
            }
            if (inDomain == true) {
                ParticlePositions.push_back(PosRow);
                ParticleVelocities.push_back(VelRow);
            }
        }

        // Create Particle container
        size_type nloc = ParticlePositions.size();
        std::cout << "rank: " << ippl::Comm->rank() << " Local number of particles: " << nloc
                  << std::endl;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        pc->create(nloc);
        pc->m = this->M_m / this->totalP_m;

        auto R_host = pc->R.getHostMirror();
        auto V_host = pc->V.getHostMirror();
        double a    = this->a_m;
        for (unsigned int i = 0; i < nloc; ++i) {
            R_host(i)[0] = ParticlePositions[i][0];
            R_host(i)[1] = ParticlePositions[i][1];
            R_host(i)[2] = ParticlePositions[i][2];
            V_host(i)[0] = ParticleVelocities[i][0] * pow(a, 1.5);
            V_host(i)[1] = ParticleVelocities[i][1] * pow(a, 1.5);
            V_host(i)[2] = ParticleVelocities[i][2] * pow(a, 1.5);
        }

        Kokkos::fence();
        ippl::Comm->barrier();
        Kokkos::deep_copy(pc->R.getView(), R_host);
        Kokkos::deep_copy(pc->V.getView(), V_host);
        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(ReadingTimer);

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        bool isFirstRepartition              = false;
        std::shared_ptr<FieldContainer_t> fc = this->fcontainer_m;
        if (this->loadbalancer_m->balance(this->totalP_m, this->it_m)) {
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            printf("first repartition works \n");
        }

        msg << "Assignment of positions and velocities done." << endl;
    }

    /**
     * @brief Advance the simulation by one time step.
     */
    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        } else {
            throw IpplException("StructureFormation ", "Step method is not set/recognized!");
        }
    }

    /**
     * @brief Perform a single LeapFrog step in the simulation.
     */
    void LeapFrogStep() {
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        static IpplTimings::TimerRef VTimer              = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");

        // Time step size is calculated according to Blanca's thesis:
        // "For the cosmological simulations, it was decided to adjust the timestep to the expansion
        // of the universe. Instead of using a fixed ∆t, a fixed ∆ log a was implemented."
        double a      = this->a_m;
        double a_i    = this->a_m;
        double a_half = a * exp(0.5 * this->Dloga);
        double a_f    = a * exp(this->Dloga);

        double H_i    = this->calculateHubble(a_i);
        double H_half = this->calculateHubble(a_half);
        double H_f    = this->calculateHubble(a_f);
        double d_drift, d_kick;

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;
        // kick (update V)
        IpplTimings::startTimer(VTimer);
        d_kick = 1. / 4 * (1 / (H_i * a_i) + 1 / (H_half * a_half)) * this->Dloga;
        pc->V  = pc->V - 4 * this->G * M_PI * pc->F * d_kick;
        IpplTimings::stopTimer(VTimer);

        // drift (update R) in comoving distances
        IpplTimings::startTimer(RTimer);
        d_drift = 1. / 6
                  * (1 / (H_i * a_i * a_i) + 4 / (H_half * a_half * a_half) + 1 / (H_f * a_f * a_f))
                  * this->Dloga;
        pc->R = pc->R + pc->V * d_drift;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
            IpplTimings::startTimer(domainDecomposition);
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        // scatter the mass onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather F field
        this->grid2par();

        // kick (update V)
        IpplTimings::startTimer(VTimer);
        d_kick = 1. / 4 * (1 / (H_half * a_half) + 1 / (H_f * a_f)) * this->Dloga;
        pc->V  = pc->V - 4 * this->G * M_PI * pc->F * d_kick;
        IpplTimings::stopTimer(VTimer);
    }

    /**
     * @brief Save the positions of particles to a file.
     *
     * @param index Current time step number
     */
    void savePositions(unsigned int index) {
        Inform msg("Saving Particles");

        static IpplTimings::TimerRef SavingTimer = IpplTimings::getTimer("Save Data");
        IpplTimings::startTimer(SavingTimer);

        msg << "snapshot " << this->it_m << endl;

        std::stringstream ss;
        if (ippl::Comm->size() == 1)
            ss << "snapshot_" << std::setfill('0') << std::setw(3) << index;
        else
            ss << "snapshot_" << ippl::Comm->rank() << "_" << std::setfill('0') << std::setw(3)
               << index;
        std::string filename = ss.str();

        std::ofstream file(this->folder + filename + ".csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening saving file!" << std::endl;
            return;
        }
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        auto Rview = this->pcontainer_m->R.getView();
        auto Vview = this->pcontainer_m->V.getView();
        auto Fview = this->pcontainer_m->F.getView();

        auto R_host = this->pcontainer_m->R.getHostMirror();
        auto V_host = this->pcontainer_m->V.getHostMirror();
        auto F_host = this->pcontainer_m->F.getHostMirror();

        Kokkos::deep_copy(R_host, Rview);
        Kokkos::deep_copy(V_host, Vview);
        Kokkos::deep_copy(F_host, Fview);

        double a = this->a_m;

        // Write data to the file
        for (unsigned int i = 0; i < pc->getLocalNum(); ++i) {
            for (unsigned int d = 0; d < Dim; ++d)
                file << R_host(i)[d] << ",";
            for (unsigned int d = 0; d < Dim; ++d)
                file << V_host(i)[d] << ",";
            for (unsigned int d = 0; d < Dim; ++d)
                file << -4 * M_PI * this->G / (a * a) * F_host(i)[d] << ",";
            file << "\n";
        }
        ippl::Comm->barrier();

        // Close the file stream
        file.close();
        msg << "done." << endl;
        IpplTimings::stopTimer(SavingTimer);
    }

    /**
     * @brief Dump the current state of the simulation.
     */
    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);
        dumpStructure(this->fcontainer_m->getF().getView());
        IpplTimings::stopTimer(dumpDataTimer);
    }

    /**
     * @brief Analyzes and logs the structure of the given field view.
     *
     * This method calculates and logs the energy and maximum norm of the field values
     * in the given view. It performs parallel reduction to compute the sum of squares
     * and maximum norm of the field values, and then reduces these values across all
     * processes. The results are written to a CSV file by the root process.
     *
     * @tparam View The type of the view to be dumped.
     * @param Fview The view whose structure is to be dumped.
     */
    template <typename View>
    void dumpStructure(const View& Fview) {
        const int nghostF = this->fcontainer_m->getF().getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Fview, nghostF),
            KOKKOS_LAMBDA(const index_array_type& args, double& F2, double& FNorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Fview, args)[0];
                double f2  = Kokkos::pow(val, 2);
                F2 += f2;

                double norm = Kokkos::fabs(ippl::apply(Fview, args)[0]);
                if (norm > FNorm) {
                    FNorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(),
                        globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << this->folder + "FieldStructure_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
