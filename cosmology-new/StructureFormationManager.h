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
private:

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
    cosmo_m.SetParameters(initializer::GlobalStuff::instance(), tfname_.c_str()); 
  }

    /**
     * @brief Destructor for StructureFormationManager.
     */
    ~StructureFormationManager() {}

#ifdef IPPL_ENABLE_TESTS
      auto getCView() -> decltype(cfield_m.getView()) { return cfield_m.getView(); }
      
      int getGhostCells() const { return cfield_m.getNghost(); } 
      
      const ippl::FieldLayout<Dim>& getLayout() const { return this->fcontainer_m->getFL(); }
      
      const ippl::Vector<int, Dim>& getNr() const noexcept {
        return this->nr_m;
      }
      
#endif

  struct MinMaxSum {
    double min;
    double max;
    double sum;
    
    KOKKOS_FUNCTION
    MinMaxSum() : min(1e20), max(-1e20), sum(0.0) {}
    
    KOKKOS_FUNCTION
    MinMaxSum& operator+=(const MinMaxSum& other) {
      min = Kokkos::min(min, other.min);
      max = Kokkos::max(max, other.max);
      sum += other.sum;
      return *this;
    }

// Only compile the global_reduce method on the host.
#ifndef __HIP_DEVICE_COMPILE__    
    void global_reduce() {
      double global_min, global_max, global_sum;

      MPI_Allreduce(&min,  &global_min,  1, MPI_DOUBLE, MPI_MIN, ippl::Comm->getCommunicator());
      MPI_Allreduce(&max,  &global_max,  1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
      MPI_Allreduce(&sum,  &global_sum,  1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());

      min = global_min;
      max = global_max;
      sum = global_sum;
    }
#else
    void global_reduce() { }
#endif
  };
  
    /**
     * @brief Pre-run setup for the simulation.
     */
    void pre_run() override {
        Inform mes("Pre Run");

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
        mes << "total mass: " << this->M_m << endl;
        mes << "mass of a single particle " << this->M_m / this->totalP_m << endl;

        this->hr_m     = this->rmax_m / this->nr_m;
        this->origin_m = this->rmin_m;
        this->it_m     = 0;

        mes << "Discretization:" << endl
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
	  readParticlesDomain();  // defines particle positions, velocities

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
	}
	else {
	  ippl::ParameterList fftParams;
	  fftParams.add("use_heffte_defaults", true);
	  Cfft_m = std::make_unique<CFFT_type>(this->fcontainer_m->getFL(), fftParams);
	  cfield_m.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());
	  Pk_m.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());
	  createParticles();
	}
        mes << "Done";
    }


  /**
     * @brief 
     */
  
  void LinearZeldoInitMP() {
    // After creating the field layout (cfield_m) and determining global grid sizes Nx, Ny, Nz:
    Inform msg("LinearZeldoInitMP ");
    
    typename CField_t::view_type& view = cfield_m.getView();
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
    const double Lx = this->rmax_m[0];
    const double Ly = this->rmax_m[1];
    const double Lz = this->rmax_m[2];

    static IpplTimings::TimerRef fourDenTimer = IpplTimings::getTimer("Fourier Density");
    IpplTimings::startTimer(fourDenTimer);
    // Initialize the Fourier density field with Gaussian random modes (Hermitian symmetric)
    ippl::parallel_for("InitDeltaField", ippl::getRangePolicy(view, ngh),
		       KOKKOS_LAMBDA(const index_array_type& idx) {
			 const double pi = Kokkos::numbers::pi_v<double>;
			 // Compute global coordinates (i,j,k) for this local index
			 int i = idx[0] - ngh + lDom[0].first();
			 int j = idx[1] - ngh + lDom[1].first();
			 int k = idx[2] - ngh + lDom[2].first();

			 // DC mode (k=0 vector) set to 0 (no DC offset)
			 if (i == 0 && j == 0 && k == 0) {
			   ippl::apply(view, idx) = Kokkos::complex<double>(0.0, 0.0);
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
			   // Choose the "key" coordinates (the global smaller of the pair) for random generation
			   int key_i = is_conjugate ? i_neg : i;
			   int key_j = is_conjugate ? j_neg : j;
			   int key_k = is_conjugate ? k_neg : k;

			   // Deterministically generate two uniform [0,1) random numbers from the key
			   uint64_t key_index = ((uint64_t)key_i * Ny + key_j) * Nz + key_k;
			   uint64_t x = key_index ^ global_seed;
			   if (x == 0ull) x = 1ull;  // avoid zero state
			   // XorShift64 steps to produce pseudorandom 64-bit values
			   x ^= x << 13;  x ^= x >> 7;  x ^= x << 17;
			   uint64_t r1 = x;
			   x ^= x << 13;  x ^= x >> 7;  x ^= x << 17;
			   uint64_t r2 = x;
			   // Map the 53 most significant bits of r1,r2 to double in [0,1)
			   const double norm = 1.0 / 9007199254740992.0;  // 1/2^53
			   double u1 = (double)(r1 >> 11) * norm;
			   double u2 = (double)(r2 >> 11) * norm;
			   
			   // Convert uniforms to Gaussian via Box-Muller
			   double R     = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
			   double theta = 2.0 * pi * u2;
			   double gauss_re = R * Kokkos::cos(theta);   // Gaussian(0,1) for real part
			   double gauss_im = R * Kokkos::sin(theta);   // Gaussian(0,1) for imaginary part
			   double Pk = 1;
                           // double Pk = ippl::apply(pkview, idx);       // power spectrum P(k) 
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
			   // Assign the complex value to this local mode
			   ippl::apply(view, idx) = Kokkos::complex<double>(val_re, val_im);
                           }
		       });

    IpplTimings::stopTimer(fourDenTimer);

    // Check whether the generated density field is Hermitian before proceeding
    static IpplTimings::TimerRef hermiticityTimer = IpplTimings::getTimer("Hermiticity Timer");     
    IpplTimings::startTimer(hermiticityTimer);

    if (isHermitian()) {
        msg << "Fourier density field is Hermitian." << endl;
    } else {
        std::cerr << "Fourier density field is NOT Hermitian!" << std::endl;
    }

    IpplTimings::stopTimer(hermiticityTimer);

    static IpplTimings::TimerRef fourDisplTimer = IpplTimings::getTimer("Fourier Displacement");
    // Store delta(k) for reuse 
    auto tmpcfield = cfield_m; 
    typename CField_t::view_type& viewtmpcfield = tmpcfield.getView();
    /*
      Now we can delete Pk and allocate the particles

    */

    const unsigned int nx = lDom[0].length();
    const unsigned int ny = lDom[1].length();
    const Vector_t<double,3> hr = this->hr_m;
    // 2–4. Loop over displacement components x(0), y(1), z(2)
    for (int dim = 0; dim < 3; ++dim) {
      IpplTimings::startTimer(fourDisplTimer);
      // Compute displacement component in k-space
      ippl::parallel_for("ComputeDisplacementComponentK", ippl::getRangePolicy(view, ngh),
			 KOKKOS_LAMBDA(const index_array_type& idx) {
			   const double pi = Kokkos::numbers::pi_v<double>;
			   int i = idx[0] - ngh + lDom[0].first();
			   int j = idx[1] - ngh + lDom[1].first();
			   int k = idx[2] - ngh + lDom[2].first();
			   
			   int kx_i = (i <= Nx / 2) ? i : i - Nx;
			   int ky_i = (j <= Ny / 2) ? j : j - Ny;
			   int kz_i = (k <= Nz / 2) ? k : k - Nz;
			   
			   double kx = 2.0 * pi * kx_i / Lx;
			   double ky = 2.0 * pi * ky_i / Ly;
			   double kz = 2.0 * pi * kz_i / Lz;
			   double k2 = kx * kx + ky * ky + kz * kz;
			   
			   Kokkos::complex<double> delta = ippl::apply(viewtmpcfield, idx);
			   Kokkos::complex<double> I(0.0, 1.0);
			   double k_comp = (dim == 0) ? kx : (dim == 1) ? ky : kz;
			   Kokkos::complex<double> result = (k2 == 0.0) ? Kokkos::complex<double>(0.0, 0.0)
			     : I * (k_comp / k2) * delta;
			   ippl::apply(view, idx) = result;
			 });
	// Inverse FFT to real space
	Cfft_m->transform(ippl::BACKWARD, cfield_m);
	IpplTimings::stopTimer(fourDisplTimer);

	static IpplTimings::TimerRef posvelInitTimer = IpplTimings::getTimer("Position/Velocity init");
	IpplTimings::startTimer(posvelInitTimer);

        // segfault fix : only loop over the cells you own
        index_type n_local = static_cast<index_type>( rView.extent(0) );

	Kokkos::parallel_for("ComputeWorldCoordinates",n_local,
			     KOKKOS_LAMBDA(const index_type n) {
			       // Convert 1D index n back to 3D indices (i, j, k)
			       const unsigned int i = n % nx;
			       const unsigned int j = (n / nx) % ny;
			       const unsigned int k = n / (nx * ny);
			       double disp = view(i, j, k).real();
			       unsigned int idx = (dim == 0) ? i : (dim == 1) ? j : k;
			       rView(n)[dim] = ((idx + 0.5) * hr[dim]) + disp;
			       vView(n)[dim] = disp;
			     });
	
	IpplTimings::stopTimer(posvelInitTimer);
    }
  }

  /**
       * @brief Check whether the complex density field delta(k) is Hermitian
       *        Compatible with multiple CPU Ranks
       *
       * A real‑space density field requires its Fourier coefficients
       * to be Hermitian, satisfying
       * \f[
       *     \delta(-\mathbf k) = \delta^*(\mathbf k) ,
       * \f]
       * where the asterisk denotes complex conjugation.
       *
       * This function loops through the indices of the complex field accessed as
       * a Kokkos view holding the complex Fourier amplitudes and returns false
       * in the case that any of the fourier modes are not Hermitian.
       *
       * @return true if the complex density field is Hermitian, false otherwise
       */
  bool isHermitian() const {
    Inform msg("isHermitian ");

    const auto& field = cfield_m.getView();
    const int Nx = this->nr_m[0], Ny = this->nr_m[1], Nz = this->nr_m[2];
    const int ngh = cfield_m.getNghost();
    const auto& layout = this->fcontainer_m->getFL();
    const ippl::NDIndex<Dim>& lDom = layout.getLocalNDIndex();
  
    const double tol = std::numeric_limits<double>::epsilon();
    const int nranks = ippl::Comm->size();
    const int myrank = ippl::Comm->rank();
  
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = typename ExecSpace::memory_space;

    
    // device view of the global index layout
    const auto& global_domains = layout.getDeviceLocalDomains();   
  
    Kokkos::View<int*, MemSpace> sendCount("sendCount", nranks);
    Kokkos::deep_copy(sendCount, 0); // zero-init in device mem
    
    // local Hermitian flag - shared across both device reductions
    int localHermitianFlag = 1;

    //Walk over local k‑space and do 2 things:
    //    (i) test Hermiticity when -k is on the same rank
    //    (ii) add count to sendCount -k lives elsewhere.
    using MDPolicy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
    MDPolicy mdp({lDom[0].first(), lDom[1].first(), lDom[2].first()},
                 {lDom[0].last()+1, lDom[1].last()+1, lDom[2].last()+1});

    Kokkos::parallel_reduce("isHermitian_countSends", mdp,
      KOKKOS_LAMBDA(const int i, const int j, const int k, int& isHermitianFlag)
      {

        if (i==0 && j==0 && k==0) return; // skip k = (0,0,0)


        // compute -k in global coordinates
        const int i_neg = (i==0 ? 0 : Nx-i);
        const int j_neg = (j==0 ? 0 : Ny-j);
        const int k_neg = (k==0 ? 0 : Nz-k);

        // figure out which rank owns -k
        int neg_k_owner = -1;
        for (int rank = 0; rank < nranks; rank++) {
          const auto& dom = global_domains(rank);   // NDIndex on the device
          if (dom[0].first() <= i_neg && i_neg <= dom[0].last() &&
              dom[1].first() <= j_neg && j_neg <= dom[1].last() &&
              dom[2].first() <= k_neg && k_neg <= dom[2].last())
            neg_k_owner = rank;
        }


        if (neg_k_owner == myrank) {
          // neg_k is local - check hermiticity directly
         
          // local k indices 
          const int li = i - lDom[0].first() + ngh;
          const int lj = j - lDom[1].first() + ngh;
          const int lk = k - lDom[2].first() + ngh;

          // local neg_k indices
          const int lni = i_neg - lDom[0].first() + ngh;
          const int lnj = j_neg - lDom[1].first() + ngh;
          const int lnk = k_neg - lDom[2].first() + ngh;

          Kokkos::complex<double> delta_k = field(li, lj, lk);
          Kokkos::complex<double> delta_neg_k = field(lni, lnj, lnk);
          auto delta_ck = Kokkos::conj(delta_k);

          if (Kokkos::abs(delta_neg_k.real()-delta_ck.real()) > tol ||
              Kokkos::abs(delta_neg_k.imag()-delta_ck.imag()) > tol) {
            isHermitianFlag = 0;
          }
        } else if (neg_k_owner >= 0) {
          // -k belongs to another rank - make space in sendcount
          Kokkos::atomic_fetch_add(&sendCount(neg_k_owner), 1);
        } else {
          // Domain decomposition bug
          if (myrank == 0) printf("Hermiticity check: no found owner rank for neg_k\n");
          isHermitianFlag = 0;
        }
      },
    Kokkos::Min<int>(localHermitianFlag));
    
    Kokkos::fence(); // make sure sendCount is ready
      
    // if only 1 rank, avoid creating send buffers
    // directly Return the hermiticity result
    if (nranks == 1){
      // global reduction of hermiticity value
      int globalResult = 1;
      MPI_Allreduce(&localHermitianFlag, &globalResult, 1, MPI_INT, MPI_MIN,
                ippl::Comm->getCommunicator());

      return globalResult != 0;
    }
    
    // Build per‑rank send counts on HOST – device to host copy
    Kokkos::View<int*, Kokkos::HostSpace> sendCount_h("sendCount_h", nranks);
    Kokkos::deep_copy(sendCount_h, sendCount);

                        
    // Prefix sums to get displacements (still on host)
    std::vector<size_t> send_disp(nranks,0);
    for (int rank = 1; rank < nranks; rank++) {
      send_disp[rank] = send_disp[rank-1] + static_cast<size_t>(sendCount_h[rank-1]);
    }
    
    const size_t total_sends = send_disp.back() + sendCount_h(nranks - 1);

    // Allocate send and receive buffer on the GPU (total_sends=total_recvs)
    Kokkos::View<HermitianPkg*, MemSpace> send_buffer_d("send_buffer_d", total_sends);
    Kokkos::View<HermitianPkg*, MemSpace> recv_buffer_d("recv_buffer_d", total_sends);
    
    // Device copy of the displacements per destination ranks
    Kokkos::View<size_t*,Kokkos::HostSpace> send_disp_h("send_disp_h", nranks);
    for(int r=0;r<nranks;++r){
      send_disp_h(r)=send_disp[r];
    }
    Kokkos::View<size_t*, MemSpace> send_disp_d("send_disp_d", nranks);
    Kokkos::deep_copy(send_disp_d, send_disp_h);
    
    // per-dest ‘how many already packed’ counters
    Kokkos::deep_copy(sendCount, 0);   // resets device view
    
    // pack each +k message into its unique slot = base+local
    Kokkos::parallel_for("isHermitian_pack_send_buffer", mdp,
      KOKKOS_LAMBDA(const int i, const int j, const int k)
      {
        if (i==0 && j==0 && k==0) return;

        // ----- same local computations as first kernel -----
        const int li = i - lDom[0].first() + ngh;
        const int lj = j - lDom[1].first() + ngh;
        const int lk = k - lDom[2].first() + ngh;
        Kokkos::complex<double> delta_k = field(li, lj, lk);

        const int i_neg = (i==0 ? 0 : Nx-i);
        const int j_neg = (j==0 ? 0 : Ny-j);
        const int k_neg = (k==0 ? 0 : Nz-k);

        int neg_k_owner = -1;
        for (int rank = 0; rank < nranks; ++rank) {
          const auto& dom = global_domains(rank);
          if (dom[0].first() <= i_neg && i_neg <= dom[0].last() &&
              dom[1].first() <= j_neg && j_neg <= dom[1].last() &&
              dom[2].first() <= k_neg && k_neg <= dom[2].last())
            neg_k_owner = rank;
        }
        
        if (neg_k_owner != myrank) {
            if (neg_k_owner >= 0) {
                // unique slot: bucket offset + per-bucket atomic increment
                const size_t base  = send_disp_d(neg_k_owner);
                const int    local = Kokkos::atomic_fetch_add(&sendCount(neg_k_owner), 1);
                const size_t slot  = base + static_cast<size_t>(local);

                send_buffer_d(slot).kx = i;
                send_buffer_d(slot).ky = j;
                send_buffer_d(slot).kz = k;
                send_buffer_d(slot).re = delta_k.real();
                send_buffer_d(slot).im = delta_k.imag();
            } else {
                if (myrank == 0) printf("Hermiticity check error: no found owner rank for neg_k\n");
            }
        }
    });
    Kokkos::fence();    // ensure send_buffer_d is filled
       
    // Communication
    std::vector<MPI_Request> mpi_requests;

    // Post all receives directly into device memory
    for (int rank = 0; rank < nranks; rank++) {
      if (sendCount_h[rank] == 0) continue;
      mpi_requests.emplace_back();
      void* recv_ptr = static_cast<void*>(recv_buffer_d.data() + send_disp[rank]);
      MPI_Irecv(recv_ptr, sendCount_h[rank] * sizeof(HermitianPkg), MPI_BYTE, rank, 
                0, ippl::Comm->getCommunicator(), &mpi_requests.back());
    }

    // Send out packages directly from device memory
    for (int rank = 0; rank < nranks; rank++) {
      if (sendCount_h[rank] == 0) continue;
      mpi_requests.emplace_back();
      const void* send_ptr = static_cast<void*>(send_buffer_d.data() + send_disp[rank]);
      MPI_Isend(send_ptr, sendCount_h[rank] * sizeof(HermitianPkg), MPI_BYTE, rank,
                0, ippl::Comm->getCommunicator(), &mpi_requests.back());
    }


    MPI_Waitall(static_cast<int>(mpi_requests.size()),
                mpi_requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence(); // ensure GPU sees new data

    // Perform final hermiticity check on remaining values
    Kokkos::parallel_reduce("finalHermiticityCheck",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(total_sends)),
      KOKKOS_LAMBDA(const int idx, int& isHermitianFlag)
      {
        // unpack the package
        const HermitianPkg p = recv_buffer_d(idx);

        // compute k coordinates (this point lies on current rank)
        const int i = (p.kx==0 ? 0 : Nx - p.kx);
        const int j = (p.ky==0 ? 0 : Ny - p.ky);
        const int k = (p.kz==0 ? 0 : Nz - p.kz);
        
        // convert to local coordinates
        const int li = i - lDom[0].first() + ngh;
        const int lj = j - lDom[1].first() + ngh;
        const int lk = k - lDom[2].first() + ngh;

        Kokkos::complex<double> delta_k = field(li, lj, lk);
        Kokkos::complex<double> delta_neg_k     = { p.re, p.im };
        auto delta_ck = Kokkos::conj(delta_k);

        if (Kokkos::abs(delta_neg_k.real() - delta_ck.real()) > tol ||
            Kokkos::abs(delta_neg_k.imag() - delta_ck.imag()) > tol) {
          isHermitianFlag = 0;
        }
      },
      Kokkos::Min<int>(localHermitianFlag));
    Kokkos::fence();
    
    // global reduction of hermiticity value
    int globalResult = 1;
    MPI_Allreduce(&localHermitianFlag, &globalResult, 1, MPI_INT, MPI_MIN,
                  ippl::Comm->getCommunicator());

    return globalResult != 0;
  }


  /**
     * @brief Create particles using Zarijas initializer
     */

  void createParticles() {
    
    Inform m2a("createParticles ",INFORM_ALL_NODES);
    Inform msg("createParticles ");

    size_type nloc = this->totalP_m / ippl::Comm->size();
    std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
    pc->create(nloc);
    pc->m = this->M_m / this->totalP_m;

    /** 
	Pk_m is constructed
     */
    initPwrSpec();

    /**
       the following code can be found
       as standalone test in test/particles/zeldo-test-mp1.cpp
     */
    LinearZeldoInitMP();

    // Load Balancer Initialisation
    auto* mesh = &this->fcontainer_m->getMesh();
    auto* FL   = &this->fcontainer_m->getFL();
    if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
      this->isFirstRepartition_m = true;
      this->loadbalancer_m->initializeORB(FL, mesh);
      this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
    }

    pc->update();
    m2a << "local number of galaxies after initializer " << pc->getLocalNum() << endl;


    
    /*  this was the old way of ding it i.e. mc4
    indens();

    test_reality();

    gravity_potential();

    set_particles();
    */  


}

  /**
     hu_sugiyama hardcoded TFFLAG==2

   */


  KOKKOS_FUNCTION static double hu_sugiyama(double k, double kh_tmp){
    
    if (k == 0.0)
      return(0.0);
    
    const double cobe_temp=2.728;  // COBE/FIRAS CMB temperature in K                                                                                                                                          
    const double tt=cobe_temp/2.7 * cobe_temp/2.7;
   
    double qkh = k*tt/kh_tmp;

    /* NOTE: the following line has 0/0 for k=0.                                                                                                                                                              
       This was taken care of at the beginning of the routine. */
  
    return Kokkos::log(1.0+2.34*qkh)/(2.34*qkh) * Kokkos::pow(1.0+3.89*qkh+Kokkos::pow(16.1*qkh, 2.0)+pow(5.46*qkh, 3.0)+Kokkos::pow(6.71*qkh, 4.0), -0.25);
  }

  KOKKOS_FUNCTION static double peacock_dodds(double k, double kh_tmp){
   
   if (k == 0.0)
     return(0.0);

   double qkh = k/kh_tmp;
   /* NOTE: the following line has 0/0 for k=0.
      This was taken care of at the beginning of the routine. */
   return Kokkos::log(1.0+2.34*qkh)/(2.34*qkh) * Kokkos::pow(1.0+3.89*qkh+Kokkos::pow(16.1*qkh,2.0)+Kokkos::pow(5.46*qkh,3.0)+Kokkos::pow(6.71*qkh,4.0), -0.25);
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

    // TFFlag == 2) Hu-Sugiyama transfer function
    const double Omega_m = initializer::GlobalStuff::instance().Omega_m;
    const double Omega_bar = initializer::GlobalStuff::instance().Omega_bar;
    const double h = initializer::GlobalStuff::instance().Hubble;
    const double akh1=pow(46.9*Omega_m*h*h, 0.670)*(1.0+pow(32.1*Omega_m*h*h, -0.532));
    const double akh2=pow(12.0*Omega_m*h*h, 0.424)*(1.0+pow(45.0*Omega_m*h*h, -0.582));
    const double alpha=pow(akh1, -1.0*Omega_bar/Omega_m)*pow(akh2, pow(-1.0*Omega_bar/Omega_m, 3.0));
    const double kh_tmp = Omega_m*h*Kokkos::sqrt(alpha);
    
    /* Set P(k)=T^2*k^n array.
       Linear array, taking care of the mirror symmetry
       (reality condition for the density field). 
    */
    
    msg << "Pulled all needed phyice quantities" << endl;
    msg << "kh_tmp= " << kh_tmp << " n_s= " << n_s << " tpiL= " << tpiL << endl;
    
    auto pkview                    = Pk_m.getView();
    auto* FL                       = &this->fcontainer_m->getFL();
    const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();  // local processor domain coordinates    
    const int nghost               = Pk_m.getNghost();
    
    ippl::parallel_for("Compute Pk_m", ippl::getRangePolicy(pkview, nghost),
		       KOKKOS_LAMBDA (const index_array_type& args) {
			 int k_i    = args[0] - nghost + ldom[0].first();
			 int k_j    = args[1] - nghost + ldom[1].first();
			 int k_k    = args[2] - nghost + ldom[2].first();
			 
			 /*
			  * we do not use symmetry
			  */
			 // if (k_k >= nq) {
			 //  k_k = -MOD(ngrid-k_k,ngrid);
			 //}
			 // without if but not sure if that is correct! 
			 // k_k -= (k_k >= nq) * ngrid;
			 
			 double kk = tpiL*Kokkos::sqrt(k_i*k_i+k_j*k_j+k_k*k_k);
			 double trans_f = StructureFormationManager<double,3>::hu_sugiyama(kk, kh_tmp);
			 double val = trans_f*trans_f*Kokkos::pow(kk, n_s);
			 ippl::apply(pkview, args) = val;
			 
			 /* for debugging 
			 int index = (k_i)
			 + (k_j) * ldom[0].length()
			 + (k_k) * ldom[0].length() * ldom[1].length();
			 
			 if ((k_i==0) && (k_j==0) && (k_k==0))
			   printf("i \t j \t k \t index \t kk \t transf \t Pk \n");
			   if ((k_i<2) && (k_j<2) && (k_k<2))
			   printf("%d \t %d \t %d \t %d \t %f \t %f \t %f \n",k_i,k_j,k_k,index,kk,trans_f,val);
			 */
		       });

    msg << "Pk created using hu_sugiyama TFFlag ==2" << endl;
    
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
  
  void indens() {
    Inform msg("indens ");
    
    auto cfview = cfield_m.getView();
    auto pkview = Pk_m.getView();
    
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345); // Seed for reproducibility

    ippl::parallel_for("random gauss field", ippl::getRangePolicy(cfview),
		       KOKKOS_LAMBDA(const index_array_type& args) {
			 double rn1, rn2, rn;
			 auto rand_gen = rand_pool.get_state();
			 do {
			   rn1 = -1.0 + 2.0 * rand_gen.drand();
			   rn2 = -1.0 + 2.0 * rand_gen.drand();
			   rn = rn1 * rn1 + rn2 * rn2;
			 } while (rn > 1.0 || rn == 0.0);		     
			 ippl::apply(cfview, args) = Kokkos::complex<double>(rn2 * Kokkos::sqrt(-2.0*Kokkos::log(rn)/rn), 0.0);
			 rand_pool.free_state(rand_gen); 		   
		       });

    msg << "FFR rho field created" << endl;
    
    Cfft_m->transform(ippl::FORWARD, cfield_m);

    msg << "FFR rho field done" << endl;

    
    // Multiply by the power spectrum:
    const double scale = Kokkos::pow(initializer::GlobalStuff::instance().ngrid, 1.5);

    ippl::parallel_for("Pk times gauss field", ippl::getRangePolicy(cfview),
		       KOKKOS_LAMBDA(const index_array_type& args) {
			 Kokkos::complex<double> tmp = ippl::apply(cfview, args); 
			 ippl::apply(cfview, args) = Kokkos::complex<double>(tmp.real()*Kokkos::sqrt(ippl::apply(pkview, args))/scale,
									     tmp.imag()*Kokkos::sqrt(ippl::apply(pkview, args))/scale);
		       }
		       );
    
    msg << "Multiply rho field by the power spectrum  done" << endl;

  }

  void test_reality(){

    Inform msg ("test_reality ");
    
    auto cfview = cfield_m.getView();  
    auto* FL   = &this->fcontainer_m->getFL();
    const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();  // local processor domain coordinates


    MinMaxSum result1;

    Kokkos::parallel_reduce("maxminsum reduce 1",
			    ippl::getRangePolicy(cfview), // assuming this is MDRangePolicy in 3 dimensions
			    KOKKOS_LAMBDA(const unsigned int i,
					  const unsigned int j,
					  const unsigned int k,
					  MinMaxSum & update) {
			      // Construct the index array expected by ippl::apply if necessary:
			      index_array_type args = { i, j, k };
			      auto tmp = ippl::apply(cfview, args);
			      double val = tmp.real();
			      update.min = Kokkos::min(update.min, val);
			      update.max = Kokkos::max(update.max, val);
			      update.sum += val;
			    },
			    result1
			    );

    result1.global_reduce();
    
    const int nghost = cfield_m.getNghost();
    const long myVol = (ldom[0].length()-nghost)*(ldom[1].length()-nghost)*(ldom[2].length()-nghost);

    msg << "Min and max value of density in k space: " << result1.min << " " << result1.max << endl;
    msg << "Average value of density in k space: " << result1.sum/myVol << endl;

    Cfft_m->transform(ippl::BACKWARD, cfield_m);
    msg << "FFT inv. done" << endl;

    double scale = Kokkos::pow(initializer::GlobalStuff::instance().box_size, 1.5); // inverse FFT scaling
    
    ippl::parallel_for("scale rho field after inv. FFT", ippl::getRangePolicy(cfview),
		       KOKKOS_LAMBDA(const index_array_type& args) {
			 Kokkos::complex<double> tmp = ippl::apply(cfview, args);
			 tmp.real() /= scale;
			 ippl::apply(cfview, args) = tmp;
		       }
		       );

    
    MinMaxSum result2;

    Kokkos::parallel_reduce("maxminsum reduce 2",
			    ippl::getRangePolicy(cfview), // assuming this is MDRangePolicy in 3 dimensions
			    KOKKOS_LAMBDA(const unsigned int i,
					  const unsigned int j,
					  const unsigned int k,
					  MinMaxSum & update) {
			      // Construct the index array expected by ippl::apply if necessary:
			      index_array_type args = { i, j, k };
			      auto tmp = ippl::apply(cfview, args);
			      double val = tmp.real();
			      update.min = Kokkos::min(update.min, val);
			      update.max = Kokkos::max(update.max, val);
			      update.sum += val;
			    },
			    result2
			    );

    result2.global_reduce(); 

    msg << "Min and max value of density in real space: " << result2.min << " " << result2.max << endl;
    msg << "Average value of density in real space: " << result2.sum/myVol << endl;
    
  /*


   for (i=0; i<My_Ng; ++i) {
      if (test_zero < fabs((real)rho[i].im))
         test_zero = fabs((real)rho[i].im);
   }

   MPI_Allreduce(MPI_IN_PLACE, &test_zero, 1, MPI_FLOAT, MPI_MAX, Parallel.GetMpiComm());	


      std::cout << std::endl << "Max value of the imaginary part of density is "
            << test_zero << std::endl;

      ave_rho = ave_rho/NumPEs;

  */
}

void gravity_potential(){
  Inform msg ("gravity_potential ");
  constexpr auto tpi             = 2*Kokkos::numbers::pi_v<double>;

  auto* FL                       = &this->fcontainer_m->getFL();
  const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();  // local processor domain coordinates
  auto cfview                    = cfield_m.getView();
  const int nghost               = cfield_m.getNghost();
  const int ngrid                = initializer::GlobalStuff::instance().ngrid;
  const int nq                   = ngrid/2;
  
  ippl::parallel_for("Multiply by the Green's function (-1/k^2)", ippl::getRangePolicy(cfview, nghost),
		     KOKKOS_LAMBDA (const index_array_type& args) {
		       int k_i    = args[0] - nghost + ldom[0].first();
		       int k_j    = args[1] - nghost + ldom[1].first();
		       int k_k    = args[2] - nghost + ldom[2].first();

		       if (k_i >= nq) {
			 k_i = -Kokkos::fmod(ngrid-k_i,ngrid);
		       }

		       if (k_j >= nq) {
			 k_j = -Kokkos::fmod(ngrid-k_j,ngrid);
		       }

		       if (k_k >= nq) {
			 k_k = -Kokkos::fmod(ngrid-k_k,ngrid);
		       }
		       
		       double kk = tpi*Kokkos::sqrt(k_i*k_i+k_j*k_j+k_k*k_k);
		       double green = 0.0;
		       if (kk==0.0)
			 green = 0.0;
		       else
			 green = -1.0/(kk*kk);
		       ippl::apply(cfview, args) *= Kokkos::complex<double>(green,green);
		       
		       if ( (k_i==0) && (k_j==0) && (k_k==0))
			 ippl::apply(cfview, args) *= Kokkos::complex<double>(0.0,0.0);

		     });
  
  msg << "G*rho done" << endl;
  
  Cfft_m->transform(ippl::BACKWARD, cfield_m);

  msg << "FFT inv. done" << endl;
  
  double scale = Kokkos::pow(initializer::GlobalStuff::instance().box_size, 1.5); // inverse FFT scaling
    
  ippl::parallel_for("scale rho field after inv. FFT", ippl::getRangePolicy(cfview),
		     KOKKOS_LAMBDA(const index_array_type& args) {
		       Kokkos::complex<double> tmp = ippl::apply(cfview, args);
		       tmp.real() /= scale;
		       ippl::apply(cfview, args) = tmp;
		     }
		     );
}
  
  
  void set_particles() {  
    Inform msg("set_particles ");
    Inform m2a("set_particles ", INFORM_ALL_NODES);
    float d_z, ddot;
    float z_in; 

    auto* FL                       = &this->fcontainer_m->getFL();
    //std::shared_ptr<ParticleContainer_t>
    auto  pc                       = this->pcontainer_m;
    const ippl::NDIndex<Dim>& ldom = FL->getLocalNDIndex();  // local processor domain coordinates
    auto cfview                    = cfield_m.getView();
    auto rView                     = pc->R.getView();            
    auto vView                     = pc->V.getView();            
    const int nghost               = cfield_m.getNghost();
    const int ngrid                = initializer::GlobalStuff::instance().ngrid;

    const double box_size          = initializer::GlobalStuff::instance().box_size;
    const double hr                = box_size / ngrid;   // 1.0 should be replaced by the box size
    
    // Growth factor for the initial redshift:
    z_in = initializer::GlobalStuff::instance().z_in;
    cosmo_m.GrowthFactor(z_in, &d_z, &ddot);
    msg << "redshift: " << z_in << "; growth factor=" << d_z << " derivative=" << ddot << endl;

    index_type lgridsize = 1;
    for (unsigned d = 0; d < Dim; d++) {
      lgridsize *= ldom[d].length();
    }
    
    if (lgridsize != pc->getLocalNum())
      m2a << "#gridpoints= " << lgridsize << " does not matcg particle container size " << pc->getLocalNum() << " this simulation does not make sense" << endl;
      
    ippl::parallel_for("Init x,v", ippl::getRangePolicy(cfview, nghost),
		       KOKKOS_LAMBDA (const index_array_type& args) {
			 const Vector_t<double,Dim>     pos = (args + ldom.first() - nghost) * hr;
			 const double                   x   = ippl::apply(cfview, args).real(); 
			 const Vector_t<double,Dim>     v   = -ddot * x;
			 const Vector_t<double,Dim>     r   = pos - d_z*x;

			 // compute the linear index n
			 const int i = args[0] - nghost;
			 const int j = args[1] - nghost;
			 const int k = args[2] - nghost;
			 index_type n = (i)
			   + (j) * ldom[0].length()
			   + (k) * ldom[0].length() * ldom[1].length();
			 rView(n) = r;
			 vView(n) = v;
		       });    
    msg << "Particles initialized ... " << endl;
    pc->update();
    msg << "Particle update done ... " << endl;
  } 


  
    /**
     * @brief Read particle data from a file.
     */
    void readParticles() {
        Inform mes("Reading Particles");

        size_type nloc = this->totalP_m / ippl::Comm->size();
        mes << "Local number of particles: " << nloc << endl;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        pc->create(nloc);
        pc->m = this->M_m / this->totalP_m;

        this->fcontainer_m->getRho() = 0.0;

        // Load Balancer Initialisation
        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            mes << "Starting first repartition" << endl;
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
        mes << "Minimum Position: " << MinPos << endl;
        mes << "Maximum Position: " << MaxPos << endl;
        mes << "Defined maximum:  " << this->rmax_m << endl;

        // Number of Particles
        if (nloc != ParticlePositions.size()) {
            std::cerr << "Error: Simulation number of particles does not match input!" << std::endl;
            std::cerr << "Input N = " << ParticlePositions.size() << ", Local N = " << nloc
                      << std::endl;
        } else
            // Particle positions and velocities, which are read in above from the initial
            // conditions file, are assigned to the particle attributes R and V in the particle
            // container.
            mes << "successfully done." << endl;

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
        if (this->loadbalancer_m->balance(this->totalP_m)) {
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            printf("first repartition works \n");
        }

        mes << "Assignment of positions and velocities done." << endl;
    }

    /**
     * @brief Read particle data from a file and assign to the domain.
     */
    void readParticlesDomain() {
        Inform mes("Reading Particles");

        this->fcontainer_m->getRho() = 0.0;

        // Load Balancer Initialisation
        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            mes << "Starting first repartition" << endl;
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
                        mes << "Particle was on edge. Shift position from " << Pos << " to "
                            << Pos * 0.9999 << endl;
                        Pos = 0.9999 * Pos;
                    }
                    if (Pos == 0) {
                        mes << "Particle was on edge. Shift position from " << Pos << " to "
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
        if (this->loadbalancer_m->balance(this->totalP_m)) {
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL   = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            printf("first repartition works \n");
        }

        mes << "Assignment of positions and velocities done." << endl;
    }

    /**
     * @brief Advance the simulation by one time step.
     */
    void advance() override {
      Inform m("advance ");
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
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP)) {
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
        Inform mes("Saving Particles");

        static IpplTimings::TimerRef SavingTimer = IpplTimings::getTimer("Save Data");
        IpplTimings::startTimer(SavingTimer);

        mes << "snapshot " << this->it_m << endl;

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
        mes << "done." << endl;
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
