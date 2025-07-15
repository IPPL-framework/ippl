// Test attribute-reduction.cpp
//   This test program sets up particle attributes and does reductions on it
//   Usage:
//     attribute-reduction 10000 --info 5
//
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#define str(x)  #x
#define xstr(x) str(x)

// dimension of our positions
#define DIM     3
constexpr unsigned Dim          = DIM;
constexpr const char* PROG_NAME = "attribute-reduction " xstr(DIM) "d";

// some typedefs
typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;
typedef ippl::UniformCartesian<double, Dim> Mesh_t;
typedef ippl::FieldLayout<Dim> FieldLayout_t;
typedef Mesh_t::DefaultCentering Centering_t;

template <typename T, unsigned Dim>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim>
using Field = ippl::Field<T, Dim, Mesh_t, Centering_t>;

typedef ippl::OrthogonalRecursiveBisection<Field<double, Dim>> ORB;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

typedef Vector<double, Dim> Vector_t;
typedef Field<double, Dim> Field_t;

double pi = Kokkos::numbers::pi_v<double>;

template <class PLayout>
class ChargedParticles : public ippl::ParticleBase<PLayout> {
public:

    Field_t sField_m;

    ORB orb;

public:

  ChargedParticles(PLayout& pl)
    : ippl::ParticleBase<PLayout>(pl) {
    
    this->setParticleBC(ippl::BC::PERIODIC);
  }

  void updateLayout(FieldLayout_t& fl, Mesh_t& mesh) {
    // Update local fields
    static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
    IpplTimings::startTimer(tupdateLayout);
    this->sField_m.updateLayout(fl);

    // Update layout with new FieldLayout
    PLayout& layout = this->getLayout();
    layout.updateLayout(fl, mesh);
    IpplTimings::stopTimer(tupdateLayout);

    static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
    IpplTimings::startTimer(tupdatePLayout);
    this->update();
    IpplTimings::stopTimer(tupdatePLayout);
  }

  void initializeORB(FieldLayout_t& fl, Mesh_t& mesh) { orb.initialize(fl, mesh, sField_m); }

  ~ChargedParticles() {}

  void repartition(FieldLayout_t& fl, Mesh_t& mesh) {
        // Repartition the domains
    bool fromAnalyticDensity = false;
    bool res                 = orb.binaryRepartition(this->R, fl, fromAnalyticDensity);

    if (res != true) {
      std::cout << "Could not repartition!" << std::endl;
      return;
    }
    this->updateLayout(fl, mesh);
    }

  bool balance(unsigned int totalP) {  //, int timestep = 1) {
    int local = 0;
    std::vector<int> res(ippl::Comm->size());
    double threshold = 1.0;
    double equalPart = (double)totalP / ippl::Comm->size();
    double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
    if (dev > threshold) {
      local = 1;
    }
    MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT, ippl::Comm->getCommunicator());
    
    for (unsigned int i = 0; i < res.size(); i++) {
      if (res[i] == 1) {
	return true;
      }
    }
    return false;
  }
  
  // @param tag
  //        2 -> uniform(0,1)
  //        1 -> normal(0,1)
  //        0 -> gridpoints
  void initPositions(FieldLayout_t& fl, Vector_t& hr, unsigned int nloc, int tag = 2) {
    Inform m("initPositions ");

    typename ippl::ParticleBase<PLayout>::particle_position_type::HostMirror R_host =
      this->R.getHostMirror();

    std::mt19937_64 eng[Dim];
    for (unsigned i = 0; i < Dim; ++i) {
      eng[i].seed(42 + i * Dim);
      eng[i].discard(nloc * ippl::Comm->rank());
    }
    
    std::mt19937_64 engN[4 * Dim];
    for (unsigned i = 0; i < 4 * Dim; ++i) {
      engN[i].seed(42 + i * Dim);
      engN[i].discard(nloc * ippl::Comm->rank());
    }
    
    auto dom                = fl.getDomain();
    unsigned int gridpoints = 1;
    for (unsigned d = 0; d < Dim; d++) {
      gridpoints *= dom[d].length();
    }
    if (tag == 0 && nloc * ippl::Comm->size() != gridpoints) {
      if (ippl::Comm->rank() == 0) {
	std::cerr << "Particle count must match gridpoint count to use gridpoint "
	  "locations. Switching to uniform distribution."
		  << std::endl;
      }
      tag = 2;
    }
    
    if (tag == 0) {
      m << "Positions are set on grid points" << endl;
      int N = fl.getDomain()[0].length();  // this only works for boxes
      const ippl::NDIndex<Dim>& lDom = fl.getLocalNDIndex();
      int size                       = ippl::Comm->size();
      using index_type               = typename ippl::RangePolicy<Dim>::index_type;
      Kokkos::Array<index_type, Dim> begin, end;
      for (unsigned d = 0; d < Dim; d++) {
	begin[d] = 0;
	end[d]   = N;
      }
      end[0] /= size;
      // Loops over particles
      using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
      ippl::parallel_for(
			 "initPositions", ippl::createRangePolicy(begin, end),
			 KOKKOS_LAMBDA(const index_array_type& args) {
			   int l = 0;
			   for (unsigned d1 = 0; d1 < Dim; d1++) {
			     int next = args[d1];
			     for (unsigned d2 = 0; d2 < d1; d2++) {
			       next *= N;
			     }
			     l += next / size;
			   }
			   R_host(l) = (0.5 + args + lDom.first()) * hr;
			 });
      
    } else if (tag == 1) {
      m << "Positions follow normal distribution" << endl;
      std::vector<double> mu = {0.5, 0.6, 0.2, 0.5, 0.6, 0.2};
      std::vector<double> sd = {0.75, 0.3, 0.2, 0.75, 0.3, 0.2};
      std::vector<double> states(Dim);
      
      Vector_t length = 1;
      
      std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
      
      double sum_coord = 0.0;
      for (unsigned long long int i = 0; i < nloc; i++) {
	for (unsigned d = 0; d < Dim; d++) {
	  double u1 = dist_uniform(engN[d * 2]);
	  double u2 = dist_uniform(engN[d * 2 + 1]);
	  states[d] =
	    sd[d] * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2) + mu[d];
	  R_host(i)[d] = std::fabs(std::fmod(states[d], length[d]));
	  sum_coord += R_host(i)[d];
	}
      }
    } else {
      double rmin = 0.0, rmax = 1.0;
      m << "Positions follow uniform distribution U(" << rmin << "," << rmax << ")" << endl;
      std::uniform_real_distribution<double> unif(rmin, rmax);
      for (unsigned long int i = 0; i < nloc; i++) {
	for (unsigned d = 0; d < Dim; d++) {
	  R_host(i)[d] = unif(eng[d]);
	}
      }
    }
    
    // Copy to device
    Kokkos::deep_copy(this->R.getView(), R_host);
  }
  
};

int main(int argc, char* argv[]) {
  ippl::initialize(argc, argv);
  {
    Inform msg(PROG_NAME);
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    ippl::Comm->setDefaultOverallocation(3.0);

    int arg = 1;
	
    ippl::Vector<int, Dim> nr;
    for (unsigned d = 0; d < Dim; d++) {
      nr[d] = 16;
    }
    
    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);
    
    const unsigned int totalP = std::atoi(argv[arg++]);
    
    msg << "attribute-reduction " << PROG_NAME << endl
	<< " Np= " << totalP << " grid = " << nr << endl;

    using bunch_type = ChargedParticles<PLayout_t>;
    std::unique_ptr<bunch_type> P;

    Vector_t rmin(0.0);
    Vector_t rmax(1.0);
    // create mesh and layout objects for this problem domain
    Vector_t hr = rmax / nr;
    
    ippl::NDIndex<Dim> domain;
    for (unsigned d = 0; d < Dim; d++) {
      domain[d] = ippl::Index(nr[d]);
    }
    
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);
    
    Vector_t origin = rmin;
    
    const bool isAllPeriodic = true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
    PLayout_t PL(FL, mesh);


    
    msg << "FIELD LAYOUT (INITIAL)" << endl;
    msg << FL << endl;
    
    P        = std::make_unique<bunch_type>(PL);
    
    P->sField_m.initialize(mesh, FL);
    
    unsigned long int nloc = totalP / ippl::Comm->size();
    
    int rest = (int)(totalP - nloc * ippl::Comm->size());

    if (ippl::Comm->rank() < rest) {
      ++nloc;
    }

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    IpplTimings::startTimer(particleCreation);
    P->create(nloc);
    // Verifying that particles are created
    double totalParticles = 0.0;
    double localParticles = P->getLocalNum();
    ippl::Comm->reduce(&localParticles, &totalParticles, 1, std::plus<double>());
    msg << "Total particles: " << totalParticles << endl;

    P->initPositions(FL, hr, nloc, 1);

    IpplTimings::stopTimer(particleCreation);

    static IpplTimings::TimerRef UpdateTimer = IpplTimings::getTimer("ParticleUpdate");
    IpplTimings::startTimer(UpdateTimer);
    P->update();
    IpplTimings::stopTimer(UpdateTimer);
    
    msg << "particles created and initial conditions assigned " << endl;
    

    P->initializeORB(FL, mesh);
    
    static IpplTimings::TimerRef domainDecomposition0 = IpplTimings::getTimer("domainDecomp0");
    IpplTimings::startTimer(domainDecomposition0);
    if (P->balance(totalP)) {
      P->repartition(FL, mesh);
    }
    IpplTimings::stopTimer(domainDecomposition0);
    msg << "Balancing finished" << endl;
    
    msg2all << "nlocal = " << P->getLocalNum() << endl;
    msg << "sum(R)     = " << P->R.sum() << endl;
    msg << "max(R)     = " << P->R.max() << endl;
    msg << "min(R)     = " << P->R.min() << endl;

    msg << "Particle test " << PROG_NAME << ": End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing" + std::to_string(ippl::Comm->size()) + "r_"
                                       + std::to_string(nr[0]) + "c.dat"));
    }
    ippl::finalize();

    return 0;
}
