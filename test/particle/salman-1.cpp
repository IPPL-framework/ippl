// -*- C++ -*-
/***********************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit http://www.
 *
 * This test program sets up a simple sine-wave electric field in 3D,
 * creates a population of particles with random q/m values (charge-to-mass
 * ratio) and velocities, and then tracks their motions in the static
 * electric field using nearest-grid-point interpolation.
 *
 * Usage:
 *
 * $MYMPIRUN -np 2 salman-1 128 128 ??? 10000 10 NGP initialx initialy --commlib mpi
 *
 *          nx,ny xx  xxx NP NTstep  Interp  xini yini dx  dy  meshorgx meshorgy hx  hy
 *          |     |       |  |       |       |    |    |   |   |        |        |   |
 * salman-1 4     xx  xxx 1  7       NGP     1.0  1.0  1.0 1.0 1.0      1.0      1.0 1.0 --commlib mpi
 *
 *
 *          nx,ny Interp  meshorgx meshorgy hx  hy  domplus steps
 * salman-1 4     CIC     -0.5     -0.5     1.0 1.0 1       1      --commlib mpi --info 0
 *
 ************************************************************************************/

#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>

// dimension of our positions
const unsigned Dim = 2;

// some typedefs
typedef ParticleSpatialLayout<double,Dim>::SingleParticlePos_t Vector_t;
typedef ParticleSpatialLayout<double,Dim> playout_t;
typedef UniformCartesian<Dim,double> Mesh_t;
typedef Vert                                      Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t> FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t>       Field_t;
typedef Field<Vector_t, Dim, Mesh_t, Center_t>     VField_t;
typedef IntCIC IntrplCIC_t;
typedef IntNGP IntrplNGP_t;

#define GUARDCELL 2

enum BC_t {OOO,OOP,PPP};
enum InterPol_t {NGP,CIC};

using namespace std;

template<class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
  ParticleAttrib<double>     qm; // charge-to-mass ratio
  typename PL::ParticlePos_t P;  // particle velocity
  typename PL::ParticlePos_t E;  // electric field at particle position
  typename PL::ParticlePos_t B;  // magnetic field at particle position

  ChargedParticles(PL* pl, InterPol_t interpol, Vector_t hr, e_dim_tag decomp[Dim], Vector_t mesh_org) :
    IpplParticleBase<PL>(pl),
    hr_m(hr),
    interpol_m(interpol),
    fieldNotInitialized_m(true),
    doRepart_m(true)
  {
    // register the particle attributes
    this->addAttribute(qm);

    NDIndex<Dim> domain = getFieldLayout().getDomain();

    for(unsigned int i=0; i<Dim; i++)
      nr_m[i] = domain[i].length();

    hr_m = hr;
    getMesh().set_meshSpacing(&(hr_m[0]));
    getMesh().set_origin(mesh_org);

    for(unsigned int i=0; i<Dim; i++)
      decomp_m[i]=decomp[i];

    setBCAllPeriodic();
  }
  
  inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }

  inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }

  inline const FieldLayout_t& getFieldLayout() const {
    return dynamic_cast<FieldLayout_t&>( this->getLayout().getLayout().getFieldLayout());
  }

  inline FieldLayout_t& getFieldLayout() {
    return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
  }

  void scatter() {
    Inform m("scatter ");
    //double initialQ = sum(qm);
    if (interpol_m==CIC)
      scatterCIC();
    else
      scatterNGP();

    /*
      now sum over all gridpoints ... a bit nasty !



    Field<double,Dim> tmpf;
    NDIndex<Dim> domain = getFieldLayout().getDomain();

    FieldLayout_t *FL  = new FieldLayout_t(getMesh(), decomp_m);
    tmpf.initialize(getMesh(), *FL);

    tmpf[domain] = rho_m[domain];

    NDIndex<Dim> idx = getFieldLayout().getLocalNDIndex();
    NDIndex<Dim> idxdom = getFieldLayout().getDomain();
    NDIndex<Dim> elem;

    double Q = 0.0;
    for (int i=idx[0].min(); i<=idx[0].max(); ++i) {
        elem[0] = Index(i,i);
        for (int j=idx[1].min(); j<=idx[1].max(); ++j) {
	    elem[1] = Index(j,j);
	    for (int k=idx[2].min(); k<=idx[2].max(); ++k) {
		elem[2] = Index(k,k);
		Q +=  tmpf.localElement(elem);
	    }
        }

    }
    reduce(Q,Q,OpAddAssign());
    m << "sum(qm)= " << initialQ << " sum(rho_m)= " << sum(rho_m) << endl;
    return initialQ-Q;
 */
  }
  
  void myInitialUpdate() {
    bounds(this->R, rmin_m, rmax_m);
    if(fieldNotInitialized_m) {

      // rescale mesh
      // getMesh().set_meshSpacing(&(hr_m[0]));
      // getMesh().set_origin( -hr_m/2.0 );
      // getMesh().set_origin(Vector_t(1.0));
      rho_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(GUARDCELL), bc_m);
      fieldNotInitialized_m=false;
    }
    this->update();
  }

  void myUpdate() {
    bounds(this->R, rmin_m, rmax_m);
    this->update();
  }


  Vector_t getRMin() { return rmin_m;}
  Vector_t getRMax() { return rmax_m;}
  Vector_t getHr() { return hr_m;}

  void setRMin(Vector_t x) { rmin_m = x; }
  void setHr(Vector_t x) { hr_m = x; }

  void savePhaseSpace(string /*fn*/, int /*idx*/) {}

  inline void setBCAllPeriodic() {
    for (unsigned int i=0; i < 2*Dim; i++) {
      this->getBConds()[i] = ParticlePeriodicBCond;
      bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
      vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
    }
  }

  void printField(Inform& m, Field_t& f)
  {
    NDIndex<Dim> domain = getFieldLayout().getDomain();
    NDIndex<Dim> loop;

    m << "sum(rho)= "<< sum(rho_m) << endl;
    for (int j=domain[1].min(); j<=domain[1].max(); ++j)
      m << "\t" << j;
    m << endl << endl;

    for (int i=domain[0].min(); i<=domain[0].max(); ++i) {
      loop[0]=Index(i,i);
      	m << i ;
      for (int j=domain[1].min(); j<=domain[1].max(); ++j) {
	loop[1]=Index(j,j);
	m << "\t" << f.localElement(loop);
      }
      m << endl;
    }
    m << endl;
  }

  Field_t rho_m;

private:

  inline void scatterCIC() {
    // create interpolater object (cloud in cell method)
    IntCIC myinterp;
    qm.scatter(rho_m, this->R, myinterp);
  }

  inline void scatterNGP() {
    // create interpolater object (cloud in cell method)
    IntNGP myinterp;
    qm.scatter(rho_m, this->R, myinterp);
  }


  BConds<double,Dim,Mesh_t,Center_t> bc_m;
  BConds<Vector_t,Dim,Mesh_t,Center_t> vbc_m;

  Vektor<int,Dim> nr_m;

  Vector_t hr_m;
  Vector_t rmin_m;
  Vector_t rmax_m;

  InterPol_t interpol_m;
  bool fieldNotInitialized_m;
  bool doRepart_m;
  e_dim_tag decomp_m[Dim];

};

int main(int argc, char *argv[]){
  Ippl ippl(argc, argv);
  Inform msg("cic ");
  Inform msg2all(argv[0],INFORM_ALL_NODES);

  Vektor<int,Dim> nr(atoi(argv[1]),atoi(argv[1]));

  // coordinate space
  size_t nx = nr[0] - 1;
  size_t ny = nr[1] - 1;

  const unsigned int totalP = nx*ny;
  Vector_t mesh_org = Vector_t(atof(argv[3]),atof(argv[4]));
  Vector_t hr = Vector_t(atof(argv[5]),atof(argv[6]));

  const unsigned int domplus = atoi(argv[7]);

  const unsigned int steps = atoi(argv[8]);

  InterPol_t myInterpol;
  if (string(argv[2])==string("CIC"))
    myInterpol = CIC;
  else
    myInterpol = NGP;

  msg << "Np= " << totalP << " grid = " << nr <<endl;
  msg << "0 <= x <= " << nx-1 << " 0 <= y <= " << ny-1 << endl;

  if (myInterpol==CIC)
    msg << "Cloud in cell (CIC) interpolation selected" << endl;
  else
    msg << "Nearest grid point (NGP) interpolation selected" << endl;

  msg << "BC: periodic in all dimensions" << endl;

  e_dim_tag decomp[Dim];
  unsigned int serialDim = Dim-1;

  msg << "Serial dimension is " << serialDim  << endl;

  Mesh_t                       *mesh;
  FieldLayout_t                *FL;
  ChargedParticles<playout_t>  *P;

  NDIndex<Dim> domain;
  for(unsigned int i=0; i<Dim; i++) {
    domain[i] = domain[i] = Index(nr[i] + domplus);
    decomp[i] = (i == serialDim) ? SERIAL : PARALLEL;
  }

  // create mesh and layout objects for this problem domain
  mesh          = new Mesh_t(domain);
  FL            = new FieldLayout_t(*mesh, decomp);
  playout_t* PL = new playout_t(*FL, *mesh);

  P = new ChargedParticles<playout_t>(PL,myInterpol,hr,decomp,mesh_org);

  P->create(totalP);

  size_t k = 0;
  for (unsigned int i=0; i<nx; i++)
    for (unsigned int j=0; j<ny; j++) {
      P->R[k]  = Vector_t(i,j);
      k++;
    }

  double q = 1.0;
  assign(P->qm,q);
  P->update();

  // redistribute particles based on spatial layout
  P->myInitialUpdate();

  msg << P->getMesh() << endl;
  msg << P->getFieldLayout() << endl;
  msg << "particles created and initial conditions assigned Q=" << sum(P->qm) << endl;

  size_t loc = (nx*ny) - 1;


  P->scatter();
  P->printField(msg, P->rho_m);
  P->myUpdate();


  for (unsigned int n=0; n<steps; n++) {
    P->rho_m= 0.0;
    P->R[loc] += Vector_t(0.0,0.1);
    P->update();

    k = 0;
    for (unsigned int i=0; i<nx; i++) {
      msg << endl;
      for (unsigned int j=0; j<ny; j++) {
	msg << P->R[k] << " ";
	k++;
      }
    }
    msg << endl << endl;

    P->scatter();
    P->myUpdate();
    P->printField(msg, P->rho_m);
  }
  Ippl::Comm->barrier();
  msg << "test  End." << endl;
  return 0;
}