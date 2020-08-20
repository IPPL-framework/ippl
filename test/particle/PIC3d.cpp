// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 *
 ***************************************************************************/

/***************************************************************************

This test program sets up a simple sine-wave electric field in 3D,
  creates a population of particles with random q/m values (charge-to-mass
  ratio) and velocities, and then tracks their motions in the static
  electric field using nearest-grid-point interpolation.

Usage:

 mpirun -np 2 PIC3d 128 128 128 10000 10 NGP OOP GUARDCELLS --commlib mpi --info 0

***************************************************************************/

#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>

// dimension of our positions
const unsigned Dim = 3;

// some typedefs
typedef ParticleSpatialLayout<double,Dim>::SingleParticlePos_t Vector_t;
typedef ParticleSpatialLayout<double,Dim> playout_t;
typedef UniformCartesian<Dim,double> Mesh_t;
typedef Cell                                       Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t> FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t>       Field_t;
typedef Field<Vector_t, Dim, Mesh_t, Center_t>     VField_t;
typedef IntCIC IntrplCIC_t;
typedef IntNGP IntrplNGP_t;

#define GUARDCELL 1

enum BC_t {OOO,OOP,PPP};
enum InterPol_t {NGP,CIC};

const double pi = acos(-1.0);
const double dt = 1.0;          // size of timestep


void dumpVTK(Field<Vektor<double,3>,3> &EFD, NDIndex<3> lDom, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

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
    vtkout << "pic3d" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
    vtkout << "ORIGIN 0 0 0" << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "POINT_DATA " << nx*ny*nz << std::endl;

    vtkout << "VECTORS E-Field float" << std::endl;
    for (int z=lDom[2].first(); z<lDom[2].last(); z++) {
        for (int y=lDom[1].first(); y<lDom[1].last(); y++) {
            for (int x=lDom[0].first(); x<lDom[0].last(); x++) {
                Vektor<double, 3> tmp = EFD[x][y][z].get();
                vtkout << tmp(0) << "\t"
                       << tmp(1) << "\t"
                       << tmp(2) << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}


void dumpVTK(Field<double,3> &EFD, NDIndex<3> lDom, int nx, int ny, int nz, int iteration,
             double dx, double dy, double dz) {

    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    //SERIAL at the moment
    //if (Ippl::myNode() == 0) {

    // open a new data file for this iteration
    // and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "toyfdtd" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
    vtkout << "ORIGIN 0 0 0" << std::endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    vtkout << "POINT_DATA " << nx*ny*nz << std::endl;

    vtkout << "SCALARS E-Field float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z=lDom[2].first(); z<=lDom[2].last(); z++) {
        for (int y=lDom[1].first(); y<=lDom[1].last(); y++) {
            for (int x=lDom[0].first(); x<=lDom[0].last(); x++) {
                vtkout << EFD[x][y][z].get() << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}




template<class PL>
class ChargedParticles : public IpplParticleBase<PL> {

  Field<Vektor<double,Dim>,Dim> EFD_m;
  Field<double,Dim> EFDMag_m;

  BConds<double,Dim,Mesh_t,Center_t> bc_m;
  BConds<Vector_t,Dim,Mesh_t,Center_t> vbc_m;

  Vektor<int,Dim> nr_m;

  BC_t bco_m;
  InterPol_t interpol_m;
  bool fieldNotInitialized_m;
  bool doRepart_m;
  bool withGuardCells_m;

  e_dim_tag decomp_m[Dim];

  Vector_t hr_m;
  Vector_t rmin_m;
  Vector_t rmax_m;


public:
    ParticleAttrib<double>     qm; // charge-to-mass ratio
    typename PL::ParticlePos_t P;  // particle velocity
    typename PL::ParticlePos_t E;  // electric field at particle position
    typename PL::ParticlePos_t B;  // magnetic field at particle position

    ChargedParticles(PL* pl, BC_t bc, InterPol_t interpol, e_dim_tag decomp[Dim], bool gCells) :
        IpplParticleBase<PL>(pl),
        bco_m(bc),
        interpol_m(interpol),
        fieldNotInitialized_m(true),
        doRepart_m(true),
        withGuardCells_m(gCells)
    {
        // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(B);
        setupBCs();
        for (unsigned int i=0; i<Dim; i++)
            decomp_m[i]=decomp[i];
    }

    /*
      In case we have OOP or PPP boundary conditions
      we must define the domain, i.e can not be deduced from the
      particles as in the OOO case.
    */

    ChargedParticles(PL* pl, BC_t bc, InterPol_t interpol, Vector_t hr, Vector_t rmin, Vector_t rmax, e_dim_tag decomp[Dim], bool gCells) :
        IpplParticleBase<PL>(pl),
        bco_m(bc),
        interpol_m(interpol),
	fieldNotInitialized_m(true),
        doRepart_m(true),
        withGuardCells_m(gCells),
	hr_m(hr),
        rmin_m(rmin),
        rmax_m(rmax)
    {
        // register the particle attributes
        this->addAttribute(qm);
        this->addAttribute(P);
        this->addAttribute(E);
        this->addAttribute(B);
        setupBCs();
        for (unsigned int i=0; i<Dim; i++)
            decomp_m[i]=decomp[i];
    }

    void setupBCs() {
        if (bco_m == OOO)
            setBCAllOpen();
        else if (bco_m == PPP)
            setBCAllPeriodic();
        else
            setBCOOP();
    }

    inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }

    inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }

    inline const FieldLayout_t& getFieldLayout() const {
        return dynamic_cast<FieldLayout_t&>( this->getLayout().getLayout().getFieldLayout());
    }

    inline FieldLayout_t& getFieldLayout() {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    void gather(int iteration) {
        if (interpol_m==CIC)
            gatherCIC();
        else
            gatherNGP();

		scatterCIC();
        NDIndex<Dim> lDom = getFieldLayout().getLocalNDIndex();
        dumpVTK(EFDMag_m,lDom,nr_m[0],nr_m[1],nr_m[2],iteration,hr_m[0],hr_m[1],hr_m[2]);
    }

    double scatter() {
        Inform m("scatter ");
        double initialQ = sum(qm);
        if (interpol_m==CIC)
            scatterCIC();
        else
            scatterNGP();

        /*
          now sum over all gridpoints ... a bit nasty !

        */

        Field<double,Dim> tmpf;
        NDIndex<Dim> domain = getFieldLayout().getDomain();

        FieldLayout_t *FL  = new FieldLayout_t(getMesh(), decomp_m);
        tmpf.initialize(getMesh(), *FL);

        tmpf[domain] = EFDMag_m[domain];

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
        m << "sum(qm)= " << initialQ << " sum(EFDMag)= " << sum(EFDMag_m) << endl;
        return initialQ-Q;
    }

    void myUpdate() {

        double hz   = hr_m[2];
        double zmin = rmin_m[2];
        double zmax = rmax_m[2];

        if (bco_m != PPP) {
            bounds(this->R, rmin_m, rmax_m);

            NDIndex<Dim> domain = this->getFieldLayout().getDomain();

            for (unsigned int i=0; i<Dim; i++)
                nr_m[i] = domain[i].length();

            for (unsigned int i=0; i<Dim; i++)
                hr_m[i] = (rmax_m[i] - rmin_m[i]) / (nr_m[i] - 1.0);

            if (bco_m == OOP) {
                rmin_m[2] = zmin;
                rmax_m[2] = zmax;
                hr_m[2] = hz;
            }

            getMesh().set_meshSpacing(&(hr_m[0]));
            getMesh().set_origin(rmin_m);

            if(withGuardCells_m) {
                EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
                EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
            }
            else {
                EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
                EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
            }
        }
        else {
            if(fieldNotInitialized_m) {
                fieldNotInitialized_m=false;
                getMesh().set_meshSpacing(&(hr_m[0]));
                getMesh().set_origin(rmin_m);
                if(withGuardCells_m) {
                    EFD_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);
                    EFDMag_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
                }
                else {
                    EFD_m.initialize(getMesh(), getFieldLayout(), vbc_m);
                    EFDMag_m.initialize(getMesh(), getFieldLayout(), bc_m);
                }
            }
        }
        this->update();
    }

    void gatherStatistics() {
        Inform m("gatherStatistics ");
        Inform m2a("gatherStatistics ",INFORM_ALL_NODES);

        double *partPerNode = new double[Ippl::getNodes()];
        double *globalPartPerNode = new double[Ippl::getNodes()];
        for (int i=0; i<Ippl::getNodes(); i++) {
            partPerNode[i] = globalPartPerNode[i] = 0.0;

        }
        partPerNode[Ippl::myNode()] = this->getLocalNum();

        reduce(partPerNode,partPerNode+Ippl::getNodes(),globalPartPerNode,OpAddAssign());

        for (int i=0; i<Ippl::getNodes(); i++)
            m << "Node " << i << " has "
              <<   globalPartPerNode[i]/this->getTotalNum()*100.0 << " \% of the total particles " << endl;
    }




    void initFields() {
        Inform m("initFields ");

        NDIndex<Dim> domain = getFieldLayout().getDomain();

        for (unsigned int i=0; i<Dim; i++)
            nr_m[i] = domain[i].length();

        int nx = nr_m[0];
        int ny = nr_m[1];
        int nz = nr_m[2];

        double phi0 = 0.1*nx;

        m << "rmin= " << rmin_m << " rmax= " << rmax_m << " h= " << hr_m << " n= " << nr_m << endl;

        Index I(nx), J(ny), K(nz);

        assign(EFD_m[I][J][K](0), -2.0*pi*phi0/nx * cos(2.0*pi*(I+0.5)/nx) * cos(4.0*pi*(J+0.5)/ny) * cos(pi*(K+0.5)/nz));

        assign(EFD_m[I][J][K](1),  4.0*pi*phi0/ny * sin(2.0*pi*(I+0.5)/nx) * sin(4.0*pi*(J+0.5)/ny));

        assign(EFD_m[I][J][K](2),  4.0*pi*phi0/ny * sin(2.0*pi*(I+0.5)/nx) * sin(4.0*pi*(J+0.5)/ny));

        assign(EFDMag_m[I][J][K],
               EFD_m[I][J][K](0) * EFD_m[I][J][K](0) +
               EFD_m[I][J][K](1) * EFD_m[I][J][K](1) +
               EFD_m[I][J][K](2) * EFD_m[I][J][K](2));
    }

    Vector_t getRMin() { return rmin_m;}
    Vector_t getRMax() { return rmax_m;}
    Vector_t getHr() { return hr_m;}

    void setRMin(Vector_t x) { rmin_m = x; }
    void setHr(Vector_t x) { hr_m = x; }

    void savePhaseSpace(std::string fn, int idx) {

        int tag = Ippl::Comm->next_tag(IPPL_APP_TAG4, IPPL_APP_CYCLE);
	std::vector<double> tmp;
        tmp.clear();

        // every node ckecks if he has to dump particles
        for (unsigned i=0; i<this->getLocalNum(); i++) {
            tmp.push_back(this->ID[i]);
            tmp.push_back(this->R[i](0));
            tmp.push_back(this->R[i](1));
            tmp.push_back(this->R[i](2));
            tmp.push_back(this->P[i](0));
            tmp.push_back(this->P[i](1));
            tmp.push_back(this->P[i](2));
        }

        if(Ippl::myNode() == 0) {
 	    std::ofstream ostr;
            std::string Fn;
            char numbuf[6];
            sprintf(numbuf, "%05d", idx);
            Fn = fn  + std::string(numbuf) + ".dat";
            ostr.open(Fn.c_str(), std::ios::out );
            ostr.precision(15);
            ostr.setf(std::ios::scientific, std::ios::floatfield);

            ostr << " x, px, y, py t, pt, id, node" << std::endl;

            unsigned int dataBlocks=0;
            double x,y,z,px,py,pz,id;
            unsigned  vn;

            for (unsigned i=0; i < tmp.size(); i+=7)
                ostr << tmp[i+1] << " " << tmp[i+4] << " " << tmp[i+2]  << " " \
                     << tmp[i+5] << " " << tmp[i+3] << " " << tmp[i+6]  << " " \
                     << tmp[i]   << " " << Ippl::myNode() << std::endl;

            int notReceived =  Ippl::getNodes() - 1;
            while (notReceived > 0) {
                int node = COMM_ANY_NODE;
                Message* rmsg =  Ippl::Comm->receive_block(node, tag);
                if (rmsg == 0)
                    ERRORMSG("Could not receive from client nodes in main." << endl);
                notReceived--;
                rmsg->get(&dataBlocks);
                rmsg->get(&vn);
                for (unsigned i=0; i < dataBlocks; i+=7) {
                    rmsg->get(&id);
                    rmsg->get(&x);
                    rmsg->get(&y);
                    rmsg->get(&z);
                    rmsg->get(&px);
                    rmsg->get(&py);
                    rmsg->get(&pz);
                    ostr << x  << "\t " << px  << "\t " << y  << "\t " \
                         << py << "\t " << z << "\t " << pz << "\t "   \
                         << id   << "\t " << vn << std::endl;
                }
                delete rmsg;
            }
            ostr.close();
        }
        else {
            unsigned dataBlocks = 0;
            Message* smsg = new Message();
            dataBlocks = tmp.size();
            smsg->put(dataBlocks);
            smsg->put(Ippl::myNode());
            for (unsigned i=0; i < tmp.size(); i++)
                smsg->put(tmp[i]);
            bool res = Ippl::Comm->send(smsg, 0, tag);
            if (! res)
                ERRORMSG("Ippl::Comm->send(smsg, 0, tag) failed " << endl;);
        }
    }

private:

    inline void setBCAllOpen() {
        for (unsigned i=0; i < 2*Dim; i++) {
            this->getBConds()[i] = ParticleNoBCond;
            bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
        }
    }

    inline void setBCAllPeriodic() {
        for (unsigned i=0; i < 2*Dim; i++) {
            this->getBConds()[i] = ParticlePeriodicBCond;
            bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
        }
    }

    inline void setBCOOP() {
        for (unsigned i=0; i < 2*Dim - 2; i++) {
            bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
            this->getBConds()[i] = ParticleNoBCond;
        }
        for (unsigned i= 2*Dim - 2; i < 2*Dim; i++) {
            bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
            this->getBConds()[i] = ParticlePeriodicBCond;
        }
    }

    inline void gatherNGP() {
        // create interpolater object (nearest-grid-point method)
        IntNGP myinterp;
        E.gather(EFD_m, this->R, myinterp);
    }

    inline void gatherCIC() {
        // create interpolater object (cloud in cell method)
        IntCIC myinterp;
        E.gather(EFD_m, this->R, myinterp);
    }

    inline void scatterCIC() {
        // create interpolater object (cloud in cell method)
        IntCIC myinterp;
        qm.scatter(EFDMag_m, this->R, myinterp);
    }

    inline void scatterNGP() {
        // create interpolater object (cloud in cell method)
        IntNGP myinterp;
        qm.scatter(EFDMag_m, this->R, myinterp);
    }

};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Vektor<int,Dim> nr(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));

    const unsigned int totalP = atoi(argv[4]);
    const unsigned int nt     = atoi(argv[5]);

    InterPol_t myInterpol;
    if (std::string(argv[6])==std::string("CIC"))
        myInterpol = CIC;
    else
        myInterpol = NGP;

    bool gCells;
    gCells =  (std::string(argv[8])==std::string("GUARDCELLS"));

    msg << "Particle test PIC3d " << endl;
    msg << "nt " << nt << " Np= " << totalP << " grid = " << nr <<endl;

    if (myInterpol==CIC)
        msg << "Cloud in cell (CIC) interpolation selected" << endl;
    else
        msg << "Nearest grid point (NGP) interpolation selected" << endl;

    if (gCells)
        msg << "Using guard cells" << endl;
    else
        msg << "Not using guard cells" << endl;

    BC_t myBC;
    if (std::string(argv[7])==std::string("OOO")) {
        myBC = OOO; // open boundary
        msg << "BC == OOO" << endl;
    }
    else if (std::string(argv[7])==std::string("OOP")) {
        myBC = OOP; // open boundary in x and y, periodic in z
        msg << "BC == OOP" << endl;
    }
    else {
        myBC = PPP; // all periodic
        msg << "BC == PPP" << endl;
    }

    e_dim_tag decomp[Dim];
    unsigned serialDim = 2;

    msg << "Serial dimension is " << serialDim  << endl;

    Mesh_t *mesh;
    FieldLayout_t *FL;
    ChargedParticles<playout_t>  *P;

    NDIndex<Dim> domain;
    if (gCells) {
        for (unsigned i=0; i<Dim; i++)
            domain[i] = domain[i] = Index(nr[i] + 1);
    }
    else {
        for (unsigned i=0; i<Dim; i++)
            domain[i] = domain[i] = Index(nr[i]);
    }

    for (unsigned d=0; d < Dim; ++d)
        decomp[d] = (d == serialDim) ? SERIAL : PARALLEL;

    // create mesh and layout objects for this problem domain
    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);
    playout_t* PL = new playout_t(*FL, *mesh);

    if (myBC==OOO)
        P = new ChargedParticles<playout_t>(PL,myBC,myInterpol,decomp,gCells);
    else {
        /*
          In case of periodic BC's define
          the domain with hr and rmin
        */

        Vector_t hr(1.0);
        Vector_t rmin(0.0);
        Vector_t rmax(nr);

        P = new ChargedParticles<playout_t>(PL,myBC,myInterpol,hr,rmin,rmax,decomp,gCells);
    }

    // initialize the particle object: do all initialization on one node,
    // and distribute to others

    unsigned long int nloc = totalP / Ippl::getNodes();

    P->create(nloc);
    for (unsigned long int i = 0; i< nloc; i++) {
        for (int d = 0; d<3; d++)
            P->R[i](d) =  IpplRandom() * nr[d];
    }

    double q = 1.0/totalP;

    // random initialization for charge-to-mass ratio
    assign(P->qm,q);

    msg << "particles created and initial conditions assigned " << endl;

    // redistribute particles based on spatial layout
    P->myUpdate();

    msg << "initial update and initial mesh done .... Q= " << sum(P->qm) << endl;
    msg << P->getMesh() << endl;
    msg << P->getFieldLayout() << endl;

    msg << "scatter test done delta= " <<  P->scatter() << endl;

    P->initFields();
    msg << "P->initField() done " << endl;

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {
        P->gatherStatistics();
        // advance the particle positions
        // basic leapfrogging timestep scheme.  velocities are offset
        // by half a timestep from the positions.
        assign(P->R, P->R + dt * P->P);

        // update particle distribution across processors
        P->myUpdate();

        // gather the local value of the E field
        P->gather(it);

        // advance the particle velocities
        assign(P->P, P->P + dt * P->qm * P->E);
        msg << "Finished iteration " << it << " - min/max r and h " << P->getRMin()
            << P->getRMax() << P->getHr() << endl;
    }
    Ippl::Comm->barrier();
    msg << "Particle test PIC3d: End." << endl;
    return 0;
}

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/