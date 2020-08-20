// -*- C++ -*-
/**************************************************************************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *

Example:
./test-scatter-1 4 4 CIC --commlib mpi --info 9 | tee output

 *************************************************************************************************************************************/

#include "Ippl.h"
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"

using namespace std;

// dimension of our positions
#define DIM 2
const unsigned Dim = 2;

// some typedefs
typedef ParticleSpatialLayout<double,Dim>::SingleParticlePos_t      Vector_t;
typedef ParticleSpatialLayout<double,Dim>                           playout_t;
typedef UniformCartesian<Dim,double>                                Mesh_t;
typedef Cell                                                        Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t>                  FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t>                        Field_t;
typedef IntCIC                                                      IntrplCIC_t;
typedef IntNGP                                                      IntrplNGP_t;

enum InterPol_t {NGP,CIC};

template<class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double>     qm;

    ChargedParticles(PL* pl, InterPol_t interpol, Vector_t nr, Vector_t hr, Vector_t rmin, Vector_t rmax, e_dim_tag decomp[Dim], bool gCells = true) :
        IpplParticleBase<PL>(pl),
        nr_m(nr),
        hr_m(hr),
        rmax_m(rmax),
        rmin_m(rmin),
        interpol_m(interpol),
        withGuardCells_m(gCells)
    {
        this->addAttribute(qm);

        for (unsigned int i=0; i < 2*Dim; i++) {
            bc_m[i]  = new ParallelPeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            this->getBConds()[i] = ParticlePeriodicBCond;
        }
        for(unsigned int i=0; i<Dim; i++)
            decomp_m[i]=decomp[i];

        getMesh().set_meshSpacing(&(hr_m[0]));
        getMesh().set_origin(rmin_m);
        if(withGuardCells_m)
            scalF_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
        else
            scalF_m.initialize(getMesh(), getFieldLayout(), bc_m);
    }
   
    void scatter() {
        Inform m ("scatter: " );
        m.precision(10);
        m.setf(ios::fixed, ios::floatfield);

        scalF_m = 0.0;
        if (interpol_m == CIC) qm.scatter(scalF_m, this->R, IntrplCIC_t());
        else qm.scatter(scalF_m, this->R, IntrplNGP_t());

        NDIndex<Dim> elem;
        for (int i = 0; i < nr_m[0]-1; ++ i)  {
            elem[0] = Index(i,i);
            for (int j = 0; j < nr_m[1]-1; ++ j)  {
                elem[1] = Index(j,j);
                    m << "(" << i << ", " << j << ") = " << scalF_m.localElement(elem) << endl;
            }
        }

        m << "Qgrid= " << sum(scalF_m) << " sum(qe)= " << sum(qm) << " deltaQ= " << (sum(qm)-sum(scalF_m)) << endl;
    }

    inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }
    inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }
    inline const FieldLayout_t& getFieldLayout() const {
        return dynamic_cast<FieldLayout_t&>( this->getLayout().getLayout().getFieldLayout());
    }
    inline FieldLayout_t& getFieldLayout() {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    Vector_t getRMin() { return rmin_m;}
    Vector_t getRMax() { return rmax_m;}
    Vector_t getHr() { return hr_m;}

    void setRMin(Vector_t x) { rmin_m = x; }
    void setHr(Vector_t x) { hr_m = x; }

private:
    Field<double, Dim> scalF_m;
    BConds<double, Dim, Mesh_t, Center_t> bc_m;

    Vektor<int,Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmax_m;
    Vector_t rmin_m;

    InterPol_t interpol_m;
    bool withGuardCells_m;
    e_dim_tag decomp_m[Dim];
};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Vektor<int,Dim> nr;
    InterPol_t myInterpol;

    // need to use preprocessor to prevent clang compiler error
#if DIM == 3
    nr = Vektor<int,Dim>(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
    if (string(argv[4])==string("CIC")) myInterpol = CIC;
    else myInterpol = NGP;
#else
    nr = Vektor<int,Dim>(atoi(argv[1]),atoi(argv[2]));
    if (string(argv[3])==string("CIC")) myInterpol = CIC;
    else myInterpol = NGP;
#endif

    e_dim_tag decomp[Dim];
    Mesh_t *mesh;
    FieldLayout_t *FL;
    ChargedParticles<playout_t>  *P;

    NDIndex<Dim> domain;
    for(unsigned int i=0; i<Dim; i++)
        domain[i] = domain[i] = Index(nr[i]);

    for (unsigned int d=0; d < Dim; ++d)
        decomp[d] = PARALLEL;

    // create mesh and layout objects for this problem domain
    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);
    playout_t* PL = new playout_t(*FL, *mesh);

    Vector_t hr(1.0);
    Vector_t rmin(0.0);
    Vector_t rmax(nr - 1.0); //XXX: rmax never used...

    P = new ChargedParticles<playout_t>(PL, myInterpol, nr, hr, rmin, rmax, decomp, true);
    INFOMSG(P->getMesh() << endl);
    INFOMSG(P->getFieldLayout() << endl);
    msg << endl << endl;

    P->create(1);
    P->R[0](0) = 0.340834;
    P->R[0](1) = 0.771409;
    msg << "at " << P->R[0] << endl;
    P->qm = 1.0;
    P->update();
    P->scatter();

    //delete P;
    //delete PL;
    //delete FL;
    //delete mesh;

    return 0;
}
