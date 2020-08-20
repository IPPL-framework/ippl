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

Usage:

 ./test --commlib mpi --info 0

Build:

 CXX=mpicc make test

***************************************************************************/

#include "Ippl.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>

using namespace std;

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

class ChargedParticles : public IpplParticleBase<playout_t> {
public:

    ChargedParticles(playout_t* pl, BC_t /*bc*/, Vector_t hr, Vector_t rmin, Vector_t rmax, e_dim_tag decomp[Dim], bool /*gCells*/) :
        IpplParticleBase<playout_t>(pl),
        hr_m(hr),
        rmin_m(rmin),
        rmax_m(rmax)
    {
        setupBCs();
        for(unsigned int i=0; i<Dim; i++)
            decomp_m[i]=decomp[i];
    }
    
    void setupBCs() {
        setBCAllOpen();
        //         setBCAllPeriodic();
        //          setBCOOP();
    }

    inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }

    inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }

    inline FieldLayout_t& getFieldLayout() {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    
    void initFields() {
        Inform m("initFields ");

        NDIndex<Dim> domain = getFieldLayout().getDomain();

        for(unsigned int i=0; i<Dim; i++)
            nr_m[i] = domain[i].length();

        int nx = nr_m[0];
        int ny = nr_m[1];
        int nz = nr_m[2];

        double phi0 = 0.1*nx;

        m << "rmin= " << rmin_m << " rmax= " << rmax_m << " h= " << hr_m << " n= " << nr_m << endl;

        Index I(nx), J(ny), K(nz);

        assign(EFD_m[I][J][K](0), -2.0*pi*phi0/nx * cos(2.*pi*(I+0.5)/nx) * cos(4.0*pi*(J+0.5)/ny) * cos(pi*(K+0.5)/nz));

        assign(EFD_m[I][J][K](1),  4.0*pi*phi0/ny * sin(2.*pi*(I+0.5)/nx) * sin(4.0*pi*(J+0.5)/ny));

        assign(EFD_m[I][J][K](2),  4.0*pi*phi0/ny * sin(2.*pi*(I+0.5)/nx) * sin(4.0*pi*(J+0.5)/ny));

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

private:

    inline void setBCAllOpen() {
        for (unsigned int i=0; i < 2*Dim; i++) {
            this->getBConds()[i] = ParticleNoBCond;
            bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
        }
    }

    inline void setBCAllPeriodic() {
        for (unsigned int i=0; i < 2*Dim; i++) {
            this->getBConds()[i] = ParticlePeriodicBCond;
            bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
        }
    }

    inline void setBCOOP() {
        for (unsigned int i=0; i < 2*Dim - 2; i++) {
            bc_m[i]  = new ZeroFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new ZeroFace<Vector_t,Dim,Mesh_t,Center_t>(i);
            this->getBConds()[i] = ParticleNoBCond;
        }
        for (unsigned int i= 2*Dim - 2; i < 2*Dim; i++) {
            bc_m[i]  = new PeriodicFace<double  ,Dim,Mesh_t,Center_t>(i);
            vbc_m[i] = new PeriodicFace<Vector_t,Dim,Mesh_t,Center_t>(i);
            this->getBConds()[i] = ParticlePeriodicBCond;
        }
    }

    Field<Vektor<double,Dim>,Dim> EFD_m;


    BConds<double,Dim,Mesh_t,Center_t> bc_m;
    BConds<Vector_t,Dim,Mesh_t,Center_t> vbc_m;

    Vektor<int,Dim> nr_m;

    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;

    e_dim_tag decomp_m[Dim];

public:
    Field<double,Dim> EFDMag_m;
};

/***************************************************************************
 * PwrSpec 
 * 
 *
 ***************************************************************************/

template <class T, unsigned int Dim>
class PwrSpec 
{
public:

    typedef Field<std::complex<double>, Dim, Mesh_t, Center_t> CxField_t;
    typedef Field<T, Dim, Mesh_t, Center_t>                 RxField_t;
    typedef FFT<CCTransform, Dim, T>                        FFT_t;

    // constructor and destructor
    PwrSpec(Mesh_t *mesh, FieldLayout_t *FL):
        mesh_m(mesh),
        layout_m(FL)
    {

        Inform msg ("FFT doInit");
        gDomain_m = layout_m->getDomain();       // global domain
        lDomain_m = layout_m->getLocalNDIndex(); // local domain


        gDomainL_m = NDIndex<Dim>(Index(gDomain_m[0].length()+1),
                                  Index(gDomain_m[1].length()+1),
                                  Index(gDomain_m[2].length()+1));

        msg << "GDomain " << gDomain_m << " GDomainL " << gDomainL_m << endl;


        for (unsigned int i=0; i < 2*Dim; ++i) {
            bc_m[i] = new ParallelPeriodicFace<T,Dim,Mesh_t,Center_t>(i);
            zerobc_m[i] = new ZeroFace<T,Dim,Mesh_t,Center_t>(i);
        }

        for(unsigned int d=0; d<Dim; d++) {
            dcomp_m[d]=layout_m->getRequestedDistribution(d);
            nr_m[d] = gDomain_m[d].length();
        }

        // create additional objects for the use in the FFT's
        rho_m.initialize(*mesh_m, *layout_m, GuardCellSizes<Dim>(1));

        // create the FFT object
        bool compressTemps = true;
        fft_m = new FFT_t(layout_m->getDomain(), compressTemps);
        fft_m->setDirectionName(+1, "forward");

        meshI_m = new Mesh_t(gDomainL_m);
        FLI_m = new FieldLayout_t(*meshI_m, dcomp_m);

        // power spectra calc
        kmax_m = nr_m[2]/2;
        spectra1D_m = (T *) malloc (kmax_m * sizeof(T));
        Nk_m = (int *) malloc (kmax_m * sizeof(int));

    }


    ~PwrSpec() {
        if (fft_m)
            delete fft_m;
        if (meshI_m)
            delete meshI_m;
        if (FLI_m)
            delete FLI_m;
    }
    
    //  void calcPwrSpecAndSave(CxField_t *f, std::string fn) {
    void calcPwrSpecAndSave(ChargedParticles *p, std::string fn) {
        Inform m("calcPwrSpecAndSave ");

        rho_m = p->EFDMag_m;

        unsigned int kk;

        T rho_0 = real(sum(rho_m)) / (nr_m[0] * nr_m[1] * nr_m[2]);
        rho_m = (rho_m - rho_0) / rho_0;

        fft_m->transform("forward", rho_m);

        rho_m = real(rho_m*conj(rho_m));

        for (int i=0;i<kmax_m;i++) {
            Nk_m[i]=0;
            spectra1D_m[i] = 0.0;
        }

        m << "Sum psp=real( ... " << sum(real(rho_m)) << " kmax= " << kmax_m << endl;

        NDIndex<Dim> loop;

        // This computes the 1-D power spectrum of a 3-D field (rho_m)
        // by binning values
        for (int i=lDomain_m[0].first(); i<=lDomain_m[0].last(); i++) {
            loop[0]=Index(i,i);
            for (int j=lDomain_m[1].first(); j<=lDomain_m[1].last(); j++) {
                loop[1]=Index(j,j);
                for (int k=lDomain_m[2].first(); k<=lDomain_m[2].last(); k++) {

                    loop[2]=Index(k,k);
                    int ii = i;
                    int jj = j;
                    int k2 = k;
                    //FIXME: this seems to have no change
                    //if(i >= (gDomain_m[0].max()+1)/2)
                    if(i >= gDomain_m[0].max()/2)
                        ii -= nr_m[0];
                    if(j >= gDomain_m[1].max()/2)
                        jj -= nr_m[1];
                    if(k >= gDomain_m[2].max()/2)
                        k2 -= nr_m[2];

                    kk=(int)nint(sqrt(ii*ii+jj*jj+k2*k2));
                    kk = min(kmax_m, (int)kk);
                    spectra1D_m[kk] += real(rho_m.localElement(loop));
                    Nk_m[kk]++;
                }
            }
        }

        INFOMSG("Loops done" << endl);

        reduce( &(Nk_m[0]), &(Nk_m[0]) + kmax_m , &(Nk_m[0]) ,OpAddAssign());
        reduce( &(spectra1D_m[0]), &(spectra1D_m[0]) + kmax_m, &(spectra1D_m[0]) ,OpAddAssign());

        Inform* fdip = new Inform(NULL,fn.c_str(),Inform::OVERWRITE,0);
        Inform& fdi = *fdip;
        setInform(fdi);
        setFormat(9,1,0);
        /*
          fdi << "# " ;
          for (int i=0; i < kmax_m; i++)
          fdi << "Nk[" << i << "]= " << Nk_m[i] << " "; 
          fdi << " sum Nk= " << sumNk << endl;

          scale = 1.0; // std::pow((T)(simData_m.rL/simData_m.ng_comp),(T)3.0);
          for (int i=1; i<=nr_m[2]/2; i++) {
          fdi << (i)*tpiL << "\t" << spectra1D_m[i]*scale << endl;
          }
        */
        delete fdip;

    }

private:    

    /// fortrans nint function                                                                                            
    inline T nint(T x)
    {
        return ceil(x + 0.5) - (fmod(x*0.5 + 0.25, 1.0) != 0);
    }
    
    FFT_t *fft_m;

    // mesh and layout objects for rho_m
    Mesh_t *mesh_m;
    FieldLayout_t *layout_m;

    // bigger mesh (length+1)
    FieldLayout_t *FLI_m;
    Mesh_t *meshI_m;
      
    /// global domain for the various fields
    NDIndex<Dim> gDomain_m;  
    /// local domain for the various fields
    NDIndex<Dim> lDomain_m;             

    /// global domain for the enlarged fields 
    NDIndex<Dim> gDomainL_m;  
  
    BConds<T,Dim,Mesh_t,Center_t> bc_m;
    BConds<T,Dim,Mesh_t,Center_t> zerobc_m;

    e_dim_tag dcomp_m[Dim];
    Vektor<int, Dim> nr_m;

    /// Fourier transformed density field
    CxField_t rho_m;
    
    /// power spectra kmax
    int kmax_m;

    /// 1D power spectra
    T *spectra1D_m;
    /// Nk power spectra
    int *Nk_m;

};




int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Vektor<int,Dim> nr(64,64,64);

    bool gCells = false;
    e_dim_tag decomp[Dim];

    msg << "PwrSpec test " << " grid = " << nr <<endl;


    if (gCells)
        msg << "Using guard cells" << endl;
    else
        msg << "Not using guard cells" << endl;

    BC_t myBC = PPP;
    msg << "BC == PPP" << endl;

    Mesh_t *mesh;
    FieldLayout_t *FL;

    ChargedParticles *P;


    NDIndex<Dim> domain;
    if (gCells) {
        for(unsigned int i=0; i<Dim; i++)
            domain[i] = domain[i] = Index(nr[i] + 1);
    }
    else {
        for(unsigned int i=0; i<Dim; i++)
            domain[i] = domain[i] = Index(nr[i]);
    }

    for (unsigned int d=0; d < Dim; ++d)
        decomp[d] = PARALLEL;

    // create mesh and layout objects for this problem domain
    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);
    playout_t* PL = new playout_t(*FL, *mesh);

    Vector_t hr(1.0);
    Vector_t rmin(0.0);
    Vector_t rmax(nr);
    
    P = new ChargedParticles(PL,myBC,hr,rmin,rmax,decomp,gCells);

    PwrSpec<double,3> *pwrSpec = new PwrSpec<double,3>(mesh,FL);


    msg << "initial update and initial mesh done " << endl;
    msg << P->getMesh() << endl;
    msg << P->getFieldLayout() << endl;
    
    P->initFields();
    msg << "P->initField() done " << endl;

    pwrSpec->calcPwrSpecAndSave(P, std::string("pwrSpec.dat"));
    msg << "calcPwrSpecAndSave  done " << endl;

    
    Ippl::Comm->barrier();
    msg << "PwrSpec test End." << endl;
    return 0;
}

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/

