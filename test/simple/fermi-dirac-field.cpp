#include "Ippl.h"
#include <iostream>

#define T double
#define Dim 3


T smpintd(int nstep, Field<T,1> &integ, T dv) {

    Index I(nstep);
    Index I2(1,2,nstep);

    FieldLayout<1> layout1(I,PARALLEL);
    Field<T,1> smp(layout1,GuardCellSizes<1>(1));

    assign(smp[I] ,4.0/3.0);
    assign(smp[I2],2.0/3.0);

    smp[1]=1.0/3.0;
    smp[nstep]=1.0/3.0;

    return dv*sum(smp*integ);
}

void cumul(Field<T,Dim> &/*vTherm1*/, Field<T,Dim> &/*vTherm2*/, int nmax,int nvdint,T hubble,T zin,T omeganu) {

  T pmax = 300000.0;
  T dv = pmax / (nvdint-1);
  T vv0 = (1.0+zin)*0.536/(omeganu*hubble*hubble);

  Index I1(nmax);

  Index I2(nvdint);

  FieldLayout<1> layout1(I1,PARALLEL);
  Field<T,1> parray(layout1,GuardCellSizes<1>(1));
  Field<T,1> carray(layout1,GuardCellSizes<1>(1));

  FieldLayout<1> layout2(I2,PARALLEL);
  Field<T,1> vv(layout2,GuardCellSizes<1>(1));
  Field<T,1> fermd(layout2,GuardCellSizes<1>(1));

  Field<T,1>::iterator fit1, fit2;

  parray = 0.0;
  carray = 0.0;

  vv = 0.0;
  fermd = 0.0;

  // forall(ii=1:nvdint) vv(ii)=(ii-1)*dv   ! velocities
  assign (vv[I2], (I2-1)*dv);

  // fermd=vv*vv/(dexp(vv/vv0)+1.d0)
  fermd = vv*vv /(exp(vv/vv0)+1);

  // call smpintd(nvdint,fermd,dv,norm)
  /*T norm = */ smpintd(nvdint,fermd,dv);

  Index I(2,1,nmax);
  assign (parray[I], (I-1)*pmax/(1.0*nmax));
  dv = pmax/nmax/(nvdint-1);

  int i,j;

  fit2=parray.begin();
  ++fit2;
  for (j=2; fit2!=carray.end(); ++fit2,++j) {
      for (fit1=vv.begin(),i=1; fit1!=vv.end(); ++fit1,++i) {
          *fit1 = ((i-1)*dv * (*fit2));
      }
  }

}

void vel(int /*np*/,int nmax,int nvdint,T hubble,T zin,T omeganu, Field<T,Dim> &vTherm) {

  INFOMSG("entering vel routine" << endl);

  //T tpi=8.0*atan(1.0);

  cumul(vTherm,vTherm,nmax,nvdint,hubble,zin,omeganu);

  INFOMSG("cumulative array generated" << endl);

  // call momentum(pp,np,carray,parray,iseed,nmax,cran)

  INFOMSG("after momentum" << endl);

  /*
  mu=2.d0*(aran-0.5d0)
  phi=tpi*bran

  vx=pp*dsqrt(1.d0-mu**2)*dcos(phi)
  vy=pp*dsqrt(1.d0-mu**2)*dsin(phi)
  vz=pp*mu

  vths(1,:)=vx(:)
  vths(2,:)=vy(:)
  vths(3,:)=vz(:)

  */



}

int main(int argc, char *argv[]) {

  Ippl ippl(argc,argv);
  Inform msg(argv[0]);

  int np=32;
  np=np*np*np;             // number of particles
  int nmax=1000;           // size of velocity array (check this!)
  int nvdint=1001;         // number of integration points
  T hubble=0.7;
  T zin=50.0;
  T omeganu=0.02;

  Index I(np);
  Index J(np);
  Index K(np);
  FieldLayout<Dim> layout(I,J,K,PARALLEL,PARALLEL,PARALLEL);
  Field<T,Dim> vTherm(layout,GuardCellSizes<Dim>(1));

  vTherm = 0.0;

  vel(np,nmax,nvdint,hubble,zin,omeganu,vTherm);


  return 0;
}