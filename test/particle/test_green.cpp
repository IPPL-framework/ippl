//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"
#include <string>
#include <vector>
#include <iostream>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <complex>
#include "H5hut.h"
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodic.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodicParallel.h"
#include "Particle/PairBuilder/PairConditions.h"
#include "Utility/PAssert.h"
#include "Utility/IpplException.h"
//#include "math.h"
#include<cmath>

#include <random>

#include "VTKFieldWriterParallel.hpp"
#include "ChargedParticleFactory.hpp"


const unsigned Dim = 3;
const double ke=2.532638e8;

typedef UniformCartesian<Dim, double>                                 Mesh_t;
typedef BoxParticleCachingPolicy<double, Dim, Mesh_t>                 CachingPolicy_t;
typedef ParticleSpatialLayout<double, Dim, Mesh_t, CachingPolicy_t>   playout_t;
typedef playout_t::SingleParticlePos_t                                Vector_t;
typedef Cell                                                          Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t>                    FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t>                          Field_t;
typedef Field<int, Dim, Mesh_t, Center_t>                             IField_t;
typedef Field<Vector_t, Dim, Mesh_t, Center_t>                        VField_t;
typedef Field<std::complex<double>, Dim, Mesh_t, Center_t>            CxField_t;
typedef FFT<CCTransform, Dim, double>                                 FFT_t;

//This is the periodic Greens function with regularization parameter epsilon.
template<unsigned int Dim>
struct SpecializedGreensFunction { };

template<>
struct SpecializedGreensFunction<3> {
    template<class T, class FT, class FT2>
    static void calculate(Vektor<T, 3> &hrsq, FT &grn, FT2 *grnI) {

        double r;
        double alpha = 1e6;

        NDIndex<3> elem0=NDIndex<3>(Index(0,0), Index(0,0),Index(0,0));
        grn = grnI[0] * hrsq[0] + grnI[1] * hrsq[1] + grnI[2] * hrsq[2];
        
        NDIndex<3> lDomain_m = grn.getLayout().getLocalNDIndex();
        NDIndex<3> elem;
        
        for (int i=lDomain_m[0].min(); i<=lDomain_m[0].max(); ++i) {
            
            elem[0]=Index(i,i);
            
            for (int j=lDomain_m[1].min(); j<=lDomain_m[1].max(); ++j) {
                
                elem[1]=Index(j,j);
                
                for (int k=lDomain_m[2].min(); k<=lDomain_m[2].max(); ++k) {
                    
                    elem[2]=Index(k,k);

                    r = real(sqrt(grn.localElement(elem)));
                    
                    if(elem==elem0) {
                        grn.localElement(elem) = 0 ;
                    } else {
                        grn.localElement(elem) = ke*std::complex<double>(std::erf(alpha*r)/(r));
                    }
                }
            }   
        }
    }
};

template<typename T>
void write(Field<T, Dim, Mesh_t, Center_t>& field) {
    NDIndex<3> myidx = (field.getLayout()).getLocalNDIndex();
    for (int z = myidx[2].first() - 1; z <= myidx[2].last() + 1; z++) {
        for (int y = myidx[1].first() - 1; y <= myidx[1].last() + 1; y++) {
            for (int x = myidx[0].first() - 1; x <= myidx[0].last() + 1; x++) {
                std::cout << field[x][y][z].get() << " ";
            }
            std::cout << std::endl;
        }
        if (z < myidx[2].last() + 1) {
            std::cout << std::endl;
        }
    }
}

template<typename T>
void write(Field<Vektor<T, 3>, Dim, Mesh_t, Center_t>& vfield) {
    NDIndex<3> myidx = (vfield.getLayout()).getLocalNDIndex();
    for (int z = myidx[2].first() - 1; z <= myidx[2].last() + 1; z++) {
        for (int y = myidx[1].first() - 1; y <= myidx[1].last() + 1; y++) {
            for (int x = myidx[0].first() - 1; x <= myidx[0].last() + 1; x++) {
                std::cout << vfield[x][y][z].get() << " ";
            }
            std::cout << std::endl;
        }
        if (z < myidx[2].last() + 1) {
            std::cout << std::endl;
        }
    }
}


int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    IpplTimings::TimerRef allTimer = IpplTimings::getTimer("AllTimer");
    IpplTimings::startTimer(allTimer);
    
    // get domain size
    Vektor<int,Dim> nr;
    nr = Vektor<int,Dim>(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));

    e_dim_tag decomp[Dim];
    Mesh_t *mesh;
    FieldLayout_t *FL;

    NDIndex<Dim> domain;
    for (unsigned i=0; i<Dim; i++)
        domain[i] = Index(nr[i]+1);

    for (unsigned d=0; d < Dim; ++d)
        decomp[d] = PARALLEL;

    // create mesh and layout objects for this problem domain
    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);

    double L = 0.5;
    Vektor<double,Dim> extend_l(-L,-L,-L);
    Vektor<double,Dim> extend_r(L,L,L);

    Vector_t hr_m;
    for (unsigned int d = 0;d<Dim;++d) {
        hr_m[d] = (extend_r[d] - extend_l[d]) / (nr[d]);
    }
    mesh->set_meshSpacing(&(hr_m[0]));
    mesh->set_origin(extend_l);

    //initialize the FFT
    FFT_t* fft = new FFT_t(FL->getDomain(),true);

    fft->setDirectionName(+1, "forward");
    fft->setDirectionName(-1, "inverse");

    // initialize fields
    Field_t rho, phi;
    VField_t efield;
    CxField_t rhocmpl, grncmpl;

    rhocmpl.initialize(*mesh, *FL, GuardCellSizes<Dim>(1));
    grncmpl.initialize(*mesh, *FL, GuardCellSizes<Dim>(1));
    rho.initialize(*mesh, *FL, GuardCellSizes<Dim>(1));
    phi.initialize(*mesh, *FL, GuardCellSizes<Dim>(1));
    efield.initialize(*mesh, *FL, GuardCellSizes<Dim>(1));

    // set rho to 2.0
    NDIndex<3> myidx = (rho.getLayout()).getLocalNDIndex();
    for (int z = myidx[2].first() - 1; z <= myidx[2].last() + 1; z++) {
        for (int y = myidx[1].first() - 1; y <= myidx[1].last() + 1; y++) {
            for (int x = myidx[0].first() - 1; x <= myidx[0].last() + 1; x++) {
                rho[x][y][z] = 2.0;
            }
        }
    }

    std::cout << "Rho: " << std::endl;
    write(rho);

    rhocmpl[FL->getDomain()] = rho[FL->getDomain()];

    fft->transform("inverse", rhocmpl);

    std::cout << "Rho trans: " << std::endl;
    write(rhocmpl);

    IField_t grnIField[3];

    for (int i = 0; i < 3; ++i) {
        grnIField[i].initialize(*mesh, *FL);
        grnIField[i][FL->getDomain()] = where(lt(FL->getDomain()[i], nr[i]/2),
                                        FL->getDomain()[i] * FL->getDomain()[i],
                                        (nr[i]-FL->getDomain()[i]) *
                                        (nr[i]-FL->getDomain()[i]));
    }
    Vector_t hrsq(hr_m * hr_m);
    SpecializedGreensFunction<3>::calculate(hrsq, grncmpl, grnIField);

    std::cout << "Green: " << std::endl;
    write(grncmpl);

    fft->transform("inverse", grncmpl);

    std::cout << "Green trans: " << std::endl;
    write(grncmpl);

    rhocmpl *= grncmpl;

    std::cout << "Convolution: " << std::endl;
    write(rhocmpl);

    fft->transform("forward", rhocmpl);

    std::cout << "Inverse fft: " << std::endl;
    write(rhocmpl);

    phi = 0.0;
    phi = real(rhocmpl)*hr_m[0]*hr_m[1]*hr_m[2];

    std::cout << "Computed phi: " << std::endl;
    write(phi);

    efield = -Grad1Ord(phi, efield);

    std::cout << "Efield: " << std::endl;
    write(efield);

    return 0;
}
