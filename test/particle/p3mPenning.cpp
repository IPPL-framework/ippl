//
// Application p3m3dHeating
//   mpirun -np 4 ./p3m3dHeating Nx Ny Nz l_beam l_box particleDensity r_cut alpha dt
//                               eps iterations charge_per_part m_part printEvery
//
//   alpha is the splitting parameter for pm and pp,
//   eps is the smoothing factor and Si are the coordinates of the charged sphere center
//
// Copyright (c) 2016, Benjamin Ulmer, ETH Zürich
// All rights reserved
//
// Implemented as part of the Master thesis
// "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
// (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
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
#include "math.h"
#include<cmath>

#include <random>

#include "VTKFieldWriterParallel.hpp"
#include "ChargedParticleFactory.hpp"


// dimension of our positions
const unsigned Dim = 3;

// Coulomb constant
//const double ke=2.532638e8; // Ulmer's
const double ke = 8.987551e9; // SI units

// some typedefs
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

typedef IntCIC                                                        IntrplCIC_t;

//This is the periodic Greens function with regularization parameter epsilon.
template<unsigned int Dim>
struct SpecializedGreensFunction { };

template<>
struct SpecializedGreensFunction<3> {
    template<class T, class FT, class FT2>
    static void calculate(Vektor<T, 3> &hrsq, FT &grn, FT2 *grnI, double alpha) {
        double r;
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
                    }
                    else
                        grn.localElement(elem) = ke*std::complex<double>(std::erf(alpha*r)/r);
                }
            }
        }
    }
};



template<class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double>      Q;
    ParticleAttrib<double>      m;
    ParticleAttrib<double>      Phi; //electrostatic potential
    ParticleAttrib<Vector_t>    EF;
    ParticleAttrib<Vector_t>    v; //velocity of the particles
    ParticleAttrib<int> ID; //velocity of the particles

    ChargedParticles(PL* pl, Vektor<double,3> nr, Vektor<double,3> extend_l_, Vektor<double,3> extend_r_) :
    IpplParticleBase<PL>(pl),
    nr_m(nr),
    rmin_m(extend_l_),
    rmax_m(extend_r_),
    extend_l(extend_l_),
    extend_r(extend_r_)
    {
        this->addAttribute(Q);
        this->addAttribute(m);
        this->addAttribute(Phi);
        this->addAttribute(EF);
        this->addAttribute(v);
        this->addAttribute(ID);

        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            //use periodic boundary conditions for the particles
            this->getBConds()[i] = ParticlePeriodicBCond;
            //boundary conditions used for interpolation kernels allow writes to ghost cells

            if (Ippl::getNodes()>1) {
                bc_m[i] = new ParallelInterpolationFace<double, Dim, Mesh_t, Center_t>(i);
                //std periodic boundary conditions for gradient computations etc.
                vbc_m[i] = new ParallelPeriodicFace<Vector_t, Dim, Mesh_t, Center_t>(i);
                bcp_m[i] = new ParallelPeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            }
            else {
                bc_m[i] = new InterpolationFace<double, Dim, Mesh_t, Center_t>(i);
                //std periodic boundary conditions for gradient computations etc.
                vbc_m[i] = new PeriodicFace<Vector_t, Dim, Mesh_t, Center_t>(i);
                bcp_m[i] = new PeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            }
        }

        this->update();

        //initialize the FFT
        bool compressTemps = true;
        fft_m = new FFT_t(domain_m,compressTemps);

        fft_m->setDirectionName(+1, "forward");
        fft_m->setDirectionName(-1, "inverse");
        INFOMSG("INIT FFT DONE"<<endl);
    }

    void update()
    {
        //should only be needed if meshspacing changes -----------
        for (unsigned int d = 0;d<Dim;++d) {
            hr_m[d] = (rmax_m[d] - rmin_m[d]) / (nr_m[d]);
        }
        this->getMesh().set_meshSpacing(&(hr_m[0]));
        this->getMesh().set_origin(rmin_m);
        //--------------------------------------------------------

        //init resets the meshes to 0 ?!
        rhocmpl_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1));
        grncmpl_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1));
        rho_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1),bc_m);
        phi_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1),bcp_m);
        eg_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);

        domain_m = this->getFieldLayout().getDomain();
        lDomain_m = this->getFieldLayout().getLocalNDIndex();

        IpplParticleBase<PL>::update();
    }

    inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }

    inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }

    inline const FieldLayout_t& getFieldLayout() const {
        return dynamic_cast<FieldLayout_t&>( this->getLayout().getLayout().getFieldLayout());
    }

    inline FieldLayout_t& getFieldLayout() {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    void calculateGridForces(double alpha) {


        // (1) scatter charge to charge density grid and transform to fourier space
        rho_m[domain_m]=0; 
        this->Q.scatter(this->rho_m, this->R, IntrplCIC_t());

        double Q_grid = sum(rho_m);
        Q_grid = std::fabs((this->total_charge - Q_grid) / this->total_charge);
        std::cout << "Rel. error in charge conservation = " << Q_grid << std::endl;
        std::cout << "Number of particles in simulation = " << this->getTotalNum() << std::endl;

        std::cout << "Rho sum before normalisation = " << sum(rho_m) << std::endl;

        rhocmpl_m[domain_m] = rho_m[domain_m]/(hr_m[0]*hr_m[1]*hr_m[2]);

        std::cout << "Rho sum after normalisation = " << sum(rhocmpl_m) << std::endl;

        dumpVTKScalar(rho_m,this,0,"rho");

        // trick for periodic BCs penning trap (subtract ion density)
        double size = 1;
        for (unsigned d = 0; d < 3; d++) {
            size *= rmax_m[d] - rmin_m[d];
        }
        std::cout << "total_charge/size = " << total_charge/size << std::endl;
        rhocmpl_m[domain_m] = rhocmpl_m[domain_m] - total_charge/size;
        ////////////////////
        
        std::cout << "Rho sum after subtraction = " << sum(rhocmpl_m) << std::endl;

        //compute rhoHat and store in rhocmpl_m
        fft_m->transform("inverse", rhocmpl_m);

        std::cout << "sum of transform = " << sum(rhocmpl_m) << std::endl;

        // (2) compute Greens function in real space and transform to fourier space
        // Fields used to eliminate excess calculation in greensFunction()
        IField_t grnIField_m[3];

        // mesh and layout objects for rho_m
        Mesh_t *mesh_m = &(getMesh());
        FieldLayout_t *layout_m = &(getFieldLayout());

        //This loop stores in grnIField_m[i] the index of the ith dimension mirrored at the central axis. e.g. grnIField_m[0]=[(0 1 2 3 ... 3 2 1) ; (0 1 2 3 ... 3 2 1; ...)]
        for (int i = 0; i < 3; ++i) {
            grnIField_m[i].initialize(*mesh_m, *layout_m);
            grnIField_m[i][domain_m] = where(lt(domain_m[i], nr_m[i]/2),
                                             domain_m[i] * domain_m[i],
                                             (nr_m[i]-domain_m[i]) *
                                             (nr_m[i]-domain_m[i]));
        }
        Vector_t hrsq(hr_m * hr_m);
        SpecializedGreensFunction<3>::calculate(hrsq, grncmpl_m, grnIField_m, alpha);

        //transform G -> Ghat and store in grncmpl_m
        fft_m->transform("inverse", grncmpl_m);

        std::cout << "sum of ghat = " << sum(grncmpl_m) << std::endl;

        //multiply in fourier space and obtain PhiHat in rhocmpl_m
        rhocmpl_m *= grncmpl_m;

        std::cout << "sum of convolution = " << sum(rhocmpl_m) << std::endl;

        // (3) Backtransformation: compute potential field in real space and E=-Grad Phi
        //compute electrostatic potential Phi in real space by FFT PhiHat -> Phi and store it in rhocmpl_m
        fft_m->transform("forward", rhocmpl_m);

        // debug
        std::cout << "sum after inverse, before proper normalisation = " << sum(rhocmpl_m) << std::endl;

        //take only the real part and store in phi_m (has periodic bc instead of interpolation bc)
        phi_m = real(rhocmpl_m)*hr_m[0]*hr_m[1]*hr_m[2];
        
        // debug
        std::cout << "Rho sum after first solve = " << sum(phi_m) << std::endl;

        dumpVTKScalar(phi_m,this,1,"phi");

        //compute Electric field on the grid by -Grad(Phi) store in eg_m
        eg_m = -Grad1Ord(phi_m, eg_m);

        //interpolate the electric field to the particle positions
        EF.gather(eg_m, this->R,  IntrplCIC_t());

        //interpolate electrostatic potential to the particle positions
        Phi.gather(phi_m, this->R, IntrplCIC_t());
    }

    Vector_t getRmin() {
        return this->rmin_m;
    }
    Vector_t getRmax() {
        return this->rmax_m;
    }

    Vector_t get_hr() { return hr_m;}

    //private:
    BConds<double, Dim, Mesh_t, Center_t> bc_m;
    BConds<double, Dim, Mesh_t, Center_t> bcp_m;
    BConds<Vector_t, Dim, Mesh_t, Center_t> vbc_m;

    CxField_t rhocmpl_m;
    CxField_t grncmpl_m;

    Field_t rho_m;
    Field_t phi_m;

    VField_t eg_m;

    Vektor<int,Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmin_m;
    Vector_t rmax_m;
    Vector_t extend_l;
    Vector_t extend_r;
    Mesh_t *mesh_m;
    FieldLayout_t *layout_m;
    NDIndex<Dim> domain_m;
    NDIndex<Dim> lDomain_m;

    FFT_t *fft_m;

    double total_charge = 0;
};

struct Newton1D {
    double tol = 1e-12;
    int max_iter = 20;
    double pi = std::acos(-1.0); 

    double mu, sigma, u;

    Newton1D() {};

    Newton1D(const double& mu_, const double& sigma_, const double& u_)
        : mu(mu_)
        , sigma(sigma_)
        , u(u_) {}

    ~Newton1D() {}

    double f(double& x) {
        double F;
        F = std::erf((x-mu) / (sigma * std::sqrt(2.0))) - 2 *u + 1;
        return F;
    }

    double fprime(double& x) {
        double Fprime;
        Fprime = (1 / sigma) * std::sqrt(2 / pi) * std::exp(-0.5 * (std::pow(((x-mu) / sigma), 2)));
        return Fprime;
    }

    void solve(double& x) {
        int iterations = 0;
        while ((iterations < max_iter) && (std::fabs(f(x)) > tol)) {
            x = x - (f(x) / fprime(x));
            iterations += 1;
        }
    }
};

template<typename Particles>
void createParticlePenning(Particles & P, Vektor<double,3> extend_l, Vektor<double,3> extend_r, unsigned Nparts) {
    Inform msg("p3mPenning ");

    double pi = std::acos(-1.0);

    msg << "Initializing Cold Sphere" << endl;
    P->total_charge = -1562.5;
    unsigned Nparticle = Nparts;
    double qi = -1562.5/Nparticle;

    Vektor<double, 3> mu, sd, length;
    length = extend_r - extend_l;
    for (unsigned d = 0; d < 3; d++) {
        mu[d] = 0.5 * length[d];
    }
    sd[0] = 0.15 * length[0];
    sd[1] = 0.05 * length[1];
    sd[2] = 0.20 * length[2];

    Vektor<double, 3> minU, maxU;
    for (unsigned d = 0; d < 3; d++) {
        minU[d] = 0.5 * (1.0 + std::erf((extend_l[d] - mu[d]) / (sd[d] * std::sqrt(2)))); 
        maxU[d] = 0.5 * (1.0 + std::erf((extend_r[d] - mu[d]) / (sd[d] * std::sqrt(2)))); 
    }

    P->create(Nparticle);
    P->update();
    std::cout << "Create particles = " << P->getTotalNum() << std::endl;

    Vektor<double, 3> minR = 0.0;
    Vektor<double, 3> maxR = 0.0;

    for (unsigned i = 0; i < Nparticle; ++i) {
        Vektor<double, 3> pos;

        for (unsigned d = 0; d < 3; d++) {

            std::default_random_engine generator(42+d+i);
            std::uniform_real_distribution<double> unidistribution(minU[d], maxU[d]);
            auto uni = std::bind(unidistribution, generator);
            double u = uni();

            pos[d] = (std::sqrt(pi / 2) * (2 * u - 1)) * sd[d] + mu[d];
            Newton1D solver(mu[d], sd[d], u);
            solver.solve(pos[d]);

            if (pos[d] < minR[d])
                minR[d] = pos[d];

            if (pos[d] > maxR[d])
                maxR[d] = pos[d];
        }

        P->Q[i] = qi;
        P->R[i] = pos;

        std::cout << "i = " << i << ", qi = " << qi << ", pos = " << pos << std::endl;
    }

    std::cout << "Minimum position = " << minR << std::endl;
    std::cout << "Maximum position = " << maxR << std::endl;
}

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Vektor<int,Dim> nr;

    nr = Vektor<int,Dim>(std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3]));
    int Nparticle = std::atol(argv[4]);

    ///////// setup the initial layout ///////////////////////////////////////
    e_dim_tag decomp[Dim];
    Mesh_t *mesh;
    FieldLayout_t *FL;
    ChargedParticles<playout_t>  *P;

    NDIndex<Dim> domain;
    for (unsigned i=0; i<Dim; i++)
        domain[i] = Index(nr[i]+1);

    for (unsigned d=0; d < Dim; ++d)
        decomp[d] = PARALLEL;

    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);
    playout_t* PL = new playout_t(*FL, *mesh);

    double L = 20.0;
    Vektor<double,Dim> extend_l(0.0,0.0,0.0);
    Vektor<double,Dim> extend_r(L,L,L);

    msg << "INPUT: Nr = " << nr << ", Npart = " << Nparticle << endl;

    P = new ChargedParticles<playout_t>(PL, nr, extend_l, extend_r);
    createParticlePenning(P, extend_l, extend_r, Nparticle);
    Ippl::Comm->barrier();

    msg << "AFTER PARTICLES: " << endl;
    msg << "Number of particles = " << P->getTotalNum() << endl;
    msg << "Total charge Q      = " << P->total_charge << endl;
    msg << "Mesh Spacing        = " << P->get_hr() << endl;
    msg << "Domain              = " << P->getRmin() << P->getRmax() << endl;

    double alpha = 1e6;
    P->calculateGridForces(alpha);

    msg << "end" << endl;

    return 0;
}
