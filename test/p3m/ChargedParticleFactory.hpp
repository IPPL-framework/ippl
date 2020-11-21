//
// File ChargedParticleFactory
//   These functions are used in the P3M applications to generate the initial
//   particle distribution.
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
#ifndef CHARGED_PARTICLE_FACTORY_
#define CHARGED_PARTICLE_FACTORY_


template<typename Particles>
void createParticleDistribution(Particles & P, std::string distribution, unsigned count, double qi, Vektor<double,3> extend_l,Vektor<double,3> extend_r, Vektor<double,3> source, double sphere_radius=1.,unsigned n_dummies=0 ) {

        enum DistType {UNIFORM, RANDOM, EVEN, MANUAL, LINE,TWOSTREAM, LANDAU};
        DistType type = UNIFORM;

        if(distribution == std::string("uniform"))
        {
                type = UNIFORM;
        }
        else if(distribution == std::string("random"))
        {
                type = RANDOM;
        }
        else if(distribution == std::string("even"))
        {
                type = EVEN;
        }
        else if(distribution == std::string("manual"))
        {
                type = MANUAL;
        }
        else if(distribution == std::string("line"))
        {
                type = LINE;
        }
        else if(distribution == std::string("twostream"))
        {
                type = TWOSTREAM;
        }
        else if(distribution == std::string("landau"))
        {
                type = LANDAU;
        }
        else {
                std::cout << "Unrecognized distribution " << distribution << std::endl;
        }

        if (P->singleInitNode())
        {
                size_t index = 0;
                if (type == UNIFORM) {
                        std::cout << "SAMPLING UNIFORM DISTRIBUTION" << std::endl;
                        double density = count*3./(sphere_radius*sphere_radius*sphere_radius*4*3.14159);
                        double pdist = std::pow(1./density,1/3.);

                        //SET TO FIXED DIST:
                        pdist = 4.*sphere_radius/15/4;

                        for (double x=-2.*sphere_radius; x<2*sphere_radius; x+=pdist){
                                for (double y=-2.*sphere_radius; y<2*sphere_radius; y+=pdist){
                                        for (double z=-2.*sphere_radius; z<2*sphere_radius; z+=pdist){
                                                Vektor<double, 3> pos(x,y,z);
                                                //msg << "checking particle pos = " << pos << endl;
                                                if (dot(source-pos,source-pos)<sphere_radius*sphere_radius){
                                                        //msg << " Particle accepted " << endl;
                                                        P->create(1);
                                                        P->Q[index]=qi;
                                                        P->R[index++]=pos;
                                                }
                                        }
                                }
                        }

                        if (n_dummies>0) {
                                n_dummies=0;
                                for (double x=-2.*sphere_radius; x<2*sphere_radius; x+=pdist){
                                        for (double y=-2.*sphere_radius; y<2*sphere_radius; y+=pdist){
                                                for (double z=-2.*sphere_radius; z<2*sphere_radius; z+=pdist){
                                                        Vektor<double, 3> pos(x,y,z);
                                                        //msg << "checking particle pos = " << pos << endl;
                                                        if (dot(source-pos,source-pos)>sphere_radius*sphere_radius){
                                                                //msg << " Particle accepted " << endl;
                                                                P->create(1);
                                                                P->Q[index]=0;
                                                                P->R[index++]=pos;
                                                                n_dummies++;
                                                        }
                                                }
                                        }
                                }
                        }

                }
                if (type == RANDOM) {
                        std::cout << "SAMPLING RANDOM DISTRIBUTION" << std::endl;
                        //create n particles within a sphere of radius R around source and charge 1/n
                        P->create(count);
                        std::default_random_engine generator;
                        //uniform distributed between 0 and 1
                        std::uniform_real_distribution<double> unidistribution(0,1);
                        //std::uniform_real_distribution<double> unidistribution(source[0]-sphere_radius, source[0]+sphere_radius);
                        auto uni = std::bind(unidistribution, generator);
                        //normal distribution
                        std::normal_distribution<double> normdistribution(0,1.0);

                        auto normal = std::bind(normdistribution, generator);


                        //use rejection method to place particles uniformly in sphere
                        double density = count*3./(sphere_radius*sphere_radius*sphere_radius*4*3.14159);
                        double pdist = std::pow(1./density,1/3.);

                        for (unsigned i = 0; i<count; ++i) {

                                Vektor<double, 3> X(normal(),normal(),normal());
                                double U = uni();
                                Vektor<double, 3> pos = source + sphere_radius*std::pow(U,1./3.)/std::sqrt(dot(X,X))*X;
                                //BEGIN reject close particles
                                bool reject = 0;
                                for (unsigned k=0; k<index; ++k) {
                                        if (sqrt(dot(P->R[k]-pos,P->R[k]-pos))<0.5*pdist)
                                                reject = 1;
                                }
                                if (!reject){
                                        Vektor<double, 3> vel(1.,1.,1.); //normal velocity distribution in z dir
                                        //Vektor<double, 3> vel(0,0,0); //normal velocity distribution in z dir
                                        P->Q[index]=-qi;
                                        P->m[index]=1;
                                        P->total_charge+=qi;
                                        P->v[index]=vel;
                                        P->R[index++]=pos;
                                }
                                else {
                                        i--;
                                }
                        }
                        //place particles with 0 charge outside the sphere

                        P->create(n_dummies);
                        std::uniform_real_distribution<double> o_unidistribution(-2.*sphere_radius,2.*sphere_radius);
                        auto o_uni = std::bind(o_unidistribution, generator);

                        for (unsigned i = 0; i<n_dummies; ++i) {

                                //Vektor<double, Dim> X(normal(),normal(),normal());
                                bool check_out_of_sphere = 1;
                                while(check_out_of_sphere){
                                        Vektor<double, 3> pos(o_uni(),o_uni(),o_uni());
                                        if (dot(source-pos,source-pos)>sphere_radius*sphere_radius){
                                                std::cout << "dummy placed " << std::endl;
                                                //P->Q[index]=1./count;
                                                P->Q[index]=0;
                                                P->R[index++]=pos;
                                                check_out_of_sphere=0;
                                        }
                                }
                        }
                }

                if (type == EVEN) {
                        std::cout << "SAMPLING EVEN DISTRIBUTION" << std::endl;
                        P->create(count);
                        std::default_random_engine generator;
                        //uniform distributed between 0 and 1
                        std::uniform_real_distribution<double> unidistribution(extend_l[0],extend_r[0]);

                        std::normal_distribution<double> normdistribution(0,1.0);
                        auto normal = std::bind(normdistribution, generator);

                        //std::uniform_real_distribution<double> unidistribution(source[0]-sphere_radius, source[0]+sphere_radius);
                        auto uni = std::bind(unidistribution, generator);

                        for (unsigned i = 0; i<count; ++i) {
                                Vektor<double, 3> X(uni(),uni(),uni());
                                Vektor<double, 3> pos = X;
                                Vektor<double, 3> vel(normal(),normal(),normal());
                                //Vektor<double, 3> vel(0,0,0);
                                P->Q[index]=qi;
                                P->R[index++]=pos;
                        }
                }

                if (type == MANUAL) {

                        std::cout << "SAMPLING MANUAL DISTRIBUTION" << std::endl;
                        for (unsigned i = 0; i<count; ++i) {
                                std::cout << "please give coordinates of particle " << i << std::endl;
                                double x,y,z;
                                std::cin >> x; std::cin >> y; std::cin >>z;
                                Vektor<double, 3> pos(0,0,0);
                                pos[0]=x; pos[1]=y; pos[2]=z;
                                std::cout << "pos = " << pos << std::endl;
                                P->create(1);
                                P->Q[index]=qi;
                                Vektor<double, 3> vel(0,0,0.1); //normal velocity distribution in z dir
                                P->v[index]=vel;
                                P->R[index++]=pos;
                        }

                }

                if (type == LINE) {
                        std::cout << "SAMPLING LINE DISTRIBUTION" << std::endl;
                        P->create(count);
                        std::default_random_engine generator;
                        //uniform distributed between 0 and 1
                        std::uniform_real_distribution<double> unidistribution(extend_l[0],extend_r[0]);
                        std::normal_distribution<double> normaldist(1,.5);

                        auto uni = std::bind(unidistribution, generator);
                        auto normal = std::bind(normaldist, generator);

                        for (unsigned i = 0; i<count; ++i) {
                                Vektor<double, 3> X(0,0,uni()); //place all particles on z axis
                                Vektor<double, 3> vel(0,0,normal()); //normal velocity distribution in z dir
                                Vektor<double, 3> pos = X;
                                P->Q[index]=qi;
                                //P->v[index]=vel;
                                //P->ID[index]=i;
                                //std::cout << P->ID[index] << std::endl;
                                P->R[index++]=pos;
                        }
                }
        }


        P->update();

}

template<typename Particles>
void createParticleDistributionTwoStream(Particles & P,
                                         Vektor<double,3> extend_l,
                                         Vektor<double,3> extend_r,
                                         Vektor<int,3> Nx,
                                         Vektor<int,3> Nv,
                                         Vektor<double,3>Vmax,
                                         double alpha = 0.05,
                                         double kk = 0.5)
{
    Vektor<double,3>L = extend_r - extend_l;
    //Vektor<double,3>Nx(2,2,16);
    //Vektor<double,3>Nv(4,4,32);
    //Vektor<double,3>Vmax(6,6,6);
    Vektor<double,3>Vmin = - Vmax;
    Vektor<double,3> hx = L / Nx;
    Vektor<double,3> hv = ( Vmax - Vmin ) / Nv;

    std::cout << "hx = " << hx << std::endl;
    std::cout << "hv = " << hv << std::endl;
    double thresh = 1e-12;
    P->total_charge=0;


    Vektor<double,3>pos = extend_l;
    std::cout << "pos = " << pos << std::endl;
    std::cout << "Initializing TwoStream instability" << std::endl;
    size_t index = 0;
    if ( P->singleInitNode() ) {
        for (int i = 0; i < Nx[0]; ++i) {
            for (int j = 0; j < Nx[1]; ++j) {
                for (int k = 0; k < Nx[2]; ++k) {
                    pos = Vektor<double,3> ( (.5 + i) * hx[0] + extend_l[0],
                                             (.5 + j) * hx[1] + extend_l[1],
                                             (.5 + k) * hx[2] + extend_l[2]);
                    //std::cout << "pos = " << pos << std::endl;
                    for (int iv = 0; iv < Nv[0]; ++iv) {
                        for (int jv = 0; jv < Nv[1]; ++jv) {
                            for (int kv = 0; kv < Nv[2]; ++kv) {
                                //double vx = (iv+.5)*hv+vmin; double vy = (jv+.5)*hv+vmin; double vz = (kv+.5)*hv+vmin;
                                Vektor<double,3>vel = Vektor<double,3>(iv + .5,
                                                                       jv + .5,
                                                                       kv + .5) * hv + Vmin;
                                double v2 = vel[0] * vel[0] +
                                            vel[1] * vel[1] +
                                            vel[2] * vel[2];

                                double f = ( 1. / ( 30 * M_PI ) ) *
                                           exp( -0.5 * v2 ) *
                                           ( 1. + alpha * cos( kk * pos[2] ) ) *
                                           ( 1.0 + 0.5 * vel[2] * vel[2]);

                                //std::cout << "f = " << f << std::endl;
                                double m = hx[0] * hv[0] *
                                           hx[1] * hv[1] *
                                           hx[2] * hv[2] * f;

                                if ( m > thresh ) {
                                    double q =-m;
                                    P->create(1);
                                    P->Q[index] = q;
                                    P->m[index]=m;
                                    P->total_charge+=q;
                                    P->v[index]= vel;
                                    P->R[index]=pos;
                                    index++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dumpParticlesCSV(P,0);
    P->update();
}


template<typename Particles>
void createParticleDistributionLandau(Particles & P,
                                      Vektor<double,3> extend_l,
                                      Vektor<double,3> extend_r,
                                      Vektor<int,3> Nx,
                                      Vektor<int,3> Nv,
                                      Vektor<double,3>Vmax,
                                      double alpha = 0.05,
                                      double kk = 0.5)
{
    Vektor<double,3>L = extend_r - extend_l;
    Vektor<double,3>Vmin = - Vmax;

    Vektor<double,3> hx = L / Nx;
    Vektor<double,3> hv = ( Vmax - Vmin) / Nv;

    std::cout << "hx = " << hx << std::endl;
    std::cout << "hv = " << hv << std::endl;

    double thresh = 1e-12;
    P->total_charge=0;

    Vektor<double,3>pos = extend_l;
    std::cout << "pos = " << pos << std::endl;
    std::cout << "Initializing Landau Damping" << std::endl;
    size_t index = 0;
    if ( P->singleInitNode() ) {
        for (int i = 0; i < Nx[0]; ++i) {
            for (int j = 0; j < Nx[1]; ++j) {
                for (int k = 0; k < Nx[2]; ++k) {
                    pos = Vektor<double,3> ( (.5 + i) * hx[0] + extend_l[0],
                                             (.5 + j) * hx[1] + extend_l[1],
                                             (.5 + k) * hx[2] + extend_l[2]
                                           );

                    for (int iv = 0; iv < Nv[0]; ++iv) {
                        for (int jv = 0; jv < Nv[1]; ++jv) {
                            for (int kv = 0; kv < Nv[2]; ++kv) {
                                Vektor<double,3>vel = Vektor<double,3>(iv + .5,
                                                                       jv + .5,
                                                                       kv + .5) * hv + Vmin;
                                double v2 = vel[0] * vel[0] +
                                            vel[1] * vel[1] +
                                            vel[2] * vel[2];

                                double f = ( 1. / ( 2. * M_PI * sqrt( 2.* M_PI ) ) ) *
                                           exp( -0.5 *v2 ) *
                                           ( 1. + alpha * ( cos( kk * pos[2] ) +
                                                            cos( kk * pos[1] ) +
                                                            cos( kk * pos[0] )
                                                          )
                                           );

                                double m = hx[0] * hv[0] *
                                           hx[1] * hv[1] *
                                           hx[2] * hv[2] * f;

                                if ( m > thresh ) {
                                    double q = -m;
                                    P->create(1);
                                    P->Q[index] = q;
                                    P->m[index] = m;
                                    P->total_charge+=q;
                                    P->v[index]= vel;
                                    P->R[index++]=pos;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    P->update();
}


template<typename Particles>
void createParticleDistributionRecurrence(Particles & P,
                                          Vektor<double,3> extend_l,
                                          Vektor<double,3> extend_r,
                                          Vektor<int,3> Nx,
                                          Vektor<int,3> Nv,
                                          Vektor<double,3>Vmax,
                                          double alpha = 0.05,
                                          double kk = 0.5)
{
    Vektor<double,3>L = extend_r - extend_l;
    Vektor<double,3>Vmin = - Vmax;

    Vektor<double,3> hx = L / Nx;
    Vektor<double,3> hv = ( Vmax - Vmin ) / Nv;

    std::cout << "hx = " << hx << std::endl;
    std::cout << "hv = " << hv << std::endl;

    double thresh = 1e-12;
    P->total_charge=0;

    Vektor<double,3>pos = extend_l;
    std::cout << "pos = " << pos << std::endl;
    std::cout << "Initializing free streaming (recurrence time)" << std::endl;
    size_t index = 0;
    if ( P->singleInitNode() ) {
        for (int i = 0; i < Nx[0]; ++i) {
            for (int j = 0; j < Nx[1]; ++j) {
                for (int k = 0; k < Nx[2]; ++k) {
                    pos = Vektor<double,3> ( (.5 + i) * hx[0] + extend_l[0],
                                             (.5 + j) * hx[1] + extend_l[1],
                                             (.5 + k) * hx[2] + extend_l[2]);

                    for (int iv = 0; iv < Nv[0]; ++iv) {
                        for (int jv = 0; jv < Nv[1]; ++jv) {
                            for (int kv = 0; kv < Nv[2]; ++kv) {
                                Vektor<double,3>vel = Vektor<double,3>(iv + .5,
                                                                       jv + .5,
                                                                       kv + .5) * hv + Vmin;

                                double v2 = vel[0] * vel[0] +
                                            vel[1] * vel[1] +
                                            vel[2] * vel[2];

                                //Free-streaming:
                                double f = alpha *
                                           ( 1. / ( 2. * M_PI * sqrt( 2. * M_PI ) ) ) *
                                           exp( -0.5 * v2 ) *
                                           (
                                               cos( kk * pos[2] ) +
                                               cos( kk * pos[1] ) +
                                               cos( kk * pos[0] )
                                           );

                                double m = hx[0] * hv[0] *
                                           hx[1] * hv[1] *
                                           hx[2] * hv[2] * f;

                                if ( m > thresh ) {
                                    double q = -m;
                                    P->create(1);
                                    P->Q[index] = q;
                                    P->m[index] = m;
                                    P->total_charge+=q;
                                    P->v[index]= vel;
                                    P->R[index++]=pos;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    P->update();
}


template<typename Particles>
void createParticleDistributionMicrobunching(Particles & P, unsigned seedID=0 ) {
        std::cout << "Initializing Microbunching" << std::endl;
        //generate seed sequence
        std::seed_seq seq{1,2,3,4,5};
        std::vector<std::uint32_t> seeds(50);
        seq.generate(seeds.begin(), seeds.end());


        std::default_random_engine generator(seeds[seedID]);
        std::cout << "The seed value chosen is =" << seeds[seedID] << std::endl;
        std::cout << std::setprecision(20);
        P->total_charge=0;
        //Number of particles is set fixed here, TODO change this!
        //unsigned Nparticle = 2.49835e6;

        unsigned Nparticle=P->Npart;
        std::cout << "charge per particle [e] = " << P->q << std::endl;
        //spread of the second order moment
        double sigmaXprime = P->emittance/(P->sigmaX*P->gamma*P->beta0);
        std::cout << "sigma_x_prime [rad] = " << sigmaXprime << std::endl;
        //the corresponding momentum spread is
        double sigmaPx = sigmaXprime*P->gamma*P->m0*P->beta0;
        std::cout << "sigma_px [MeV/c] = " << sigmaPx << std::endl;
        //the momenta are normally distributed with std deviation sigma_px
        std::normal_distribution<double> normdistpx(0,sigmaPx);
        std::normal_distribution<double> normdistpy(0,sigmaPx);

        //longitudinal momenta
        std::cout << "gamma=" << P->gamma << std::endl;
        std::cout << "deltagamma=" << P->deltagamma << std::endl;
        std::cout << "sigmaP=" << P->deltagamma/P->gamma << std::endl;
        //std::normal_distribution<double> normdistdeltagamma(0,P->deltagamma);
        std::normal_distribution<double> normdistdeltagamma(0,P->deltagamma/P->gamma);

        //setup rng for uniform distributions of particles in x,y,z direction
        std::uniform_real_distribution<double> unidistx(P->extend_l[0],P->extend_r[0]);
        std::uniform_real_distribution<double> unidisty(P->extend_l[1],P->extend_r[1]);
        std::uniform_real_distribution<double> unidistz(P->extend_l[2],P->extend_r[2]);

        //double k = 2.*M_PI/P->lambda;
        //std::complex<double> imag = -1.;
        //imag = sqrt(imag);
        //std::cout << "imag=" << imag << std::endl;
std::cout<< P->beta0 << std::endl;
        double pz,deltagamma_dist;
        if (P->singleInitNode()) {
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        deltagamma_dist=normdistdeltagamma(generator);
                        //deltapz in the lab frame in units of [MeV/c] becomes
                        pz = deltagamma_dist*P->beta0*P->m0;
                        //std::cout <<"pzprime= " << pzPrime << std::endl;
                        //do lorentz transformation
                        double z = unidistz(generator);
                        //Vektor<double, 3> momentumPrime(P->m0*normdistpx(generator),P->m0*normdistpy(generator),deltapz/P->gamma);
                        Vektor<double, 3> momentumPrime(normdistpx(generator),normdistpy(generator),pz/P->gamma);
                        Vektor<double, 3> posPrime(unidistx(generator),unidisty(generator),P->gamma*z);
                        P->Q[i]=-P->q;
                        P->m[i]= P->m0;
                        P->total_charge+=-P->q;
                        P->p[i]=momentumPrime;
                        P->R[i]=posPrime;
/*
                        for (int j=0;j<10;++j){
                                //std::cout << std::exp(-imag*k) << P->b0[j] << std::endl;
                                P->b0[j]+=std::exp(-imag*k*(z+P->R[i][0]*sin(P->theta[j])));
                        }
*/
                }
        }
        //transform computational domain from lab to beamframe
//      for (int j=0;j<10;++j)
//              P->b0[j]/=Nparticle;
        P->extend_r[2]=P->gamma*P->extend_r[2];
        //dumpParticlesCSVp(P,0);
        P->update();
}


template<typename Particles>
void createParticleDistributionEquiPart(Particles & P, Vektor<double,3> /*extend_l*/, Vektor<double,3> /*extend_r*/, double beam_length, double part_density,double qi, double mi, int seed=0) {
        std::cout << "Initializing Equipartitioning" << std::endl;
        P->total_charge=0;
        const double c = 299792458000;
        unsigned Nparticle = beam_length*beam_length*beam_length*part_density;
        std::cout << "Number of particles = " << Nparticle << std::endl;
        //the momenta are normally distributed with std deviation sigma_px

        std::default_random_engine generator(seed);
        //setup rng for uniform distributions of particles in x,y,z direction
        std::uniform_real_distribution<double> unidist_spacial(-beam_length/2.,beam_length/2.);
        std::uniform_real_distribution<double> vel_trans(-c*0.0015,c*0.0015);
        //std::uniform_real_distribution<double> vel_trans(-sqrt(3),sqrt(3));
        //std::uniform_real_distribution<double> vel_trans(-5.00346e-14,5.00346e-14);
        std::uniform_real_distribution<double> vel_long(-c*0.003,c*0.003);
        //std::uniform_real_distribution<double> vel_long(-sqrt(3),sqrt(3));
        //std::uniform_real_distribution<double> vel_long(-1.00069e-13,1.00069e-13);

        if (P->singleInitNode()) {
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        Vektor<double, 3> pos(unidist_spacial(generator),unidist_spacial(generator),unidist_spacial(generator));
                        Vektor<double, 3> vel(vel_trans(generator),vel_trans(generator),vel_long(generator));
                        P->Q[i]=-qi;
                        P->m[i]= mi;
                        P->total_charge+=-qi;
                        P->v[i]=vel;
                        P->R[i]=pos;
                }
        }
        P->update();
}

template<typename Particles>
void createParticleDistributionEquiPartSphere(Particles & P, Vektor<double,3> /*extend_l*/, Vektor<double,3> /*extend_r*/, double beam_length, unsigned Nparts,double qi, double mi, int seed=0) {
        std::cout << "Initializing Equipartitioning Sphere" << std::endl;
        P->total_charge=0;
        const double c = 299792458000;
        unsigned Nparticle = Nparts;
        double sphere_radius=beam_length/2.;
        std::cout << "Number of particles = " << Nparticle << std::endl;
        //the momenta are normally distributed with std deviation sigma_px

        std::default_random_engine generator(seed);
        std::normal_distribution<double> normdistribution(0,1.0);
        auto normal = std::bind(normdistribution, generator);
        std::uniform_real_distribution<double> unidistribution(0,1);
        auto uni = std::bind(unidistribution, generator);

        Vektor<double, 3> source(0,0,0);
        //setup rng for uniform distributions of particles in x,y,z direction
        std::uniform_real_distribution<double> vel_trans(-c*0.0015,c*0.0015);
        //std::uniform_real_distribution<double> vel_trans(-sqrt(3),sqrt(3));
        //std::uniform_real_distribution<double> vel_trans(-5.00346e-14,5.00346e-14);
        std::uniform_real_distribution<double> vel_long(-c*0.003,c*0.003);
        //std::uniform_real_distribution<double> vel_long(-sqrt(3),sqrt(3));
        //std::uniform_real_distribution<double> vel_long(-1.00069e-13,1.00069e-13);

        if (P->singleInitNode()) {
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        Vektor<double, 3> X(normal(),normal(),normal());
                        double U = uni();
                        Vektor<double, 3> pos = source + sphere_radius*std::pow(U,1./3.)/std::sqrt(dot(X,X))*X;

                        Vektor<double, 3> vel(vel_trans(generator),vel_trans(generator),vel_long(generator));
                        P->Q[i]=-qi;
                        P->m[i]= mi;
                        P->total_charge+=-qi;
                        P->v[i]=vel;
                        P->R[i]=pos;
                }
        }
        P->update();
}


template<typename Particles>
void createParticleDistributionHeating(Particles & P, Vektor<double,3> /*extend_l*/, Vektor<double,3> /*extend_r*/, double beam_radius, unsigned Nparts,double qi, double mi) {
    Inform msg("p3m3dHeating ");

    msg << "Initializing Cold Sphere" << endl;
        P->total_charge=0;
        unsigned Nparticle = Nparts;

        //the momenta are normally distributed with std deviation sigma_px

        std::default_random_engine generator(0);
        std::normal_distribution<double> normdistribution(0,1.0);
        auto normal = std::bind(normdistribution, generator);
        std::uniform_real_distribution<double> unidistribution(0,1);
        auto uni = std::bind(unidistribution, generator);

        Vektor<double, 3> source(0,0,0);

        if (P->singleInitNode()) {
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        Vektor<double, 3> X(normal(),normal(),normal());
                        double U = uni();
                        Vektor<double, 3> pos = source + beam_radius*std::pow(U,1./3.)/std::sqrt(dot(X,X))*X;

                        Vektor<double, 3> vel(0,0,0);
                        P->Q[i]=-qi;
                        P->m[i]= mi;
                        P->total_charge+=-qi;
                        P->v[i]=vel;
                        P->R[i]=pos;
                }
        }
        P->update();
}




template<typename Particles>
void createParticleDistributionPerformance(Particles & P, Vektor<double,3> extend_l, Vektor<double,3> extend_r, unsigned Nparts,double qi, double mi, double vMin, double vMax) {
        P->total_charge=0;
        unsigned Nparticle = Nparts;
        //the momenta are normally distributed with std deviation sigma_px

        std::default_random_engine generator(0);
        std::uniform_real_distribution<double> unidistributionVel(vMin,vMax);
        std::uniform_real_distribution<double> unidistributionX(extend_l[0],extend_r[0]);
        std::uniform_real_distribution<double> unidistributionY(extend_l[1],extend_r[1]);
        std::uniform_real_distribution<double> unidistributionZ(extend_l[2],extend_r[2]);
        auto uniVel = std::bind(unidistributionVel, generator);

        if (P->singleInitNode()) {
                P->create(Nparticle);
                for (unsigned i = 0; i<Nparticle; ++i) {
                        Vektor<double, 3> pos(unidistributionX(generator),unidistributionY(generator),unidistributionZ(generator));
                        Vektor<double, 3> vel(uniVel(),uniVel(),uniVel());
                        P->Q[i]=-qi;
                        P->m[i]= mi;
                        P->total_charge+=-qi;
                        P->v[i]=vel;
                        P->R[i]=pos;
                }
        }
        P->update();
}

#endif
