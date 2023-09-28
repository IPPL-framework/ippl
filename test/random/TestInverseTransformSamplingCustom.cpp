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
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "Utility/IpplTimings.h"
#include "Ippl.h"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"

const int Dim = 2;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

const double pi    = Kokkos::numbers::pi_v<double>;

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

struct custom_cdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params) const {
           if(d==0){
               return ippl::random::normal_cdf_func<double>(x, params[0], params[1]);
           }
           else{
               return x + (params[2] / params[3]) * Kokkos::sin(params[3] * x);
           }
       }
};

struct custom_pdf{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params) const {
           if(d==0){
               return ippl::random::normal_pdf_func<double>(x, params[0], params[1]);
           }
           else{
               return  1.0 + params[2] * Kokkos::cos(params[3] * x);
           }
       }
};

struct custom_estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params) const {
            if(d==0){
                return ippl::random::normal_estimate_func<double>(u, params[0], params[1]);
            }
            else{
                return u;
            }
        }
};

KOKKOS_FUNCTION unsigned int doublefactorial(unsigned int n)
{
    if (n == 0 || n==1)
      return 1;
    return n*doublefactorial(n-2);
}

KOKKOS_FUNCTION double NormDistCentMom(double stdev, unsigned int p){
    if(p%2==0){
        return pow(stdev, p)*doublefactorial(p-1);
    }
    else{
        return 0.;
    }
}

KOKKOS_FUNCTION void NormDistCentMoms(double stdev, const int P, double *moms){
    for(int p=1; p<P; p++){
        moms[p] = NormDistCentMom(stdev, p+1);
    }
}

void MomentsFromSamples(view_type position, int d, int ntotal, const int P, double *moms){
    double temp = 0.0;
    Kokkos::parallel_reduce("moments", position.extent(0),
                            KOKKOS_LAMBDA(const int i, double& valL) {
        double myVal = position(i)[d];
        valL += myVal;
        },
	Kokkos::Sum<double>(temp));
    Kokkos::fence();
    double gtemp = 0.0;
    MPI_Reduce(&temp, &gtemp, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
    double mean = gtemp/ntotal;
    moms[0] = mean;
    for(int p=1; p<P; p++){
        temp = 0.0;
        Kokkos::parallel_reduce("moments", position.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = pow(position(i)[d]-mean, p+1);
            valL += myVal;
            },
            Kokkos::Sum<double>(temp));
        Kokkos::fence();
        MPI_Reduce(&temp, &gtemp, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        moms[p] = gtemp/(ntotal-1); // Bessel's correction
    }
}

void WriteErrorInMoments(double *moms, double *moms_ref, int P){
    Inform csvout(NULL, "data/error_moments_custom_dist.csv", Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);
    for(int i=0; i<P; i++){
        csvout << moms_ref[i] << " " << moms[i] << " " << fabs(moms_ref[i] - moms[i]) << endl;
    }
    ippl::Comm->barrier();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        ippl::Vector<int, 2> nr   = {20, 20};
        const unsigned int ntotal = 1000000;

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        ippl::Vector<double, Dim> rmin   = -4.;
        ippl::Vector<double, Dim> rmax   = 4.;
        ippl::Vector<double, Dim> length = rmax - rmin;

        ippl::Vector<double, Dim> hr     = length / nr;
        ippl::Vector<double, Dim> origin = rmin;

        const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);

        ippl::FieldLayout<Dim> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, Dim, Mesh_t> rlayout(fl, mesh);

        int seed = 42;
        using size_type = ippl::detail::size_type;
        unsigned int nlocal;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));
        
        // example of sampling normal/uniform in one and harmonic in another with custom functors
        const int DimP = 4;
        const double mu = 1.0;
        const double sd = 0.9;
        const double parH[DimP] = {mu, sd, 0.5, 2.*pi/(rmax[1]-rmin[1])*4.0};
        
        using DistH_t = ippl::random::Distribution<double, Dim, DimP, custom_pdf, custom_cdf, custom_estimate>;
        using samplingH_t = ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace, DistH_t>;

        DistH_t distH(parH);
        samplingH_t samplingH(distH, rmax, rmin, rlayout, ntotal);
        nlocal = samplingH.getLocalNum();
        view_type positionH("positionH", nlocal);
        samplingH.generate(positionH, rand_pool64);
        
        const int P = 6;
        double moms1_ref[P];
        double moms1[P];
        
        // compute error in moments of 1st dimension
        moms1_ref[0] = mu;
        NormDistCentMoms(sd, P, moms1_ref);
        MomentsFromSamples(positionH, 0, ntotal, P, moms1);
        WriteErrorInMoments(moms1, moms1_ref, P);
        
        // next, compute error in moments of 2nd dimension
        //double moms2_ref[P];
        //double moms2[P];
    }
    ippl::finalize();
    return 0;
}

