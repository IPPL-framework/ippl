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

// Testing the inverse transform sampling method for Normal Distribution on bounded domains
//     Example:
//     srun ./TestInverseTransformSamplingNormal --overallocate 2.0 --info 10

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
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"

const int Dim = 2;

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

using size_type = ippl::detail::size_type;

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

void MomentsFromSamples(view_type position, int d, int ntotal, int P, double *moms){
    double temp = 0.0;
    Kokkos::parallel_reduce("moments", position.extent(0),
                            KOKKOS_LAMBDA(const int i, double& valL) {
        double myVal = position(i)[d];
        valL += myVal;
        },
        Kokkos::Sum<double>(temp));
    Kokkos::fence();
    MPI_Reduce(&temp, &moms[0], 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
    moms[0] = moms[0]/ntotal;
    
    for(int p=1; p<P; p++){
        temp = 0.0;
        Kokkos::parallel_reduce("moments", position.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = pow(position(i)[d]-moms[0], p+1);
            valL += myVal;
            },
            Kokkos::Sum<double>(temp));
        Kokkos::fence();
        MPI_Reduce(&temp, &moms[p], 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
        moms[p] = moms[p]/(ntotal-1); // Bessel's correction
    }
}

void WriteErrorInMoments(double *moms, double *moms_ref, int P){
    Inform csvout(NULL, "data/error_moments_normal_dist.csv", Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);

    for(int i=0; i<P; i++){
        csvout << fabs( moms_ref[i] - moms[i] )  << endl;
    }
    ippl::Comm->barrier();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        ippl::Vector<int, 2> nr   = {20, 20};
        const unsigned int ntotal = 100000;
        
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

        unsigned int nlocal;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        // example of sampling normal in both dimensions
        const double mu1 = 1.0;
        const double sd1 = 0.8;
        const double mu2 = -2.0;
        const double sd2 = 0.4;
        const double par[4] = {mu1, sd1, mu2, sd2};
        using Dist_t = ippl::random::NormalDistribution<double, Dim>;
        using sampling_t = ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace, Dist_t>;

        Dist_t dist(par);
        sampling_t sampling(dist, rmax, rmin, rlayout, ntotal);
        nlocal = sampling.getLocalNum();
        view_type position("position", nlocal);
        sampling.generate(position, rand_pool64);
        
        const int P = 4;
        double moms1_ref[P], moms2_ref[P];
        double moms1[P], moms2[P];
        
        moms1_ref[0] = mu1;
        NormDistCentMoms(sd1, P, moms1_ref);
        MomentsFromSamples(position, 0, ntotal, P, moms1);
        
        moms2_ref[0] = mu2;
        NormDistCentMoms(sd2, P, moms2_ref);
        MomentsFromSamples(position, 1, ntotal, P, moms2);
        
        WriteErrorInMoments(moms1, moms1_ref, P);
        WriteErrorInMoments(moms2, moms2_ref, P);

    }
    ippl::finalize();
    return 0;
}

