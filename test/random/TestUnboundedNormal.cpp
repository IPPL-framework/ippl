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
#include "Random/NormalDistribution.h"

const int Dim = 2;

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

using size_type = ippl::detail::size_type;

using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

KOKKOS_FUNCTION unsigned int doublefactorial(unsigned int n)
{
    if (n == 0 || n==1)
      return 1;
    return n*doublefactorial(n-2);
}

KOKKOS_FUNCTION double NormDistCentMom(double stdev, unsigned int p){
    // returns the central moment E[(x-\mu)^p] for Normal distribution function
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
    Inform csvout(NULL, "data/error_moments_unbounded_normal_dist.csv", Inform::APPEND);
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
        ippl::Vector<int, 2> nr   = {100, 100};
        size_type ntotal = 1000000;

        int seed = 42;

        GeneratorPool rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        double mu[Dim] = {0.0, -1.0};
        double sd[Dim] = {1.0, 0.5};

        view_type position("position", ntotal);
        Kokkos::parallel_for(
            ntotal, ippl::random::randn<double, Dim>(position, rand_pool64, mu, sd)
        );

        Kokkos::fence();
        ippl::Comm->barrier();

        const int P = 6; // number of moments to check, i.e. E[x^i] for i = 1,...,P
        double moms1_ref[P], moms2_ref[P];
        double moms1[P], moms2[P];

        moms1_ref[0] = mu[0];
        NormDistCentMoms(sd[0], P, moms1_ref);
        MomentsFromSamples(position, 0, ntotal, P, moms1);

        moms2_ref[0] = mu[1];
        NormDistCentMoms(sd[1], P, moms2_ref);
        MomentsFromSamples(position, 1, ntotal, P, moms2);

        WriteErrorInMoments(moms1, moms1_ref, P);
        WriteErrorInMoments(moms2, moms2_ref, P);

    }
    ippl::finalize();
    return 0;
}

