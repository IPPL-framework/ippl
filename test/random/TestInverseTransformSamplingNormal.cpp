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
    int d_ = d;
    int ntotal_ = ntotal;
    double temp = 0.0;
    Kokkos::parallel_reduce("moments", position.extent(0),
                            KOKKOS_LAMBDA(const int i, double& valL) {
        double myVal = position(i)[d_];
        valL += myVal;
    }, Kokkos::Sum<double>(temp));

    double mean = temp / ntotal_;
    moms[0] = mean;

    for (int p = 1; p < P; p++) {
        temp = 0.0;
        Kokkos::parallel_reduce("moments", position.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = pow(position(i)[d_] - mean, p + 1);
            valL += myVal;
        }, Kokkos::Sum<double>(temp));
        moms[p] = temp / ntotal_;
    }
}

void WriteErrorInMoments(double *moms, double *moms_ref, int P){
    Inform csvout(NULL, "data/error_moments_normal_dist.csv", Inform::APPEND);
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
        Inform m("test ITS normal");

        ippl::Vector<int, 2> nr   = {100, 100};
        size_type ntotal = 100000;

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<double, Dim> rmin   = -4.;
        ippl::Vector<double, Dim> rmax   = 4.;
        ippl::Vector<double, Dim> length = rmax - rmin;
        ippl::Vector<double, Dim> hr     = length / nr;
        ippl::Vector<double, Dim> origin = rmin;

        const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);

        ippl::FieldLayout<Dim> fl(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);

        ippl::detail::RegionLayout<double, Dim, Mesh_t> rlayout(fl, mesh);

        int seed = 42;

        GeneratorPool rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        const double mu1 = 0.1;
        const double sd1 = 0.5;
        const double mu2 = -0.1;
        const double sd2 = 1.0;
        const double par[4] = {mu1, sd1, mu2, sd2};
        using Dist_t = ippl::random::NormalDistribution<double, Dim>;
        using sampling_t = ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace, Dist_t>;

        Dist_t dist(par);
        sampling_t sampling(dist, rmax, rmin, rlayout, ntotal);
        ntotal = sampling.getLocalNum();
        view_type position("position", ntotal);
        sampling.generate(position, rand_pool64);

        const int P = 6; // number of moments to check, i.e. E[x^i] for i = 1,...,P
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
