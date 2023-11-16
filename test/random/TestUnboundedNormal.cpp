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
#include "Random/Randn.h"

const int Dim = 2;

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

using size_type = ippl::detail::size_type;

using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

KOKKOS_FUNCTION unsigned int get_double_factorial(unsigned int n)
{
    if (n == 0 || n==1)
      return 1;
    return n*get_double_factorial(n-2);
}

KOKKOS_FUNCTION double get_norm_dist_cent_mom(double stdev, unsigned int p){
    // returns the central moment E[(x-\mu)^p] for Normal distribution function
    if(p%2==0){
        return pow(stdev, p)*get_double_factorial(p-1);
    }
    else{
	return 0.;
    }
}

KOKKOS_FUNCTION void get_norm_dist_cent_moms(double stdev, const int P, double *moms_p){
    for(int p=1; p<P; p++){
        moms_p[p] = get_norm_dist_cent_mom(stdev, p+1);
    }
}

void get_moments_from_samples(view_type position, int d, int ntotal, const int P, double *moms_p){
    int d_ = d;
    int ntotal_ = ntotal;
    double temp = 0.0;
    Kokkos::parallel_reduce("moments", position.extent(0),
                            KOKKOS_LAMBDA(const int i, double& valL) {
        double myVal = position(i)[d_];
        valL += myVal;
    }, Kokkos::Sum<double>(temp));

    double mean = temp / ntotal_;
    moms_p[0] = mean;

    for (int p = 1; p < P; p++) {
        temp = 0.0;
        Kokkos::parallel_reduce("moments", position.extent(0),
                                KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = pow(position(i)[d_] - mean, p + 1);
            valL += myVal;
        }, Kokkos::Sum<double>(temp));
        moms_p[p] = temp / ntotal_;
    }
}

void write_error_in_moments(double *moms_p, double *moms_ref_p, int P){
    Inform csvout(NULL, "data/error_moments_normal_dist.csv", Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);
    for(int i=0; i<P; i++){
        csvout << moms_ref_p[i] << " " << moms_p[i] << " " << fabs(moms_ref_p[i] - moms_p[i]) << endl;
    }
    ippl::Comm->barrier();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
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
        get_norm_dist_cent_moms(sd[0], P, moms1_ref);
        get_moments_from_samples(position, 0, ntotal, P, moms1);

        moms2_ref[0] = mu[1];
        get_norm_dist_cent_moms(sd[1], P, moms2_ref);
        get_moments_from_samples(position, 1, ntotal, P, moms2);

        write_error_in_moments(moms1, moms1_ref, P);
        write_error_in_moments(moms2, moms2_ref, P);

    }
    ippl::finalize();
    return 0;
}

