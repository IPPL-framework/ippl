// Testing the customized range policy of the inverse transform sampling method for Normal
// Distribution
//     Example:
//     srun ./TestInverseTransformSamplingSpecificRange --overallocate 2.0 --info 10

#include "Ippl.h"

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

#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"

const int Dim = 2;

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using size_type = ippl::detail::size_type;

using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

KOKKOS_FUNCTION unsigned int get_double_factorial(unsigned int n) {
    if (n == 0 || n == 1)
        return 1;
    return n * get_double_factorial(n - 2);
}

KOKKOS_FUNCTION double get_norm_dist_cent_mom(double stdev, unsigned int p) {
    // returns the central moment E[(x-\mu)^p] for Normal distribution function
    if (p % 2 == 0) {
        return pow(stdev, p) * get_double_factorial(p - 1);
    } else {
        return 0.;
    }
}

KOKKOS_FUNCTION void get_norm_dist_cent_moms(double stdev, const int P, double* moms_p) {
    for (int p = 1; p < P; p++) {
        moms_p[p] = get_norm_dist_cent_mom(stdev, p + 1);
    }
}

void get_moments_from_samples(view_type position, int d, int start, int end, int P,
                              double* moms_p) {
    int d_      = d;
    double temp = 0.0, mean = 0.0;
    std::vector<double> locmoms(P, 0.0);

    int gNpart = 0, locNpart = end - start;

    for (int p = 0; p < P; p++) {
        moms_p[p] = 0.0;
    }

    Kokkos::parallel_reduce(
        "moments", Kokkos::RangePolicy<>(start, end),
        KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = position(i)[d_];
            valL += myVal;
        },
        Kokkos::Sum<double>(temp));
    Kokkos::fence();

    locmoms[0] = temp;

    MPI_Allreduce(&temp, &mean, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
    MPI_Allreduce(&locNpart, &gNpart, 1, MPI_INT, MPI_SUM, ippl::Comm->getCommunicator());
    ippl::Comm->barrier();

    mean = mean / gNpart;

    for (int p = 1; p < P; p++) {
        temp = 0.0;
        Kokkos::parallel_reduce(
            "moments", Kokkos::RangePolicy<>(start, end),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = pow(position(i)[d_] - mean, p + 1);
                valL += myVal;
            },
            Kokkos::Sum<double>(temp));
        Kokkos::fence();

        locmoms[p] = temp;
    }

    MPI_Allreduce(locmoms.data(), moms_p, P, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
    ippl::Comm->barrier();

    for (int p = 0; p < P; p++) {
        moms_p[p] = moms_p[p] / gNpart;
    }
}

void write_error_in_moments(double* moms_p, double* moms_ref_p, int P) {
    Inform csvout(NULL, "data/error_moments_normal_dist.csv", Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);
    for (int i = 0; i < P; i++) {
        csvout << moms_ref_p[i] << " " << moms_p[i] << " " << fabs(moms_ref_p[i] - moms_p[i])
               << endl;
    }
    csvout << endl;
    ippl::Comm->barrier();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform m("test ITS normal");
        // set up ITS as other examples to sample position
        ippl::Vector<int, 2> nr = {100, 100};
        size_type ntotal        = 1000000;

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<double, Dim> rmin = -3.;
        ippl::Vector<double, Dim> rmax = 3.;

        int seed = 42;

        GeneratorPool rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        // sample the first half of particles with a distribution
        double mu1          = 0.1;
        double sd1          = 0.25;
        double mu2          = -0.2;
        double sd2          = 0.5;
        const double par[4] = {mu1, sd1, mu2, sd2};
        using Dist_t        = ippl::random::NormalDistribution<double, Dim>;
        using sampling_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   Dist_t>;

        Dist_t dist(par);
        sampling_t sampling(dist, rmax, rmin, rmax, rmin, ntotal);
        size_type nlocal = sampling.getLocalSamplesNum();
        view_type position("position", nlocal);

        size_type startIndex = 0;
        size_type endIndex   = floor(nlocal / 2);
        sampling.generate(position, startIndex, endIndex, rand_pool64);

        // compute error in moments in the first half of particles
        const int P = 6;  // number of moments to check, i.e. E[x^i] for i = 1,...,P
        double moms1_ref[P], moms2_ref[P];
        double moms1[P], moms2[P];

        moms1_ref[0] = mu1;
        get_norm_dist_cent_moms(sd1, P, moms1_ref);
        get_moments_from_samples(position, 0, startIndex, endIndex, P, moms1);

        moms2_ref[0] = mu2;
        get_norm_dist_cent_moms(sd2, P, moms2_ref);
        get_moments_from_samples(position, 1, startIndex, endIndex, P, moms2);

        write_error_in_moments(moms1, moms1_ref, P);
        write_error_in_moments(moms2, moms2_ref, P);

        // sample the second half of particles with a different distribution
        mu1 = -0.1;
        sd1 = 0.5;
        mu2 = 0.25;
        sd2 = 0.1;

        const double par2[4] = {mu1, sd1, mu2, sd2};
        Dist_t dist2(par2);
        sampling_t sampling2(dist2, rmax, rmin, rmax, rmin, ntotal);

        startIndex = floor(nlocal / 2);
        endIndex   = nlocal;
        sampling2.generate(position, startIndex, endIndex, rand_pool64);

        // compute error in moments in the second half of particles
        moms1_ref[0] = mu1;
        get_norm_dist_cent_moms(sd1, P, moms1_ref);
        get_moments_from_samples(position, 0, startIndex, endIndex, P, moms1);

        moms2_ref[0] = mu2;
        get_norm_dist_cent_moms(sd2, P, moms2_ref);
        get_moments_from_samples(position, 1, startIndex, endIndex, P, moms2);

        write_error_in_moments(moms1, moms1_ref, P);
        write_error_in_moments(moms2, moms2_ref, P);
    }
    ippl::finalize();
    return 0;
}
