// Testing Correlated Normal Distribution on unbounded domains
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
#include "Random/CorrRandn.h"

const int Dim = 2;

using view_type  = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

using size_type = ippl::detail::size_type;

using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

using Vector_t = ippl::Vector<double, Dim>;

using Matrix_t = ippl::Vector< ippl::Vector<double, Dim>, Dim>;

void computeMeanCovariance(Vector_t &mu, Matrix_t &cov, view_type position, size_type ntotal) {

    mu(0.0);
    for (unsigned int i = 0; i < ntotal; i++) {
        for (unsigned int j = 0; j < Dim; j++) {
            mu[j] += position(i)[j];
        }
    }
    for (unsigned int j = 0; j < Dim; j++) {
        mu[j] /= ntotal;
    }

    cov(0.0);
    for (unsigned int i = 0; i < ntotal; i++) {
        for (unsigned int j = 0; j < Dim; j++) {
            for (unsigned int k = 0; k < Dim; k++) {
                cov[j][k] += (position(i)[j] - mu[j]) * (position(i)[k] - mu[k]);
            }
        }
    }
    for (unsigned int j = 0; j < Dim; j++) {
        for (unsigned int k = 0; k < Dim; k++) {
            cov[j][k] /= (ntotal - 1); // Unbiased estimator
        }
    }
}

void write_error_in_moments(Vector_t &mu, Vector_t &mu_est, Matrix_t &cov, Matrix_t &cov_est){
    Inform csvout(NULL, "data/error_mean_CorrRandn.csv", Inform::APPEND);
    csvout.precision(16);
    csvout.setf(std::ios::scientific, std::ios::floatfield);
    for(int i=0; i<Dim; i++){
        csvout << mu[i] << " " << mu_est[i] << " " << fabs(mu[i] - mu_est[i]) << endl;
    }

    Inform csvoutcov(NULL, "data/error_cov_CorrRandn.csv", Inform::APPEND);
    csvoutcov.precision(16);
    csvoutcov.setf(std::ios::scientific, std::ios::floatfield);
    for(int i=0; i<Dim; i++){
        for(int j=0; j<Dim; j++){
            csvoutcov << cov[i][j] << " " << cov_est[i][j] << " " << fabs(cov[i][j] - cov_est[i][j]) << endl;
        }
    }

    ippl::Comm->barrier();
}


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        size_type ntotal = 1000000;

        int seed = 42;

        GeneratorPool rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using Vector_t = ippl::Vector<double, Dim>;
        using Matrix_t = ippl::Vector< ippl::Vector<double, Dim>, Dim>;

        Vector_t mu;
        Matrix_t cov;

        mu[0] = 1.;
        mu[1] = -1;

        cov[0][0] = 1.;
        cov[0][1] = 0.5;
        cov[1][0] = 0.5;
        cov[1][1] = 1.25;

        view_type position("position", ntotal);
        Kokkos::parallel_for(
            ntotal, ippl::random::CorrRandn<double, Dim>(position, rand_pool64, mu, cov)
        );

        Kokkos::fence();
        ippl::Comm->barrier();

        Vector_t mu_est;
        Matrix_t cov_est;
        computeMeanCovariance(mu_est, cov_est, position, ntotal);

        Inform m("CorrRandn ");
        m << "mean:" << endl;
        for (unsigned int j = 0; j < Dim; j++) {
          m << mu[j] << " ";
        }
        m << endl;

        for (unsigned int j = 0; j < Dim; j++) {
          m << mu_est[j] << " ";
        }
        m << endl;

        m << "covariance:" << endl;
        for (unsigned int j = 0; j < Dim; j++) {
          for (unsigned int k = 0; k < Dim; k++) {
             m << cov[j][k] << " ";
          }
          m << endl;
        }


        for (unsigned int j = 0; j < Dim; j++) {
          for (unsigned int k = 0; k < Dim; k++) {
             m << cov_est[j][k] << " ";
          }
          m << endl;
        }

        write_error_in_moments(mu, mu_est, cov, cov_est);

    }
    ippl::finalize();
    return 0;
}

