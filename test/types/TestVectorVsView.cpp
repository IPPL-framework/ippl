//
// Created by Bob Schreiner on 13.10.2025.
//

#include <Ippl.h>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <memory>

#include "Types/Tuple.h"
// #include <Kokkos_Core.hpp>
#define alwaysAssert(X)                                                                    \
    do {                                                                                   \
        if (!(X)) {                                                                        \
            std::cerr << "Assertion " << #X << " failed: " << __FILE__ << ": " << __LINE__ \
                      << '\n';                                                             \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (false)

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // We want to test if ippl::Vector accumulates more memory on GPU's then Kokkos::View
        // We also want to check if they stall computation

        constexpr int N = 50000;
        constexpr int M = 3000;
        constexpr int K = 300;

        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

        double pi = 3.14;


        Kokkos::View<size_t**> idx_list_view("idx" ,N,M);
        Kokkos::View<double*> input_view("input",K);
        Kokkos::View<double*> result_view("result",K);

        // start a timer
        static IpplTimings::TimerRef fill_view_timer = IpplTimings::getTimer("fill view");
        IpplTimings::startTimer(fill_view_timer);

        Kokkos::parallel_for("fill idx list view" , N , KOKKOS_LAMBDA(int i) {

            auto generator = random_pool.get_state();
            for (int j = 0; j < M; ++j) {
                idx_list_view(i,j) = generator.urand(0,K-1);
            }
            random_pool.free_state(generator);
        });

        IpplTimings::stopTimer(fill_view_timer);

        // start a timer
        static IpplTimings::TimerRef input_view_timer = IpplTimings::getTimer("input view");
        IpplTimings::startTimer(input_view_timer);

        Kokkos::parallel_for("fill input view" , K , KOKKOS_LAMBDA(int k) {
            auto generator = random_pool.get_state();
            input_view(k) = generator.drand(0.,1.);
            random_pool.free_state(generator);
        });

        IpplTimings::stopTimer(input_view_timer);

        static IpplTimings::TimerRef output_view_timer = IpplTimings::getTimer("output view");
        IpplTimings::startTimer(output_view_timer);

        Kokkos::parallel_for("compute view " , N , KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < M; ++j) {
                int idx = idx_list_view(i,j);
                Kokkos::atomic_add(&result_view(idx) , pi*input_view(idx));
            }
        });
        IpplTimings::stopTimer(output_view_timer);


        //ippl::Vector<ippl::Vector<size_t, M>, N> idx_list_vector(0);
        ippl::Vector<double , K> input_vector(0.);
        ippl::Vector<double , K> result_vector(0.);

        /*
        // start a timer
        static IpplTimings::TimerRef fill_vector_timer = IpplTimings::getTimer("fill vector");
        IpplTimings::startTimer(fill_vector_timer);

        Kokkos::parallel_for("fill idx list vector" , N , KOKKOS_LAMBDA(int i) {

            auto generator = random_pool.get_state();
            for (int j = 0; j < M; ++j) {
                idx_list_vector[i][j] = generator.urand(0,K-1);
            }
            random_pool.free_state(generator);
        });

        IpplTimings::stopTimer(fill_vector_timer);

        */

        // start a timer
        static IpplTimings::TimerRef input_vector_timer = IpplTimings::getTimer("input vector");
        IpplTimings::startTimer(input_vector_timer);

        for (int k = 0; k < K; ++k){
            auto generator = random_pool.get_state();
            input_vector[k] = generator.drand(0.,1.);
            random_pool.free_state(generator);
        }

        IpplTimings::stopTimer(input_vector_timer);

        static IpplTimings::TimerRef output_vector_timer = IpplTimings::getTimer("output vector");
        IpplTimings::startTimer(output_vector_timer);

        Kokkos::parallel_for("compute vector " , N , KOKKOS_LAMBDA(int i) {
            auto generator = random_pool.get_state();
            ippl::Vector<double, M> local_idx(0);
            for (int j = 0; j<M;++j) {
                local_idx[j] = generator.urand(0.,K-1);
            }
            random_pool.free_state(generator);

            for (int j = 0; j < M; ++j) {
                int idx = local_idx[j];
                Kokkos::atomic_add(&result_view(idx), pi * input_vector[idx]);
            }
        });
        IpplTimings::stopTimer(output_vector_timer);


        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));


    }
    ippl::finalize();
    return 0;
}