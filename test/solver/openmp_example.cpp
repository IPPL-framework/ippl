#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

int main() {

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        if (thread_id == 0) {
            std::cout << "There are " << num_threads << " threads" << std::endl;
        }
    }

    double time1 = omp_get_wtime();

    // Run the parallel_for test 1000 times
    for (int iteration = 0; iteration < 1000; ++iteration) {

        const int num_elements = 1000000;
        std::vector<double> results(num_elements);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937 generator(thread_id + static_cast<unsigned>(time(0)));
            std::uniform_real_distribution<double> distribution(0.0, 1.0);

            #pragma omp for
            for (int i = 0; i < num_elements; ++i) {
                double rand_num = distribution(generator);
                double exp_result = exp(rand_num);
                double sin_result = sin(rand_num);
                results[i] = exp_result + sin_result;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < num_elements; ++i) {
            sum += results[i];
        }
        //std::cout << "Sum of results: " << sum << std::endl;
        
    }

    double time2 = omp_get_wtime();

    std::cout << "Time = " << time2 - time1 << std::endl;

    return 0;
}
