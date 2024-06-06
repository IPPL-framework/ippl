#include <Kokkos_Core.hpp>
#include <iostream>

// Kernel function to be executed in parallel
void parallel_for_test(int iteration) {
    Kokkos::parallel_for("ExampleKernel", Kokkos::RangePolicy<Kokkos::OpenMP>(0, 10000), KOKKOS_LAMBDA(int i) {
        int j = i*i;
    });
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        Kokkos::Timer timer;
        double time1 = timer.seconds();

        // Run the parallel_for test 1000 times
        for (int iteration = 0; iteration < 1000; ++iteration) {
            parallel_for_test(iteration);
        }

        double time2 = timer.seconds();
        std::cout << "Time = " << time2 - time1 << std::endl;
    }
    // Finalize Kokkos
    Kokkos::finalize();

    return 0;
}
