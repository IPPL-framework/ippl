#include <Kokkos_Core.hpp>
#include "dims.h"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using view_type = Kokkos::View<NPtr<double, 2>::type>;
    view_type view("Matrix", 3, 3);
    auto begin = Kokkos::Array<unsigned int, 2>{};
    auto end = Kokkos::Array<unsigned int, 2>{};
    for (int i = 0; i < 2; i++) {
        begin[i] = 0;
        end[i] = view.extent(i);
    }
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(begin, end);
    Kokkos::parallel_for("Init", policy,
          KOKKOS_LAMBDA(const size_t i,
              const size_t j) {
            view(i, j) = i;
          });

    auto reducer = ConvenientReducer<2, double>(view);
    //auto reducer = Reducer<
    //  Coords<unsigned int, 2>::type, 2, double
    //  >(view);

    double sum = 0;
    Kokkos::parallel_reduce("Reduce", policy, reducer,
            Kokkos::Sum<double>(sum));

    printf("Sum: %f\n", sum);
  }
  Kokkos::finalize();
}
