#include "Ippl.h"

#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    typedef ippl::Vector<double, 10> vector_t;

    vector_t x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector_t y = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    double a   = 2.0;

    vector_t z = a * x * y * a;

    z = a * x * y * a;

    std::cout << z << std::endl;

    ippl::finalize();

    return 0;
}
