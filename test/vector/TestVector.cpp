#include "Ippl.h"

#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        typedef ippl::Vector<double, 10> vector_t;

        vector_t x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        vector_t y = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        double a   = 2.0;

        vector_t z = a * x * y * a;

        z = a * x * y * a;

        std::cout << z << std::endl;

        std::cout << x.dot(y) << std::endl;

        vector_t w(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);

        std::cout << w << std::endl;

        w += 1;

        std::cout << w << std::endl;

        w = w + 1;

        std::cout << w << std::endl;

        w = w - 1;

        std::cout << w << std::endl;

        w -= 1;

        std::cout << w << std::endl;

        x = y + a;

        std::cout << x << std::endl;

        x = y - a;

        std::cout << x << std::endl;

        y *= a;

        std::cout << y << std::endl;

        y /= a;

        std::cout << y << std::endl;

        y /= (1.0 / a);

        std::cout << y << std::endl;
    }
    ippl::finalize();

    return 0;
}
