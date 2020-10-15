#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::IpplParticleBase<double, 3> bunch_type;

    bunch_type p;

    std::cout << p.getLocalNum() << std::endl;

    p.create(10);

    std::cout << p.getLocalNum() << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << p.ID(i) << std::endl;
    }


    return 0;
}