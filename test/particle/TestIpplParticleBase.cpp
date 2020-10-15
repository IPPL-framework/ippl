#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::IpplParticleBase<double, 3> bunch_type;

    bunch_type p;

    std::cout << p.getLocalNum() << std::endl;

    p.create(10);

    std::cout << p.getLocalNum() << std::endl;


    return 0;
}