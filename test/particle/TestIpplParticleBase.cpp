#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::detail::ParticleLayout<double, 3> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    playout_type pl;

    bunch_type p(pl);

    std::cout << p.getLocalNum() << std::endl;

    p.create(10);

    std::cout << p.getLocalNum() << std::endl;

    return 0;
}
