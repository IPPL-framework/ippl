#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::ParticleLayout<double, 3> playout;
    typedef ippl::ParticleBase<playout> bunch_type;

    std::shared_ptr<playout> pl = std::make_shared<playout>();

    bunch_type p(pl);

    std::cout << p.getLocalNum() << std::endl;

    p.create(10);

    p.destroy();

    std::cout << p.getLocalNum() << std::endl;

    return 0;
}
