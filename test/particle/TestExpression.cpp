#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::detail::ParticleLayout<double, 3> playout;
    typedef ippl::ParticleBase<playout> bunch_type;

    std::shared_ptr<playout> pl = std::make_shared<playout>();

    bunch_type p(pl);

    p.create(10);

    p.ID.print();

    p.ID = 1;

    p.ID.print();

    p.ID = p.ID + 1;

    p.ID.print();

    return 0;
}
