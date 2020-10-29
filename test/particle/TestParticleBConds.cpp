#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    enum { dim = 3 };
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    std::unique_ptr<bunch_type> bunch;
    double len = 0.2;
    ippl::NDRegion<double, dim>  nr;
    typename bunch_type::particle_position_type::HostMirror HostR;
    playout_type pl_m;



    double shift = 0.1;
    //    setup(len + shift);                                                                                                                                                                                                          
    double pos = len + shift;
    bunch = std::make_unique<bunch_type>(pl_m);

    size_t nParticles = 1000;
    bunch->create(nParticles);

    HostR = bunch->R.getHostMirror();

    for (size_t i = 0; i < nParticles; ++i) {
        HostR(i) = ippl::Vector<double, dim>({pos, pos, pos});
    }

    Kokkos::deep_copy(bunch->R.getView(), HostR);

    // domain                                                                                                                                                                                                                     
    ippl::PRegion<double> region(0.0, 0.2);

    nr = ippl::NDRegion<double, dim>(region, region, region);

    using BC = ippl::BC;

    bunch_type::bc_container_type bcs = {
        BC::PERIODIC,
        BC::REFLECTIVE,
        BC::SINK,
        BC::PERIODIC,
        BC::REFLECTIVE,
        BC::NO
    };

    bunch->setBConds(bcs);

    bunch->getLayout().applyBC(bunch->R, nr);    

    Kokkos::deep_copy(HostR, bunch->R.getView());

    ippl::Vector<double, dim> expected = {shift, shift, shift};

    for (unsigned i = 0; i < 3; ++i) {
        std::cout << HostR(0)[i] << std::endl;
    }

    return 0;
}
