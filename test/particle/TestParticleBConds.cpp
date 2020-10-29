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

    bunch->create(1);

    std::cout << "Hi 1" << std::endl;

    HostR = bunch->R.getHostMirror();

    HostR(0) = ippl::Vector<double, dim>({pos, pos, pos});

    Kokkos::deep_copy(bunch->R.getView(), HostR);

    std::cout << "Hi 2" << std::endl;

    // domain                                                                                                                                                                                                                     
    ippl::PRegion<double> region(0.0, 0.2);

    nr = ippl::NDRegion<double, dim>(region, region, region);

    std::cout << "Hi 3" << std::endl;
                                                                                                                                                                                                                                        
    /*    for (unsigned i = 0; i < 2 * dim; i++) {                                                                                                                                                                                                 
        bunch->setBCond(ippl::ParticlePeriodicBCond<double>, i);                                                                                                                                                                             
	} */                               

    std::cout << "Hi 4" << std::endl;
                                                                                                                                                                                                       
                                                                                                                                                                                                                                             
    bunch->getLayout().applyBC(bunch->R, nr);    

    std::cout << "Hi 5" << std::endl;
                                                                                                                                                                                      
    Kokkos::deep_copy(HostR, bunch->R.getView());

    std::cout << "Hi 6" << std::endl;

    ippl::Vector<double, dim> expected = {shift, shift, shift};

    for (unsigned i = 0; i < 3; ++i) {
	std::cout << HostR(0)[i] << std::endl;
    }

    return 0;
}
