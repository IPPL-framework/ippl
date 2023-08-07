#include "Ippl.h"

#include "Stream/Hdf5ParticleStream.h"

using Mesh = ippl::UniformCartesian<double, 3>;
using PLayout = ippl::ParticleSpatialLayout<double, 3, Mesh>;


class Particles : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:

    Particles(PLayout& p) : Base(p)
                          , q("q", "charge", "C")
                          , v("v", "veloctity", "m/s"), E("E", "electric field", "V/m") {
        this->addAttribute(q);
        this->addAttribute(v);
        this->addAttribute(E);
    }


    ippl::ParticleAttrib<double> q;
    typename Base::particle_position_type v;
    typename Base::particle_position_type E;  // electric field at particle position
};


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {

        try {
            ippl::NDIndex<3> domain;
            for (unsigned i = 0; i < 3; i++) {
                domain[i] = ippl::Index(32);
            }

            ippl::e_dim_tag decomp[3];
            for (unsigned d = 0; d < 3; ++d) {
                decomp[d] = ippl::PARALLEL;
            }


            ippl::Vector<double, 3> hr     = 1.0 / 32;
            ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
            Mesh mesh(domain, hr, origin);
            ippl::FieldLayout<3> fl(domain, decomp);
            PLayout pl(fl, mesh);


            Particles part(pl);

            part.create(50);

            ippl::hdf5::ParticleStream<Particles> ps;

            std::filesystem::path filename = "test.hdf5";

            ps.create(filename);

            ps.open(filename, 'w');

            ps << part;

            ps.close();

        } catch(const IpplException& ex) {
            std::cout << ex.what() << std::endl;
        }

    }
    ippl::finalize();

    return 0;
}
