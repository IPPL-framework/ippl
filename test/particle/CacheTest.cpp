
// -*- C++ -*-
/**************************************************************************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *

Example:
               grid size
                 / \
./CacheTest      4 4 --commlib mpi --info 9 | tee output

 *************************************************************************************************************************************/

#include "Ippl.h"
#include <string>
#include <vector>
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/CellParticleCachingPolicy.h"

#include "mpi.h"

// dimension of our positions
#define DIM 2
const unsigned Dim = 2;

// some typedefs
typedef UniformCartesian<Dim, double>                               Mesh_t;
typedef ParticleSpatialLayout < double, Dim, Mesh_t,
        CellParticleCachingPolicy<double, Dim, Mesh_t> >             playout_t;
typedef playout_t::SingleParticlePos_t                              Vector_t;
typedef Cell                                                        Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t>                  FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t>                        Field_t;

template<class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double>     qm;

    ChargedParticles(PL *pl, Vector_t nr, Vector_t hr, Vector_t rmin, e_dim_tag decomp[Dim], bool gCells = true) :
        IpplParticleBase<PL>(pl),
        nr_m(nr),
        hr_m(hr),
        rmin_m(rmin),
        withGuardCells_m(gCells) {
        this->addAttribute(qm);

        for(unsigned int i = 0; i < 2 * Dim; i++) {
            bc_m[i]  = new ParallelPeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            this->getBConds()[i] =  ParticlePeriodicBCond;//ParticleNoBCond;//
        }
        for(unsigned int i = 0; i < Dim; i++)
            decomp_m[i] = decomp[i];

        getMesh().set_meshSpacing(&(hr_m[0]));
        getMesh().set_origin(rmin_m);
    }

    inline const Mesh_t &getMesh() const { return this->getLayout().getLayout().getMesh(); }
    inline Mesh_t &getMesh() { return this->getLayout().getLayout().getMesh(); }
    inline const FieldLayout_t &getFieldLayout() const {
        return dynamic_cast<FieldLayout_t &>(this->getLayout().getLayout().getFieldLayout());
    }

    inline FieldLayout_t &getFieldLayout() {
        return dynamic_cast<FieldLayout_t &>(this->getLayout().getLayout().getFieldLayout());
    }

    NDRegion<double, Dim> getLocalRegion() {
        return (*(this->getLayout().getLayout().begin_iv())).second->getDomain();
    }

    bool checkParticles() {
        Inform msg("CheckParticles", INFORM_ALL_NODES);
        bool ok = true;
        unsigned int i = 0;
        NDRegion<double, Dim> region = getLocalRegion();
        for(; i < this->getLocalNum(); ++i) {
            NDRegion<double, Dim> ppos;
            for(unsigned int d = 0; d < Dim; ++d)
                ppos[d] = PRegion<double>(this->R[i][d], this->R[i][d]);

            if(!region.contains(ppos)) {
                ok = false;
                msg << "Particle misplaced!\nPosition: "
                    << this->R[i] << "\nRegion: " << region << endl;
            }
        }
        for(; i < this->getLocalNum() + this->getGhostNum(); ++i) {
            NDRegion<double, Dim> ppos;
            for(unsigned int d = 0; d < Dim; ++d)
                ppos[d] = PRegion<double>(this->R[i][d], this->R[i][d]);

            if(region.contains(ppos)) {
                ok = false;
                msg << "Ghost Particle misplaced!\nPosition: "
                    << this->R[i] << "\nRegion: " << region << endl;
            }
        }
        return ok;
    }

    void printParticles() {
        Inform msg("IpplParticleBase", INFORM_ALL_NODES);
        Ippl::Comm->barrier();

        for(int i = 0; i < Ippl::getNodes(); ++i) {
            if(i == Ippl::myNode()) {
                msg << "local region: " << getLocalRegion() << '\n';
                msg << "local particles:\n";
                int i = 0;
                for(; i < this->getLocalNum(); ++i) {
                    msg << '\t' << this->R[i];
                }
                msg << "\nghost particles\n";
                for(; i < this->getLocalNum() + this->getGhostNum(); ++i) {
                    msg << '\t' << this->R[i];
                }
                msg << endl << endl;
            }
            Ippl::Comm->barrier();
        }
    }

private:
    BConds<double, Dim, Mesh_t, Center_t> bc_m;
    Vektor<int, Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmin_m;

    bool withGuardCells_m;
    e_dim_tag decomp_m[Dim];
};

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);
    Inform msg(argv[0]);
    Inform msg2all(argv[0], INFORM_ALL_NODES);

    Vektor<int, Dim> nr;

    // need to use preprocessor to prevent clang compiler error
#if DIM == 3
    nr = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
#else
    nr = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]));
#endif

    e_dim_tag decomp[Dim];
    Mesh_t *mesh;
    FieldLayout_t *FL;
    ChargedParticles<playout_t>  *P;

    NDIndex<Dim> domain;
    for(unsigned int i = 0; i < Dim; i++)
        domain[i] = domain[i] = Index(nr[i]);

    for(unsigned int d = 0; d < Dim; ++d)
        decomp[d] = PARALLEL;

    // create mesh and layout objects for this problem domain
    mesh          = new Mesh_t(domain);
    FL            = new FieldLayout_t(*mesh, decomp);
    playout_t *PL = new playout_t(*FL, *mesh);

    Vector_t hr(1.0);
    Vector_t rmin(0.0);

    // fixme: do we have this ??? PL->setAllCacheDimensions();

    PL->setAllCacheCellRanges(2);
    PL->enableCaching();

    P = new ChargedParticles<playout_t>(PL, nr, hr, rmin, decomp, true);
    INFOMSG(P->getMesh() << endl);
    INFOMSG(P->getFieldLayout() << endl);
    msg << endl << endl;

    Ippl::Comm->barrier();

    size_t size = size_t(nr[0] - 1) * size_t(nr[1] - 1);
    size_t begin = size * Ippl::myNode() / Ippl::getNodes();
    size_t end = size * (Ippl::myNode() + 1) / Ippl::getNodes();
    P->create(end - begin);
    size_t j = 0;
    for(size_t i = begin; i < end; ++i, ++j) {
        P->R[j](0) = double(0.5 + i % size_t(nr[0] - 1));
        P->R[j](1) = double(0.5 + i / size_t(nr[0] - 1));
    }


    P->qm = 1.0;

    Ippl::Comm->barrier();

    unsigned expected_particles = P->getLocalRegion().volume();
    unsigned expected_ghosts = 16;
    for(unsigned int d = 0; d < Dim; ++d) {
        expected_ghosts += P->getLocalRegion()[d].length() * 4;
    }

    msg2all << "pre update: " << P->getLocalNum() << " particles and " << P->getGhostNum() << " ghost particles" << endl;
    Vector_t min, max;
    bounds(P->R, min, max);
    msg << min << ' ' << max << endl;
    P->update();

    bounds(P->R, min, max);
    msg << min << ' ' << max << endl;

    BinaryRepartition(*P);
    P->update();

    bounds(P->R, min, max);
    msg << min << ' ' << max << endl;
    Ippl::Comm->barrier();

    msg2all << "post update: " << P->getLocalNum() << " particles and " << P->getGhostNum() << " ghost particles. ";
    bool count_ok = true;
    if(P->getLocalNum() != expected_particles) {
        count_ok = false;
        msg2all << "wrong amount of particles, expected " << expected_particles << endl;
    }
    if(P->getGhostNum() != expected_ghosts) {
        count_ok = false;
        msg2all << "wrong amount of ghost particles, expected " << expected_ghosts << endl;
    }
    if(count_ok) {
        msg2all << "particle count ok" << endl;
    }

    Ippl::Comm->barrier();
    if(!P->checkParticles()) {
        msg2all << "CheckParticle failed!" << endl;
    } else {
        msg2all << "CheckParticle successful" << endl;
    }
    delete P;
    delete FL;
    delete mesh;
    return 0;
}

