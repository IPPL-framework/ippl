//
// Application p3m3d
//   Example:      interaction radius
//                      /
//           grid size /  particles    distribution
//            / |  \  /    /             /
//   ./p3m3d 16 16 16 5. 1000 [uniform|random|point] --commlib mpi --info 9 | tee field.txt
//
//   using the "point" distribution will only place one particle
//
// Benjamin Ulmer, ETH Zürich (2016)
// Implemented as part of the Master thesis
// "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
// (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
//
#include "Ippl.h"

#include <cfloat>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilder.h"
#include "Particle/PairBuilder/PairConditions.h"
#include "Particle/PairBuilder/SortingPairBuilder.h"

// dimension of our positions
const unsigned Dim = 3;

// some typedefs
typedef UniformCartesian<Dim, double> Mesh_t;
typedef BoxParticleCachingPolicy<double, Dim, Mesh_t> CachingPolicy_t;
typedef ParticleSpatialLayout<double, Dim, Mesh_t, CachingPolicy_t> playout_t;
typedef playout_t::SingleParticlePos_t Vector_t;
typedef Cell Center_t;
typedef CenteredFieldLayout<Dim, Mesh_t, Center_t> FieldLayout_t;
typedef Field<double, Dim, Mesh_t, Center_t> Field_t;
typedef Field<int, Dim, Mesh_t, Center_t> IField_t;
typedef Field<Vector_t, Dim, Mesh_t, Center_t> VField_t;
typedef Field<std::complex<double>, Dim, Mesh_t, Center_t> CxField_t;
typedef FFT<RCTransform, Dim, double> FFT_t;
typedef IntCIC IntrplCIC_t;
typedef IntNGP IntrplNGP_t;

template <unsigned int Dim>
struct SpecializedGreensFunction {};

/*
  template<>
  struct SpecializedGreensFunction<3>;
*/

template <>
struct SpecializedGreensFunction<3> {
    template <class T, class FT, class FT2>

    static void calculate(Vektor<T, 3>& hrsq, FT& grn, FT2* grnI) {
        grn          = grnI[0] * hrsq[0] + grnI[1] * hrsq[1] + grnI[2] * hrsq[2];
        grn          = 1.0 / sqrt(grn);
        grn[0][0][0] = grn[0][0][1];
    }

    // order two transition-function
    template <class T, class FT, class FT2>
    static void calculate(Vektor<T, 3>& hrsq, FT& grn, FT2* grnI, double R) {
        grn          = grnI[0] * hrsq[0] + grnI[1] * hrsq[1] + grnI[2] * hrsq[2];
        grn          = where(lt(R * R, grn), 1. / sqrt(grn),
                             ((grn * sqrt(grn)) / R - 2 * grn) / (R * R * R) + 2 / R);
        grn[0][0][0] = grn[0][0][1];
    }
};

template <class T>
struct ApplyField;

template <class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double> Q;
    ParticleAttrib<Vector_t> EF;

    ChargedParticles(PL* pl, Vector_t nr, e_dim_tag decomp[Dim])
        : IpplParticleBase<PL>(pl)
        , nr_m(nr) {
        this->addAttribute(Q);
        this->addAttribute(EF);

        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bc_m[i]              = new ZeroFace<double, Dim, Mesh_t, Center_t>(i);
            vbc_m[i]             = new ZeroFace<Vector_t, Dim, Mesh_t, Center_t>(i);
            this->getBConds()[i] = ParticleNoBCond;
        }

        for (unsigned int i = 0; i < Dim; i++)
            decomp_m[i] = decomp[i];

        for (unsigned int d = 0; d < Dim; ++d) {
            rmax_m[d] = 1.0;
            rmin_m[d] = 0.0;
            hr_m[d]   = (rmax_m[d] - rmin_m[d]) / (nr_m[d] - 1.0);
        }
        this->getMesh().set_meshSpacing(&(hr_m[0]));
        this->getMesh().set_origin(rmin_m);

        rho_m.initialize(this->getMesh(), this->getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
        eg_m.initialize(this->getMesh(), this->getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);

        domain_m = this->getFieldLayout().getDomain();
    }

    inline const Mesh_t& getMesh() const { return this->getLayout().getLayout().getMesh(); }

    inline Mesh_t& getMesh() { return this->getLayout().getLayout().getMesh(); }

    inline const FieldLayout_t& getFieldLayout() const {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    inline FieldLayout_t& getFieldLayout() {
        return dynamic_cast<FieldLayout_t&>(this->getLayout().getLayout().getFieldLayout());
    }

    void update() {
        Inform msg("update");
        bounds(this->R, rmin_m, rmax_m);

        Vector_t stretch = 0.6 / nr_m;
        Vector_t diff    = rmax_m - rmin_m;
        rmax_m += stretch * diff;
        rmin_m -= stretch * diff;

        for (unsigned int d = 0; d < Dim; ++d) {
            hr_m[d] = (rmax_m[d] - rmin_m[d]) / (nr_m[d] - 1.0);
        }
        this->getMesh().set_meshSpacing(&(hr_m[0]));
        this->getMesh().set_origin(rmin_m);

        rho_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
        eg_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);

        domain_m = this->getFieldLayout().getDomain();

        IpplParticleBase<PL>::update();
    }

    void calculatePairForces(double interaction_radius);

    // setup and use the FFT solver
    void calculateGridForces(double interaction_radius) {
        this->Q.scatter(this->rho_m, this->R, IntrplCIC_t());
        // this->Q.scatter(this->rho_m, this->R, IntrplNGP_t());

        // rho2_m is the charge-rho_m field with mesh doubled in each dimension
        Field_t rho2_m;

        // real field with layout of complex field: domain3_m
        Field_t greentr_m;

        // rho2tr_m is the Fourier transformed charge-rho_m field
        // domain3_m and mesh3_ are used
        CxField_t rho2tr_m;
        CxField_t imgrho2tr_m;

        // grntr_m is the Fourier transformed Green's function
        // domain3_m and mesh3_ are used
        CxField_t grntr_m;

        // Fields used to eliminate excess calculation in greensFunction()
        // mesh2_m and layout2_m are used
        IField_t grnIField_m[3];

        // the FFT object
        FFT_t* fft_m;

        // mesh and layout objects for rho_m
        Mesh_t* mesh_m          = &(getMesh());
        FieldLayout_t* layout_m = &(getFieldLayout());

        // mesh and layout objects for rho2_m
        Mesh_t* mesh2_m;
        FieldLayout_t* layout2_m;

        //
        Mesh_t* mesh3_m;
        FieldLayout_t* layout3_m;

        // tmp
        Field_t tmpgreen;

        // domains for the various fields
        NDIndex<3> domain_m;  // original domain, gridsize
        // mesh and gridsize defined outside of FFT class, given as
        // parameter to the constructor (mesh and layout object).
        NDIndex<3> domain2_m;  // doubled gridsize (2*Nx,2*Ny,2*Nz)
        NDIndex<3> domain3_m;  // field for the complex values of the RC transformation
        // (2*Nx,Ny,2*Nz)
        NDIndex<3> domainFFTConstruct_m;

        domain_m = layout_m->getDomain();

        // For efficiency in the FFT's, we can use a parallel decomposition
        // which can be serial in the first dimension.
        e_dim_tag decomp[3];
        e_dim_tag decomp2[3];

        for (int d = 0; d < 3; ++d) {
            decomp[d]  = layout_m->getRequestedDistribution(d);
            decomp2[d] = layout_m->getRequestedDistribution(d);
        }

        // The FFT's require double-sized field sizes in order to (more closely
        // do not understand this ...)
        // simulate an isolated system.  The FFT of the charge density field, rho,
        // would otherwise mimic periodic boundary conditions, i.e. as if there were
        // several beams set a periodic distance apart.  The double-sized fields
        // alleviate this problem.
        for (int i = 0; i < 3; i++) {
            hr_m[i]      = mesh_m->get_meshSpacing(i);
            nr_m[i]      = domain_m[i].length();
            domain2_m[i] = Index(2 * nr_m[i] + 1);
        }

        // create double sized mesh and layout objects for the use in the FFT's
        mesh2_m   = new Mesh_t(domain2_m);
        layout2_m = new FieldLayout_t(*mesh2_m, decomp);
        rho2_m.initialize(*mesh2_m, *layout2_m);

        NDIndex<3> tmpdomain;
        // Create the domain for the transformed (complex) fields.  Do this by
        // taking the domain from the doubled mesh, permuting it to the right, and
        // setting the 2nd dimension to have n/2 + 1 elements.
        domain3_m[0] = Index(2 * nr_m[3 - 1] + 1);
        domain3_m[1] = Index(nr_m[0] + 2);

        for (int i = 2; i < 3; ++i)
            domain3_m[i] = Index(2 * nr_m[i - 1] + 1);

        // create mesh and layout for the new real-to-complex FFT's, for the
        // complex transformed fields
        mesh3_m   = new Mesh_t(domain3_m);
        layout3_m = new FieldLayout_t(*mesh3_m, decomp2);
        rho2tr_m.initialize(*mesh3_m, *layout3_m);
        imgrho2tr_m.initialize(*mesh3_m, *layout3_m);
        grntr_m.initialize(*mesh3_m, *layout3_m);

        // helper field for sin
        greentr_m.initialize(*mesh3_m, *layout3_m);

        // create a domain used to indicate to the FFT's how to construct it's
        // temporary fields.  This is the same as the complex field's domain,
        // but permuted back to the left.
        tmpdomain = layout3_m->getDomain();
        for (int i = 0; i < 3; ++i)
            domainFFTConstruct_m[i] = tmpdomain[(i + 1) % 3];

        // create the FFT object
        fft_m = new FFT_t(layout2_m->getDomain(), domainFFTConstruct_m);

        // these are fields that are used for calculating the Green's function.
        // they eliminate some calculation at each time-step.
        for (int i = 0; i < 3; ++i) {
            grnIField_m[i].initialize(*mesh2_m, *layout2_m);
            grnIField_m[i][domain2_m] =
                where(lt(domain2_m[i], nr_m[i]), domain2_m[i] * domain2_m[i],
                      (2 * nr_m[i] - domain2_m[i]) * (2 * nr_m[i] - domain2_m[i]));
        }

        tmpgreen.initialize(*mesh2_m, *layout2_m);

        // use grid of complex doubled in both dimensions
        // and store rho in lower left quadrant of doubled grid
        rho2_m = 0.0;

        rho2_m[domain_m] = rho_m[domain_m];

        // needed in greens function
        // hr_m = hr;
        // FFT double-sized charge density
        // we do a backward transformation so that we dont have to account for the normalization
        // factor that is used in the forward transformation of the IPPL FFT
        fft_m->transform(-1, rho2_m, rho2tr_m);

        // must be called if the mesh size has changed
        // have to check if we can do G with h = (1,1,1)
        // and rescale later

        Vector_t hrsq(hr_m * hr_m);

        SpecializedGreensFunction<3>::calculate(hrsq, rho2_m, grnIField_m, interaction_radius);

        fft_m->transform(-1, rho2_m, grntr_m);

        // multiply transformed charge density
        // and transformed Green function
        // Don't divide by (2*nx_m)*(2*ny_m), as Ryne does;
        // this normalization is done in POOMA's fft routine.
        rho2tr_m *= grntr_m;

        // inverse FFT, rho2_m equals to the electrostatic potential
        fft_m->transform(+1, rho2tr_m, rho2_m);
        // end convolution

        // back to physical grid
        // reuse the charge density field to store the electrostatic potential
        rho_m[domain_m] = rho2_m[domain_m];

        // delete the FFT object
        delete fft_m;

        // delete the mesh and layout objects
        if (mesh2_m != 0)
            delete mesh2_m;
        if (layout2_m != 0)
            delete layout2_m;
        if (mesh3_m != 0)
            delete mesh3_m;
        if (layout3_m != 0)
            delete layout3_m;

        eg_m = -Grad(rho_m, eg_m);

        EF.gather(eg_m, this->R, IntrplCIC_t());
        // EF.gather(eg_m, this->R,  IntrplNGP_t());
    }

    void writeFields(unsigned idx) {
        INFOMSG("*** START DUMPING SCALAR FIELD ***" << endl);

        std::ostringstream istr;
        istr << idx;
        std::string rho_fn = std::string("rhofield-" + std::string(istr.str()) + ".dat");

        Inform fstr2(NULL, rho_fn.c_str(), Inform::OVERWRITE);
        fstr2.precision(9);

        NDIndex<3> myidx = getFieldLayout().getLocalNDIndex();
        for (int x = myidx[0].first(); x <= myidx[0].last(); x++) {
            for (int y = myidx[1].first(); y <= myidx[1].last(); y++) {
                for (int z = myidx[2].first(); z <= myidx[2].last(); z++) {
                    fstr2 << x + 1 << " " << y + 1 << " " << z + 1 << " " << rho_m[x][y][z].get()
                          << endl;
                }
            }
        }

        INFOMSG("*** START DUMPING E FIELD ***" << endl);
        std::string e_field = std::string("efield-" + std::string(istr.str()) + ".dat");

        Inform fstr(NULL, e_field.c_str(), Inform::OVERWRITE);
        fstr.precision(9);
        NDIndex<3> myidxx = getFieldLayout().getLocalNDIndex();
        for (int x = myidxx[0].first(); x <= myidxx[0].last(); x++) {
            for (int y = myidxx[1].first(); y <= myidxx[1].last(); y++) {
                for (int z = myidxx[2].first(); z <= myidxx[2].last(); z++) {
                    fstr << x + 1 << " " << y + 1 << " " << z + 1 << " " << eg_m[x][y][z].get()
                         << endl;
                }
            }
        }
        INFOMSG("*** FINISHED DUMPING E FIELD ***" << endl);
    }

private:
    BConds<double, Dim, Mesh_t, Center_t> bc_m;
    BConds<Vector_t, Dim, Mesh_t, Center_t> vbc_m;

    Field_t rho_m;
    VField_t eg_m;

    Vektor<int, Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmax_m;
    Vector_t rmin_m;

    NDIndex<Dim> domain_m, domain2_m, domain3_m, domainFFTConstruct_m;

    Mesh_t *mesh2_m, *mesh3_m;
    FieldLayout_t *layout2_m, *layout3_m;

    FFT_t* fft_m;

    e_dim_tag decomp_m[Dim];
};

template <class T>
struct ApplyField {
    ApplyField(T c, double r)
        : C(c)
        , R(r) {}
    void operator()(std::size_t i, std::size_t j, ChargedParticles<playout_t>& P) const {
        const Vector_t diff = P.R[i] - P.R[j];
        double sqr          = 0;

        // const double sqr =  dot(diff,diff);

        for (unsigned d = 0; d < Dim; ++d)
            sqr += diff[d] * diff[d];

        if (sqr != 0) {
            double r = std::sqrt(sqr);

            // for order two transition
            Vector_t Fij =
                C * (diff / r) * (1 / sqr - (-3 / (R * R * R * R) * r * r + 4 / (R * R * R) * r));

            P.EF[i] -= P.Q[j] * Fij;
            P.EF[j] += P.Q[i] * Fij;
        }
    }
    T C;
    double R;
};

template <class PL>
void ChargedParticles<PL>::calculatePairForces(double interaction_radius) {
    HashPairBuilder<ChargedParticles<playout_t> > HPB(*this);
    HPB.for_each(RadiusCondition<double, Dim>(interaction_radius),
                 ApplyField<double>(-1, interaction_radius));
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        IpplTimings::TimerRef allTimer = IpplTimings::getTimer("AllTimer");
        IpplTimings::startTimer(allTimer);

        Vektor<int, Dim> nr;

        unsigned param = 1;

        if (Dim == 3) {
            nr    = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
            param = 4;
        } else {
            nr    = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]));
            param = 3;
        }

        double interaction_radius = atof(argv[param++]);
        size_t count              = atoi(argv[param++]);

        enum DistType {
            UNIFORM,
            RANDOM,
            POINT
        };
        DistType type = UNIFORM;

        if (argv[param] == std::string("uniform")) {
            type = UNIFORM;
        } else if (argv[param] == std::string("random")) {
            type = RANDOM;
        } else if (argv[param] == std::string("point")) {
            type = POINT;
        }

        e_dim_tag decomp[Dim];
        Mesh_t* mesh;
        FieldLayout_t* FL;
        ChargedParticles<playout_t>* P;

        NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++)
            domain[i] = domain[i] = Index(nr[i]);

        for (unsigned d = 0; d < Dim; ++d)
            decomp[d] = PARALLEL;

        // create mesh and layout objects for this problem domain
        mesh          = new Mesh_t(domain);
        FL            = new FieldLayout_t(*mesh, decomp);
        playout_t* PL = new playout_t(*FL, *mesh);

        PL->setAllCacheDimensions(interaction_radius);
        PL->enableCaching();

        P = new ChargedParticles<playout_t>(PL, nr, decomp);
        INFOMSG(P->getMesh() << endl);
        INFOMSG(P->getFieldLayout() << endl);
        msg << endl << endl;
        ippl::Comm->barrier();

        // center of charge distribution
        Vektor<double, Dim> source(0.1 + (32) / 2, 0.3 + (32) / 2, 0.7 + (32) / 2);

        double total_charge = 0;

        if (Ippl::myNode() == 0) {
            size_t index  = 0;
            double radius = 6;

            double density = count * 3 / (radius * radius * radius * 4 * 3.14159);
            double pdist   = std::pow(1 / density, 1 / 3.);

            // create sample particles with charge 0
            for (int i = 0; i < 32; ++i)
                for (int j = 0; j < 32; ++j)
                    for (int k = 0; k < 32; ++k) {
                        Vektor<double, Dim> pos(i, j, k);

                        if (type != POINT
                            && dot(pos - source, pos - source)
                                   <= (radius + pdist / 3) * (radius + pdist / 3))
                            continue;
                        P->create(1);
                        P->R[index] = pos;
                        P->Q[index] = 0;
                        ++index;
                    }

            switch (type) {
                case UNIFORM:

                    for (double x = 0; x < 32; x += pdist)
                        for (double y = 0; y < 32; y += pdist)
                            for (double z = 0; z < 32; z += pdist) {
                                Vektor<double, Dim> pos(x, y, z);

                                if (dot(pos - source, pos - source) <= radius * radius) {
                                    P->create(1);
                                    P->R[index] = pos;
                                    P->Q[index] = 1. / count;
                                    total_charge += 1. / count;
                                    ++index;
                                }
                            }
                    break;

                case RANDOM:

                    IpplRandom.SetSeed(42);

                    P->create(count);
                    for (unsigned i = 0; i < count; ++i) {
                        Vektor<double, Dim> pos;
                        do {
                            pos = 2
                                  * Vektor<double, Dim>((IpplRandom() - 0.5), (IpplRandom() - 0.5),
                                                        (IpplRandom() - 0.5));
                        } while (dot(pos, pos) > 1);

                        P->R[index] = source + pos * radius;
                        P->Q[index] = 1. / count;
                        total_charge += 1. / count;
                        ++index;
                    }

                    break;
                case POINT:
                    P->create(1);
                    P->R[index] = source;
                    P->Q[index] = 1;
                    total_charge += 1.;
                    break;
            }
        }

        P->update();

        msg << "calculating grid" << endl;
        IpplTimings::TimerRef gridTimer = IpplTimings::getTimer("GridTimer");
        IpplTimings::startTimer(gridTimer);

        P->calculateGridForces(interaction_radius);

        IpplTimings::stopTimer(gridTimer);

        msg << "calculating pairs" << endl;

        IpplTimings::TimerRef particleTimer = IpplTimings::getTimer("ParticleTimer");
        IpplTimings::startTimer(particleTimer);

        P->calculatePairForces(interaction_radius);

        IpplTimings::stopTimer(particleTimer);

        // P->writeFields(1);

        double error = 0;
        int n        = 0;
        for (unsigned i = 0; i < P->getLocalNum(); ++i) {
            double radius = std::sqrt(dot(source - P->R[i], source - P->R[i]));
            double E      = std::sqrt(dot(P->EF[i], P->EF[i]));
            // if(radius < 2*interaction_radius)
            // msg2all << radius << ' ' << E << ' ' << P->R[i][0] << ' ' << P->R[i][1] << ' ' <<
            // P->R[i][2] << endl;

            double diff = 0;

            if (type != POINT && radius <= 6) {
                diff = E - radius / (6 * 6 * 6);
            } else {
                if (radius > 0)
                    diff = E - 1 / (radius * radius);
            }
            error += diff * diff;
            n++;
        }

        double total_error = 0;
        reduce(error, total_error, OpAddAssign());

        total_error = std::sqrt(total_error) / n;

        IpplTimings::stopTimer(allTimer);

        IpplTimings::print();

        msg << "total charge: " << total_charge << endl;
        msg << "Error: " << total_error << endl;

        delete P;
        delete FL;
        delete mesh;
    }
    ippl::finalize();

    return 0;
}
