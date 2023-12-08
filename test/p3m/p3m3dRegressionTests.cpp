//
// Application p3m3dRegressionTests
//   mpirun -np 4 ./p3m3dRegressionTests testcase Nx Ny Nz rcut alpha dt eps timesteps Sx Sy Sz
//   alpha is the splitting parameter for pm and pp, eps is the smoothing factor and Si are the
//   coordinates of the charged sphere center
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
#include <random>
#include <string>
#include <vector>

#include "ChargedParticleFactory.hpp"
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodic.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodicParallel.h"
#include "Particle/PairBuilder/PairConditions.h"
#include "VTKFieldWriterParallel.hpp"
#include "math.h"

// dimension of our positions
const unsigned Dim = 3;
const double ke    = 1. / (4. * M_PI);

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
typedef FFT<CCTransform, Dim, double> FFT_t;

typedef IntCIC IntrplCIC_t;
typedef IntNGP IntrplNGP_t;
typedef IntTSC IntrplTSC_t;

typedef UniformCartesian<2, double> Mesh2d_t;
typedef CenteredFieldLayout<2, Mesh2d_t, Center_t> FieldLayout2d_t;
typedef Field<double, 2, Mesh2d_t, Center_t> Field2d_t;

template <class T>
struct ApplyField;

// This is the periodic Greens function with regularization parameter epsilon.
template <unsigned int Dim>
struct SpecializedGreensFunction {};

template <>
struct SpecializedGreensFunction<3> {
    template <class T, class FT, class FT2>
    static void calculate(Vektor<T, 3>& hrsq, FT& grn, FT2* grnI, double alpha, double eps) {
        double r;
        NDIndex<3> elem0     = NDIndex<3>(Index(0, 0), Index(0, 0), Index(0, 0));
        grn                  = grnI[0] * hrsq[0] + grnI[1] * hrsq[1] + grnI[2] * hrsq[2];
        NDIndex<3> lDomain_m = grn.getLayout().getLocalNDIndex();
        NDIndex<3> elem;
        for (int i = lDomain_m[0].min(); i <= lDomain_m[0].max(); ++i) {
            elem[0] = Index(i, i);
            for (int j = lDomain_m[1].min(); j <= lDomain_m[1].max(); ++j) {
                elem[1] = Index(j, j);
                for (int k = lDomain_m[2].min(); k <= lDomain_m[2].max(); ++k) {
                    elem[2] = Index(k, k);
                    r       = real(sqrt(grn.localElement(elem)));
                    if (elem == elem0)
                        // grn.localElement(elem) = 0 ;
                        // grn.localElement(elem) = ke*std::complex<double>(2*alpha/sqrt(M_PI)) ;
                        grn.localElement(elem) =
                            ke * std::complex<double>(erf(alpha * r) / (r + eps));
                    else
                        grn.localElement(elem) = ke * std::complex<double>(erf(alpha * r) / (r));
                }
            }
        }

        // grn[0][0][0] = grn[0][0][1];
    }
};

/*
template<>
struct SpecializedGreensFunction<3> {
        template<class T, class FT, class FT2>
                static void calculate(Vektor<T, 3> &hrsq, FT &grn, FT2 *grnI, double alpha,double
eps) { double r; grn = grnI[0] * hrsq[0] + grnI[1] * hrsq[1] + grnI[2] * hrsq[2]; NDIndex<3>
lDomain_m = grn.getLayout().getLocalNDIndex(); NDIndex<3> elem; for (int i=lDomain_m[0].min();
i<=lDomain_m[0].max(); ++i) { elem[0]=Index(i,i); for (int j=lDomain_m[1].min();
j<=lDomain_m[1].max(); ++j) { elem[1]=Index(j,j); for (int k=lDomain_m[2].min();
k<=lDomain_m[2].max(); ++k) { elem[2]=Index(k,k); r = real(sqrt(grn.localElement(elem)));
                                                grn.localElement(elem)
= 1./(4.*M_PI)*std::complex<double>(erf(alpha*r)/(r+eps));
                                        }
                                }
                        }
                //grn[0][0][0] = grn[0][0][1];
                }
};
*/

template <class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double> Q;
    ParticleAttrib<double> m;
    ParticleAttrib<double> SpaceQ;
    ParticleAttrib<double> Phi;  // electrostatic potential
    ParticleAttrib<Vector_t> EF;
    ParticleAttrib<Vector_t> v;                 // velocity of the particles
    ParticleAttrib<int> ID;                     // velocity of the particles
    ParticleAttrib<Vektor<double, 2> > Rphase;  // velocity of the particles

    ChargedParticles(PL* pl, Vektor<double, 3> nr, e_dim_tag /*decomp*/[Dim],
                     Vektor<double, 3> extend_l_, Vektor<double, 3> extend_r_, Vektor<int, 3> Nx_,
                     Vektor<int, 3> Nv_, Vektor<double, 3> Vmax_)
        : IpplParticleBase<PL>(pl)
        , nr_m(nr)
        , extend_l(extend_l_)
        , extend_r(extend_r_)
        , Nx(Nx_)
        , Nv(Nv_)
        , Vmax(Vmax_) {
        this->addAttribute(Q);
        this->addAttribute(m);
        this->addAttribute(SpaceQ);
        this->addAttribute(Phi);
        this->addAttribute(EF);
        this->addAttribute(v);
        this->addAttribute(ID);
        this->addAttribute(Rphase);

        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            // use periodic boundary conditions for the particles
            this->getBConds()[i] = ParticlePeriodicBCond;
            // boundary conditions used for interpolation kernels allow writes to ghost cells

            if (Ippl::getNodes() > 1) {
                bc_m[i] = new ParallelInterpolationFace<double, Dim, Mesh_t, Center_t>(i);
                // std periodic boundary conditions for gradient computations etc.
                vbc_m[i] = new ParallelPeriodicFace<Vector_t, Dim, Mesh_t, Center_t>(i);
                bcp_m[i] = new ParallelPeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            } else {
                bc_m[i] = new InterpolationFace<double, Dim, Mesh_t, Center_t>(i);
                // std periodic boundary conditions for gradient computations etc.
                vbc_m[i] = new PeriodicFace<Vector_t, Dim, Mesh_t, Center_t>(i);
                bcp_m[i] = new PeriodicFace<double, Dim, Mesh_t, Center_t>(i);
            }
        }

        for (unsigned int d = 0; d < Dim; ++d) {
            rmax_m[d] = extend_r[d];
            rmin_m[d] = extend_l[d];
        }
        // Initialize the meshes and layouts for 2D phase space interpolation:
        double spacings[2] = {(extend_r[2] - extend_l[2]) / (Nx[2]), 2. * Vmax[2] / (Nv[2])};
        Vektor<double, 2> origin;

        origin(0) = extend_l[2];
        origin(1) = -Vmax[2];
        Index I(Nx[2] + 1);
        Index J(Nv[2] + 1);
        domain2d_m[0] = I;
        domain2d_m[1] = J;

        mesh2d_m   = Mesh2d_t(domain2d_m, spacings, origin);
        layout2d_m = new FieldLayout2d_t(mesh2d_m);

        BConds<double, 2, UniformCartesian<2, double>, Cell> BC;
        if (Ippl::getNodes() > 1) {
            BC[0] = new ParallelInterpolationFace<double, 2, Mesh2d_t, Cell>(0);
            BC[1] = new ParallelInterpolationFace<double, 2, Mesh2d_t, Cell>(1);
            BC[2] = new ParallelInterpolationFace<double, 2, Mesh2d_t, Cell>(2);
            BC[3] = new ParallelInterpolationFace<double, 2, Mesh2d_t, Cell>(3);
        } else {
            BC[0] = new InterpolationFace<double, 2, Mesh2d_t, Cell>(0);
            BC[1] = new InterpolationFace<double, 2, Mesh2d_t, Cell>(1);
            BC[2] = new InterpolationFace<double, 2, Mesh2d_t, Cell>(2);
            BC[3] = new InterpolationFace<double, 2, Mesh2d_t, Cell>(3);
        }
        // set origin and spacing is needed for correct results, even if mesh was created with these
        // paremeters ?!
        mesh2d_m.set_meshSpacing(&(spacings[0]));
        mesh2d_m.set_origin(origin);

        domain2d_m = layout2d_m->getDomain();
        // f_m is used for twostream instability as 2D phase space mesh
        f_m.initialize(mesh2d_m, *layout2d_m, GuardCellSizes<2>(1), BC);

        domain_m  = this->getFieldLayout().getDomain();
        lDomain_m = this->getFieldLayout().getLocalNDIndex();  // local domain

        // initialize the FFT
        bool compressTemps = true;
        fft_m              = new FFT_t(domain_m, compressTemps);

        fft_m->setDirectionName(+1, "forward");
        fft_m->setDirectionName(-1, "inverse");
        INFOMSG("INIT FFT DONE" << endl);

        /*
                                ///////test code to compute local domain boundaries
           //////////////////// domain_m = this->getFieldLayout().getDomain(); lDomain_m =
           this->getFieldLayout().getLocalNDIndex(); // local domain for (unsigned int d =
           0;d<Dim;++d) { hr_m[d] = (extend_r[d] - extend_l[d]) / (nr_m[d]);
                        }
                        std::cout <<"Node " << Ippl::myNode() << " has ldomain " << lDomain_m <<
           std::endl; std::cout <<"Node " << Ippl::myNode() << " has x slice (" <<
           lDomain_m[0].first()*hr_m[0] << " , " <<  lDomain_m[0].last()*hr_m[0] << ")" <<
           std::endl; std::cout <<"Node " << Ippl::myNode() << " has y slice (" <<
           lDomain_m[1].first()*hr_m[1] << " , " <<  lDomain_m[1].last()*hr_m[0] << ")" <<
           std::endl; std::cout <<"Node " << Ippl::myNode() << " has z slice (" <<
           lDomain_m[2].first()*hr_m[2] << " , " <<  lDomain_m[2].last()*hr_m[0] << ")" <<
           std::endl;

                        ////////////////////////////////////////////////////////////////////////
        */
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
        // should only be needed if meshspacing changes -----------
        for (unsigned int d = 0; d < Dim; ++d) {
            hr_m[d] = (extend_r[d] - extend_l[d]) / (nr_m[d]);
        }
        this->getMesh().set_meshSpacing(&(hr_m[0]));
        this->getMesh().set_origin(extend_l);
        //--------------------------------------------------------

        // init resets the meshes to 0 ?!
        rhocmpl_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1));
        grncmpl_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1));
        rho_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bc_m);
        phi_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), bcp_m);
        eg_m.initialize(getMesh(), getFieldLayout(), GuardCellSizes<Dim>(1), vbc_m);

        domain_m  = this->getFieldLayout().getDomain();
        lDomain_m = this->getFieldLayout().getLocalNDIndex();

        IpplParticleBase<PL>::update();
    }

    void interpolate_distribution(Vektor<double, 3> dx, Vektor<double, 3> dv) {
        f_m = 0;
        for (unsigned i = 0; i < this->getLocalNum(); ++i) {
            SpaceQ[i] = Q[i] / (dx[0] * dx[1] * dv[0] * dv[1]);
            Rphase[i] = Vektor<double, 2>(this->R[i][2], v[i][2]);
        }
        // this->SpaceQ.scatter(this->f_m, this->Rphase, IntrplCIC_t());
        this->Q.scatter(this->f_m, this->Rphase, IntrplCIC_t());
    }

    void calc_kinetic_energy() {
        kinetic_energy = 0;
        double v2;
        for (unsigned i = 0; i < this->getTotalNum(); ++i) {
            v2 = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
            kinetic_energy += 0.5 * v2;
        }
    }

    void calc_field_energy() {
        NDIndex<3> elem;
        double cell_volume = hr_m[0] * hr_m[1] * hr_m[2];
        field_energy       = 0;
        field_energy       = 0.5 * cell_volume * sum(dot(eg_m, eg_m));

        rhomax = max(abs(rho_m)) / (hr_m[0] * hr_m[1] * hr_m[2]);
        // rhomax=max(rho_m);
        integral_phi_m = 0.5 * sum(rho_m * phi_m);
    }

    void calc_potential_energy() {
        potential_energy = 0;
        for (unsigned i = 0; i < this->getLocalNum(); ++i) {
            potential_energy += 0.5 * (Q[i]) * Phi[i];
        }
    }

    void calc_Amplitude_E() {
        // computes the maximum amplitude in the electric field
        AmplitudeEfield = max(sqrt(dot(eg_m, eg_m)));
        eg_m            = eg_m * Vektor<double, 3>(0, 0, 1);
        AmplitudeEFz    = max(sqrt(dot(eg_m, eg_m)));
    }

    void calculatePairForces(double interaction_radius, double eps, double alpha);

    // compute Greens function without making use of index computations
    void calcGrealSpace(double alpha, double eps) {
        NDIndex<3> elem, elem_mirr, elem_all;
        Vector_t hrsq(hr_m * hr_m);
        int N = domain_m[0].max();
        // Loop over first octant of domain
        for (int i = lDomain_m[0].min(); i <= (lDomain_m[0].max()) / 2. + 1; ++i) {
            elem[0]      = Index(i, i);
            elem_mirr[0] = Index(N - i + 1, N - i + 1);
            for (int j = lDomain_m[1].min(); j <= (lDomain_m[1].max()) / 2. + 1; ++j) {
                elem[1]      = Index(j, j);
                elem_mirr[1] = Index(N - j + 1, N - j + 1);
                for (int k = lDomain_m[2].min(); k <= (lDomain_m[2].max()) / 2. + 1; ++k) {
                    elem[2]      = Index(k, k);
                    elem_mirr[2] = Index(N - k + 1, N - k + 1);

                    double val = 0;
                    double r   = 0;
                    r          = sqrt((i) * (i)*hr_m[0] * hr_m[0] + (j) * (j)*hr_m[1] * hr_m[1]
                                      + (k) * (k)*hr_m[2] * hr_m[2]);
                    val        = 1. / (4. * M_PI) * erf(alpha * r) / (r + eps);

                    elem_all                         = elem;
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on x axis
                    elem_all[0]                      = elem_mirr[0];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on y axis
                    elem_all[0]                      = elem[0];
                    elem_all[1]                      = elem_mirr[1];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on z axis
                    elem_all[1]                      = elem[1];
                    elem_all[2]                      = elem_mirr[2];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on x,y axis
                    elem_all[0]                      = elem_mirr[0];
                    elem_all[1]                      = elem_mirr[1];
                    elem_all[2]                      = elem[2];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on y,z axis
                    elem_all[0]                      = elem[0];
                    elem_all[1]                      = elem_mirr[1];
                    elem_all[2]                      = elem_mirr[2];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on z,x axis
                    elem_all[0]                      = elem_mirr[0];
                    elem_all[1]                      = elem[1];
                    elem_all[2]                      = elem_mirr[2];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                    // mirror on x,y,z axis
                    elem_all[0]                      = elem_mirr[0];
                    elem_all[1]                      = elem_mirr[1];
                    elem_all[2]                      = elem_mirr[2];
                    grncmpl_m.localElement(elem_all) = std::complex<double>(val);
                }
            }
        }
    }

    void calculateGridForces(double /*interaction_radius*/, double alpha, double eps, int it = 0,
                             bool normalizeSphere = 0) {
        // (1) scatter charge to charge density grid and transform to fourier space
        // this->Q.scatter(this->rho_m, this->R, IntrplTSC_t());
        rho_m[domain_m] = 0;  //!!!!!! there has to be a better way than setting rho to 0 every time
        this->Q.scatter(this->rho_m, this->R, IntrplCIC_t());
        // this->Q.scatter(this->rho_m, this->R, IntrplNGP_t());
        dumpVTKScalar(rho_m, this, it, "RhoInterpol");

        // rhocmpl_m[domain_m] = rho_m[domain_m];
        rhocmpl_m[domain_m] = rho_m[domain_m] / (hr_m[0] * hr_m[1] * hr_m[2]);
        RhoSum              = sum(real(rhocmpl_m));

        std::cout << "total charge in densitty field before ion subtraction is"
                  << sum(real(rhocmpl_m)) << std::endl;
        std::cout << "max total charge in densitty field before ion subtraction is"
                  << max(real(rhocmpl_m)) << std::endl;
        // subtract the background charge of the ions
        if (normalizeSphere)
            rhocmpl_m[domain_m] = 1. / (hr_m[0] * hr_m[1] * hr_m[2] * (nr_m[0] * nr_m[1] * nr_m[2]))
                                  + rhocmpl_m[domain_m];
        else
            rhocmpl_m[domain_m] = 1. + rhocmpl_m[domain_m];
        std::cout << "total charge in densitty field after ion subtraction is"
                  << sum(real(rhocmpl_m)) << std::endl;

        // compute rhoHat and store in rhocmpl_m
        fft_m->transform("inverse", rhocmpl_m);

        // (2) compute Greens function in real space and transform to fourier space
        // calcGrealSpace(alpha,eps);

        /////////compute G with Index Magic///////////////////
        // Fields used to eliminate excess calculation in greensFunction()
        IField_t grnIField_m[3];

        // mesh and layout objects for rho_m
        Mesh_t* mesh_m          = &(getMesh());
        FieldLayout_t* layout_m = &(getFieldLayout());

        // This loop stores in grnIField_m[i] the index of the ith dimension mirrored at the central
        // axis. e.g. grnIField_m[0]=[(0 1 2 3 ... 3 2 1) ; (0 1 2 3 ... 3 2 1; ...)]
        for (int i = 0; i < 3; ++i) {
            grnIField_m[i].initialize(*mesh_m, *layout_m);
            grnIField_m[i][domain_m] =
                where(lt(domain_m[i], nr_m[i] / 2), domain_m[i] * domain_m[i],
                      (nr_m[i] - domain_m[i]) * (nr_m[i] - domain_m[i]));
        }
        Vector_t hrsq(hr_m * hr_m);
        SpecializedGreensFunction<3>::calculate(hrsq, grncmpl_m, grnIField_m, alpha, eps);
        /////////////////////////////////////////////////

        // transform G -> Ghat and store in grncmpl_m
        fft_m->transform("inverse", grncmpl_m);
        // multiply in fourier space and obtain PhiHat in rhocmpl_m
        rhocmpl_m *= grncmpl_m;

        // (3) Backtransformation: compute potential field in real space and E=-Grad Phi
        // compute electrostatic potential Phi in real space by FFT PhiHat -> Phi and store it in
        // rhocmpl_m
        fft_m->transform("forward", rhocmpl_m);

        // take only the real part and store in phi_m (has periodic bc instead of interpolation bc)
        phi_m = real(rhocmpl_m) * hr_m[0] * hr_m[1] * hr_m[2];
        dumpVTKScalar(phi_m, this, it, "Phi_m");

        // compute Electric field on the grid by -Grad(Phi) store in eg_m
        std::cout << "compute gradient now" << std::endl;
        eg_m = -Grad1Ord(phi_m, eg_m);

        // interpolate the electric field to the particle positions
        EF.gather(eg_m, this->R, IntrplCIC_t());
        // interpolate electrostatic potenital to the particle positions
        Phi.gather(phi_m, this->R, IntrplCIC_t());
    }

    Vector_t getRmin() { return this->rmin_m; }
    Vector_t getRmax() { return this->rmax_m; }

    Vector_t get_hr() { return hr_m; }

    // private:
    BConds<double, Dim, Mesh_t, Center_t> bc_m;
    BConds<double, Dim, Mesh_t, Center_t> bcp_m;
    BConds<Vector_t, Dim, Mesh_t, Center_t> vbc_m;

    CxField_t rhocmpl_m;
    CxField_t grncmpl_m;

    Field_t rho_m;
    Field_t phi_m;

    VField_t eg_m;

    Vektor<int, Dim> nr_m;
    Vector_t hr_m;
    Vector_t rmax_m;
    Vector_t rmin_m;
    Vektor<double, Dim> extend_l;
    Vektor<double, Dim> extend_r;
    Mesh_t* mesh_m;
    FieldLayout_t* layout_m;
    NDIndex<Dim> domain_m;
    NDIndex<Dim> lDomain_m;

    double kinetic_energy;
    double field_energy;
    double field_energy_gather;
    double integral_phi_m;
    double potential_energy;
    double AmplitudeEfield;
    double AmplitudeEFz;
    double total_charge;
    double rhomax;

    FFT_t* fft_m;

    e_dim_tag decomp_m[Dim];

    Vektor<int, Dim> Nx;
    Vektor<int, Dim> Nv;
    Vektor<double, Dim> Vmax;
    // Fields for tracking distribution function
    Field2d_t f_m;
    Mesh2d_t mesh2d_m;
    NDIndex<2> domain2d_m;
    FieldLayout2d_t* layout2d_m;
    // TEMP debug variable
    double RhoSum = 0;
};

template <class T>
struct ApplyField {
    ApplyField(T c, double r, double epsilon, double alpha)
        : C(c)
        , R(r)
        , eps(epsilon)
        , a(alpha) {}
    void operator()(std::size_t i, std::size_t j, ChargedParticles<playout_t>& P,
                    Vektor<double, 3>& shift) const {
        Vector_t diff = P.R[i] - (P.R[j] + shift);
        double sqr    = 0;

        for (unsigned d = 0; d < Dim; ++d)
            sqr += diff[d] * diff[d];

        // compute r with softening parameter, unsoftened r is obtained by sqrt(sqr)
        if (sqr != 0) {
            double r = std::sqrt(sqr + eps * eps);

            // for order two transition
            if (P.Q[i] != 0 && P.Q[j] != 0) {
                // compute potential energy
                // double phi =1./(4.*M_PI)*(1.-erf(a*sqrt(sqr)))/r;
                double phi = ke * (1. - erf(a * sqrt(sqr))) / r;

                // compute force
                // Vector_t Fij
                // = 1./(4.*M_PI)*C*(diff/sqrt(sqr))*((2.*a*exp(-a*a*sqr))/(sqrt(M_PI)*r)+(1.-erf(a*sqrt(sqr)))/(r*r));
                Vector_t Fij = ke * C * (diff / sqrt(sqr))
                               * ((2. * a * exp(-a * a * sqr)) / (sqrt(M_PI) * r)
                                  + (1. - erf(a * sqrt(sqr))) / (r * r));

                // Actual Force is F_ij multiplied by Qi*Qj
                // The electrical field on particle i is E=F/q_i and hence:
                P.EF[i] -= P.Q[j] * Fij;
                P.EF[j] += P.Q[i] * Fij;
                // update potential per particle
                P.Phi[i] += P.Q[j] * phi;
                P.Phi[j] += P.Q[i] * phi;
            }
        }
    }
    T C;
    double R;
    double eps;
    double a;
};

template <class PL>
void ChargedParticles<PL>::calculatePairForces(double interaction_radius, double eps,
                                               double alpha) {
    if (interaction_radius > 0) {
        if (Ippl::getNodes() > 1) {
            HashPairBuilderPeriodicParallel<ChargedParticles<playout_t> > HPB(*this);
            HPB.for_each(RadiusCondition<double, Dim>(interaction_radius),
                         ApplyField<double>(-1, interaction_radius, eps, alpha), extend_l,
                         extend_r);
        } else {
            HashPairBuilderPeriodic<ChargedParticles<playout_t> > HPB(*this);
            HPB.for_each(RadiusCondition<double, Dim>(interaction_radius),
                         ApplyField<double>(-1, interaction_radius, eps, alpha), extend_l,
                         extend_r);
        }
    }
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        IpplTimings::TimerRef allTimer = IpplTimings::getTimer("AllTimer");
        IpplTimings::startTimer(allTimer);

        Vektor<int, Dim> nr;

        enum TestType {
            SPHERE,
            RECURRENCE,
            LANDAU,
            TWOSTREAM
        };
        TestType test  = SPHERE;
        unsigned param = 1;
        if (argv[param] == std::string("sphere"))
            test = SPHERE;
        else if (argv[param] == std::string("recurrence"))
            test = RECURRENCE;
        else if (argv[param] == std::string("landau"))
            test = LANDAU;
        else if (argv[param] == std::string("twostream"))
            test = TWOSTREAM;
        else if (argv[param] == std::string("recurrence"))
            std::cout
                << "Please call the program with a valid test from [sphere, recurrence, landau, "
                   "twostream]"
                << std::endl;

        nr    = Vektor<int, Dim>(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
        param = 5;

        double interaction_radius = atof(argv[param++]);

        // read the remaining sim params
        double alpha   = atof(argv[param++]);
        double dt      = atof(argv[param++]);
        double eps     = atof(argv[param++]);
        int iterations = atoi(argv[param++]);
        Vektor<double, 3> source(0, 0, 0);
        source[0] = atof(argv[param++]);
        source[1] = atof(argv[param++]);
        source[2] = atof(argv[param++]);

        ///////// setup the initial layout ///////////////////////////////////////
        e_dim_tag decomp[Dim];
        Mesh_t* mesh;
        FieldLayout_t* FL;
        ChargedParticles<playout_t>* P;

        NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++)
            domain[i] = domain[i] = Index(nr[i] + 1);

        for (unsigned d = 0; d < Dim; ++d)
            decomp[d] = PARALLEL;

        // create mesh and layout objects for this problem domain
        mesh          = new Mesh_t(domain);
        FL            = new FieldLayout_t(*mesh, decomp);
        playout_t* PL = new playout_t(*FL, *mesh);

        PL->setAllCacheDimensions(interaction_radius);
        PL->enableCaching();

        /////////// Create the particle distribution
        ////////////////////////////////////////////////////////

        switch (test) {
            case TWOSTREAM: {
                double k = 0.5;
                Vektor<double, Dim> extend_r(2. * M_PI / k, 2. * M_PI / k, 2. * M_PI / k);
                Vektor<double, Dim> extend_l(0, 0, 0);

                double ampl_alpha = 0.05;
                Vektor<int, Dim> Nx(4, 4, 32);
                Vektor<int, Dim> Nv(8, 8, 128);
                Vektor<double, Dim> Vmax(6, 6, 6);
                P = new ChargedParticles<playout_t>(PL, nr, decomp, extend_l, extend_r, Nx, Nv,
                                                    Vmax);
                createParticleDistributionTwoStream(P, extend_l, extend_r, Nx, Nv, Vmax,
                                                    ampl_alpha);
                P->interpolate_distribution((extend_r - extend_l) / (Nx), 2. * Vmax / (Nv));
                write_f_field(P->f_m, P, 0, "f_m");
                break;
            }

            case RECURRENCE: {
                double k = 0.5;
                Vektor<double, Dim> extend_l(0, 0, 0);
                Vektor<double, Dim> extend_r(2. * M_PI / k, 2. * M_PI / k, 2. * M_PI / k);

                double ampl_alpha = 0.01;
                Vektor<int, Dim> Nx(8, 8, 8);
                Vektor<int, Dim> Nv(32, 32, 32);
                Vektor<double, Dim> Vmax(6, 6, 6);
                P = new ChargedParticles<playout_t>(PL, nr, decomp, extend_l, extend_r, Nx, Nv,
                                                    Vmax);
                createParticleDistributionRecurrence(P, extend_l, extend_r, Nx, Nv, Vmax,
                                                     ampl_alpha);
                break;
            }
            case LANDAU: {
                double k = 0.5;
                Vektor<double, Dim> extend_l(0, 0, 0);
                Vektor<double, Dim> extend_r(2. * M_PI / k, 2. * M_PI / k, 2. * M_PI / k);

                double ampl_alpha = 0.05;
                Vektor<int, Dim> Nx(8, 8, 8);
                Vektor<int, Dim> Nv(32, 32, 32);
                Vektor<double, Dim> Vmax(6, 6, 6);
                P = new ChargedParticles<playout_t>(PL, nr, decomp, extend_l, extend_r, Nx, Nv,
                                                    Vmax);
                createParticleDistributionLandau(P, extend_l, extend_r, Nx, Nv, Vmax, ampl_alpha);

                break;
            }
            case SPHERE: {
                double L = 4.;
                Vektor<double, Dim> extend_l(-L, -L, -L);
                Vektor<double, Dim> extend_r(L, L, L);

                Vektor<int, Dim> Nx(16, 16, 16);
                Vektor<int, Dim> Nv(16, 16, 16);
                Vektor<double, Dim> Vmax(6, 6, 6);
                P = new ChargedParticles<playout_t>(PL, nr, decomp, extend_l, extend_r, Nx, Nv,
                                                    Vmax);
                createParticleDistribution(P, "random", 20000, 0.00005, extend_l, extend_r, source,
                                           1., 0);

                break;
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////

        /////// Print mesh informations ////////////////////////////////////////////////////////////
        INFOMSG(P->getMesh() << endl);
        INFOMSG(P->getFieldLayout() << endl);
        msg << endl << endl;
        ippl::Comm->barrier();

        // dumpParticlesCSV(P,0);

        INFOMSG(P->getMesh() << endl);
        INFOMSG(P->getFieldLayout() << endl);
        msg << endl << endl;

        msg << "number of particles = " << endl;
        msg << P->getTotalNum() << endl;
        msg << "Total charge Q = " << endl;
        msg << P->total_charge << endl;
        ////////////////////////////////////////////////////////////////////////////////////////////

        msg << "Starting iterations ..." << endl;

        // calculate initial grid forces
        if (test == SPHERE)
            P->calculateGridForces(interaction_radius, alpha, eps, 0, 1);
        else
            P->calculateGridForces(interaction_radius, alpha, eps, 0, 0);

        dumpVTKVector(P->eg_m, P, 0, "EFieldAfterPMandPP");

        // compute quantities to check correctness:
        P->calc_field_energy();
        P->calc_potential_energy();
        P->calc_kinetic_energy();
        writeEnergy(P, 0);

        if (test == TWOSTREAM) {
            P->calc_Amplitude_E();
            writeEamplitude(P, 0);
        }
        for (int it = 0; it < iterations; it++) {
            // advance the particle positions
            // basic leapfrogging timestep scheme.  velocities are offset
            // by half a timestep from the positions.

            assign(P->R, P->R + dt * P->v);
            // update particle distribution across processors
            P->update();

            // compute the electric field
            msg << "calculating grid" << endl;
            IpplTimings::TimerRef gridTimer = IpplTimings::getTimer("GridTimer");
            IpplTimings::startTimer(gridTimer);

            if (test == SPHERE)
                P->calculateGridForces(interaction_radius, alpha, eps, it + 1, 1);
            else
                P->calculateGridForces(interaction_radius, alpha, eps, it + 1, 0);

            IpplTimings::stopTimer(gridTimer);

            msg << "calculating pairs" << endl;

            IpplTimings::TimerRef particleTimer = IpplTimings::getTimer("ParticleTimer");
            IpplTimings::startTimer(particleTimer);

            P->calculatePairForces(interaction_radius, eps, alpha);
            IpplTimings::stopTimer(particleTimer);

            P->update();

            if (test == SPHERE) {
                if (it % 1 == 0) {
                    // dumpVTKVector(P->eg_m, P,it+1,"EFieldAfterPMandPP");
                }
            }

            // second part of leapfrog: advance velocitites
            if (test != SPHERE)
                assign(P->v, P->v + dt * P->Q / P->m * P->EF);

            if (test == SPHERE) {
                // Print the particle positions
                dumpConservedQuantities(P, it + 1);
                if (it % 1 == 0) {
                    // dumpParticlesCSV(P,it+1);
                }
            }

            // compute quantities
            P->calc_field_energy();
            P->calc_kinetic_energy();
            P->calc_potential_energy();
            writeEnergy(P, it + 1);
            if (test == TWOSTREAM) {
                P->calc_Amplitude_E();
                writeEamplitude(P, it + 1);
                msg << "start interpolation to phase space " << endl;
                P->interpolate_distribution((P->extend_r - P->extend_l) / (P->Nx),
                                            2. * P->Vmax / (P->Nv));
                write_f_field(P->f_m, P, it + 1, "f_m");
            }
            msg << "Finished iteration " << it << endl;
        }
        ippl::Comm->barrier();

        msg << "number of particles = " << endl;
        msg << P->getTotalNum() << endl;

        IpplTimings::stopTimer(allTimer);

        IpplTimings::print();

        if (test == SPHERE) {
            double sphere_radius = 1;
            std::cout << "total E-Field = " << sum(dot(P->EF, P->EF)) << std::endl;
            std::cout << "total Potential = " << sum(P->Phi) << std::endl;
            std::cout << "source = " << source << std::endl;
            std::cout << "RhoSum = " << P->RhoSum << std::endl;

            double error = 0, errorV = 0;
            double l2_exact = 0, l2_V_exact = 0;
            double total_E_exact = 0, total_V_exact = 0;
            double total_E = 0, total_V = 0;

            double total_charge = 1.;
            double k0           = 1. / (4. * M_PI);

            for (unsigned i = 0; i < P->getLocalNum(); ++i) {
                double radius = std::sqrt(dot(source - P->R[i], source - P->R[i]));
                double E      = std::sqrt(dot(P->EF[i], P->EF[i]));
                double V      = P->Phi[i];

                double diff = 0, diffV = 0;
                double exact = 0, exactV = 0;

                if (radius <= sphere_radius) {
                    exact =
                        k0
                        * (total_charge * radius / (sphere_radius * sphere_radius * sphere_radius));
                    diff = E - exact;

                    exactV = k0 * total_charge / (2. * sphere_radius)
                             * (3. - radius * radius / (sphere_radius * sphere_radius));
                    diffV = V - exactV;
                } else {
                    if (radius > 0) {
                        exact = k0 * (total_charge * 1. / (radius * radius));
                        diff  = E - exact;

                        exactV = k0 * total_charge / radius;
                        diff   = V - exactV;
                    }
                }

                total_E += E;
                total_V += V;
                total_E_exact += exact;
                total_V_exact += exactV;

                error += diff * diff;
                errorV += diffV * diffV;

                l2_exact += exact * exact;
                l2_V_exact += exactV * exactV;
            }
            // reduce all relevant quantities:
            double Error, L2_exact, ErrorV, L2_V_exact, Total_E, Total_V, Total_E_exact,
                Total_V_exact, Potential_energy;
            reduce(error, Error, OpAddAssign());
            reduce(l2_exact, L2_exact, OpAddAssign());
            reduce(errorV, ErrorV, OpAddAssign());
            reduce(l2_V_exact, L2_V_exact, OpAddAssign());

            reduce(total_E, Total_E, OpAddAssign());
            reduce(total_V, Total_V, OpAddAssign());
            reduce(total_E_exact, Total_E_exact, OpAddAssign());
            reduce(total_V_exact, Total_V_exact, OpAddAssign());

            reduce(P->potential_energy, Potential_energy, OpAddAssign());

            double Relative_error   = std::sqrt(Error) / std::sqrt(L2_exact);
            double Relative_V_error = std::sqrt(ErrorV) / std::sqrt(L2_V_exact);
            // double U = k0*3./5.*total_charge*total_charge/sphere_radius; //electric energy stored
            // in solid charged sphere for infinite domain

            double Ufinite =
                0.0394785;  // electric energy stored in solid charged sphere for finite domain 8^3
            double U = Ufinite;

            if (Ippl::myNode() == 0) {
                std::cout << "master node prints: Q = " << total_charge << std::endl;
                Inform ofs(NULL, "data/statistics.txt", Inform::APPEND);
                // mesh size , n particle, r_cut, alpha, smoothing eps, absolut_err, relative error,
                // relative error in total E-field, absolut_V_err, relative V error, relative error
                // in total V, deviation in sum(U) from solid sphere
                ofs << nr[0] << "," << P->getTotalNum() << "," << interaction_radius << "," << eps
                    << "," << alpha << "," << Error << "," << Relative_error << ","
                    << fabs(Total_E - Total_E_exact) / Total_E_exact << "," << ErrorV << ","
                    << Relative_V_error << "," << fabs(Total_V - Total_V_exact) / Total_V_exact
                    << "," << std::abs(Potential_energy - U) / U << std::endl;
            }
        }

        delete P;
        delete FL;
        delete mesh;
    }
    ippl::finalize();

    return 0;
}
