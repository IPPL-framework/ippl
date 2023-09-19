//
// Application p3m3dHeating
//   mpirun -np 4 ./p3m3dHeating Nx Ny Nz l_beam l_box particleDensity r_cut alpha dt
//                               eps iterations charge_per_part m_part printEvery
//
//   alpha is the splitting parameter for pm and pp,
//   eps is the smoothing factor and Si are the coordinates of the charged sphere center
//
// Benjamin Ulmer, ETH Zürich (2016)
// Implemented as part of the Master thesis
// "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
// (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
//
#include "Ippl.h"

#include <cfloat>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Utility/PAssert.h"

#include "ChargedParticleFactory.hpp"
#include "H5hut.h"
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodic.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodicParallel.h"
#include "Particle/PairBuilder/PairConditions.h"
#include "VTKFieldWriterParallel.hpp"
#include "math.h"

// dimension of our positions
const unsigned Dim = 3;
// const double ke=1./(4.*M_PI*8.8e-14);
const double ke = 2.532638e8;
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
                    if (elem == elem0) {
                        // grn.localElement(elem) = ke*std::complex<double>(2*alpha/sqrt(M_PI)) ;
                        grn.localElement(elem) = 0;
                    } else
                        grn.localElement(elem) =
                            ke * std::complex<double>(erf(alpha * r) / (r + eps));
                }
            }
        }
    }
};

template <class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double> Q;
    ParticleAttrib<double> m;
    ParticleAttrib<double> Phi;  // electrostatic potential
    ParticleAttrib<Vector_t> EF;
    ParticleAttrib<Vector_t> v;  // velocity of the particles
    ParticleAttrib<int> ID;      // velocity of the particles

    ChargedParticles(PL* pl, Vektor<double, 3> nr, e_dim_tag /*decomp*/[Dim],
                     Vektor<double, 3> extend_l_, Vektor<double, 3> extend_r_)
        : IpplParticleBase<PL>(pl)
        , nr_m(nr)
        , extend_l(extend_l_)
        , extend_r(extend_r_) {
        this->addAttribute(Q);
        this->addAttribute(m);
        this->addAttribute(Phi);
        this->addAttribute(EF);
        this->addAttribute(v);
        this->addAttribute(ID);

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

        domain_m  = this->getFieldLayout().getDomain();
        lDomain_m = this->getFieldLayout().getLocalNDIndex();  // local domain

        // initialize the FFT
        bool compressTemps = true;
        fft_m              = new FFT_t(domain_m, compressTemps);

        fft_m->setDirectionName(+1, "forward");
        fft_m->setDirectionName(-1, "inverse");
        INFOMSG("INIT FFT DONE" << endl);
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

    void compute_temperature() {
        Inform m("compute_temperature ");
        double loc_temp[Dim]    = {0.0, 0.0, 0.0};
        double loc_avg_vel[Dim] = {0.0, 0.0, 0.0};

        for (unsigned long k = 0; k < this->getLocalNum(); ++k) {
            for (unsigned i = 0; i < Dim; i++) {
                loc_avg_vel[i] += this->v[k](i);
            }
        }
        reduce(&(loc_avg_vel[0]), &(loc_avg_vel[0]) + Dim, &(avg_vel[0]), OpAddAssign());

        const double N = static_cast<double>(this->getTotalNum());
        avg_vel[0]     = avg_vel[0] / N;
        avg_vel[1]     = avg_vel[1] / N;
        avg_vel[2]     = avg_vel[2] / N;

        m << "avg_vel[0]= " << avg_vel[0] << " avg_vel[1]= " << avg_vel[1]
          << " avg_vel[2]= " << avg_vel[2] << endl;

        for (unsigned long k = 0; k < this->getLocalNum(); ++k) {
            for (unsigned i = 0; i < Dim; i++) {
                loc_temp[i] += (this->v[k](i) - avg_vel[i]) * (this->v[k](i) - avg_vel[i]);
            }
        }
        reduce(&(loc_temp[0]), &(loc_temp[0]) + Dim, &(temperature[0]), OpAddAssign());
        temperature[0] = temperature[0] / N;
        temperature[1] = temperature[1] / N;
        temperature[2] = temperature[2] / N;
    }

    void calcMoments() {
        double part[2 * Dim];

        double loc_centroid[2 * Dim]        = {};
        double loc_moment[2 * Dim][2 * Dim] = {};
        double moments[2 * Dim][2 * Dim]    = {};

        for (unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for (unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = loc_moment[i][j];
            }
        }

        // double p0=m0*gamma*beta0;
        for (unsigned long k = 0; k < this->getLocalNum(); ++k) {
            part[1] = this->v[k](0);
            part[3] = this->v[k](1);
            part[5] = this->v[k](2);
            part[0] = this->R[k](0);
            part[2] = this->R[k](1);
            part[4] = this->R[k](2);

            for (unsigned i = 0; i < 2 * Dim; i++) {
                loc_centroid[i] += part[i];
                for (unsigned j = 0; j <= i; j++) {
                    loc_moment[i][j] += part[i] * part[j];
                }
            }
        }

        for (unsigned i = 0; i < 2 * Dim; i++) {
            for (unsigned j = 0; j < i; j++) {
                loc_moment[j][i] = loc_moment[i][j];
            }
        }

        reduce(&(loc_moment[0][0]), &(loc_moment[0][0]) + 2 * Dim * 2 * Dim, &(moments[0][0]),
               OpAddAssign());

        reduce(&(loc_centroid[0]), &(loc_centroid[0]) + 2 * Dim, &(centroid_m[0]), OpAddAssign());

        for (unsigned i = 0; i < 2 * Dim; i++) {
            for (unsigned j = 0; j <= i; j++) {
                moments_m[i][j] = moments[i][j];
                moments_m[j][i] = moments[i][j];
            }
        }
    }

    void computeBeamStatistics() {
        // const size_t locNp = this->getLocalNum();
        const double N    = static_cast<double>(this->getTotalNum());
        const double zero = 0.0;

        Vector_t eps2, fac, rsqsum, vsqsum, rvsum;
        for (unsigned int i = 0; i < Dim; i++) {
            rmean_m(i) = centroid_m[2 * i] / N;
            vmean_m(i) = centroid_m[(2 * i) + 1] / N;
            rsqsum(i)  = moments_m[2 * i][2 * i] - N * rmean_m(i) * rmean_m(i);
            vsqsum(i)  = moments_m[(2 * i) + 1][(2 * i) + 1] - N * vmean_m(i) * vmean_m(i);
            if (vsqsum(i) < 0)
                vsqsum(i) = 0;
            rvsum(i) = moments_m[(2 * i)][(2 * i) + 1] - N * rmean_m(i) * vmean_m(i);
        }

        eps2 = (rsqsum * vsqsum - rvsum * rvsum) / (N * N);
        rvsum /= N;

        for (unsigned int i = 0; i < Dim; i++) {
            rrms_m(i)  = sqrt(rsqsum(i) / N);
            vrms_m(i)  = sqrt(vsqsum(i) / N);
            eps_m(i)   = std::sqrt(std::max(eps2(i), zero));
            double tmp = rrms_m(i) * vrms_m(i);
            fac(i)     = (tmp == 0) ? zero : 1.0 / tmp;
        }
        rvrms_m = rvsum * fac;
    }

    void calc_kinetic_energy() {
        double loc_kinetic_energy = 0;
        double v2;
        for (unsigned i = 0; i < this->getLocalNum(); ++i) {
            v2 = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
            loc_kinetic_energy += 0.5 * m[i] * v2;
        }
        reduce(loc_kinetic_energy, kinetic_energy, OpAddAssign());
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

    void computeAvgSpaceChargeForces() {
        Inform m("computeAvgSpaceChargeForces ");

        const double N       = static_cast<double>(this->getTotalNum());
        double locAvgEF[Dim] = {};
        for (unsigned i = 0; i < this->getLocalNum(); ++i) {
            locAvgEF[0] += fabs(EF[i](0));
            locAvgEF[1] += fabs(EF[i](1));
            locAvgEF[2] += fabs(EF[i](2));
        }

        reduce(&(locAvgEF[0]), &(locAvgEF[0]) + Dim, &(globSumEF[0]), OpAddAssign());

        m << "globSumEF = " << globSumEF[0] << "\t" << globSumEF[1] << "\t" << globSumEF[2] << endl;

        avgEF[0] = globSumEF[0] / N;
        avgEF[1] = globSumEF[1] / N;
        avgEF[2] = globSumEF[2] / N;
    }

    void applyConstantFocusing(double f, double beam_radius) {
        double focusingForce = sqrt(dot(avgEF, avgEF));
        for (unsigned i = 0; i < this->getLocalNum(); ++i) {
            EF[i] += this->R[i] / beam_radius * f * focusingForce;
        }
    }

    void calculatePairForces(double interaction_radius, double eps, double alpha);

    void calculateGridForces(double /*interaction_radius*/, double alpha, double eps,
                             int /*it*/ = 0, bool /*normalizeSphere*/ = 0) {
        // (1) scatter charge to charge density grid and transform to fourier space
        // this->Q.scatter(this->rho_m, this->R, IntrplTSC_t());
        rho_m[domain_m] = 0;  //!!!!!! there has to be a better way than setting rho to 0 every time
        this->Q.scatter(this->rho_m, this->R, IntrplCIC_t());
        // this->Q.scatter(this->rho_m, this->R, IntrplNGP_t());
        // dumpVTKScalar(rho_m,this,it,"RhoInterpol");

        // rhocmpl_m[domain_m] = rho_m[domain_m];
        rhocmpl_m[domain_m] = rho_m[domain_m] / (hr_m[0] * hr_m[1] * hr_m[2]);
        RhoSum              = sum(real(rhocmpl_m));

        // std::cout << "total charge in densitty field before ion subtraction is" <<
        // sum(real(rhocmpl_m))<< std::endl; std::cout << "max total charge in densitty field before
        // ion subtraction is" << max(real(rhocmpl_m)) << std::endl; subtract the background charge
        // of the ions
        /*
         if (normalizeSphere)
         rhocmpl_m[domain_m]=1./(hr_m[0]*hr_m[1]*hr_m[2]*(nr_m[0]*nr_m[1]*nr_m[2]))+rhocmpl_m[domain_m];
         else
         rhocmpl_m[domain_m]=1.+rhocmpl_m[domain_m];
         */

        // std::cout << "total charge in densitty field after ion subtraction is" <<
        // sum(real(rhocmpl_m)) << std::endl;

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
        // dumpVTKScalar( phi_m, this,it, "Phi_m") ;

        // compute Electric field on the grid by -Grad(Phi) store in eg_m
        eg_m = -Grad1Ord(phi_m, eg_m);

        // interpolate the electric field to the particle positions
        EF.gather(eg_m, this->R, IntrplCIC_t());
        // interpolate electrostatic potenital to the particle positions
        Phi.gather(phi_m, this->R, IntrplCIC_t());
    }

    Vector_t getRmin() { return this->rmin_m; }
    Vector_t getRmax() { return this->rmax_m; }

    Vector_t get_hr() { return hr_m; }

    void closeH5() { H5CloseFile(H5f_m); }

    void openH5(std::string fn) {
        h5_prop_t props = H5CreateFileProp();
        MPI_Comm comm   = ippl::Comm->getCommunicator();
        h5_err_t h5err  = H5SetPropFileMPIOCollective(props, &comm);
#if defined(NDEBUG)
        (void)h5err;
#endif
        PAssert(h5err != H5_ERR);
        H5f_m = H5OpenFile(fn.c_str(), H5_O_RDONLY, props);
        PAssert(H5f_m != (h5_file_t)H5_ERR);
        H5CloseProp(props);
    }

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

    h5_file_t H5f_m;
    double temperature[Dim];
    double avg_vel[Dim];

    // Moment calculations:
    /// 6x6 matrix of the moments of the beam
    // FMatrix<double, 2 * Dim, 2 * Dim> moments_m;
    double moments_m[2 * Dim][2 * Dim];
    /// holds the centroid of the beam
    double centroid_m[2 * Dim];
    /// rms beam size (m)
    Vector_t rrms_m;
    /// rms momenta
    Vector_t vrms_m;
    /// mean position (m)
    Vector_t rmean_m;
    /// mean momenta
    Vector_t vmean_m;
    /// rms emittance (not normalized)
    Vector_t eps_m;
    /// rms correlation
    Vector_t rvrms_m;

    Vektor<double, Dim> avgEF;
    double globSumEF[Dim];
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
                double phi = ke * (1. - erf(a * sqrt(sqr))) / r;

                // compute force
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

        nr                 = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        int param          = 4;
        double beam_radius = atof(argv[param++]);
        double box_length  = atof(argv[param++]);
        // double part_density =atof(argv[param++]);
        int Nparticle             = atoi(argv[param++]);
        double interaction_radius = atof(argv[param++]);
        // read the remaining sim params
        double alpha           = atof(argv[param++]);
        double dt              = atof(argv[param++]);
        double eps             = atof(argv[param++]);
        int iterations         = atoi(argv[param++]);
        double charge_per_part = atof(argv[param++]);
        double mass_per_part   = atof(argv[param++]);
        double focusingForce   = atof(argv[param++]);
        int print_every        = atof(argv[param++]);

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
        double L = box_length / 2.;
        Vektor<double, Dim> extend_l(-L, -L, -L);
        Vektor<double, Dim> extend_r(L, L, L);

        Vektor<double, Dim> Vmax(6, 6, 6);
        P = new ChargedParticles<playout_t>(PL, nr, decomp, extend_l, extend_r);
        createParticleDistributionHeating(P, extend_l, extend_r, beam_radius, Nparticle,
                                          charge_per_part, mass_per_part);

        // COmpute and write temperature
        P->compute_temperature();
        writeTemperature(P, 0);
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

        msg << "number of particles = " << P->getTotalNum() << endl;
        msg << "Total charge Q      = " << P->total_charge << endl;

        ////////////////////////////////////////////////////////////////////////////////////////////
        std::string fname;
        fname = "data/particleData";
        fname += ".h5part";

        P->openH5(fname);
        dumpH5partVelocity(P, 0);
        unsigned printid = 1;

        msg << "Starting iterations ..." << endl;
        P->compute_temperature();
        // calculate initial space charge forces
        P->calculateGridForces(interaction_radius, alpha, 0, 0, 0);
        P->calculatePairForces(interaction_radius, eps, alpha);

        // avg space charge forces for constant focusing
        P->computeAvgSpaceChargeForces();

        // dumpVTKVector(P->eg_m, P,0,"EFieldAfterPMandPP");

        // compute quantities to check correctness:
        /*
        P->calc_field_energy();
        P->calc_potential_energy();
        P->calc_kinetic_energy();
        writeEnergy(P,0);
        */

        IpplTimings::TimerRef gridTimer     = IpplTimings::getTimer("GridTimer");
        IpplTimings::TimerRef particleTimer = IpplTimings::getTimer("ParticleTimer");

        for (int it = 0; it < iterations; it++) {
            /*
            P->calcMoments();
            P->computeBeamStatistics();
            writeBeamStatisticsVelocity(P,it);

            P->calc_kinetic_energy();
            P->calc_field_energy();
            writeEnergy(P,it);
            */
            // advance the particle positions
            // basic leapfrogging timestep scheme.  velocities are offset
            // by half a timestep from the positions.

            assign(P->R, P->R + dt * P->v);
            // update particle distribution across processors
            P->update();

            // compute the electric field

            IpplTimings::startTimer(gridTimer);
            P->calculateGridForces(interaction_radius, alpha, 0, it + 1, 0);
            IpplTimings::stopTimer(gridTimer);

            IpplTimings::startTimer(particleTimer);
            P->calculatePairForces(interaction_radius, eps, alpha);
            IpplTimings::stopTimer(particleTimer);

            // P->update();

            // second part of leapfrog: advance velocitites
            // P->computeAvgSpaceChargeForces();
            // if (Ippl::myNode()==0)
            // std::cout <<"avg E-Field = " << P->avgEF << std::endl;

            P->applyConstantFocusing(focusingForce, beam_radius);

            assign(P->v, P->v + dt * P->Q / P->m * (P->EF));

            P->compute_temperature();

            if (it % print_every == 0) {
                // dumpConservedQuantities(P,printid);
                // compute quantities
                /*
                P->calc_field_energy();
                P->calc_kinetic_energy();
                P->calc_potential_energy();
                writeEnergy(P,printid);
                */
                P->compute_temperature();
                writeTemperature(P, it + 1);

                dumpH5partVelocity(P, printid++);
            }

            msg << "Finished iteration " << it << endl;
        }
        ippl::Comm->barrier();

        P->closeH5();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(allTimer);

        IpplTimings::print();

        delete P;
        delete FL;
        delete mesh;
    }
    ippl::finalize();

    return 0;
}
