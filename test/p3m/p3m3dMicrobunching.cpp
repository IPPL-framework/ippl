//
// Application p3m3dMicrobunching
//   mpirun -n 32 ./p3m3dMicrobunching ${Nx} ${Ny} ${Nz} ${r_cut} ${alpha} ${epsilon} ${Nsteps}
//   $SeedID} ${printSteps} Nx,Ny,Nx is the poisson solver grid size, r_cut is the cutoff for pp
//   interaction, alpha is the splitting parameter, epsilon is the softening parameter,
//   printSteps=10 prints every tenth step
//
// Benjamin Ulmer, ETH Zürich (2016)
// Implemented as part of the Master thesis
// "The P3M Model on Emerging Computer Architectures With Application to Microbunching"
// (http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projectscompleted/cse/thesisBUlmer.pdf)
//
#include "Ippl.h"

#include <cfloat>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Utility/PAssert.h"

#include "H5hut.h"
#include "Particle/BoxParticleCachingPolicy.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodic.h"
#include "Particle/PairBuilder/HashPairBuilderPeriodicParallel.h"
#include "Particle/PairBuilder/PairConditions.h"
#include "math.h"
// #include "FixedAlgebra/FMatrix.h"

#include <random>

#include "ChargedParticleFactory.hpp"
#include "VTKFieldWriterParallel.hpp"

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
typedef FFT<CCTransform, Dim, double> FFT_t;

typedef IntCIC IntrplCIC_t;
// typedef IntNGP                                                        IntrplNGP_t;
// typedef IntTSC                                                        IntrplTSC_t;

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
    static void calculate(Vektor<T, 3>& hrsq, FT& grn, FT2* grnI, double alpha, double eps,
                          double ke) {
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

template <class CharT, class Traits>
double readNextBeamParamValue(std::basic_istream<CharT, Traits>& input) {
    std::basic_string<CharT, Traits> line;
    std::getline(input, line);
    // std::istringstream iss(line);
    // std::basic_string<CharT,Traits> number;
    // iss >> number;
    if (Ippl::myNode() == 0) {
        std::cout << "the line read is" << line << std::endl;
        std::cout << "the number is " << std::stod(line) << std::endl;
    }
    return std::stod(line);
}

template <class PL>
class ChargedParticles : public IpplParticleBase<PL> {
public:
    ParticleAttrib<double> Q;     // Charge [elementary charge e]
    ParticleAttrib<double> m;     // rest mass [MeV/c^2]
    ParticleAttrib<double> Phi;   // electrostatic potential
    ParticleAttrib<Vector_t> EF;  // Electric field [MeV/(sec)]
    ParticleAttrib<Vector_t> p;   // momentum [MeV/c]
    ParticleAttrib<int> ID;       // unique ID for debugging reasons => remove for production

    ChargedParticles(PL* pl, Vektor<double, 3> nr, e_dim_tag /*decomp*/[Dim], unsigned seedID_ = 0)
        : IpplParticleBase<PL>(pl)
        , nr_m(nr)
        , seedID(seedID_) {
        this->addAttribute(Q);
        this->addAttribute(m);
        this->addAttribute(Phi);
        this->addAttribute(EF);
        this->addAttribute(p);
        this->addAttribute(ID);

        // read beam parameters from input file:

        if (Ippl::myNode() == 0) {
            std::cout << "we are reading the following beam parameters" << std::endl;
        }

        std::ifstream input("BeamParams.in");
        gamma       = readNextBeamParamValue(input);
        deltagamma  = readNextBeamParamValue(input);
        I           = readNextBeamParamValue(input);
        extend_r[2] = readNextBeamParamValue(input);
        extend_r[1] = readNextBeamParamValue(input);
        extend_r[0] = extend_r[1];
        Ld          = readNextBeamParamValue(input);
        sigmaX      = readNextBeamParamValue(input);
        emittance   = readNextBeamParamValue(input);
        R56         = readNextBeamParamValue(input);
        q           = readNextBeamParamValue(input);
        // Npart = readNextBeamParamValue(input);
        m0 = readNextBeamParamValue(input);
        ke = readNextBeamParamValue(input);
        c  = readNextBeamParamValue(input);

        double NpartTotal = extend_r[2] * I / (c * 1.6e-19);
        std::cout << "total number of particles is = " << NpartTotal << std::endl;
        double particleDensity = NpartTotal / extend_r[2] * 1 / (2 * M_PI * sigmaX * sigmaX);
        std::cout << "particle density = " << particleDensity << std::endl;
        Npart = particleDensity * extend_r[0] * extend_r[1] * extend_r[2];
        std::cout << "number of particles in simulation domain is = " << Npart << std::endl;
        // q=I*extend_r[2]/double(Npart);

        // wavelength of interest
        lambda = 0.5e-6;

        beta0 = sqrt(1. - 1. / (gamma * gamma));
        for (unsigned j = 0; j < 10; j++)
            theta[j] = 0.001 * double(j);
        extend_l[0] = 0;
        extend_l[1] = 0;
        extend_l[2] = 0;

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

    void calcMoments() {
        double part[2 * Dim];

        double loc_centroid[2 * Dim];
        double loc_moment[2 * Dim][2 * Dim];
        double moments[2 * Dim][2 * Dim];

        for (unsigned i = 0; i < 2 * Dim; i++) {
            loc_centroid[i] = 0.0;
            for (unsigned j = 0; j <= i; j++) {
                loc_moment[i][j] = 0.0;
                loc_moment[j][i] = loc_moment[i][j];
            }
        }

        // double p0=m0*gamma*beta0;
        for (unsigned long k = 0; k < this->getLocalNum(); ++k) {
            part[1] = this->p[k](0);
            part[3] = this->p[k](1);
            part[5] = (gamma * this->p[k](2));
            part[0] = this->R[k](0);
            part[2] = this->R[k](1);
            part[4] = this->R[k](2) / gamma;

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

    // compute the determinant of a matrix with dimensions up to 2*Dimx2*dim
    double det(int n, double mat[2 * Dim][2 * Dim]) {
        double d = 0;
        int c, subi, i, j, subj;
        double submat[2 * Dim][2 * Dim];
        if (n == 2)
            return ((mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1]));
        else {
            for (c = 0; c < n; c++) {
                subi = 0;
                for (i = 1; i < n; i++) {
                    subj = 0;
                    for (j = 0; j < n; j++) {
                        if (j == c)
                            continue;
                        submat[subi][subj] = mat[i][j];
                        subj++;
                    }
                    subi++;
                }
                d = d + (pow(-1, c) * mat[0][c] * det(n - 1, submat));
            }
        }
        return d;
    }

    // compute the full determinant of the 6x6 momentsmatrix to get the emittance
    double computeEmittance() {
        const double N = static_cast<double>(this->getTotalNum());
        double moments[2 * Dim][2 * Dim];

        for (unsigned i = 0; i < 2 * Dim; i++) {
            for (unsigned j = 0; j < 2 * Dim; j++) {
                moments[i][j] = moments_m[i][j] / N - centroid_m[i] * centroid_m[j] / (N * N);
            }
        }

        double eps = sqrt(det(2 * Dim, moments));
        return eps;
    }

    void computeBeamStatistics() {
        const size_t locNp = this->getLocalNum();
        const double N     = static_cast<double>(this->getTotalNum());
        const double zero  = 0.0;

        Vector_t eps2, fac, rsqsum, psqsum, rpsum;
        for (unsigned int i = 0; i < Dim; i++) {
            rmean_m(i) = centroid_m[2 * i] / N;
            pmean_m(i) = centroid_m[(2 * i) + 1] / N;
            rsqsum(i)  = moments_m[2 * i][2 * i] - N * rmean_m(i) * rmean_m(i);
            psqsum(i)  = moments_m[(2 * i) + 1][(2 * i) + 1] - N * pmean_m(i) * pmean_m(i);
            if (psqsum(i) < 0)
                psqsum(i) = 0;
            rpsum(i) = moments_m[(2 * i)][(2 * i) + 1] - N * rmean_m(i) * pmean_m(i);
        }

        eps2 = (rsqsum * psqsum - rpsum * rpsum) / (N * N);
        rpsum /= N;

        for (unsigned int i = 0; i < Dim; i++) {
            rrms_m(i)  = sqrt(rsqsum(i) / N);
            prms_m(i)  = sqrt(psqsum(i) / N);
            eps_m(i)   = std::sqrt(std::max(eps2(i), zero));
            double tmp = rrms_m(i) * prms_m(i);
            fac(i)     = (tmp == 0) ? zero : 1.0 / tmp;
        }
        rprms_m = rpsum * fac;

        // Find normalized emittance.
        double actual_gamma = 0.0;
        for (size_t i = 0; i < locNp; i++)
            actual_gamma += sqrt(1.0
                                 + (gamma * p[i](2) + m0 * gamma * beta0)
                                       * (gamma * p[i](2) + m0 * gamma * beta0) / m0 / m0
                                 + p[i](1) * p[i](1) / m0 / m0 + p[i](0) * p[i](0) / m0 / m0);

        reduce(actual_gamma, actual_gamma, OpAddAssign());
        actual_gamma /= N;

        // eps_norm_m = eps_m *actual_gamma*beta0;
        eps_norm_m          = eps_m / m0;
        eps6x6_m            = computeEmittance();
        eps6x6_normalized_m = eps6x6_m * actual_gamma * beta0;
    }

    void calculatePairForces(double interaction_radius, double eps, double alpha);

    void calculateGridForces(double /*interaction_radius*/, double alpha, double eps,
                             int /*it*/ = 0) {
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
        // sum(real(rhocmpl_m))<< std::endl; subtract the background charge of the ions
        // rhocmpl_m[domain_m]=1.+rhocmpl_m[domain_m];
        // std::cout << "total charge in densitty field after ion subtraction is" <<
        // sum(real(rhocmpl_m)) << std::endl;

        // compute rhoHat and store in rhocmpl_m
        fft_m->transform("inverse", rhocmpl_m);
        // (2) compute Greens function in real space and transform to fourier space
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
        SpecializedGreensFunction<3>::calculate(hrsq, grncmpl_m, grnIField_m, alpha, eps, ke);
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

    void closeH5() { H5CloseFile(H5f_m); }

    void openH5(std::string fn) {
        h5_prop_t props = H5CreateFileProp();
        MPI_Comm comm   = ippl::Comm->getCommunicator();
        h5_err_t h5err  = H5SetPropFileMPIOCollective(props, &comm);
        PAssert(h5err != H5_ERR);
        H5f_m = H5OpenFile(fn.c_str(), H5_O_WRONLY, props);
    }

    const Vector_t get_hr() { return hr_m; }

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

    double total_charge;
    FFT_t* fft_m;
    e_dim_tag decomp_m[Dim];

    // Beam parameter:
    double gamma;       // energy [1]
    double deltagamma;  // longitdnl. energy spread [1]
    double I;           // beam current [A]
    double Ld;          // drift length [m]
    double sigmaX;      // rms envelope size [m]
    double emittance;   // transverse emittance [m rad]
    double q;           // charge per particle [e]
    double m0;          // particle rest mass [MeV/c^2]
    double ke;          // coulomb constant [m^2MeV/(se^2c)]
    double R56;         // energy-position coupling [m]
    double c;           // speed of light [m/s]
    int Npart;          // number of particles

    double beta0;  // relative velocity of the beam

    // TEMP debug variable
    double RhoSum = 0;

    h5_file_t H5f_m;
    double lambda;
    double theta[10];
    std::complex<double> b0[10];
    std::complex<double> bend[10];
    std::complex<double> MBgain[10];
    unsigned seedID;

    // Moment calculations:
    /// 6x6 matrix of the moments of the beam
    // FMatrix<double, 2 * Dim, 2 * Dim> moments_m;
    double moments_m[2 * Dim][2 * Dim];
    /// holds the centroid of the beam
    double centroid_m[2 * Dim];
    /// rms beam size (m)
    Vector_t rrms_m;
    /// rms momenta
    Vector_t prms_m;
    /// mean position (m)
    Vector_t rmean_m;
    /// mean momenta
    Vector_t pmean_m;
    /// rms emittance (not normalized)
    Vector_t eps_m;
    /// emittance including correlations: Det(whole 6x6 matrix)
    double eps6x6_normalized_m;
    double eps6x6_m;
    /// rms normalized emittance
    Vector_t eps_norm_m;
    /// rms correlation
    Vector_t rprms_m;
};

template <class T>
struct ApplyField {
    ApplyField(T c, double r, double epsilon, double alpha, double coulombConst)
        : C(c)
        , R(r)
        , eps(epsilon)
        , a(alpha)
        , ke(coulombConst) {}
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
    double ke;
};

template <class PL>
void ChargedParticles<PL>::calculatePairForces(double interaction_radius, double eps,
                                               double alpha) {
    if (interaction_radius > 0) {
        if (Ippl::getNodes() > 1) {
            HashPairBuilderPeriodicParallel<ChargedParticles<playout_t> > HPB(*this);
            HPB.for_each(RadiusCondition<double, Dim>(interaction_radius),
                         ApplyField<double>(-1, interaction_radius, eps, alpha, ke), extend_l,
                         extend_r);
        } else {
            HashPairBuilderPeriodic<ChargedParticles<playout_t> > HPB(*this);
            HPB.for_each(RadiusCondition<double, Dim>(interaction_radius),
                         ApplyField<double>(-1, interaction_radius, eps, alpha, ke), extend_l,
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

        nr        = Vektor<int, Dim>(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        int param = 4;

        double interaction_radius = atof(argv[param++]);
        double alpha              = atof(argv[param++]);
        double eps                = atof(argv[param++]);
        int iterations            = atoi(argv[param++]);
        unsigned myseedID         = atoi(argv[param++]);
        int printEvery            = atoi(argv[param++]);

        // double R56 =  atof(argv[param++]); //coupling constant in m for real frame
        ///////// setup the initial layout ///////////////////////////////////////
        e_dim_tag decomp[Dim];
        Mesh_t* mesh;
        FieldLayout_t* FL;
        ChargedParticles<playout_t>* P;

        NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++)
            domain[i] = domain[i] = Index(nr[i] + 1);

        for (unsigned d = 0; d < Dim; ++d)
            decomp[d] = SERIAL;
        decomp[2] = PARALLEL;
        // decomp[1]=PARALLEL;

        // create mesh and layout objects for this problem domain
        mesh          = new Mesh_t(domain);
        FL            = new FieldLayout_t(*mesh, decomp);
        playout_t* PL = new playout_t(*FL, *mesh);
        // define beam parameters:

        /////////// Create the particle distribution
        ////////////////////////////////////////////////////////
        P = new ChargedParticles<playout_t>(PL, nr, decomp, myseedID);
        INFOMSG(P->getMesh() << endl);
        INFOMSG(P->getFieldLayout() << endl);
        msg << endl << endl;

        double tend = P->Ld / (P->beta0);
        std::cout << "tend = " << tend << std::endl;
        // lorentz transform to beam frame:
        //////// TODO check lorentz transformation of time
        // tend = P->gamma*(tend-P->beta0*P->Ld);
        tend /= P->gamma;
        std::cout << "tend' = " << tend << std::endl;
        double dt = tend / iterations;
        std::cout << "TIMESTEP dt = " << dt << std::endl;
        createParticleDistributionMicrobunching(P, myseedID);
        /////////////////////////////////////////////////////////////////////////////////////////////
        PL->setAllCacheDimensions(interaction_radius);
        PL->enableCaching();

        /////// Print mesh informations ////////////////////////////////////////////////////////////
        ippl::Comm->barrier();
        // dumpParticlesCSVp(P,0);

        INFOMSG(P->getMesh() << endl);
        INFOMSG(P->getFieldLayout() << endl);
        msg << endl << endl;

        msg << "number of particles = " << endl;
        msg << P->getTotalNum() << endl;
        msg << "Total charge Q = " << endl;
        msg << P->total_charge << endl;
        ////////////////////////////////////////////////////////////////////////////////////////////
        std::string fname;
        fname = "data/particleData_seedID_";
        fname += std::to_string(P->seedID);
        fname += ".h5part";

        P->openH5(fname);
        dumpH5part(P, 0);
        unsigned printid = 1;
        msg << "Starting iterations ..." << endl;

        // calculate initial grid forces
        P->calculateGridForces(interaction_radius, alpha, eps, 0);
        // dumpVTKVector(P->eg_m, P,0,"EFieldAfterPMandPP");

        P->calcMoments();
        P->computeBeamStatistics();
        writeBeamStatistics(P, 0);

        for (int it = 0; it < iterations; it++) {
            // advance the particle positions
            // basic leapfrogging timestep scheme.  velocities are offset
            // by half a timestep from the positions.

            // energy position coupling:
            /*
            Vektor<double,3> kHat;
            kHat[0]=0; kHat[1]=0; kHat[2]=1.;
            */
            // assign(P->R, P->R + dt *
            // P->p/(P->gamma*P->m0)+rearrangez*1./P->gamma*(1.-sqrt(1.+dot(P->p,P->p)/(P->m0*P->m0*P->c*P->c)))*P->R56);
            assign(P->R, P->R + dt * P->p / P->m0);
            // shift particle due to longitudinal dispersion
            // assign(P->R, P->R + kHat*P->gamma*P->R56*P->p/(P->beta0*P->m0));
            /*
            for (unsigned i=0; i<P->getLocalNum(); ++i) {
            P->R[i][2]+=(sqrt(P->p[i][2]+P->m0*P->m0)/(P->m0)-1.)*1./P->gamma*P->R56;
            }
            */
            // update particle distribution across processors
            msg << "do particle update" << endl;
            IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("UpdateTimer");
            IpplTimings::startTimer(updateTimer);
            P->update();
            IpplTimings::stopTimer(updateTimer);

            msg << "done particle update" << endl;
            // compute the electric field
            msg << "calculating grid" << endl;
            IpplTimings::TimerRef gridTimer = IpplTimings::getTimer("GridTimer");
            IpplTimings::startTimer(gridTimer);

            P->calculateGridForces(interaction_radius, alpha, eps, it + 1);

            IpplTimings::stopTimer(gridTimer);

            msg << "calculating pairs" << endl;

            IpplTimings::TimerRef particleTimer = IpplTimings::getTimer("ParticleTimer");
            IpplTimings::startTimer(particleTimer);

            P->calculatePairForces(interaction_radius, eps, alpha);
            IpplTimings::stopTimer(particleTimer);

            // P->update();

            // dumpVTKVector(P->eg_m, P,it+1,"EFieldAfterPMandPP");
            // dumpVTKScalar(P->rho_m,P,it+1,"RhoInterpol");

            // second part of leapfrog: advance velocitites
            assign(P->p, P->p + dt * P->Q * P->EF);

            if ((it + 1) % printEvery == 0) {
                // dumpVTKVector(P->eg_m, P,printid,"EFieldAfterPMandPP");
                dumpH5part(P, printid++);
            }
            // dumpParticlesCSVp(P,it+1);

            P->calcMoments();
            P->computeBeamStatistics();
            writeBeamStatistics(P, it + 1);

            msg << "Finished iteration " << it << endl;
        }

        // print final state
        dumpH5part(P, printid++);

        // P->computeBunchingGain();

        P->closeH5();
        ippl::Comm->barrier();

        msg << "number of particles = " << endl;
        msg << P->getTotalNum() << endl;

        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();

        delete P;
        delete FL;
        delete mesh;
    }
    ippl::finalize();

    return 0;
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
