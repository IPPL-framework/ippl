/** @file MithraBunch.h
 * @brief Host sampling and moving-frame initialization of the FEL particle bunch.
 * @ingroup fel_particles
 *
 * The active path samples transverse Gaussians and a longitudinal uniform core
 * with tapered tails, imposes the configured-in-code wavelength modulation,
 * applies the MITHRA-style initial boost, and copies positions and normalized
 * momenta into IPPL attributes. Sampling uses host containers and the C random
 * generator; only the final attribute assignment uses a Kokkos kernel.
 *
 * This code uses FEL's internal unit system from units.h, not SI storage.
 * Gamma is dimensionless and momentum means gamma*beta=p/(m*c). The current
 * manager calls the complete generator on rank zero and distributes particles
 * afterwards; the rank/size sampling arguments are not used for distributed
 * generation by that path.
 */
#ifndef IPPL_FEL_MITHRA_BUNCH_H
#define IPPL_FEL_MITHRA_BUNCH_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <list>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

#include "Config.h"
#include "LorentzTransform.h"
#include "datatypes.h"
#include "units.h"

#ifndef assert_isreal
/** @brief Debug assertion that a host scalar is neither NaN nor infinity.
 * @param X Scalar expression tested with std::isnan and std::isinf.
 * @note Disabled with NDEBUG; this is not run-time input validation.
 */
#define assert_isreal(X) assert(!std::isnan(X) && !std::isinf(X))
#endif

/** @brief Three-component host sampling vector.
 * @tparam scalar Component type; dimensions depend on the containing member.
 */
template <typename scalar>
using FieldVector = ippl::Vector<scalar, 3>;
/** @brief Host-side parameters for the retained MITHRA-style sampling routines.
 * @ingroup fel_particles
 * @tparam scalar Floating-point sampling type.
 *
 * generate_mithra_config() fills the subset used by the FEL mini-app. This
 * aggregate alone does not initialize or validate its scalar members. Several
 * compatibility members describe features not implemented by the active path.
 */
template <typename scalar>
struct BunchInitialize {
    /** Longitudinal core type: "uniform" or "gaussian"; transverse sampling is Gaussian.
     */
    std::string distribution_;

    /** Sampling generator; the implemented initializeBunchEllipsoid path requires "random".
     */
    std::string generator_;

    /** Nominal total sample count; locally rounded up to a multiple of four before generation. */
    unsigned int numberOfParticles_;

    /** Total charge parameter in internal charge units, copied from config::charge, not pC. */
    scalar cloudCharge_;

    /** Dimensionless laboratory Lorentz factor gamma, not a kinetic energy in MeV. */
    scalar initialGamma_;

    /** Dimensionless laboratory speed beta=v/c, stored by the configuration adapter. */
    scalar initialBeta_;

    /** Nominal unit direction; the active adapter fixes this to +z. */
    FieldVector<scalar> initialDirection_;

    /** Laboratory central position in internal length units, before boost and recentering. */
    FieldVector<scalar> position_;

    /** Unused crystal-generator compatibility counts; the active adapter sets zero. */
    FieldVector<unsigned int> numbers_;

    /** Unused crystal-generator lattice lengths; the active adapter sets zero. */
    FieldVector<scalar> latticeConstants_;

    /** Internal-length sampling scales. x/y are Gaussian rms widths. z is a
     * uniform half-width or Gaussian rms width according to distribution_, before
     * rejection, wavelength modulation and any added uniform-core tails.
     */
    FieldVector<scalar> sigmaPosition_;

    /** Gaussian rms widths of the dimensionless components gamma*beta=p/(m*c). */
    FieldVector<scalar> sigmaGammaBeta_;

    /** Strict absolute cutoff applied separately to both x and y sampling offsets.
     * In internal length units; copied from the x component of the input truncations.
     */
    scalar tranTrun_;

    /** Strict absolute z-offset cutoff in internal length units, before modulation.
     */
    scalar longTrun_;

    /** Unused distribution-file compatibility name; no file is read by this sampler.
     */
    std::string fileName_;

    /** Laboratory resonant-wavelength parameter in internal length units.
     * Zero disables quarter-wavelength particle grouping and modulation.
     */
    scalar lambda_;

    /** Dimensionless longitudinal modulation parameter, accepted in [0,2].
     * The adapter fixes 0.01; this parameter is not a measured post-sampling bunching factor.
     */
    scalar bF_;

    /** Deterministic modulation phase in degrees; the adapter fixes zero.
     */
    scalar bFP_;

    /** Enable the retained shot-noise branch; false in the active FEL adapter.
     */
    bool shotNoise_;

    /** Dimensionless mean laboratory velocity vector; initialGamma_*betaVector_ sets mean momentum.
     */
    FieldVector<scalar> betaVector_;

};

/** @brief Adapt the parsed FEL configuration to the active bunch-sampling settings.
 * @ingroup fel_particles
 * @tparam scalar Floating-point type of the sampling parameters and frame.
 * @param cfg Parsed FEL configuration with internal-unit lengths and charge.
 * @return Parameters for random sampling, a uniform longitudinal core, +z motion,
 * 0.01 deterministic modulation, zero phase, and disabled shot noise.
 *
 * The first argument cfg already stores internal-unit lengths and total charge;
 * sigma_momentum is dimensionless gamma*beta spread. The second, unnamed frame
 * argument is retained for compatibility but is not read. The wavelength is
 * recomputed from
 * @f$\gamma_f=\gamma_b/\sqrt{1+K^2/2}@f$ and
 * @f$\lambda=\lambda_u/(2\gamma_f^2)@f$.
 * Unlike the manager's boost factor, this local gamma_f is not clamped to one.
 * No MeV, coulomb or SI-length conversion takes place here.
 *
 * Only position_truncations[0] and [2] are used; the x cutoff also applies to y.
 * numberOfParticles_ is unsigned int even though config::num_particles is wider.
 * Gaussian longitudinal sampling and shot noise are not selected by JSON keys
 * in this adapter; changing those settings requires an explicit code extension.
 */
template <typename scalar>
BunchInitialize<scalar> generate_mithra_config(
    const config& cfg, const ippl::UniaxialLorentzframe<scalar>& /*frame_boost unused*/) {
    using vec3         = ippl::Vector<scalar, 3>;
    scalar frame_gamma = cfg.bunch_gamma / std::sqrt(1 + 0.5 * cfg.undulator_K * cfg.undulator_K);
    BunchInitialize<scalar> init;
    init.generator_        = "random";
    init.distribution_     = "uniform";
    init.initialDirection_ = vec3{0, 0, 1};
    init.initialGamma_     = cfg.bunch_gamma;
    init.initialBeta_      = cfg.bunch_gamma == scalar(1)
                                 ? 0
                                 : (sqrt(cfg.bunch_gamma * cfg.bunch_gamma - 1) / cfg.bunch_gamma);
    init.sigmaGammaBeta_   = vector_cast<scalar>(cfg.sigma_momentum);
    init.sigmaPosition_    = vector_cast<scalar>(cfg.sigma_position);

    // Initial bunching factor.
    init.bF_                = 0.01;
    init.bFP_               = 0;
    init.shotNoise_         = false;
    init.cloudCharge_       = cfg.charge;
    init.lambda_            = cfg.undulator_period / (2 * frame_gamma * frame_gamma);
    init.longTrun_          = cfg.position_truncations[2];
    init.tranTrun_          = cfg.position_truncations[0];
    init.position_          = vector_cast<scalar>(cfg.mean_position);
    init.betaVector_        = ippl::Vector<scalar, 3>{0, 0, init.initialBeta_};
    init.numberOfParticles_ = cfg.num_particles;

    init.numbers_          = 0;              // UNUSED
    init.latticeConstants_ = vec3{0, 0, 0};  // UNUSED

    return init;
}
/** @brief Temporary host sample before copying into IPPL particle attributes.
 * @ingroup fel_particles
 * @tparam Double Floating-point component type.
 */
template <typename Double>
struct Charge {
    Double q;                     ///< Nominal sample charge in the same internal units as cloudCharge_.
    /** Positions in internal length units: rnp is populated and boosted; rnm is unused here. */
    FieldVector<Double> rnp, rnm;
    FieldVector<Double> gb;       ///< Dimensionless momentum gamma*beta, initially in the laboratory frame.

    /** Unused compatibility flag for undulator entrance crossing, initialized to zero.
     */
    Double e = 0.0;
};
/** @brief Host linked list used while sampling and boosting temporary particles.
 * @tparam scalar Floating-point type of each Charge.
 */
template <typename scalar>
using ChargeVector = std::list<Charge<scalar>>;
/** @brief Append accepted laboratory-frame samples using the retained host generator.
 * @ingroup fel_particles
 * @tparam Double Floating-point sampling type.
 * @param bunchInit Sampling settings, copied by value; count corrections do not update the caller.
 * @param[in,out] chargeVector Host list receiving accepted samples and any added tail samples.
 * @param rank Starting sample-group index for strided rank/size generation.
 * @param size Positive stride between sample groups; current manager passes one.
 * @param ia Component of position_ used by the retained uniform-tail scalar-offset expression.
 * The current adapter passes zero; the core uses the full position_ vector instead.
 * @pre The active supported call starts with an empty list, generator_="random",
 * nonzero nominal particle count, positive uniform half-width and valid rank/size.
 * @throws IpplException If bF_ is outside [0,2]. Indexed random access may throw
 * std::out_of_range for unsupported combinations; invalid distribution names exit.
 *
 * Gaussian variates use Box-Muller sampling. The longitudinal uniform core spans
 * @f$[-\sigma_z,\sigma_z]@f$ before strict coordinate cutoffs. Rejected groups
 * are not replaced. If lambda_ is nonzero, each accepted base sample produces
 * four particles separated initially by lambda_/4 and then sinusoidally shifted.
 * The uniform path additionally samples Gaussian tails beyond the core ends.
 * Actual particle count therefore need not equal numberOfParticles_.
 *
 * Every call resets the process-global C RNG with srand(42), then allocates a
 * host random-number pool proportional to the nominal count. Reproducibility
 * depends on the C library's rand() sequence; this is not a device RNG. The
 * active manager uses rank=0,size=1 and later performs MPI particle distribution.
 *
 * @note The dormant shot-noise normalization treats cloudCharge_ as an electron
 * count, whereas the current adapter supplies internal charge units. That branch
 * is disabled and requires a unit audit before enabling it.
 */
template <typename Double>
void initializeBunchEllipsoid(BunchInitialize<Double> bunchInit, ChargeVector<Double>& chargeVector,
                              int rank, int size, int ia) {
    /* Correct the number of particles if it is not a multiple of four.
     */
    if (bunchInit.numberOfParticles_ % 4 != 0) {
        unsigned int n = bunchInit.numberOfParticles_ % 4;
        bunchInit.numberOfParticles_ += 4 - n;
        Inform msg("BunchInitialize");
        msg << "Warning: number of particles in the bunch is not a multiple of four; "
            << "corrected to " << bunchInit.numberOfParticles_ << "." << endl;
    }

    /* Save the initially given number of particles. */
    unsigned int Np = bunchInit.numberOfParticles_, i, Np0 = chargeVector.size();

    /* Declare the required parameters for the initialization of charge vectors. */
    Charge<Double> charge;
    charge.q               = bunchInit.cloudCharge_ / Np;
    FieldVector<Double> gb = bunchInit.initialGamma_ * bunchInit.betaVector_;
    FieldVector<Double> r(0.0);
    FieldVector<Double> t(0.0);
    Double t0;  //, g;
    Double zmin = 1e100;
    Double Ne;
    Double bF = bunchInit.bF_;
    Double bFi;
    unsigned int bmi;
    std::vector<Double> randomNumbers;

    /* The initialization in group of four particles should only be done if there exists an
     * undulator in the interaction.
     */
    unsigned int ng = (bunchInit.lambda_ == 0.0) ? 1 : 4;

    /* Check the bunching factor. */
    if (bunchInit.bF_ > 2.0 || bunchInit.bF_ < 0.0) {
        throw IpplException("initializeBunchEllipsoid",
                            "The bunching factor must be between 0 and 2.");
    }

    /* All invocations build the same seeded sequence; rank-strided sample indices
     * select different groups when this low-level routine is used on several ranks.
     */
    if (bunchInit.generator_ == "random") {
        /* Initialize the random number generator with a fixed seed so the
         * generated bunch is reproducible across runs.
         */
        srand(42);
        /* Reserve twenty random values per nominal sample group.
         */
        randomNumbers.resize(Np / ng * 20, 0.0);
        for (unsigned int ri = 0; ri < Np / ng * 20; ri++)
            randomNumbers[ri] =
                (float)std::min(1 - 1e-7, std::max(1e-7, ((double)rand()) / RAND_MAX));
    }

    /* Declare the generator function depending on the input.
     */
    auto generate = [&](unsigned int n, unsigned int m) {
        return (randomNumbers.at(n * 2 * Np / ng + m));
    };

    /* Declare the function for injecting the shot noise.
     */
    auto insertCharge = [&](Charge<Double> q) {
        for (unsigned int ii = 0; ii < ng; ii++) {
            /* The random modulation is introduced depending on the shot-noise being activated.
             */
            if (bunchInit.shotNoise_) {
                /* Obtain the number of beamlet.
                 */
                bmi = int((charge.rnp[2] - zmin) / bunchInit.lambda_);

                /* Obtain the phase and amplitude of the modulation.
                 */
                bFi = bF * sqrt(-2.0 * log(generate(8, bmi)));

                q.rnp[2] = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

                q.rnp[2] -= bunchInit.lambda_ / M_PI * bFi
                            * sin(2.0 * M_PI / bunchInit.lambda_ * q.rnp[2]
                                  + 2.0 * M_PI * generate(9, bmi));
            } else if (bunchInit.lambda_ != 0.0) {
                q.rnp[2] = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

                q.rnp[2] -= bunchInit.lambda_ / M_PI * bunchInit.bF_
                            * sin(2.0 * M_PI / bunchInit.lambda_ * q.rnp[2]
                                  + bunchInit.bFP_ * M_PI / 180.0);
            }

            /* Set this charge into the charge vector. */
            chargeVector.push_back(q);
        }
    };

    /* If the shot noise is on, we need the minimum value of the bunch z coordinate to be able to
     * calculate the FEL bucket number. */
    if (bunchInit.shotNoise_) {
        for (i = 0; i < Np / ng; i++) {
            if (bunchInit.distribution_ == "uniform")
                zmin = std::min(
                    Double(2.0 * generate(2, i + Np0) - 1.0) * bunchInit.sigmaPosition_[2], zmin);
            else if (bunchInit.distribution_ == "gaussian")
                zmin = std::min(
                    (Double)(bunchInit.sigmaPosition_[2] * sqrt(-2.0 * log(generate(2, i + Np0)))
                             * sin(2.0 * M_PI * generate(3, i + Np0))),
                    zmin);
            else {
                std::cout << std::string(
                    "The longitudinal type is not correctly given to the code !!!\n");
                exit(1);
            }
        }

        if (bunchInit.distribution_ == "uniform")
            for (; i < unsigned(Np / ng
                                * (1.0
                                   + 2.0 * bunchInit.lambda_ * sqrt(2.0 * M_PI)
                                         / (2.0 * bunchInit.sigmaPosition_[2])));
                 i++) {
                t0 = 2.0 * bunchInit.lambda_ * sqrt(-2.0 * log(generate(2, i + Np0)))
                     * sin(2.0 * M_PI * generate(3, i + Np0));
                t0 += (t0 < 0.0) ? (-bunchInit.sigmaPosition_[2]) : (bunchInit.sigmaPosition_[2]);

                zmin = std::min(t0, zmin);
            }

        zmin = zmin + bunchInit.position_[2];

        /* Obtain the average number of electrons per FEL beamlet.
         */
        Ne = bunchInit.cloudCharge_ * bunchInit.lambda_ / (2.0 * bunchInit.sigmaPosition_[2]);

        /* Set the bunching factor level for the shot noise depending on the given values.
         */
        bF = (bunchInit.bF_ == 0.0) ? 1.0 / sqrt(Ne) : bunchInit.bF_;
    }

    /* Determine the properties of each charge point and add them to the charge vector. */
    for (i = rank; i < Np / ng; i += size) {
        /* Determine the transverse coordinate. */
        r[0] = bunchInit.sigmaPosition_[0] * sqrt(-2.0 * log(generate(0, i + Np0)))
               * cos(2.0 * M_PI * generate(1, i + Np0));
        r[1] = bunchInit.sigmaPosition_[1] * sqrt(-2.0 * log(generate(0, i + Np0)))
               * sin(2.0 * M_PI * generate(1, i + Np0));

        /* Determine the longitudinal coordinate. */
        if (bunchInit.distribution_ == "uniform")
            r[2] = (2.0 * generate(2, i + Np0) - 1.0) * bunchInit.sigmaPosition_[2];
        else if (bunchInit.distribution_ == "gaussian")
            r[2] = bunchInit.sigmaPosition_[2] * sqrt(-2.0 * log(generate(2, i + Np0)))
                   * sin(2.0 * M_PI * generate(3, i + Np0));
        else {
            exit(1);
        }

        /* Determine the transverse momentum. */
        t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt(-2.0 * log(generate(4, i + Np0)))
               * cos(2.0 * M_PI * generate(5, i + Np0));
        t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt(-2.0 * log(generate(4, i + Np0)))
               * sin(2.0 * M_PI * generate(5, i + Np0));
        t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt(-2.0 * log(generate(6, i + Np0)))
               * cos(2.0 * M_PI * generate(7, i + Np0));

        if (fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_
            && fabs(r[2]) < bunchInit.longTrun_) {
            /* Shift the generated charge to the center position and momentum space.
             */
            charge.rnp = bunchInit.position_;
            charge.rnp += r;

            charge.gb = gb;
            charge.gb += t;
            if (std::isinf(gb[2])) {
                std::cerr << "[Warning] Gammabeta obtained an klonked here\n";
            }

            /* Insert this charge and the mirrored ones into the charge vector.
             */
            insertCharge(charge);
        }
    }

    /* If the longitudinal type of the bunch is uniform a tapered part needs to be added to remove
     * the coherent spontaneous emission (CSE) from the tail of the bunch.
     */
    if (bunchInit.distribution_ == "uniform") {
        for (; i < unsigned(uint32_t(Np / ng)
                            * (1.0
                               + 2.0 * bunchInit.lambda_ * sqrt(2.0 * M_PI)
                                     / (2.0 * bunchInit.sigmaPosition_[2])));
             i += size) {
            r[0] = bunchInit.sigmaPosition_[0] * sqrt(-2.0 * log(generate(0, i + Np0)))
                   * cos(2.0 * M_PI * generate(1, i + Np0));
            r[1] = bunchInit.sigmaPosition_[1] * sqrt(-2.0 * log(generate(0, i + Np0)))
                   * sin(2.0 * M_PI * generate(1, i + Np0));

            /* Determine the longitudinal coordinate. */
            r[2] = 2.0 * bunchInit.lambda_ * sqrt(-2.0 * log(generate(2, i + Np0)))
                   * sin(2.0 * M_PI * generate(3, i + Np0));
            r[2] += (r[2] < 0.0) ? (-bunchInit.sigmaPosition_[2]) : (bunchInit.sigmaPosition_[2]);

            /* Determine the transverse momentum.
             */
            t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt(-2.0 * log(generate(4, i + Np0)))
                   * cos(2.0 * M_PI * generate(5, i + Np0));
            t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt(-2.0 * log(generate(4, i + Np0)))
                   * sin(2.0 * M_PI * generate(5, i + Np0));
            t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt(-2.0 * log(generate(6, i + Np0)))
                   * cos(2.0 * M_PI * generate(7, i + Np0));
            if (fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_
                && fabs(r[2]) < bunchInit.longTrun_) {
                /* Shift the generated charge to the center position and momentum space.
                 */
                charge.rnp = bunchInit.position_[ia];
                charge.rnp += r;

                charge.gb = gb;

                charge.gb += t;
                /* Insert this charge and the mirrored ones into the charge vector.
                 */
                insertCharge(charge);
            }
        }
    }

    /* Reset the value for the number of particle variable according to the installed number of
     * macro-particles and perform the corresponding changes. */
    bunchInit.numberOfParticles_ = chargeVector.size();
}

/** @brief Apply the MITHRA-style initial +z boost and position resynchronization on the host.
 * @ingroup fel_particles
 * @tparam Double Floating-point particle type.
 * @param[in,out] chargeVectorn_ Laboratory samples on entry; moving-frame positions
 * and gamma*beta momenta on return. Charge weights are unchanged.
 * @param frame_gamma Finite frame Lorentz factor at least one, with positive boost velocity.
 * @pre The input list contains finite momenta and positions; gamma is not validated here.
 *
 * Let @f${\bf u}={\bf p}/(mc)@f$ denote particle momentum
 * in units of m*c. The first pass leaves transverse u unchanged and computes
 * @f[
 * z' = \gamma_f z,\qquad
 * u'_z=\gamma_f(u_z-\beta_f\sqrt{1+|{\bf u}|^2}).
 * @f]
 * With @f$z'_{\max}@f$ from this list and
 * @f$\gamma'_p=\sqrt{1+|{\bf u}'|^2}@f$, the second pass shifts all coordinates by
 * @f$\Delta{\bf r}'=({\bf u}'/\gamma'_p)\beta_f(z'-z'_{\max})@f$.
 * This describes the implemented initialization convention, not a general
 * spacetime-event transform. No MPI reduction is performed for z'_{max}; the
 * active manager supplies the complete bunch on rank zero before distribution.
 *
 * @note Nonfinite detected states print diagnostics and call abort(). This path
 * performs host loops only; the conversion to device attributes is separate.
 */
template <typename Double>
void boost_bunch(ChargeVector<Double>& chargeVectorn_, Double frame_gamma) {
    Double frame_beta = std::sqrt((double)frame_gamma * frame_gamma - 1.0) / double(frame_gamma);
    Double zmaxL      = -1.0e100, zmaxG;
    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++) {
        Double g = std::sqrt(1.0 + iterQ->gb.dot(iterQ->gb));
        if (std::isinf(g)) {
            std::cerr << __FILE__ << ": " << __LINE__ << " inf gb: " << iterQ->gb << ", g = " << g
                      << "\n";
            abort();
        }
        Double bz = iterQ->gb[2] / g;
        iterQ->rnp[2] *= frame_gamma;

        iterQ->gb[2] = frame_gamma * g * (bz - frame_beta);

        zmaxL = std::max(zmaxL, iterQ->rnp[2]);
    }
    zmaxG = zmaxL;
    struct {
        Double zu_;
        Double beta_;
    } bunch_;
    bunch_.zu_   = zmaxG;
    bunch_.beta_ = frame_beta;

    /****************************************************************************************************/

    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++) {
        Double g = std::sqrt(1.0 + iterQ->gb.dot(iterQ->gb));
        iterQ->rnp[0] += iterQ->gb[0] / g * (iterQ->rnp[2] - bunch_.zu_) * frame_beta;
        iterQ->rnp[1] += iterQ->gb[1] / g * (iterQ->rnp[2] - bunch_.zu_) * frame_beta;
        iterQ->rnp[2] += iterQ->gb[2] / g * (iterQ->rnp[2] - bunch_.zu_) * frame_beta;
        if (std::isnan(iterQ->rnp[2])) {
            std::cerr << iterQ->gb[2] << ", " << g << ", " << iterQ->rnp[2] << ", " << bunch_.zu_
                      << ", " << frame_beta << "\n";
            std::cerr << __FILE__ << ": " << __LINE__ << " Particle has NaN velocity or position\n";
            abort();
        }
    }
}

/** @brief Generate, boost, and copy a complete temporary bunch into IPPL attributes.
 * @ingroup fel_particles
 * @tparam bunch_type Particle container exposing create(), getLocalNum(), R, R_nm1,
 * and gamma_beta with compatible three-component Kokkos views.
 * @tparam scalar Floating-point type shared by sampling and particle vectors.
 * @param[in,out] bunch Destination container; the active manager calls this on rank zero.
 * @param bunchInit Host sampling settings used with rank=0, size=1 and ia=0.
 * @param frame_gamma Lorentz factor for boost_bunch(), in the physical range gamma>=1.
 * @return Actual local generated sample count, including rejection, grouping and tails.
 * @pre Destination storage is initially empty or compatible with the generated count.
 * This routine grows storage when needed but does not shrink an oversized bunch.
 *
 * Sampling and boost use host lists; data are staged in HostSpace views and
 * explicitly copied to the default Kokkos memory space. A kernel sets R and
 * R_nm1 to the same initial positions and gamma_beta to boosted normalized
 * momenta, then fences. It does not assign charge, mass, or perform MPI migration.
 * The manager assigns total charge/mass divided by the returned count, recentres
 * the bunch, and calls the particle update to distribute it.
 *
 * @note Debug assertions require finite states and nonzero boosted longitudinal
 * momentum. NDEBUG removes those assertions. Charge::q is not copied to Q;
 * nominal weights in the temporary generator do not set the final charge sum.
 */
template <typename bunch_type, typename scalar>
size_t initialize_bunch_mithra(bunch_type& bunch, const BunchInitialize<scalar>& bunchInit,
                               scalar frame_gamma) {
    ChargeVector<scalar> temporary_charge_list;
    initializeBunchEllipsoid(bunchInit, temporary_charge_list, 0, 1, 0);
    for (auto& c : temporary_charge_list) {
        if (std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
        if (std::isinf(c.rnp[0]) || std::isinf(c.rnp[1]) || std::isinf(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
    }
    boost_bunch(temporary_charge_list, frame_gamma);
    for (auto& c : temporary_charge_list) {
        if (std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2])) {
            std::cout << "Pos after boost: " << c.rnp << "\n";
            break;
        }
    }
    Kokkos::View<ippl::Vector<scalar, 3>*, Kokkos::HostSpace> positions("", temporary_charge_list.size());
    Kokkos::View<ippl::Vector<scalar, 3>*, Kokkos::HostSpace> gammabetas("", temporary_charge_list.size());
    auto iterQ = temporary_charge_list.begin();
    for (size_t i = 0; i < temporary_charge_list.size(); i++) {
        assert_isreal(iterQ->gb[0]);
        assert_isreal(iterQ->gb[1]);
        assert_isreal(iterQ->gb[2]);
        assert(iterQ->gb[2] != 0.0f);
        scalar g = std::sqrt(1.0 + iterQ->gb.dot(iterQ->gb));
        assert_isreal(g);
        scalar bz = iterQ->gb[2] / g;
        assert_isreal(bz);
        (void)bz;
        positions(i)  = iterQ->rnp;
        gammabetas(i) = iterQ->gb;
        ++iterQ;
    }
    if (temporary_charge_list.size() > bunch.getLocalNum()) {
        bunch.create(temporary_charge_list.size() - bunch.getLocalNum());
    }
    Kokkos::View<ippl::Vector<scalar, 3>*> dpositions("", temporary_charge_list.size());
    Kokkos::View<ippl::Vector<scalar, 3>*> dgammabetas("", temporary_charge_list.size());

    Kokkos::deep_copy(dpositions, positions);
    Kokkos::deep_copy(dgammabetas, gammabetas);
    Kokkos::deep_copy(bunch.R_nm1.getView(), positions);
    Kokkos::deep_copy(bunch.gamma_beta.getView(), gammabetas);
    auto rview = bunch.R.getView(), rm1view = bunch.R_nm1.getView(),
         gbview = bunch.gamma_beta.getView();
    ;
    Kokkos::parallel_for(
        temporary_charge_list.size(), KOKKOS_LAMBDA(size_t i) {
            rview(i)   = dpositions(i);
            rm1view(i) = dpositions(i);
            gbview(i)  = dgammabetas(i);
        });
    Kokkos::fence();

    return temporary_charge_list.size();
}

#endif
