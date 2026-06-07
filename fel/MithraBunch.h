#ifndef IPPL_FEL_MITHRA_BUNCH_H
#define IPPL_FEL_MITHRA_BUNCH_H

// MITHRA-style relativistic bunch initialization.
//
// Ported faithfully from the original FreeElectronLaser.cpp. Generates an
// ellipsoidal electron bunch (Gaussian/uniform distributions, optional shot
// noise / bunching factor / tail tapering), boosts it into the moving frame,
// and copies the result into an ippl particle bunch.
//
// Behaviour is unchanged from the original; only the surrounding includes and
// the assert_isreal helper were adapted (the old MaxwellSolvers/FDTD.h that
// defined it is no longer part of the tree).

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
#define assert_isreal(X) assert(!std::isnan(X) && !std::isinf(X))
#endif

template <typename scalar>
using FieldVector = ippl::Vector<scalar, 3>;
template <typename scalar>
struct BunchInitialize {
    /* Type of the bunch which is one of the manual, ellipsoid, cylinder, cube, and 3D-crystal. If
     * it is manual the charge at points of the position vector will be produced.
     */
    // std::string     			bunchType_;

    /* Type of the distributions (transverse or longitudinal) in the bunch.
     */
    std::string distribution_;

    /* Type of the generator for creating the bunch distribution.
     */
    std::string generator_;

    /* Total number of macroparticles in the bunch. */
    unsigned int numberOfParticles_;

    /* Total charge of the bunch in pC. */
    scalar cloudCharge_;

    /* Initial energy of the bunch in MeV. */
    scalar initialGamma_;

    /* Initial normalized speed of the bunch. */
    scalar initialBeta_;

    /* Initial movement direction of the bunch, which is a unit vector. */
    FieldVector<scalar> initialDirection_;

    /* Position of the center of the bunch in the unit of length scale. */
    // std::vector<FieldVector<scalar> >	position_;
    FieldVector<scalar> position_;

    /* Number of macroparticles in each direction for 3Dcrystal type. */
    FieldVector<unsigned int> numbers_;

    /* Lattice constant in x, y, and z directions for 3D crystal type. */
    FieldVector<scalar> latticeConstants_;

    /* Spread in position for each of the directions in the unit of length scale. For the 3D crystal
     * type, it will be the spread in position for each micro-bunch of the crystal.
     */
    FieldVector<scalar> sigmaPosition_;

    /* Spread in energy in each direction. */
    FieldVector<scalar> sigmaGammaBeta_;

    /* Store the truncation transverse distance for the electron generation.
     */
    scalar tranTrun_;

    /* Store the truncation longitudinal distance for the electron generation.
     */
    scalar longTrun_;

    /* Name of the file for reading the electrons distribution from.
     */
    std::string fileName_;

    /* The radiation wavelength corresponding to the bunch length outside the undulator
     */
    scalar lambda_;

    /* Bunching factor for the initialization of the bunch.
     */
    scalar bF_;

    /* Phase of the bunching factor for the initialization of the bunch.
     */
    scalar bFP_;

    /* Boolean flag determining the activation of shot-noise.
     */
    bool shotNoise_;

    /* Initial beta vector of the bunch, which is obtained as the product of beta and direction.
     */
    FieldVector<scalar> betaVector_;

    /* Initialize the parameters for the bunch initialization to some first values. */
    // BunchInitialize ();
};

// LORENTZ FRAME AND UNDULATOR

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

    // TODO: Initial bunching factor huh
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
template <typename Double>
struct Charge {
    Double q;                     /* Charge of the point in the unit of electron charge.	*/
    FieldVector<Double> rnp, rnm; /* Position vector of the charge.			*/
    FieldVector<Double> gb;       /* Normalized velocity vector of the charge.		*/

    /* Double flag determining if the particle is passing the entrance point of the undulator. This
     * flag can be used for better boosting the bunch to the moving frame. We need to consider it to
     * be double, because this flag needs to be communicated during bunch update.
     */
    Double e;

    // Charge();
};
template <typename scalar>
using ChargeVector = std::list<Charge<scalar>>;
template <typename Double>
void initializeBunchEllipsoid(BunchInitialize<Double> bunchInit, ChargeVector<Double>& chargeVector,
                              int rank, int size, int ia) {
    /* Correct the number of particles if it is not a multiple of four.
     */
    if (bunchInit.numberOfParticles_ % 4 != 0) {
        unsigned int n = bunchInit.numberOfParticles_ % 4;
        bunchInit.numberOfParticles_ += 4 - n;
        // printmessage(std::string(__FILE__), __LINE__, std::string("Warning: The number of
        // particles in the bunch is not a multiple of four. ") +
        //     std::string("It is corrected to ") +  std::to_string(bunchInit.numberOfParticles_) );
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
    Double Ne, bF, bFi;
    unsigned int bmi;
    std::vector<Double> randomNumbers;

    /* The initialization in group of four particles should only be done if there exists an
     * undulator in the interaction.
     */
    unsigned int ng = (bunchInit.lambda_ == 0.0) ? 1 : 4;

    /* Check the bunching factor. */
    if (bunchInit.bF_ > 2.0 || bunchInit.bF_ < 0.0) {
        // printmessage(std::string(__FILE__), __LINE__, std::string("The bunching factor can not be
        // larger than one or a negative value !!!") ); exit(1);
    }

    /* If the generator is random we should make sure that different processors do not produce the
     * same random numbers.
     */
    if (bunchInit.generator_ == "random") {
        /* Initialize the random number generator.
         */
        srand(time(NULL));
        /* Np / ng * 20 is the maximum number of particles.
         */
        randomNumbers.resize(Np / ng * 20, 0.0);
        for (unsigned int ri = 0; ri < Np / ng * 20; ri++)
            randomNumbers[ri] =
                (float)std::min(1 - 1e-7, std::max(1e-7, ((double)rand()) / RAND_MAX));
    }

    /* Declare the generator function depending on the input.
     */
    auto generate = [&](unsigned int n, unsigned int m) {
        // if 	( bunchInit.generator_ == "random" )
        return (randomNumbers.at(n * 2 * Np / ng + m));
        // else
        //   return  ( randomNumbers[ n * 2 * Np/ng + m ] );
        // TODO: Return halton properly
        // return 	( halton(n,m) );
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
     * the CSE from the tail of the bunch.
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
