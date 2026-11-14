//
// Class TruncatedGreenParticleInteraction
//   This class implements the short-range part of the Ewald Green function split by evaluating the
//   gradient of forceConstant * erfc(alpha * r) / r. Below regularization_cutoff, the singular
//   Coulomb contribution is replaced by a linear force that matches continuously at the cutoff.
//   The long-range part is handled by the periodic or open-boundary P3M mesh solver.
//
//   The pairs come from the ParticleSpatialOverlapLayout of the container. In the moving-source
//   mode every source also has a momentum, and the class adds the lab-frame E and B of each source
//   moving with its own velocity.
//

#ifndef IPPL_TRUNCATEDGREEN_SHORTRANGE_H
#define IPPL_TRUNCATEDGREEN_SHORTRANGE_H

#include <sstream>

#include "ParticleInteractionBase.h"

namespace ippl {
    namespace detail {
        /*!
         * Per-target sums of the moving-source pair loop: e is the electric contribution, b the
         * sum of u_j x F_ij that becomes the magnetic field after division by the speed of light.
         */
        template <typename Vector_t>
        struct MovingSourceSums {
            Vector_t e;
            Vector_t b;

            KOKKOS_INLINE_FUNCTION explicit MovingSourceSums(typename Vector_t::value_type value)
                : e(value)
                , b(value) {}
        };
    }  // namespace detail

    /*!
     * TruncatedGreenParticleInteraction class definition.
     * @tparam ParticleContainer particle container type
     * @tparam VectorAttribute type to store vector valued attributes
     * @tparam ScalarAttribute type to store scalar valued attributes
     * evaluates truncated short range interaction F = - q * forceConstant grad [(1 - erf(alpha *
     * r)) / r].
     */
    template <typename ParticleContainer, typename VectorAttribute, typename ScalarAttribute>
    class TruncatedGreenParticleInteraction : public ParticleInteractionBase<ParticleContainer> {
    public:
        using Base            = ParticleInteractionBase<ParticleContainer>;
        using Vector_t        = typename VectorAttribute::value_type;
        using Scalar_t        = typename ScalarAttribute::value_type;
        using execution_space = typename VectorAttribute::execution_space;
        static_assert(std::is_same_v<execution_space, typename ScalarAttribute::execution_space>);

        /*!
         * Values of the `pair_kernel` parameter. OWNED_PARTICLE (default) runs one thread per
         * owned particle and sums in registers
         * (ParticleSpatialOverlapLayout::forEachOwnedParticle); CELL_PAIRS is the per-cell
         * traversal with one atomic per pair (forEachPair). Both visit the same pairs, so the
         * fields agree up to summation order.
         */
        enum PairKernel {
            CELL_PAIRS     = 0,
            OWNED_PARTICLE = 1
        };

    public:
        /*!
         * @param pc Particle container
         * @param F Field attribute, where the evaluated field will be added to
         * @param R Position attribute
         * @param QM Charge or Mass like attribute determining the force magnitued
         * @param params Parameters, containing at least 'alpha', 'force_constant' and 'rcut'. alpha
         * controls the Ewald split, force_constant scales the interaction, and rcut determines the
         * maximal contributing pair distance; solve() rejects an rcut above the cutoff of the
         * particle layout. The optional positive regularization_cutoff controls the near-origin
         * softening and defaults to 1e-9; the optional pair_kernel selects a PairKernel.
         */
        TruncatedGreenParticleInteraction(const ParticleContainer& pc, VectorAttribute& F,
                                          const VectorAttribute& R, const ScalarAttribute& QM,
                                          const ParameterList& params)
            : Base(pc, params)
            , Field_m(F)
            , R_m(R)
            , QM_m(QM) {
            if (!this->params_m.contains("regularization_cutoff")) {
                this->params_m.template add<Scalar_t>("regularization_cutoff",
                                                      static_cast<Scalar_t>(1.0e-9));
            }
            if (!this->params_m.contains("pair_kernel")) {
                this->params_m.add("pair_kernel", static_cast<int>(OWNED_PARTICLE));
            }
        }

        /*!
         * @brief Moving-source mode: every source j moves with its own velocity.
         *
         * The truncated field of source j is evaluated at the separation in its rest frame,
         * \f$ r'^2 = r^2 + (\mathbf{u}_j \cdot \mathbf{r})^2 \f$, and transformed to the lab: the
         * electric contribution is \f$ \gamma_j \f$ times the static one with \f$ r' \f$ in the
         * radial term, and the magnetic one is \f$ \mathbf{u}_j / (\gamma_j c) \f$ cross that
         * electric contribution. Pairs are skipped once \f$ r' \f$ reaches rcut. With all
         * \f$ \mathbf{u}_j = 0 \f$ the field equals the one of the static constructor and B is
         * unchanged. Requires the additional positive parameter 'speed_of_light' c, in the units
         * that make \f$ \mathbf{B} = \boldsymbol{\beta} \times \mathbf{E} / c \f$. Three dimensions
         * only.
         * @param B Magnetic field attribute, where the evaluated field will be added to
         * @param P Normalized momentum \f$ \mathbf{u} = \beta\gamma \f$ of every particle; ghosts
         * carry it like any other attribute
         */
        TruncatedGreenParticleInteraction(const ParticleContainer& pc, VectorAttribute& F,
                                          VectorAttribute& B, const VectorAttribute& R,
                                          const VectorAttribute& P, const ScalarAttribute& QM,
                                          const ParameterList& params)
            : TruncatedGreenParticleInteraction(pc, F, R, QM, params) {
            static_assert(Vector_t::dim == 3, "Moving sources need three dimensions");
            B_m = &B;
            P_m = &P;
        }

        ~TruncatedGreenParticleInteraction() override = default;

        /*!
         * Evaluates the short range interactions F = - q * forceConstant grad [(1 - erf(alpha *
         * r)) / r], and in the moving-source mode also the magnetic field.
         */
        void solve() override;

    private:
        /*!
         * Helper function to compute the field = - q * forceConstant grad [(1 - erf(alpha *
         * r)) / r]. generated by an interacting particle
         */
        KOKKOS_INLINE_FUNCTION static constexpr Vector_t fieldFromPair(const Vector_t& dist,
                                                                       Scalar_t r2, Scalar_t alpha,
                                                                       Scalar_t forceConstant,
                                                                       Scalar_t qm, Scalar_t reg);

        /*!
         * Contribution of a source with normalized momentum u at the lab separation dist with
         * \f$ |dist|^2 = r^2 \f$: e is subtracted from the field, b / c from B. Returns false if
         * the rest-frame distance reaches rcut. With u = 0, e equals fieldFromPair() exactly.
         */
        KOKKOS_INLINE_FUNCTION static bool movingSourceFromPair(
            const Vector_t& dist, Scalar_t r2, const Vector_t& u, Scalar_t rcut2, Scalar_t alpha,
            Scalar_t forceConstant, Scalar_t qm, Scalar_t reg, Vector_t& e, Vector_t& b);

        ///! The electric or gravitational field
        VectorAttribute& Field_m;
        ///! Positions of the particles
        const VectorAttribute& R_m;
        ///! Charge or Mass of the particles
        const ScalarAttribute& QM_m;
        ///! Magnetic field, only in the moving-source mode
        VectorAttribute* B_m = nullptr;
        ///! Normalized momenta of the sources, only in the moving-source mode
        const VectorAttribute* P_m = nullptr;
    };
}  // namespace ippl

#include "TruncatedGreenParticleInteraction.hpp"

#endif  // IPPL_TRUNCATEDGREEN_SHORTRANGE_H
