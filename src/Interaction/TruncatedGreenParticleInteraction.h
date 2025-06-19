#ifndef IPPL_TRUNCATEDGREEN_SHORTRANGE_H
#define IPPL_TRUNCATEDGREEN_SHORTRANGE_H

#include "ParticleInteractionBase.h"

namespace ippl {
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

    public:
        /*!
         * @param pc Particle container
         * @param F Force attribute, where the evaluated force will be added to
         * @param R Position attribute
         * @param QM Charge or Mass like attribute determining the force magnitued
         * @param params Parameters, containing at least 'alpha', 'force_constant' and 'rcut'. alpha
         * controls the truncation strength. force_constant to be multiplied with the force. rcut
         * determines the maximal distance between two particles to contribute to the forces.
         */
        TruncatedGreenParticleInteraction(const ParticleContainer& pc, VectorAttribute& F,
                                          const VectorAttribute& R, const ScalarAttribute& QM,
                                          const ParameterList& params)
            : Base(pc, params)
            , F_m(F)
            , R_m(R)
            , QM_m(QM) {}

        ~TruncatedGreenParticleInteraction() override = default;

        /*!
         * Evaluates the short range interactions F = - q * forceConstant grad [(1 - erf(alpha *
         * r)) / r].
         */
        void solve() override;

    private:
        /*!
         * Helper function to compute the force F = - q * forceConstant grad [(1 - erf(alpha *
         * r)) / r].
         */
        KOKKOS_INLINE_FUNCTION static constexpr Vector_t pairForce(const Vector_t& dist,
                                                                   Scalar_t r2, Scalar_t alpha,
                                                                   Scalar_t forceConstant,
                                                                   Scalar_t qm2 = 1);

        VectorAttribute& F_m;
        const VectorAttribute& R_m;
        const ScalarAttribute& QM_m;
    };
}  // namespace ippl

#include "TruncatedGreenParticleInteraction.hpp"

#endif  // IPPL_TRUNCATEDGREEN_SHORTRANGE_H
