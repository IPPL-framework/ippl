#ifndef IPPL_TRUNCATEDGREEN_SHORTRANGE_H
#define IPPL_TRUNCATEDGREEN_SHORTRANGE_H

#include "ParticleInteractionBase.h"

namespace ippl {
    template<typename ParticleContainer, typename VectorAttribute, typename ScalarAttribute>
    class TruncatedGreenInteraction : public ParticleInteractionBase<ParticleContainer> /*ScaledForce, Position, Charge/Mass*/ {
    public:
        using Base = ParticleInteractionBase<ParticleContainer>;
        using Vector_t = typename VectorAttribute::value_type;
        using Scalar_t = typename ScalarAttribute::value_type;
        using execution_space = typename VectorAttribute::execution_space;
        static_assert(std::is_same_v<execution_space, typename ScalarAttribute::execution_space>);

    public:
        TruncatedGreenInteraction(const ParticleContainer &pc, VectorAttribute &F, const VectorAttribute &R,
                                 const ScalarAttribute &QM, const ParameterList &params) : Base(pc, params),
            F_m(&F), R_m(&R), QM_m(&QM) {
        }

        ~TruncatedGreenInteraction() override = default;

        void solve() override;

    private:

        KOKKOS_INLINE_FUNCTION static constexpr Vector_t pairForce(const Vector_t& dist, Scalar_t r2, Scalar_t alpha, Scalar_t forceConstant, Scalar_t qm2 = 1);

        VectorAttribute *F_m;
        const VectorAttribute *R_m;
        const ScalarAttribute *QM_m;
    };
}

#include "TruncatedGreenInteraction.hpp"

#endif //IPPL_TRUNCATEDGREEN_SHORTRANGE_H
