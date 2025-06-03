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

    public:
        TruncatedGreenInteraction(const ParticleContainer &pc, VectorAttribute &F, const VectorAttribute &R,
                                 const ScalarAttribute &QM, const ParameterList &params) : Base(pc, params),
            F_m(&F), R_m(&R), QM_m(&QM) {
        }

        ~TruncatedGreenInteraction() override = default;

        void solve() override;

    private:
        VectorAttribute *F_m;
        const VectorAttribute *R_m;
        const ScalarAttribute *QM_m;
    };
}

#include "TruncatedGreenInteraction.hpp"

#endif //IPPL_TRUNCATEDGREEN_SHORTRANGE_H
