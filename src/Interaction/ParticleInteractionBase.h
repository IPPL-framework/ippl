//
// Class ParticleInteractionBase
//   Base class for all particle-particle interactions.
//
//   Inheriting classes need to implement solve(), which evaluates all pairwise interactions between
//   particles. These (short-range) interaction happen on the particle container and its attributes
//   and not on a field.
//

#ifndef IPPL_PARTICLEINTERACTIONBASE_H
#define IPPL_PARTICLEINTERACTIONBASE_H

#include <string>

#include "Utility/ParameterList.h"

namespace ippl {
    template <typename ParticleContainer>
    class ParticleInteractionBase {
    public:
        ParticleInteractionBase(const ParticleContainer& pc, const ParameterList& params)
            : pc_m(pc) {
            ParticleInteractionBase::setDefaultParameters();
            params_m.merge(params);
        }

        explicit ParticleInteractionBase(const ParticleContainer& pc)
            : pc_m(pc) {
            ParticleInteractionBase::setDefaultParameters();
        }

    public:
        /*!
         * Update one of the solver's parameters
         * @param key The parameter key
         * @param value The new value
         * @throw IpplException Fails if there is no existing parameter with the given key
         */
        template <typename T>
        void updateParameter(const std::string& key, const T& value) {
            params_m.update<T>(key, value);
        }

        /*!
         * Updates all solver parameters based on values in another parameter set
         * @param params Parameter list with updated values
         * @throw IpplException Fails if the provided parameter list includes keys not already
         * present
         */
        void updateParameters(const ParameterList& params) { params_m.update(params); }

        /*!
         * Merges another parameter set into the solver's parameters, overwriting
         * existing parameters in case of conflict
         * @param params Parameter list with desired values
         */
        void mergeParameters(const ParameterList& params) { params_m.merge(params); }

        virtual void solve() = 0;

        virtual ~ParticleInteractionBase() = default;

    protected:
        const ParticleContainer& pc_m;
        ParameterList params_m;

        /*!
         * Utility function for initializing a solver's default
         * parameters (to be overridden for each base class)
         */
        virtual void setDefaultParameters() {}

    private:
    };
}  // namespace ippl

#endif  // IPPL_PARTICLEINTERACTIONBASE_H
