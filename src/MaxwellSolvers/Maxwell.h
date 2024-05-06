//
// Class Maxwell
//   Base class for solvers for Maxwell's equations
//

#ifndef IPPL_MAXWELL_H
#define IPPL_MAXWELL_H

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename EMField, typename SourceField>
    class Maxwell {
    public:
        // typedefs for the different fields and vector fields
        using typeR = typename SourceField::value_type;
        using typeL = typename EMField::value_type;

        constexpr static unsigned Dim = EMField::dim;

        // type for communication buffers
        using memory_space = typename SourceField::memory_space;
        using buffer_type  = Communicate::buffer_type<memory_space>;

        /*!
         * Default constructor for Maxwell solvers;
         */
        Maxwell() {}

        Maxwell(SourceField& four_current, EMField& E, EMField& B) {
            setSources(four_current);
            setEMFields(E, B);
        }

        /*!
         * Set the problem RHS (charge & current densities)
         * @param four_current Reference to the four current field (rho, J)
         */
        virtual void setSources(SourceField& four_current) { JN_mp = &four_current; }

        /*!
         * Set the problem LHS (electromagnetic fields)
         * @param E Reference to electric field
         * @param B Reference to magnetic field
         */
        void setEMFields(EMField& E, EMField& B) {
            En_mp = &E;
            Bn_mp = &B;
        }

        /*!
         * Solve the electromagnetic problem (Maxwell's eqs)
         */
        virtual void solve() = 0;

        virtual ~Maxwell() {}

    protected:
        // fields containing reference to four-current (rho, J)
        SourceField* JN_mp = nullptr;

        // E and B fields
        EMField* En_mp = nullptr;
        EMField* Bn_mp = nullptr;
    };
}  // namespace ippl

#endif
