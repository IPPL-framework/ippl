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
        constexpr static unsigned Dim = EMField::dim;

        /*!
         * Default constructor for Maxwell solvers;
         */
        Maxwell() {}

        /*!
         * Constructor which allows to initialize the field pointers
         * (J, E, B) in the Maxwell solvers class
         * @param four_current The four current field (rho, J)
         * @param E The electric field
         * @param B The magnetic field
         */
        Maxwell(SourceField& four_current, EMField& E, EMField& B) {
            setSources(four_current);
            setEMFields(E, B);
        }

        /*!
         * Set the problem RHS (charge & current densities)
         * @param four_current The four current field (rho, J)
         */
        virtual void setSources(SourceField& four_current) { JN_mp = &four_current; }

        /*!
         * Set the problem LHS (electromagnetic fields)
         * @param E The electric field
         * @param B The magnetic field
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
        // Field for four-current (rho, J)
        SourceField* JN_mp = nullptr;

        // E and B fields
        EMField* En_mp = nullptr;
        EMField* Bn_mp = nullptr;
    };
}  // namespace ippl

#endif
