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

    template <typename EMField, typename FourField>
    class Maxwell {
    public:
        // typedefs for the different fields and vector fields
        using typeR = typename FourField::value_type;

        constexpr static unsigned Dim = EMField::dim;
        typedef typename EMField::Mesh_t Mesh_t;
        typedef typename EMField::Centering_t Centering_t;
        typedef Vector<typeR, Dim> Vector_t;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // type for communication buffers
        using memory_space = typename FourField::memory_space;
        //using buffer_type  = Communicate::buffer_type<memory_space>;

        /*!
         * Default constructor for Maxwell solvers;
         */
        Maxwell() {}

        Maxwell(FourField& source, EMField& E, EMField& B) {
            setSource(source);
            setEMFields(E, B);
        }

        /*!
         * Set the problem RHS (charge & current densities)
         * @param charge Reference to rho field
         * @param current Reference to J field
         */
        virtual void setSource(FourField& charge) {
            source_mp = &charge;
        }

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
        // fields containing reference to charge and current
        FourField* source_mp = nullptr;

        // E and B fields
        EMField* En_mp = nullptr;
        EMField* Bn_mp = nullptr;
    };
}  // namespace ippl

#endif
