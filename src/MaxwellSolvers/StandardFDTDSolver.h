/**
 * @file StandardFDTDSolver.h
 * @brief Defines the StandardFDTDSolver class for solving Maxwell's equations using the FDTD
 * method.
 */

#ifndef IPPL_STANDARD_FDTD_SOLVER_H
#define IPPL_STANDARD_FDTD_SOLVER_H

#include <cstddef>
using std::size_t;
#include "Types/Vector.h"

#include "FieldLayout/FieldLayout.h"
#include "MaxwellSolvers/AbsorbingBC.h"
#include "MaxwellSolvers/FDTDSolverBase.h"
#include "MaxwellSolvers/Maxwell.h"
#include "Meshes/UniformCartesian.h"
#include "Particle/ParticleBase.h"

namespace ippl {

    /**
     * @class StandardFDTDSolver
     * @brief A solver for Maxwell's equations using the Finite-Difference Time-Domain (FDTD)
     * method.
     *
     * @tparam EMField The type representing the electromagnetic field.
     * @tparam SourceField The type representing the source field.
     * @tparam boundary_conditions The boundary conditions to be applied (default is periodic).
     */
    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions = periodic>
    class StandardFDTDSolver : public FDTDSolverBase<EMField, SourceField, boundary_conditions> {
    public:
        /**
         * @brief Constructs a StandardFDTDSolver.
         *
         * @param source The source field.
         * @param E The electric field.
         * @param B The magnetic field.
         */
        StandardFDTDSolver(SourceField& source, EMField& E, EMField& B);

        constexpr static unsigned Dim = EMField::dim;  // Dimension of the electromagnetic field.
        using scalar = typename EMField::value_type::value_type;  // Scalar type used in the
                                                                  // electromagnetic field.
        using Vector_t = Vector<typename EMField::value_type::value_type,
                                Dim>;  // Vector type used in the electromagnetic field.
        using SourceVector_t =
            typename SourceField::value_type;  // Vector type used in the source field.

        /**
         * @brief Advances the simulation by one time step.
         */
        virtual void step() override;
        /**
         * @brief Initializes the solver.
         */
        virtual void initialize() override;
    };
}  // namespace ippl

#include "StandardFDTDSolver.hpp"
#endif