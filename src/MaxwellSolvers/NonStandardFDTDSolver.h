/**
 * @file NonStandardFDTDSolver.h
 * @brief Defines the NonStandardFDTDSolver class for solving Maxwell's equations using a
 * non-standard Finite-Difference Time-Domain (FDTD) method. The method and derivation of the
 * discretization are based on:
 * - A. Taflove, *Computational Electrodynamics: The Finite-difference Time-domain Method*, Artech
 * House, 1995
 * - A. Fallahi, MITHRA 2.0: *A Full-Wave Simulation Tool for Free Electron Lasers*, (2020)
 */

#ifndef IPPL_NON_STANDARD_FDTD_SOLVER_H
#define IPPL_NON_STANDARD_FDTD_SOLVER_H

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
     * @class NonStandardFDTDSolver
     * @brief A solver for Maxwell's equations using a non-standard Finite-Difference Time-Domain
     * (FDTD) method.
     *
     * @tparam EMField The type representing the electromagnetic field.
     * @tparam SourceField The type representing the source field.
     * @tparam boundary_conditions The boundary conditions to be applied (default is periodic).
     */

    template <typename EMField, typename SourceField, fdtd_bc boundary_conditions = periodic>
    class NonStandardFDTDSolver : public FDTDSolverBase<EMField, SourceField, boundary_conditions> {
    public:
        /**
         * @brief Constructs a NonStandardFDTDSolver.
         *
         * @param source The source field.
         * @param E The electric field.
         * @param B The magnetic field.
         */
        NonStandardFDTDSolver(SourceField& source, EMField& E, EMField& B);

        constexpr static unsigned Dim = EMField::dim;  // Dimension of the electromagnetic field.
        using scalar = typename EMField::value_type::value_type;  // Scalar type used in the
                                                                  // electromagnetic field.
        using Vector_t = Vector<typename EMField::value_type::value_type,
                                Dim>;  // Vector type used in the electromagnetic field.
        using SourceVector_t =
            typename SourceField::value_type;  // Vector type used in the source field.

        /**
         * @brief A structure representing nondispersive coefficients.
         *
         * This structure holds the coefficients used in the nondispersive
         * formulation of the solver. These coefficients are used to update
         * the electromagnetic fields during the simulation.
         *
         * @tparam scalar The type of the coefficients.
         */
        template <typename scalar>
        struct nondispersive {
            scalar a1;
            scalar a2;
            scalar a4;
            scalar a6;
            scalar a8;
        };

        /**
         * @brief Advances the simulation by one time step.
         */
        void step() override;
        /**
         * @brief Initializes the solver.
         */
        void initialize() override;
    };
}  // namespace ippl

#include "NonStandardFDTDSolver.hpp"
#endif