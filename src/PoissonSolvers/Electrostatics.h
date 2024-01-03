//
// Class Electrostatics
//   Base class for solvers for electrostatics problems
//
// Copyright (c) 2021 Alessandro Vinciguerra, ETH Zürich, Zurich, Switzerland
// All rights reserved
//
// This file is part of IPPL.

#ifndef IPPL_ELECTROSTATICS_H
#define IPPL_ELECTROSTATICS_H

#include "Solver/Solver.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class Electrostatics : public Solver<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;
        using Trhs                    = typename FieldRHS::value_type;
        using Base                    = Solver<FieldLHS, FieldRHS>;

    public:
        using grad_type = Field<Vector<Tlhs, Dim>, Dim, typename FieldLHS::Mesh_t,
                                typename FieldLHS::Centering_t>;
        using typename Base::lhs_type, typename Base::rhs_type;

        /*!
         * Represents the types of fields that should
         * be output by the solver
         */
        enum OutputType {
            SOL          = 0b01,
            GRAD         = 0b10,
            SOL_AND_GRAD = 0b11
        };

        /*!
         * Default constructor for electrostatic solvers;
         * desired output type defaults to solution only
         */
        Electrostatics()
            : Base()
            , grad_mp(nullptr) {
            static_assert(std::is_floating_point<Trhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        Electrostatics(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , grad_mp(nullptr) {
            static_assert(std::is_floating_point<Trhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        /*!
         * Set the field in which the gradient of the computed potential
         * should be stored
         * @param grad Reference to field in which to store the gradient
         */
        void setGradient(grad_type& grad) { grad_mp = &grad; }

        /*!
         * Solve the electrostatics problem described by
         * -laplace(lhs) = rhs
         */
        virtual void solve() = 0;

        virtual ~Electrostatics() {}

    protected:
        grad_type* grad_mp;

        virtual void setDefaultParameters() override { this->params_m.add("output_type", SOL); }
    };
}  // namespace ippl

#endif
