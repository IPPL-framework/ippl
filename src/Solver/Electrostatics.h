//
// Class Electrostatics
//   Base class for solvers for electrostatics problems
//
// Copyright (c) 2021 Alessandro Vinciguerra, ETH Zürich, Zurich, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef IPPL_ELECTROSTATICS_H
#define IPPL_ELECTROSTATICS_H

#include "Solver/Solver.h"

namespace ippl {

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Cell>
    class Electrostatics : public Solver<Tlhs, Trhs, Dim, Mesh, Cell> {
    public:
        using grad_type = Field<Vector<Tlhs, Dim>, Dim, Mesh, Cell>;
        using lhs_type  = typename Solver<Tlhs, Trhs, Dim, Mesh, Cell>::lhs_type;
        using rhs_type  = typename Solver<Tlhs, Trhs, Dim, Mesh, Cell>::rhs_type;

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
            : Solver<Tlhs, Trhs, Dim, Mesh, Cell>()
            , grad_mp(nullptr) {
            setDefaultParameters();
        }

        Electrostatics(lhs_type& lhs, rhs_type& rhs)
            : Solver<Tlhs, Trhs, Dim, Mesh, Cell>(lhs, rhs)
            , grad_mp(nullptr) {
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
