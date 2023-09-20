//
// Class FDTDSolver
//   Finite Differences Time Domain electromagnetic solver.
//
// Copyright (c) 2022, Sonali Mayani, PSI, Villigen, Switzerland
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

#ifndef FDTD_SOLVER_H_
#define FDTD_SOLVER_H_

#include "Solver/Maxwell.h"

namespace ippl {
    template <typename EMField, typename SourceField>
    class FDTDSolver : public Maxwell<EMField, SourceField> {
    public:
        constexpr static unsigned Dim = EMField::dim;
        using typeR                   = typename SourceField::value_type;
        using Base                    = Maxwell<EMField, SourceField>;
        using VectorSourceField_t     = typename Base::VectorSourceField_t;
        using Mesh_t                  = typename Base::Mesh_t;
        using FieldLayout_t           = typename Base::FieldLayout_t;
        using Vector_t                = typename Base::Vector_t;

        // constructor and destructor
        FDTDSolver()
            : Base() {}
        FDTDSolver(SourceField& charge, VectorSourceField_t& current, EMField& E, EMField& B,
                   double timestep = 0.05, bool seed_ = false);
        ~FDTDSolver() = default;

        // finite differences time domain solver for potentials (A and phi)
        void solve() override;

        // evaluates E and B fields using computed potentials
        void field_evaluation();

        // gaussian pulse
        double gaussian(size_t it);

        // initialization of FDTD solver
        void initialize();

    private:
        // mesh and layout objects
        Mesh_t* mesh_mp;
        FieldLayout_t* layout_mp;

        // computational domain
        NDIndex<Dim> domain_m;

        // mesh spacing and mesh size
        Vector_t hr_m;
        Vector<int, Dim> nr_m;

        // size of timestep
        double dt;

        // seed flag
        bool seed;

        // iteration number for gaussian seed
        size_t iteration = 0;

        // scalar and vector potentials at n-1, n, n+1 times
        SourceField phiNm1_m;
        SourceField phiN_m;
        SourceField phiNp1_m;
        VectorSourceField_t aNm1_m;
        VectorSourceField_t aN_m;
        VectorSourceField_t aNp1_m;

        // buffer for communication
        detail::FieldBufferData<typeR> fd_m;
    };
}  // namespace ippl

#include "FDTDSolver.hpp"

#endif
