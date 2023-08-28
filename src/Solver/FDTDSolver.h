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

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {
    template <typename Tfields, unsigned Dim, class M = UniformCartesian<double, Dim>,
              class C = typename M::DefaultCentering>
    class FDTDSolver {
    public:
        // define a type for scalar field (e.g. charge density field)
        // define a type for vectors
        // define a type for vector field
        typedef Field<Tfields, Dim, M, C> Field_t;
        typedef Vector<Tfields, Dim> Vector_t;
        typedef Field<Vector_t, Dim, M, C> VField_t;
        using memory_space = typename Field_t::memory_space;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // type for communication buffers
        using buffer_type = Communicate::buffer_type<memory_space>;

        // constructor and destructor
        FDTDSolver(Field_t& charge, VField_t& current, VField_t& E, VField_t& B,
                   double timestep = 0.05, bool seed_ = false);
        ~FDTDSolver();

        // finite differences time domain solver for potentials (A and phi)
        void solve();

        // evaluates E and B fields using computed potentials
        void field_evaluation();

        // gaussian pulse
        double gaussian(size_t it);

        // initialization of FDTD solver
        void initialize();

    private:
        // mesh and layout objects
        M* mesh_mp;
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

        // fields containing reference to charge and current
        Field_t* rhoN_mp;
        VField_t* JN_mp;

        // scalar and vector potentials at n-1, n, n+1 times
        Field_t phiNm1_m;
        Field_t phiN_m;
        Field_t phiNp1_m;
        VField_t aNm1_m;
        VField_t aN_m;
        VField_t aNp1_m;

        // E and B fields
        VField_t* En_mp;
        VField_t* Bn_mp;

        // buffer for communication
        detail::FieldBufferData<Tfields> fd_m;
    };
}  // namespace ippl

#include "FDTDSolver.hpp"

#endif
