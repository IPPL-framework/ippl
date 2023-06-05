//
// TestP3MSolver
// This program tests the P3MSolver with a constant source rho = 2.
// This is for comparison purposes with a reference implementation in ippl_orig.
// I/O output is only enabled when running serially.
//   Usage:
//     srun ./TestP3MSolver <nx> <ny> <nz> --info 5
//     nx = No. cell-centered points in the x-direction
//     ny = No. cell-centered points in the y-direction
//     nz = No. cell-centered points in the z-direction
//
//     Example:
//       srun ./TestP3MSolver 16 16 16 --info 5
//
// Copyright (c) 2023, Sonali Mayani,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include "Ippl.h"

#include <iostream>

#include "P3MSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        int ranks = ippl::Comm->size();

        constexpr unsigned int dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Vector<double, dim> Vector_t;
        typedef ippl::Field<Vector_t, dim, Mesh_t, Centering_t> VField_t;
        typedef ippl::P3MSolver<VField_t, Field_t> Solver_t;

        // get the gridsize from the user
        ippl::Vector<int, dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // domain
        ippl::NDIndex<dim> owned;
        for (unsigned i = 0; i < dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[dim];
        for (unsigned int d = 0; d < dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        // unit box
        double dx       = 1.0 / nr[0];
        double dy       = 1.0 / nr[1];
        double dz       = 1.0 / nr[2];
        Vector_t hr     = {dx, dy, dz};
        Vector_t origin = {-0.5, -0.5, -0.5};

        Mesh_t mesh(owned, hr, origin);
        ippl::FieldLayout<dim> layout(owned, decomp);

        Field_t field;
        field.initialize(mesh, layout);

        VField_t efield;
        efield.initialize(mesh, layout);

        ippl::ParameterList params;
        params.add("use_heffte_defaults", false);
        params.add("use_pencils", true);
        // params.add("use_reorder", false);
        params.add("use_gpu_aware", true);
        params.add("comm", ippl::a2av);
        params.add("r2c_direction", 0);
        params.add("output_type", Solver_t::SOL_AND_GRAD);

        // assign the rho field with 2.0
        typename Field_t::view_type view_rho = field.getView();

        Kokkos::parallel_for(
            "Assign rho field", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) { view_rho(i, j, k) = 2.0; });

        if (ranks == 1) {
            msg << "Rho: " << endl;
            field.write();
        }

        Solver_t solver;

        solver.mergeParameters(params);

        solver.setLhs(efield);
        solver.setRhs(field);

        solver.solve();

        if (ranks == 1) {
            msg << "Computed phi: " << endl;
            field.write();
        }

        if (ranks == 1) {
            msg << "Efield: " << endl;
            efield.write();
        }

        msg << "End of test" << endl;
    }
    ippl::finalize();

    return 0;
}
