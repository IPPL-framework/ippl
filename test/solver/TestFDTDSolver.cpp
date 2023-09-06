//
// TestFDTDSolver
// This programs tests the FDTD electromagnetic solver with a
// sinusoidal pulse at the center, and absorbing boundaries.
//   Usage:
//     srun ./TestFDTDSolver <nx> <ny> <nz> <timesteps> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//     timesteps = No. of timesteps
//     (the timestep size is computed using the CFL condition)
//
//     Example:
//       srun ./TestFDTDSolver 25 25 25 150 --info 5
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

#include <cstdlib>
#include <fstream>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Solver/FDTDSolver.h"

KOKKOS_INLINE_FUNCTION double sine(double n, double dt) {
    return 100 * std::sin(n * dt);
}
template<typename T>
KOKKOS_INLINE_FUNCTION auto sq(const T& x) -> decltype(std::declval<T>() * std::declval<T>()) {
    return x * x;
}


void dumpVTK(ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>,
                         ippl::UniformCartesian<double, 3>::DefaultCentering>& E,
             int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {

    //return;
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<ippl::Vector<double, 3>, 3, Mesh_t, Centering_t> VField_t;
    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    std::ofstream vtkout(fname.str().c_str());
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    #define endl '\n'
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    #undef endl
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                       << host_view(x, y, z)[2] << '\n';
            }
        }
    }
    vtkout << std::endl;
}

void dumpVTK(ippl::Field<double, 3, ippl::UniformCartesian<double, 3>,
                         ippl::UniformCartesian<double, 3>::DefaultCentering>& rho,
             int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<double, 3, Mesh_t, Centering_t> Field_t;
    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z) << '\n';
            }
        }
    }
    vtkout << endl;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    for(int ext = 75;ext < 80;ext += 5){
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);


        const unsigned int Dim = 3;

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {ext, ext, ext};
        //ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // get the total simulation time from the user
        //const unsigned int iterations = std::atof(argv[4]);

        using Mesh_t      = ippl::UniformCartesian<double, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t> VField_t;

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        ippl::e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        // unit box
        double dx                        = 1.0 / nr[0];
        double dy                        = 1.0 / nr[1];
        double dz                        = 1.0 / nr[2];
        ippl::Vector<double, Dim> hr     = {dx, dy, dz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);

        // CFL condition lambda = c*dt/h < 1/sqrt(d) = 0.57 for d = 3
        // we set a more conservative limit by choosing lambda = 0.5
        // we take h = minimum(dx, dy, dz)
        const double c = 1.0;  // 299792458.0;
        double dt      = std::min({dx, dy, dz}) * 0.1 / c;
        

        //Simulation should run for 1.0 time-units
        const double total_time = 1.0;

        //Current simulation time (will increase by dt every timestep)
        double simulated_time = 0.0;

        double timesteps_nonintegral = total_time / dt; //To be ceiled
        double timesteps_ceiled = std::ceil(timesteps_nonintegral);
        dt = total_time / timesteps_ceiled;
        
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);

        // define the R (rho) field
        Field_t rho;
        rho.initialize(mesh, layout);

        // define the Vector field E (LHS)
        VField_t fieldE, fieldB;
        fieldE.initialize(mesh, layout);
        fieldB.initialize(mesh, layout);
        fieldE = 0.0;
        fieldB = 0.0;

        // define current = 0
        VField_t current;
        current.initialize(mesh, layout);
        current = 0.0;

        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object
        ippl::FDTDSolver<double, Dim> solver(rho, current, fieldE, fieldB, dt, seed);

        if (!seed) {
            // add pulse at center of domain
            auto view_rho    = rho.getView();
            solver.aNm1_m = 0.0;
            auto vector_potential_at_minus_one_view = solver.aNm1_m.getView();
            auto vector_potential_at_now_view = solver.aN_m.getView();

            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();

            Kokkos::parallel_for(
                "Assign sinusoidal source at center", ippl::getRangePolicy(vector_potential_at_now_view, 0)/*rho.getFieldRangePolicy()*/,
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];
                    //std::cout << y << " Y\n";
                    if ((x == 0.5) && (y == 0.5) && (z == 0.5)){
                        //vector_potential_at_now_view(i, j, k) = 100.0;
                        //view_rho(i, j, k) = sine(0, dt);
                    }
                    if (y >= 0.4 && y <= 0.6){
                        vector_potential_at_now_view(i, j, k) = Kokkos::exp(-sq((y - 0.5) * 20.0));
                    }
            });
            //vector_potential_at_now_view(7,7,0) = 1000.0;
        }
        solver.field_evaluation();
        {
            auto vector_potential_at_minus_one_view = solver.aNm1_m.getView();
            auto vector_potential_at_now_view = solver.aN_m.getView();
            solver.field_evaluation();
            auto electric_field_view = fieldE.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();
            double error_sum = 0.0;
            Kokkos::parallel_reduce(
                "Assign sinusoidal source at center", ippl::getRangePolicy(vector_potential_at_now_view, 1)/*rho.getFieldRangePolicy()*/,
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum_ref) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    if ((x == 0.5) && (y == 0.5) && (z == 0.5)){
                        //vector_potential_at_now_view(i, j, k) = 100.0;
                        //view_rho(i, j, k) = sine(0, dt);
                    }
                    ippl::Vector<double, 3> expected_E_field = 0.0;
                    if (y >= 0.4 && y <= 0.6){
                        expected_E_field = Kokkos::exp(-sq((y - 0.5) * 20.0));
                    }
                    
                    expected_E_field[0] *= 1.0 / dt;
                    expected_E_field[1] *= 1.0 / dt;
                    expected_E_field[2] *= 1.0 / dt;

                    ippl::Vector<double, 3> error = electric_field_view(i, j, k) - expected_E_field;
                    //if(std::abs(error[0]) > 0.0){
                    //    std::ostringstream ostr;
                    //    ostr << electric_field_view(i, j, k) << " vs " << expected_E_field << "\n";
                    //    std::cout << ostr.str();
                    //}
                    
                    sum_ref += std::abs(error[0]);
                    sum_ref += std::abs(error[1]);
                    sum_ref += std::abs(error[2]);
            
            }, error_sum);
            std::cout << ext << " Direkt nachher: " << error_sum << std::endl;
            //return 0;
        }
        dumpVTK(fieldE, nr[0], nr[1], nr[2], 0, hr[0], hr[1], hr[2]);

        msg << "Timestep number = " << 0 << " , time = " << 0 << endl;
        solver.solve();
        
        simulated_time += dt;

        // time-loop
        double every_dt_output = 0.005;
        int every_step_output = every_dt_output / dt;

        
        //Simple incrementor for now
        unsigned int it = 1;
        for (;simulated_time < total_time;it++) {
            msg << "Timestep number = " << it << " , time = " << it * dt << endl;
            //solver.dt = std::min(dt, total_time - simulated_time);
            /*if (false && !seed) {
                // add pulse at center of domain
                auto view_rho    = rho.getView();
                const int nghost = rho.getNghost();
                auto ldom        = layout.getLocalNDIndex();

                Kokkos::parallel_for(
                    "Assign sine source at center", ippl::getRangePolicy(view_rho, nghost),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        const int ig = i + ldom[0].first() - nghost;
                        const int jg = j + ldom[1].first() - nghost;
                        const int kg = k + ldom[2].first() - nghost;

                        // define the physical points (cell-centered)
                        double x = (ig + 0.5) * hr[0] + origin[0];
                        double y = (jg + 0.5) * hr[1] + origin[1];
                        double z = (kg + 0.5) * hr[2] + origin[2];

                        if ((x == 0.5) && (y == 0.5) && (z == 0.5))
                            view_rho(i, j, k) = sine(it, dt);
                });
            }*/
            
            solver.solve();
            simulated_time += solver.dt;
            //double time = it * dt, ptime = (it - 1) * dt;
            std::cout << it << ": " << fieldE(nr[0] / 2, nr[1] / 2, nr[2] / 2) << "\n";
            if(it % every_step_output == 0/*std::fmod(time, every_dt_output) - std::fmod(ptime, every_dt_output) < 0*/){
                dumpVTK(fieldE, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
            }
            //return 0;
        }
        std::cout << "Simulation time: " << simulated_time << std::endl;
        if(!seed){ // Well and otherwise there's no error analysis
            auto vector_potential_at_minus_one_view = solver.aNm1_m.getView();
            auto vector_potential_at_now_view = solver.aN_m.getView();
            //solver.field_evaluation();
            auto electric_field_view = fieldE.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();
            double error_sum = 0.0;
            Kokkos::parallel_reduce(
                "Assign sinusoidal source at center", ippl::getRangePolicy(vector_potential_at_now_view, 1)/*rho.getFieldRangePolicy()*/,
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum_ref) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    if ((x == 0.5) && (y == 0.5) && (z == 0.5)){
                        //vector_potential_at_now_view(i, j, k) = 100.0;
                        //view_rho(i, j, k) = sine(0, dt);
                    }
                    ippl::Vector<double, 3> expected_E_field = 0.0;
                    if (y >= 0.4 && y <= 0.6){
                        expected_E_field = Kokkos::exp(-sq((y - 0.5) * 20.0));
                    }
                    
                    expected_E_field[0] *= -1.0 / dt;
                    expected_E_field[1] *= -1.0 / dt;
                    expected_E_field[2] *= -1.0 / dt;

                    ippl::Vector<double, 3> error = electric_field_view(i, j, k) - expected_E_field;
                    if(std::abs(error[0]) > 1.0){
                        std::ostringstream ostr;
                        ostr << electric_field_view(i, j, k) << " vs " << expected_E_field << "\n";
                        //std::cout << ostr.str();
                    }
                    
                    sum_ref += std::abs(error[0]);
                    sum_ref += std::abs(error[1]);
                    sum_ref += std::abs(error[2]);
            
            }, error_sum);
            std::cout << ext << ": " << error_sum / double(ext * ext * ext) << std::endl;
        }
        std::cout << "Breaking after one loop\n";
        break;
    }
    ippl::finalize();

    return 0;
}
