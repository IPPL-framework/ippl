//
// TestCopy
// This program tests the assignment of a field from an N^3 grid to
// a (2N)^3 grid and vice-versa.
// Non-trivial MPI communication is involved.
// The copy is iterated 5 times for the purpose of timing studies.
//   Usage:
//     srun ./TestCopy <nx> <ny> <nz> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//
//     For more info on the heffte parameters, see:
//     https://github.com/icl-utk-edu/heffte
//
//     Example:
//       srun ./TestCopy 64 64 64 HOCKNEY --info 5
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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Solver/FFTPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const unsigned int Dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
        using memory_space = typename field::memory_space;
        using buffer_type  = ippl::Communicate::buffer_type<memory_space>;
        using Trhs = double;

        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // print out info and title for the relative error (L2 norm)
        msg << "TestCopy, grid = " << nr << endl;

        // domain
        ippl::NDIndex<Dim> owned, owned2;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i]  = ippl::Index(nr[i]);
            owned2[i] = ippl::Index(2 * nr[i]);
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
        Mesh_t mesh2(owned2, hr, origin);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);
        ippl::FieldLayout<Dim> layout2(owned2, decomp);

        // define the R (rho) field
        field rho, rho2;
        rho.initialize(mesh, layout);
        rho2.initialize(mesh2, layout2);

        // communication buffers
        ippl::detail::FieldBufferData<double> fd;

        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {

            // initialize rho to something constant
            rho = 1.0;

            // start a timer
            static IpplTimings::TimerRef stod = IpplTimings::getTimer("Solve: Physical to double");
            IpplTimings::startTimer(stod);

            // store rho (RHS) in the lower left quadrant of the doubled grid
            // with or without communication (if only 1 rank)

            const int ranks = ippl::Comm->size();

            auto view2 = rho2.getView();
            auto view1 = rho.getView();

            const int nghost2 = rho2.getNghost();
            const int nghost1 = rho.getNghost();

            const auto& ldom2 = layout2.getLocalNDIndex();
            const auto& ldom1 = layout.getLocalNDIndex();

            if (ranks > 1) {
                // COMMUNICATION
                const auto& lDomains2 = layout2.getHostLocalDomains();

                // send
                std::vector<MPI_Request> requests(0);

                for (int i = 0; i < ranks; ++i) {
                    if (lDomains2[i].touches(ldom1)) {
                        auto intersection = lDomains2[i].intersect(ldom1);

                        requests.resize(requests.size() + 1);

                        ippl::Communicate::size_type nsends;
                        pack(intersection, view1, fd, nghost1, ldom1, nsends);

                        buffer_type buf =
                            ippl::Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_SEND + i, nsends);

                        ippl::Comm->isend(i, OPEN_SOLVER_TAG, fd, *buf, requests.back(), nsends);
                        buf->resetWritePos();
                    }
                }

                // receive
                const auto& lDomains1 = layout.getHostLocalDomains();
                int myRank            = ippl::Comm->rank();

                for (int i = 0; i < ranks; ++i) {
                    if (lDomains1[i].touches(ldom2)) {
                        auto intersection = lDomains1[i].intersect(ldom2);

                        ippl::Communicate::size_type nrecvs;
                        nrecvs = intersection.size();

                        buffer_type buf =
                            ippl::Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_RECV + myRank, nrecvs);

                        ippl::Comm->recv(i, OPEN_SOLVER_TAG, fd, *buf, nrecvs * sizeof(Trhs), nrecvs);
                        buf->resetReadPos();

                        unpack(intersection, view2, fd, nghost2, ldom2);
                    }
                }

                // wait for all messages to be received
                if (requests.size() > 0) {
                    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                }
                ippl::Comm->barrier();

            } else {
                Kokkos::parallel_for(
                    "Write rho on the doubled grid", rho.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        const size_t ig2 = i + ldom2[0].first() - nghost2;
                        const size_t jg2 = j + ldom2[1].first() - nghost2;
                        const size_t kg2 = k + ldom2[2].first() - nghost2;

                        const size_t ig1 = i + ldom1[0].first() - nghost1;
                        const size_t jg1 = j + ldom1[1].first() - nghost1;
                        const size_t kg1 = k + ldom1[2].first() - nghost1;

                        // write physical rho on [0,N-1] of doubled field
                        const bool isQuadrant1 = ((ig1 == ig2) && (jg1 == jg2) && (kg1 == kg2));
                        view2(i, j, k)         = view1(i, j, k) * isQuadrant1;
                    });
            }
            IpplTimings::stopTimer(stod);
            
            Kokkos::fence();
            ippl::Comm->barrier();

            rho2 = 2.0 * rho2;

            // start a timer
            static IpplTimings::TimerRef dtos = IpplTimings::getTimer("Solve: Double to physical");
            IpplTimings::startTimer(dtos);

            // get the physical part only --> physical electrostatic potential is now given in RHS
            // need communication if more than one rank

            if (ranks > 1) {
                // COMMUNICATION

                // send
                const auto& lDomains1 = layout.getHostLocalDomains();

                std::vector<MPI_Request> requests(0);

                for (int i = 0; i < ranks; ++i) {
                    if (lDomains1[i].touches(ldom2)) {
                        auto intersection = lDomains1[i].intersect(ldom2);

                        requests.resize(requests.size() + 1);

                        ippl::Communicate::size_type nsends;
                        pack(intersection, view2, fd, nghost2, ldom2, nsends);

                        buffer_type buf =
                            ippl::Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_SEND + i, nsends);

                        ippl::Comm->isend(i, OPEN_SOLVER_TAG, fd, *buf, requests.back(), nsends);
                        buf->resetWritePos();
                    }
                }

                // receive
                const auto& lDomains2 = layout2.getHostLocalDomains();
                int myRank            = ippl::Comm->rank();

                for (int i = 0; i < ranks; ++i) {
                    if (ldom1.touches(lDomains2[i])) {
                        auto intersection = ldom1.intersect(lDomains2[i]);

                        ippl::Communicate::size_type nrecvs;
                        nrecvs = intersection.size();

                        buffer_type buf =
                            ippl::Comm->getBuffer<memory_space, Trhs>(IPPL_SOLVER_RECV + myRank, nrecvs);

                        ippl::Comm->recv(i, OPEN_SOLVER_TAG, fd, *buf, nrecvs * sizeof(Trhs), nrecvs);
                        buf->resetReadPos();

                        unpack(intersection, view1, fd, nghost1, ldom1);
                    }
                }

                // wait for all messages to be received
                if (requests.size() > 0) {
                    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                }
                ippl::Comm->barrier();

            } else {
                Kokkos::parallel_for(
                    "Write the solution into the LHS on physical grid",
                    rho.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        const int ig2 = i + ldom2[0].first() - nghost2;
                        const int jg2 = j + ldom2[1].first() - nghost2;
                        const int kg2 = k + ldom2[2].first() - nghost2;

                        const int ig = i + ldom1[0].first() - nghost1;
                        const int jg = j + ldom1[1].first() - nghost1;
                        const int kg = k + ldom1[2].first() - nghost1;

                        // take [0,N-1] as physical solution
                        const bool isQuadrant1 = ((ig == ig2) && (jg == jg2) && (kg == kg2));
                        view1(i, j, k)         = view2(i, j, k) * isQuadrant1;
                    });
            }
            IpplTimings::stopTimer(dtos);

            Kokkos::fence();
            ippl::Comm->barrier();
        }

        // compute difference
        rho = rho - 2.0;
        double err = norm(rho);
        msg << "Error = " << err << endl;

        // stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
