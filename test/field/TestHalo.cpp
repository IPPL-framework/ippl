// Tests the halo cell update functions to verify that the
// correct data is copied to halo cells of neighboring MPI ranks
#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("TestHalo");

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
        IpplTimings::startTimer(mainTimer);

        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        //     std::array<int, dim> pt = {8, 7, 13};
        std::array<int, dim> pt = {4, 4, 4};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        typedef ippl::FieldLayout<dim> Layout_t;
        Layout_t layout(MPI_COMM_WORLD, owned, isParallel);

        std::array<double, dim> dx = {
            1.0 / double(pt[0]),
            1.0 / double(pt[1]),
            1.0 / double(pt[2]),
        };
        ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;

        field_type field(mesh, layout);

        field      = ippl::Comm->rank();
        int myRank = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == ippl::Comm->rank()) {
                const auto& neighbors = layout.getNeighbors();
                for (unsigned i = 0; i < neighbors.size(); i++) {
                    const auto& n = neighbors[i];
                    if (n.size() > 0) {
                        unsigned dim = 0;
                        for (unsigned idx = i; idx > 0; idx /= 3) {
                            dim += idx % 3 == 2;
                        }
                        std::cout << "My Rank: " << myRank;
                        switch (dim) {
                            case 0:
                                std::cout << " vertex: ";
                                break;
                            case 1:
                                std::cout << " edge: ";
                                break;
                            case 2:
                                std::cout << " face: ";
                                break;
                        }
                        std::cout << i << " neighbors: ";
                        for (const auto& nrank : n) {
                            std::cout << nrank << ' ';
                        }
                        std::cout << std::endl;
                    }
                }
            }
            ippl::Comm->barrier();
        }

        auto& domains = layout.getHostLocalDomains();

        for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
            if (rank == ippl::Comm->rank()) {
                auto& neighbors = layout.getNeighbors();

                int nFaces = 0, nEdges = 0, nVertices = 0;
                for (unsigned i = 0; i < neighbors.size(); i++) {
                    if (neighbors[i].size() > 0) {
                        unsigned dim = 0;
                        for (unsigned idx = i; idx > 0; idx /= 3) {
                            dim += idx % 3 == 2;
                        }
                        switch (dim) {
                            case 0:
                                nVertices++;
                                break;
                            case 1:
                                nEdges++;
                                break;
                            case 2:
                                nFaces++;
                                break;
                        }
                    }
                }

                std::cout << "rank " << rank << ": " << std::endl
                          << " - domain:   " << domains[rank] << std::endl
                          << " - faces:    " << nFaces << std::endl
                          << " - edges:    " << nEdges << std::endl
                          << " - vertices: " << nVertices << std::endl
                          << "--------------------------------------" << std::endl;
            }
            ippl::Comm->barrier();
        }

        int nsteps = 300;

        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        for (int nt = 0; nt < nsteps; ++nt) {
            field.accumulateHalo();
            ippl::Comm->barrier();
            field.fillHalo();
            ippl::Comm->barrier();
            msg << "Update: " << nt + 1 << endl;
        }
        IpplTimings::stopTimer(fillHaloTimer);

        for (int rank = 0; rank < nRanks; ++rank) {
            if (rank == ippl::Comm->rank()) {
                std::string fname = "field_nRanks_" + std::to_string(nRanks) + "_rank_"
                                    + std::to_string(rank) + ".dat";
                Inform out("Output", fname.c_str(), Inform::OVERWRITE, rank);
                field.write(out);
            }
            ippl::Comm->barrier();
        }

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
