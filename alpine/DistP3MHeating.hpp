#ifndef IPPL_P3M_BENCH_MANAGER_HPP
#define IPPL_P3M_BENCH_MANAGER_HPP

// includes
#include <memory>
#include <string>
#include <iostream>
#include <Kokkos_Random.hpp>
#include <Kokkos_ScatterView.hpp>

// Alpine Headers
// #include "../alpine/LoadBalancer.hpp"
#include "FieldContainer.hpp"

// P3M Headers
#include "Manager/P3M3DManager.h"
#include "PoissonSolvers/P3MSolver.h"
#include "../src/P3M/P3MParticleContainer.hpp"

// Distribution functions
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/UniformDistribution.h"

// Required Datatypes
template <typename T, unsigned Dim>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::P3MSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T, unsigned Dim, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, 3>, 1>::view_type;

// Kokkos Device and Host Spaces
using Device = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;

// physical constants
// const double ke = 2.532638e8;

/**
 * @class P3M3DBenchManager
 * @brief A class that benchmarks the P3M Method
 * 
 * @tparam T the data dype for simulation variables
 * @tparam Dim the dimensionality of the simulation
*/
template <typename T, unsigned Dim>
class P3M3DBenchManager 
    : public ippl::P3M3DManager<T, Dim, FieldContainer<T, Dim>> {
public:

    using ParticleContainer_t = P3MParticleContainer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using FieldContainer_t = FieldContainer<T, Dim>;

protected:

    size_type totalP_m;         // Total number of particles
    int nt_m;                   // Total number of time steps
    double dt_m;                // Time step size
    Vector_t<int, Dim> nr_m;    // Domain granularity
    double rcut_m;              // Interaction cutoff radius
    std::string solver_m;       // solver is P3MSolver
    double beamRad_m;           // beam radius
    double focusingF_m;         // constant focusing force
    double boxlen_m;            // box length
    
public:
    P3M3DBenchManager(size_type totalP_, int nt_, double dt_, Vector_t<int, Dim>& nr_, double rcut_, double alpha_, double beamRad_, double focusingF_, double boxlen_) 
        : ippl::P3M3DManager<T, Dim, FieldContainer<T, Dim> >() 
        , totalP_m(totalP_), nt_m(nt_), dt_m(dt_), nr_m(nr_), rcut_m(rcut_), alpha_m(alpha_), solver_m("P3M"), beamRad_m(beamRad_), focusingF_m(focusingF_), boxlen_m(boxlen_)
        {
            this->preallocatedSendBuffer_m = 1000;
            this->sendBuffer_m = new T[preallocatedSendBuffer_m];
            this->preallocatedRecvBuffer_m = 1000;
            this->recvBuffer_m = new T[preallocatedRecvBuffer_m];
        }

    ~P3M3DBenchManager(){}

protected:
    double time_m;                  // Simulation time
    double it_m;                    // Iteration counter
    double alpha_m;                 // Green's function splitting parameter
    double epsilon_m;               // Regularization for PP interaction
    Vector_t<double, Dim> rmin_m;   // minimum domain extend
    Vector_t<double, Dim> rmax_m;   // maximum domain extend
    Vector_t<double, Dim> hr_m;     // PM Meshwidth
    Vector_t<int, Dim> nCells_m;    // Number of cells in each dimension
    double Q_m;                     // Particle Charge
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;    // Domain as index range
    std::array<bool, Dim> decomp_m; // Domain Decomposition
    double rhoNorm_m;               // Rho norm, required for scatterCIC
    MPI_Comm graph_comm_m;          // MPI Graph communicator
    unsigned preallocatedSendBuffer_m;  // Preallocated buffer size
    T* sendBuffer_m;                // Send buffer
    unsigned preallocatedRecvBuffer_m;  // Preallocated buffer size
    T* recvBuffer_m;                // Recieve buffer
    Kokkos::View<ippl::Vector<double, 3> *, Host> haloE_m;

public: 
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    void setupMPI() {
        auto neighbors = this->fcontainer_m->getFL().getNeighbors();
        int sources[26];
        int it = 0;

        unsigned nx = nCells_m[0];
        unsigned ny = nCells_m[1];
        unsigned nz = nCells_m[2];

        for (const auto& componentNeighbors : neighbors) {
            for(const auto& neighbor : componentNeighbors){
                // std::cerr << "Rank " << ippl::Comm->rank() << " has neighbor " << neighbor << std::endl;
                sources[it++] = neighbor;
            }
        }

        assert(it == 26 && "Invalid mesh topology");

        int weights[26] = {1, 1, nx, 1, 1, nx, ny, ny, ny*nx, 1, 1, nx, 1, 1, nx, ny, ny, ny*nx, nz, nz, nx*nz, nz, nz, nx*nz, ny*nz, ny*nz};

        MPI_Comm dist_graph_comm;
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 26, sources, weights, 26, sources, weights, MPI_INFO_NULL, 0, &dist_graph_comm);
        this->graph_comm_m = dist_graph_comm;

        std::cerr << "MPI Graph Communicator Setup Done" << std::endl;

    }

    void particleExchange() {
        Inform m("Setup MPI");

        // get communicator size and rank
        int commSize = ippl::Comm->size();
        int rank = ippl::Comm->rank();

        const double ke = 2.532638e8;
        // get domain decomposition
        auto hLocalRegions = this->pcontainer_m->getLayout().getRegionLayout().gethLocalRegions();
        auto neighbors = this->fcontainer_m->getFL().getNeighbors();
        auto cellStartingIdx = Kokkos::create_mirror_view(this->pcontainer_m->getNL());
        // auto cellStartingIdx = this->pcontainer_m->getNL();
        const unsigned nx = nCells_m[0];
        const unsigned ny = nCells_m[1];
        const unsigned nz = nCells_m[2];
        const unsigned nzm1 = nz - 1;
        const double rcut = hLocalRegions(rank)[0].length() / (double)nx;

        // particle exchange facilitated in 3 steps
        // 1. Compute number of particles to be sent to each neighbor
        // 2. Build send buffer
        // 3. Send and Recieve Particles

        // 1. Compute number of particles to be sent to each neighbor
        // This is split up into 4 parts
        // I.   We compute the number of particles to be sent to each corner
        // II.  We compute the number of particles to be sent to each edge in z direction and faces in y-z plane
        // III. We compute the number of particles to be sent to each edge in x direction and faces in x-z plane
        // IV.  We compute the number of particles to be sent to each edge in y direction and faces in x-y plane
        // We do this to make use of the fact that particles are stored in an ordered manner in the particle container

        // I. Compute the number of particles to be sent to each corner
        unsigned neighbor00_Idx, neighbor01_Idx, neighbor03_Idx, neighbor04_Idx, neighbor09_Idx, neighbor10_Idx, neighbor12_Idx, neighbor13_Idx;
        unsigned nParticles00, nParticles01, nParticles03, nParticles04, nParticles09, nParticles10, nParticles12, nParticles13;

        // all 8 corners only require a single gridCell in the PP grid
        neighbor00_Idx = 0; nParticles00 = cellStartingIdx(1);
        neighbor01_Idx = (nx - 1) * ny * nz; nParticles01 = cellStartingIdx(neighbor01_Idx + 1) - cellStartingIdx(neighbor01_Idx);
        neighbor03_Idx = (ny - 1) * nz; nParticles03 = cellStartingIdx(neighbor03_Idx + 1) - cellStartingIdx(neighbor03_Idx);
        neighbor04_Idx = neighbor01_Idx + neighbor03_Idx; nParticles04 = cellStartingIdx(neighbor04_Idx + 1) - cellStartingIdx(neighbor04_Idx);
        neighbor09_Idx = neighbor00_Idx + nzm1; nParticles09 = cellStartingIdx(neighbor09_Idx + 1) - cellStartingIdx(neighbor09_Idx);
        neighbor10_Idx = neighbor01_Idx + nzm1; nParticles10 = cellStartingIdx(neighbor10_Idx + 1) - cellStartingIdx(neighbor10_Idx);
        neighbor12_Idx = neighbor03_Idx + nzm1; nParticles12 = cellStartingIdx(neighbor12_Idx + 1) - cellStartingIdx(neighbor12_Idx);
        neighbor13_Idx = neighbor04_Idx + nzm1; nParticles13 = cellStartingIdx(neighbor13_Idx + 1) - cellStartingIdx(neighbor13_Idx);

        // required to build buffer
        unsigned cornerIdx[8] = {neighbor00_Idx, neighbor01_Idx, neighbor03_Idx, neighbor04_Idx, neighbor09_Idx, neighbor10_Idx, neighbor12_Idx, neighbor13_Idx};
        unsigned cornerCounts[8] = {nParticles00, nParticles01, nParticles03, nParticles04, nParticles09, nParticles10, nParticles12, nParticles13};
        unsigned cornerIdentifiers[8] = {0, 1, 3, 4, 9, 10, 12, 13};

        // std::cerr << "Checkpoint 1" << std::endl;

        // II. Compute the number of particles to be sent to each edge in z direction and faces in y-z plane
        unsigned nParticles18, nParticles19, nParticles21, nParticles22, nParticles24, nParticles25;

        // cellStartingIdx is consecutive in z direction, thus simplifying the computation for 4 of the edges
        nParticles18 = cellStartingIdx(neighbor09_Idx + 1); // cellStartingIdx(0) = 0
        nParticles19 = cellStartingIdx(neighbor10_Idx + 1) - cellStartingIdx(neighbor01_Idx);
        nParticles21 = cellStartingIdx(neighbor12_Idx + 1) - cellStartingIdx(neighbor03_Idx);
        nParticles22 = cellStartingIdx(neighbor13_Idx + 1) - cellStartingIdx(neighbor04_Idx);

        // 2 of the faces are also straightforward
        nParticles24 = cellStartingIdx(neighbor12_Idx + 1);
        nParticles25 = cellStartingIdx(neighbor13_Idx + 1) - cellStartingIdx(neighbor01_Idx);

        // required to build buffer
        unsigned zTopologyIdx[6] = {0, neighbor01_Idx, neighbor03_Idx, neighbor04_Idx, 0, neighbor01_Idx};
        unsigned zTopologyCounts[6] = {nParticles18, nParticles19, nParticles21, nParticles22, nParticles24, nParticles25};
        unsigned zTopologyIdentifiers[6] = {18, 19, 21, 22, 24, 25};

        // std::cerr << "Checkpoint 2" << std::endl;

        // III. Compute the number of particles to be sent to each edge in x direction and faces in x-z plane
        unsigned nParticles02 = 0, nParticles11 = 0, nParticles05 = 0, nParticles14 = 0, nParticles20 = 0, nParticles23 = 0;

        // replace with Kokkos parallel_for or reduce : TODO
        for(int x_Idx = 0; x_Idx < nx; ++x_Idx){
            // edges in x direction (xx2, x <2)
            nParticles02 += cellStartingIdx(x_Idx * ny * nz + 1) - cellStartingIdx(x_Idx * ny * nz);
            nParticles11 += cellStartingIdx(x_Idx * ny * nz + nz) - cellStartingIdx(x_Idx * ny * nz + nzm1);
            nParticles05 += cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + 1) - cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz);
            nParticles14 += cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nz) - cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nzm1);

            // faces in x-z plane
            nParticles20 += cellStartingIdx(x_Idx * ny * nz + nz) - cellStartingIdx(x_Idx * ny * nz);
            nParticles23 += cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nz) - cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz);
        }

        // std::cerr << "Checkpoint 3" << std::endl;

        // IV. Compute the number of particles to be sent to each edge in y direction and faces in x-y plane
        unsigned nParticles06 = 0, nParticles15 = 0, nParticles07 = 0, nParticles16 = 0, nParticles08 = 0, nParticles17 = 0;

        // rest of the topology requires a bit more work
        for(int y_Idx = 0; y_Idx < ny; ++y_Idx){
            // edges in y direction (x2x, x < 2)
            nParticles06 += cellStartingIdx(y_Idx * nz + 1) - cellStartingIdx(y_Idx * nz);
            nParticles15 += cellStartingIdx(y_Idx * nz + nz) - cellStartingIdx(y_Idx * nz + nzm1);
            nParticles07 += cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + 1) - cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz);
            nParticles16 += cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + nz) - cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + nzm1);
            
            for(int x_Idx = 0; x_Idx < nx; ++x_Idx){
                // faces in x-y plane
                nParticles08 += cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + 1) - cellStartingIdx(x_Idx * ny * nz + y_Idx * nz);
                nParticles17 += cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + nz) - cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + nzm1);
            }
        }

        // std::cerr << "Checkpoint 4" << std::endl;

        // required to facilitate exchange
        unsigned nTotal = nParticles00 + nParticles01 + nParticles02 + nParticles03 + nParticles04 + nParticles05 + nParticles06 + nParticles07 + nParticles08 + nParticles09 + nParticles10 + nParticles11 + nParticles12 + nParticles13 
                        + nParticles14 + nParticles15 + nParticles16 + nParticles17 + nParticles18 + nParticles19 + nParticles20 
                        + nParticles21 + nParticles22 + nParticles23 + nParticles24 + nParticles25;

        int sendCounts[26] = {4*nParticles00, 4*nParticles01, 4*nParticles02, 4*nParticles03, 4*nParticles04, 4*nParticles05, 4*nParticles06, 
                                4*nParticles07, 4*nParticles08, 4*nParticles09, 4*nParticles10, 4*nParticles11, 4*nParticles12, 4*nParticles13, 
                                4*nParticles14, 4*nParticles15, 4*nParticles16, 4*nParticles17, 4*nParticles18, 4*nParticles19, 4*nParticles20, 
                                4*nParticles21, 4*nParticles22, 4*nParticles23, 4*nParticles24, 4*nParticles25};

        // DEBUG OUTPUT
        // for(int i = 0; i < 26; ++i){
        //     std::cerr << "sendCounts[" << i << "]: " << sendCounts[i] << std::endl;
        // }
        
        // if nTotal larger than preallocated buffer, reallocate
        if(nTotal*4 > this->preallocatedSendBuffer_m){
            // reallocate buffer
            this->preallocatedSendBuffer_m = 4*nTotal + 1000; // overallocate
            delete[] this->sendBuffer_m;
            this->sendBuffer_m = new T[preallocatedSendBuffer_m];
        }

        // DEBUG OUTPUT
        // std::cout << "nTotal: " << nTotal << std::endl;

        // compute displacements
        int displacements[26];
        displacements[0] = 0;
        for (int i = 1; i < 26; ++i) {
            displacements[i] = displacements[i-1] + sendCounts[i-1];
        }


        // BUILD SEND BUFFER
        // 1. Corners
        // 2. 4 Edges in z direction, 2 faces in y-z plane
        // 3. 4 Edges in x direction, 2 faces in x-z plane
        // 4. 4 Edges in y direction, 2 faces in x-y plane
        // SEND BUFFER LAYOUT: [x, y, z, Q]        

        // required particle data
        auto R_host = Kokkos::create_mirror_view(this->pcontainer_m->R.getView());
        auto Q_host = Kokkos::create_mirror_view(this->pcontainer_m->Q.getView());

        // 1. Corners
        for(int i = 0; i < 8; ++i){
            unsigned cornerIndex = cornerIdx[i];
            unsigned cornerCount = cornerCounts[i];
            unsigned cornerStart = cellStartingIdx(cornerIndex);
            for(unsigned j = 0; j < cornerCount; ++j){
                sendBuffer_m[displacements[cornerIdentifiers[i]] + 4*j + 0] = R_host(cornerStart + j)[0];
                sendBuffer_m[displacements[cornerIdentifiers[i]] + 4*j + 1] = R_host(cornerStart + j)[1];
                sendBuffer_m[displacements[cornerIdentifiers[i]] + 4*j + 2] = R_host(cornerStart + j)[2];
                sendBuffer_m[displacements[cornerIdentifiers[i]] + 4*j + 3] = Q_host(cornerStart + j);
            }
        }

        // std::cerr << "Checkpoint 5" << std::endl;

        // 2. 4 Edges in z direction, 2 faces in y-z plane
        for(int i = 0; i < 6; ++i){
            unsigned zIdx = zTopologyIdx[i];
            unsigned zCount = zTopologyCounts[i];
            // std::cerr << "zIdx: " << zIdx << " zCount: " << zCount << std::endl;
            unsigned zStart = cellStartingIdx(zIdx);
            for(unsigned j = 0; j < zCount; ++j){
                for(int d = 0; d < Dim; ++d){
                    sendBuffer_m[displacements[zTopologyIdentifiers[i]] + 4*j + d] = R_host(zStart + j)[d];
                }
                sendBuffer_m[displacements[zTopologyIdentifiers[i]] + 4*j + 3] = Q_host(zStart + j);
            }
        }

        // std::cerr << "Checkpoint 6" << std::endl;

        // 3. 4 Edges in x direction, 2 faces in x-z plane
        // Kokkos::parallel_for("Build Send Buffer", Kokkos::RangePolicy<Host>(0, nx),
        //     KOKKOS_LAMBDA(const int& x_Idx){
        
        for(int x_Idx = 0; x_Idx < nx; ++x_Idx){
            // begin with the edges in x direction
            unsigned lowerEdgeStarts[4] = {cellStartingIdx(x_Idx * ny * nz), cellStartingIdx(x_Idx * ny * nz + nzm1), 
                                            cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz), cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nzm1)};
            unsigned lowerEdgeEnds[4] = {cellStartingIdx(x_Idx * ny * nz + 1), cellStartingIdx(x_Idx * ny * nz + nz),
                                            cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + 1), cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nz)};
            unsigned lowerEdgeIdentifiers[4] = {2, 11, 5, 14};

            for(int i = 0; i < 4; ++i){
                for(unsigned j = lowerEdgeStarts[i]; j < lowerEdgeEnds[i]; ++j){
                    for(int d = 0; d < Dim; ++d){
                        sendBuffer_m[displacements[lowerEdgeIdentifiers[i]] + 4*(j - lowerEdgeStarts[i]) + d] = R_host(j)[d];
                    }
                    sendBuffer_m[displacements[lowerEdgeIdentifiers[i]] + 4*(j - lowerEdgeStarts[i]) + 3] = Q_host(j);
                }
                displacements[lowerEdgeIdentifiers[i]] += 4*(lowerEdgeEnds[i] - lowerEdgeStarts[i]);
            }

            // lower face in x-z plane
            unsigned lowerFaceStart = cellStartingIdx(x_Idx * ny * nz);
            unsigned lowerFaceEnd = cellStartingIdx(x_Idx * ny * nz + nz);
            for(unsigned i = lowerFaceStart; i < lowerFaceEnd; ++i) {
                for(int d = 0; d < Dim; ++d){
                    sendBuffer_m[displacements[20] + 4*(i - lowerFaceStart) + d] = R_host(i)[d];
                }
                sendBuffer_m[displacements[20] + 4*(i - lowerFaceStart) + 3] = Q_host(i);
            }
            displacements[20] += 4*(lowerFaceEnd - lowerFaceStart);

            // upper face in x-z plane
            unsigned upperFaceStart = cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz);
            unsigned upperFaceEnd = cellStartingIdx(x_Idx * ny * nz + (ny-1) * nz + nz);
            for(unsigned i = upperFaceStart; i < upperFaceEnd; ++i) {
                for(int d = 0; d < Dim; ++d){
                    sendBuffer_m[displacements[23] + 4*(i - upperFaceStart) + d] = R_host(i)[d];
                }
                sendBuffer_m[displacements[23] + 4*(i - upperFaceStart) + 3] = Q_host(i);
            }
            displacements[23] += 4*(upperFaceEnd - upperFaceStart);
        }

        // std::cerr << "Checkpoint 7" << std::endl;
    
        // );
        

        // 4. 4 Edges in y direction, 2 faces in x-y plane
        // Kokkos::parallel_for("Build Send Buffer", Kokkos::RangePolicy<Host>(0, ny),
        //     KOKKOS_LAMBDA(const int& y_Idx){
        for(int y_Idx = 0; y_Idx < ny; ++y_Idx){
            // edges in y direction
            unsigned lowerEdgeStarts[4] = {cellStartingIdx(y_Idx * nz), cellStartingIdx(y_Idx * nz + nzm1), 
                                            cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz), cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + nzm1)};
            unsigned lowerEdgeEnds[4] = {cellStartingIdx(y_Idx * nz + 1), cellStartingIdx(y_Idx * nz + nz),
                                            cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + 1), cellStartingIdx(y_Idx * nz + (nx-1) * ny * nz + nz)};
            unsigned lowerEdgeIdentifiers[4] = {6, 15, 7, 16};

            for(int i = 0; i < 4; ++i){
                for(unsigned j = lowerEdgeStarts[i]; j < lowerEdgeEnds[i]; ++j){
                    for(int d = 0; d < Dim; ++d){
                        sendBuffer_m[displacements[lowerEdgeIdentifiers[i]] + 4*(j - lowerEdgeStarts[i]) + d] = R_host(j)[d];
                    }
                    sendBuffer_m[displacements[lowerEdgeIdentifiers[i]] + 4*(j - lowerEdgeStarts[i]) + 3] = Q_host(j);
                }
                displacements[lowerEdgeIdentifiers[i]] += 4*(lowerEdgeEnds[i] - lowerEdgeStarts[i]);
            }
            
            for(int x_Idx = 0; x_Idx < nx; ++x_Idx){
                // faces in x-y plane
                unsigned lowerFaceStarts[2] = {cellStartingIdx(x_Idx * ny * nz + y_Idx * nz), cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + nzm1)};
                unsigned lowerFaceEnds[2] = {cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + 1), cellStartingIdx(x_Idx * ny * nz + y_Idx * nz + nz)};
                unsigned lowerFaceIdentifiers[2] = {8, 17};

                for(int i = 0; i < 2; ++i){
                    for(int j = lowerFaceStarts[i]; j < lowerFaceEnds[i]; ++j){
                        for(int d = 0; d < Dim; ++d){
                            sendBuffer_m[displacements[lowerFaceIdentifiers[i]] + 4*(j - lowerFaceStarts[i]) + d] = R_host(j)[d];
                        }
                        sendBuffer_m[displacements[lowerFaceIdentifiers[i]] + 4*(j - lowerFaceStarts[i]) + 3] = Q_host(j);
                    }
                    displacements[lowerFaceIdentifiers[i]] += 4*(lowerFaceEnds[i] - lowerFaceStarts[i]);
                }
            }
        }

        // Kokkos::fence();

        // std::cerr << "Checkpoint 8" << std::endl;
        // return;
        // );

        // recompute displacements, changed during building buffer
        displacements[0] = 0;
        for (int i = 1; i < 26; ++i) {
            displacements[i] = displacements[i-1] + sendCounts[i-1];
        }
        

        // Send and Recieve Particles
        // 1.   MPI_Neighbor_alltoall for size information
        // 2.   Compute displacements
        // (3.) MPI_Neighbor_alltoallw when using type indexed
        // (3.) MPI_Neighbor_alltoallv when using buffers

        // 1. MPI_Neighbor_alltoall for size information
        int recvCounts[26];

        double commStart = MPI_Wtime();
        MPI_Neighbor_alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, graph_comm_m);
        double commEnd = MPI_Wtime();
        // std::cerr << "Comm Time: " << commEnd - commStart << std::endl;

        // ippl::Comm->barrier();

        // required buffer size
        unsigned nTotalRecv = 0;

        // 2. Compute displacements
        int recvDisplacements[26];
        recvDisplacements[0] = 0;
        for (int i = 1; i < 26; ++i) {
            nTotalRecv += recvCounts[i-1] / 4;
            recvDisplacements[i] = recvDisplacements[i-1] + recvCounts[i-1];
        }
        nTotalRecv += recvCounts[25] / 4;


        // if nTotalRecv larger than preallocated buffer, reallocate
        if(nTotalRecv*4 > preallocatedRecvBuffer_m){
            // reallocate buffer
            preallocatedRecvBuffer_m = 4*nTotalRecv + 1000; // overallocate
            delete[] recvBuffer_m;
            recvBuffer_m = new T[preallocatedRecvBuffer_m];
        }

        // 3. MPI_Neighbor_alltoallv to facilitate particle exchange
        double commStart2 = MPI_Wtime();
        MPI_Neighbor_alltoallv(sendBuffer_m, sendCounts, displacements, MPI_DOUBLE, recvBuffer_m, recvCounts, recvDisplacements, MPI_DOUBLE, graph_comm_m);
        double commEnd2 = MPI_Wtime();
        // std::cerr << "Comm Time 2: " << commEnd2 - commStart2 << std::endl;
        // ippl::Comm->barrier();

        // Compute Interactions - 4 Steps
        // I.   Compute interactions on the corners
        // II.  Compute interactions on the edges in z direction and faces in y-z plane
        // III. Compute interactions on the edges in x direction and faces in x-z plane
        // IV.  Compute interactions on the edges in y direction and faces in x-y plane

        // needed for all steps
        unsigned nCells = nCells_m[0] * nCells_m[1] * nCells_m[2];
        unsigned nLoc = cellStartingIdx(nCells);
        Kokkos::View<ippl::Vector<double, 3> *, Host> F_sr("PP-Halo Force", nLoc);
        double alpha = alpha_m;
        
	    // I. Compute interactions on the corners
        using team_t = Kokkos::TeamPolicy<Host>::member_type;
        Kokkos::parallel_for("PP on Corners", Kokkos::TeamPolicy<Host>(8, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();

                const int displacement = recvDisplacements[cornerIdentifiers[7-i]];
                const int haloCornerCount = recvCounts[cornerIdentifiers[7-i]] / 4;

                const int internalCornerIndex = cornerIdx[i];       // starting index in particle view
                const int internalCornerCount = cornerCounts[i];    // number of particles in corner (local)
                const int internalCornerDisplacement = cellStartingIdx(internalCornerIndex); // starting index in particle view

                // if a buffer has no particles, we have nothing to calculate
		        if (haloCornerCount == 0 || internalCornerCount == 0) return;
            
                auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, internalCornerCount, haloCornerCount);
                Kokkos::parallel_for(p, [&](const int& ii, const int& jj){
                    const int internalIdx = internalCornerDisplacement + ii;
                    const int haloIdx = displacement + 4*jj;

                    // compute distance
                    double rsq_ij = 0;
                    Vector_t<T, Dim> dist_ij;
                    for(int d = 0; d < Dim; ++d){
                        dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                        rsq_ij += dist_ij[d] * dist_ij[d];
                    }		    
                    double r_ij = Kokkos::sqrt(rsq_ij);

                    // if within cutoff
                    if(r_ij >= rcut_m) return;
                    Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                    Kokkos::atomic_sub(&F_sr(internalIdx), F_ij * recvBuffer_m[haloIdx + 3]);
                }); 
            }
        );

        // II. Compute interactions on the edges in z direction and faces in y-z plane
        // a. generate mini neighborlist
        // b. compute interactions
	
    	// std::cerr << "Checkpoint 9" << std::endl;       
        
        // int zEdgeNeighborList[4][nz+1];
        // interaction on edges in z direction
        Kokkos::parallel_for("PP interaction for z edges", Kokkos::TeamPolicy<Host>(4, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();
        
                const int displacement = recvDisplacements[zTopologyIdentifiers[3-i]];
                const int haloEdgeCount = recvCounts[zTopologyIdentifiers[3-i]] / 4;
                int localEdgeCount = sendCounts[zTopologyIdentifiers[i]] / 4;
		        if(haloEdgeCount == 0 || localEdgeCount == 0) return;

                const double edgeStart = hLocalRegions(rank)[2].min();
                Kokkos::View<unsigned*, Host> zEdgeNeighborList("NL for Halo interaction on z edges", nz+1);

                // build neighborlist for the buffer we received
                zEdgeNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloEdgeCount; ++jj){
        		    double zPosition = recvBuffer_m[displacement + jj * 4 + 2];
        		    int newCell = floor((zPosition - edgeStart)/ rcut);
                    if (newCell > currentCell){
                        ++currentCell;
                        while (currentCell < newCell){
                            zEdgeNeighborList(currentCell) = jj; // first particle outside cell
                            ++currentCell;                       // search for next cell
                        }
                        zEdgeNeighborList(currentCell) = jj;
                    }
                }
            	while (currentCell < nz){
                    currentCell++;   
                	zEdgeNeighborList(currentCell) = haloEdgeCount;
        		}

                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nz),
                    [&](const int& zCellNumber){
                        int localCellIdx = zTopologyIdx[i] + zCellNumber;

                        // starting index and count of local particles
                        unsigned localStartingIdx = cellStartingIdx(localCellIdx);
                        unsigned localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        // declare starting index and count of halo particles
                        unsigned haloStartingIdx;

                        // handle first cell in Edge
                        if(zCellNumber == 0) {
                            haloStartingIdx = 0;
                        } else {
                            haloStartingIdx = zEdgeNeighborList(zCellNumber-1);
                        }

                        unsigned haloEndingIdx;

                        if(zCellNumber == nz-1) {
                            haloEndingIdx = zEdgeNeighborList(nz);
                        } else {
                            haloEndingIdx = zEdgeNeighborList(zCellNumber+2);
                        }

                        unsigned haloCount = haloEndingIdx - haloStartingIdx;
                        if(localCount == 0 || haloCount == 0) return;
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, localCount, haloCount);
                        Kokkos::parallel_for(p, [&](const int& ii, const int& jj){
                            const int internalIdx = localStartingIdx + ii;
                            const int haloIdx = displacement + 4 * (jj + haloStartingIdx);

                            double rsq_ij = 0;
                            Vector_t<T, Dim> dist_ij;
                            for(int d = 0; d < Dim; ++d){
                                dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                rsq_ij += dist_ij[d] * dist_ij[d];
                            }
                            double r_ij = Kokkos::sqrt(rsq_ij);
                            
                            if (r_ij >= rcut_m) return;
                            Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                            Kokkos::atomic_sub(&F_sr(internalIdx), F_ij * recvBuffer_m[haloIdx + 3]);
                        });
                    }
                );
            }
        );

    	// std::cerr << "Checkpoint 10" << std::endl;

        // Interaction on faces in y-z plane
        Kokkos::parallel_for("PP interaction for y-z faces", Kokkos::TeamPolicy<Host>(2, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
            
                // upper or lower face
                const int i = team.league_rank();

                // get displacement and recieve counts to index buffer
                const int displacement = recvDisplacements[zTopologyIdentifiers[5-i]];
                const int haloFaceCount = recvCounts[zTopologyIdentifiers[5-i]] / 4;
        		int localFaceCount = sendCounts[zTopologyIdentifiers[4+i]] / 4;

                if (haloFaceCount == 0 || localFaceCount == 0) return;

                const double faceStart[2] = {hLocalRegions(rank)[1].min(), hLocalRegions(rank)[2].min()};
                Kokkos::View<unsigned*, Host> yzFaceNeighborList("NL for Halo interaction on yz face", ny*nz+1);
                
                yzFaceNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloFaceCount; ++jj){
                    // if outside of cell
                    double yPosition = recvBuffer_m[displacement + jj * 4 + 1];
                    double zPosition = recvBuffer_m[displacement + jj * 4 + 2];

                    int newCell = floor((yPosition - faceStart[0]) / rcut) * nz + floor((zPosition - faceStart[1]) / rcut);
                    if (newCell > currentCell) {
                        currentCell++;
                        // fill intermediate cells if empty, currentCell == newCell after loop
                        while (currentCell < newCell){
                            yzFaceNeighborList(currentCell) = jj;
                            currentCell++;
                        }
                        yzFaceNeighborList(currentCell) = jj;
                    }
                        
                }
        		while(currentCell < ny*nz){
                    currentCell++;
        		    yzFaceNeighborList(currentCell) = haloFaceCount;
        		}
        
                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny*nz),
                    [&](const int& cellNumber){
                        int zCellNumber = cellNumber % nz;
                        int yCellNumber = cellNumber / nz;
                        int localCellIdx = zTopologyIdx[4+i] + cellNumber;

                        // starting index and count of local particles
                        unsigned localStartingIdx = cellStartingIdx(localCellIdx);
                        unsigned localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        int neighbors[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
                        if (localCount == 0) return;                        
                        
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, 9, localCount);
                        Kokkos::parallel_for(p, [&](const int& neighborIdx, const int& ii){

                            // calculate index of halo cell
                            int haloYCellIdx = yCellNumber + neighbors[neighborIdx][0];
                            int haloZCellIdx = zCellNumber + neighbors[neighborIdx][1];
                            if(haloYCellIdx < 0 || haloYCellIdx >= ny || haloZCellIdx < 0 || haloZCellIdx >= nz) return;

                            int haloCellIdx = haloYCellIdx * nz + haloZCellIdx;
                            int haloStartingIdx = yzFaceNeighborList(haloCellIdx);
                            int haloCount = yzFaceNeighborList(haloCellIdx+1) - haloStartingIdx;

                            Vector_t<T, Dim> tempFsr = 0.0;
                            const int internalIdx = localStartingIdx + ii;

                            
                            for(unsigned jj = 0; jj < haloCount; ++jj){
                                const size_type haloIdx = displacement + 4 * (jj + haloStartingIdx);

				                double rsq_ij = 0;
                                Vector_t<T, Dim> dist_ij;
                                for(int d = 0; d < Dim; ++d){
                                    dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                    rsq_ij += dist_ij[d] * dist_ij[d];
                                }

                                double r_ij = Kokkos::sqrt(rsq_ij);
				                if (r_ij >= rcut_m) continue;
	
                                tempFsr += recvBuffer_m[haloIdx + 3] *  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                                
                            }
                            Kokkos::atomic_sub(&F_sr(internalIdx), tempFsr);
                            
                        });
                    }
                );
            }
        );

        // std::cerr << "Checkpoint 11" << std::endl;

        // III. Compute interactions on the edges in x direction and faces in x-z plane
        unsigned xEdgeIdentifiers[4] = {2, 11, 5, 14};
        unsigned xEdgeStartingIndices[4] = {0, nzm1, (ny-1) * nz, (ny-1) * nz + nzm1};
        Kokkos::parallel_for("PP interaction for x edges", Kokkos::TeamPolicy<Host>(4, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();

                const int displacement = recvDisplacements[xEdgeIdentifiers[3-i]];
                const int haloEdgeCount = recvCounts[xEdgeIdentifiers[3-i]] / 4;
                const int localEdgeCount = sendCounts[xEdgeIdentifiers[i]] /4;

                if(localEdgeCount == 0 || haloEdgeCount == 0) return;

                // lower bound of domain in x direction
                const double edgeStart = hLocalRegions(rank)[0].min();

                Kokkos::View<unsigned*, Host> xEdgeNeighborList("NL for Halo edges in x direction", nx+1);

                xEdgeNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloEdgeCount; ++jj){
        		    double xPosition = recvBuffer_m[displacement + jj * 4 + 0];
        		    int newCell = floor((xPosition - edgeStart)/ rcut);
                    if (newCell > currentCell){
                        ++currentCell;
                        while (currentCell < newCell){
                            xEdgeNeighborList(currentCell) = jj; // first particle outside cell
                            ++currentCell;                       // search for next cell
                        }
                        xEdgeNeighborList(currentCell) = jj;
                    } /*else if (newCell < currentCell) {
                        std::cerr << "wrong particle order assumed or sent" << std::endl;
                    }*/
                }
            	while (currentCell < nx){
                    currentCell++;
                	xEdgeNeighborList(currentCell) = haloEdgeCount;
                }

                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx),
                    [&](const int& xCellNumber){
                        size_type localCellIdx = xEdgeStartingIndices[i] + xCellNumber * ny * nz;

                        // starting index and count of local particles
                        size_type localStartingIdx = cellStartingIdx(localCellIdx);
                        size_type localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        // declare starting index and count of halo particles
                        unsigned haloStartingIdx;

                        // handle first cell in Edge
                        if(xCellNumber == 0) {
                            haloStartingIdx = 0;
                        } else {
                            haloStartingIdx = xEdgeNeighborList(xCellNumber-1);
                        }

                        unsigned haloEndingIdx;

                        if(xCellNumber == nx-1) {
                            haloEndingIdx = xEdgeNeighborList(nx);
                        } else {
                            haloEndingIdx = xEdgeNeighborList(xCellNumber+2);
                        }

                        unsigned haloCount = haloEndingIdx - haloStartingIdx;
                        if (localCount == 0 || haloCount == 0) return;
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, localCount, haloCount);
                        Kokkos::parallel_for(p, [&](const int& ii, const int& jj){
                            const int internalIdx = localStartingIdx + ii;
                            const int haloIdx = displacement + 4 * (jj + haloStartingIdx);

                            double rsq_ij = 0;
                            Vector_t<T, Dim> dist_ij;
                            for(int d = 0; d < Dim; ++d){
                                dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                rsq_ij += dist_ij[d] * dist_ij[d];
                            }
                            double r_ij = Kokkos::sqrt(rsq_ij);
                            
			                if (r_ij >= rcut_m) return;
                            Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                            Kokkos::atomic_sub(&F_sr(internalIdx), F_ij * recvBuffer_m[haloIdx + 3]);
                        });
                    }
                );
            }
        );

        // std::cerr << "Checkpoint 13 " << std::endl;

        // Interaction on faces in x-z plane
        int xzTopologyIdentifiers[2] = {20, 23};
        int xzTopologyIndex[2] = {0, (ny-1) * nz};
        Kokkos::parallel_for("PP interaction for x-z faces", Kokkos::TeamPolicy<Host>(2, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();

                const int displacement = recvDisplacements[xzTopologyIdentifiers[1-i]];
                const int haloFaceCount = recvCounts[xzTopologyIdentifiers[1-i]] / 4;
                const int localFaceCount = sendCounts[xzTopologyIdentifiers[i]] / 4;

                if (haloFaceCount == 0 || localFaceCount == 0) return;

                const double faceStart[2] = {hLocalRegions(rank)[0].min(), hLocalRegions(rank)[2].min()};
                Kokkos::View<unsigned*, Host> xzFaceNeighborList("NL for Halo interaction on xz face", nx*nz+1);
                
                xzFaceNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloFaceCount; ++jj){
                    // if outside of cell
                    double xPosition = recvBuffer_m[displacement + jj * 4 + 0];
                    double zPosition = recvBuffer_m[displacement + jj * 4 + 2];

                    int newCell = floor((xPosition - faceStart[0]) / rcut) * nz + floor((zPosition - faceStart[1]) / rcut);
                    if (newCell > currentCell) {
                        currentCell++;
                        // fill intermediate cells if empty, currentCell == newCell after loop
                        while (currentCell < newCell){
                            xzFaceNeighborList(currentCell) = jj;
                            currentCell++;
                        }
                        xzFaceNeighborList(currentCell) = jj;
                    }
                        
                }
                while (currentCell < nx*nz){
                    ++currentCell;
                    xzFaceNeighborList(currentCell) = haloFaceCount;
                }

                
                
                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx*nz),
                    [&](const int& cellNumber){
                        int xCellNumber = cellNumber / nz;
                        int zCellNumber = cellNumber % nz;
                        int localCellIdx = xzTopologyIndex[i] + xCellNumber * ny * nz + zCellNumber;

                        // starting index and count of local particles
                        unsigned localStartingIdx = cellStartingIdx(localCellIdx);
                        unsigned localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        int neighbors[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
                        if (localCount == 0) return;                        
                        
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, 9, localCount);
                        Kokkos::parallel_for(p, [&](const int& neighborIdx, const int& ii){

                            // calculate index of halo cell
                            int haloXCellIdx = xCellNumber + neighbors[neighborIdx][0];
                            int haloZCellIdx = zCellNumber + neighbors[neighborIdx][1];
                            if(haloXCellIdx < 0 || haloXCellIdx >= nx || haloZCellIdx < 0 || haloZCellIdx >= nz) return;

                            int haloCellIdx = haloXCellIdx * nz + haloZCellIdx;
                            unsigned haloStartingIdx = xzFaceNeighborList(haloCellIdx);
                            unsigned haloCount = xzFaceNeighborList(haloCellIdx+1) - haloStartingIdx;
            			    if (haloCount == 0) return;


                            // DEBUG output
                            // if (rank == 0) std::cerr << haloCount << std::endl;

                            Vector_t<T, Dim> tempFsr = 0.0;
                            const int internalIdx = localStartingIdx + ii;

                            
                            for(unsigned jj = 0; jj < haloCount; ++jj){
                                const int haloIdx = displacement + 4 * (jj + haloStartingIdx);

                                double rsq_ij = 0;
                                Vector_t<T, Dim> dist_ij;
                                for(int d = 0; d < Dim; ++d){
                                    dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                    rsq_ij += dist_ij[d] * dist_ij[d];
                                }

                                double r_ij = Kokkos::sqrt(rsq_ij);
                				if (r_ij >= rcut_m) continue; 
                                tempFsr += recvBuffer_m[haloIdx + 3] *  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                            }
                            Kokkos::atomic_sub(&F_sr(internalIdx), tempFsr);
                            
                        });
                    }
                );
            }
        );

        // std::cerr << "Checkpoint 14 " << std::endl;

        // IV. Compute interactions on the edges in y direction and faces in x-y plane
        unsigned yEdgeIdentifiers[4] = {6, 15, 7, 16};
        unsigned yEdgeStartingIndices[4] = {0, nzm1, (nx-1) * ny * nz, (nx-1) * ny * nz + nzm1};
        Kokkos::parallel_for("PP interaction for y edges", Kokkos::TeamPolicy<Host>(4, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();

                const int displacement = recvDisplacements[yEdgeIdentifiers[3-i]];
                const int haloEdgeCount = recvCounts[yEdgeIdentifiers[3-i]] / 4;
                const int localEdgeCount = sendCounts[yEdgeIdentifiers[i]] /4;

                if(localEdgeCount == 0 || haloEdgeCount == 0) return;
                const double edgeStart = hLocalRegions(rank)[1].min();

                Kokkos::View<unsigned*, Host>  yEdgeNeighborList("NL for edges in y direction", ny+1);

                yEdgeNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloEdgeCount; ++jj){
        		    double yPosition = recvBuffer_m[displacement + jj * 4 + 1];
        		    int newCell = floor((yPosition - edgeStart)/ rcut);
                    if (newCell > currentCell){
                        ++currentCell;
                        while (currentCell < newCell){
                            yEdgeNeighborList(currentCell) = jj; // first particle outside cell
                            ++currentCell;                       // search for next cell
                        }
                        yEdgeNeighborList(currentCell) = jj;
                    } /*else if (newCell < currentCell) {
                        std::cerr << "wrong particle order assumed or sent" << std::endl;
                    }*/
                }
            	while (currentCell < ny){
                    currentCell++;
                	yEdgeNeighborList(currentCell) = haloEdgeCount;
                }

                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                    [&](const int& yCellNumber){
                        int localCellIdx = yEdgeStartingIndices[i] + yCellNumber * nz;

                        // starting index and count of local particles
                        unsigned localStartingIdx = cellStartingIdx(localCellIdx);
                        unsigned localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        // declare starting index and count of halo particles
                        unsigned haloStartingIdx;

                        // handle first cell in Edge
                        if(yCellNumber == 0) {
                            haloStartingIdx = 0;
                        } else {
                            haloStartingIdx = yEdgeNeighborList(yCellNumber-1);
                        }

                        unsigned haloEndingIdx;

                        if(yCellNumber == ny-1) {
                            haloEndingIdx = yEdgeNeighborList(ny);
                        } else {
                            haloEndingIdx = yEdgeNeighborList(yCellNumber+2);
                        }

                        unsigned haloCount = haloEndingIdx - haloStartingIdx;
                        if (haloCount == 0 || localCount == 0) return;
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, localCount, haloCount);
                        Kokkos::parallel_for(p, [&](const int& ii, const int& jj){
                            const int internalIdx = localStartingIdx + ii;
                            const int haloIdx = displacement + 4 * (jj + haloStartingIdx);

                            double rsq_ij = 0;
                            Vector_t<T, Dim> dist_ij;
                            for(int d = 0; d < Dim; ++d){
                                dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                rsq_ij += dist_ij[d] * dist_ij[d];
                            }

                            double r_ij = Kokkos::sqrt(rsq_ij);
                            if (r_ij >= rcut_m) return;
                            Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                            Kokkos::atomic_sub(&F_sr(internalIdx), F_ij * recvBuffer_m[haloIdx + 3]);
                        });
                    }
                );
            }
        );

        // std::cerr << "Checkpoint 15 " << std::endl;

        // interaction on faces in xy plane
        int xyTopologyIdentifiers[2] = {8, 17};
        int xyTopologyIndex[2] = {0, nzm1};
        Kokkos::parallel_for("PP interaction for x-y faces", Kokkos::TeamPolicy<Host>(2, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_t& team){
                const int i = team.league_rank();

                const int displacement = recvDisplacements[xyTopologyIdentifiers[1-i]];
                const int haloFaceCount = recvCounts[xyTopologyIdentifiers[1-i]] / 4;

                if (haloFaceCount == 0) return;

                const double faceStart[2] = {hLocalRegions(rank)[0].min(), hLocalRegions(rank)[1].min()};
                Kokkos::View<unsigned*, Host> xyFaceNeighborList("NL for Halo interaction on xy face", nx*ny+1);
                
                xyFaceNeighborList(0) = 0;
                int currentCell = 0;
                for(int jj = 0; jj < haloFaceCount; ++jj){
                    // if outside of cell
                    double xPosition = recvBuffer_m[displacement + jj * 4 + 0];
                    double yPosition = recvBuffer_m[displacement + jj * 4 + 1];

                    int newCell = floor((xPosition - faceStart[0]) / rcut) + floor((yPosition - faceStart[1]) / rcut) * nx;
                    if (newCell > currentCell) {
                        currentCell++;
                        // fill intermediate cells if empty, currentCell == newCell after loop
                        while (currentCell < newCell){
                            xyFaceNeighborList(currentCell) = jj;
                            currentCell++;
                        }
                        xyFaceNeighborList(currentCell) = jj;
                    }/* else if (newCell < currentCell) {
                        // std::cerr << "wrong particle order assumed or sent" << std::endl;
                    }*/
                        
                }
                while (currentCell < nx*ny){
                    currentCell++;
                    xyFaceNeighborList(currentCell) = haloFaceCount;
                }
                
                // particle particle interaction
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx*ny),
                    [&](const int& cellNumber){
                        int yCellNumber = cellNumber / nx;
                        int xCellNumber = cellNumber % nx;
                        int localCellIdx = xyTopologyIndex[i] + xCellNumber * ny * nz + yCellNumber * nz;

                        // starting index and count of local particles
                        unsigned localStartingIdx = cellStartingIdx(localCellIdx);
                        unsigned localCount = cellStartingIdx(localCellIdx+1) - localStartingIdx;

                        int neighbors[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
                        if (localCount == 0) return;                        
                        
                        auto p = Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, 9, localCount);
                        Kokkos::parallel_for(p, [&](const int& neighborIdx, const int& ii){

                            // calculate index of halo cell
                            int haloXCellIdx = xCellNumber + neighbors[neighborIdx][0];
                            int haloYCellIdx = yCellNumber + neighbors[neighborIdx][1];
                            if(haloXCellIdx < 0 || haloXCellIdx >= nx || haloYCellIdx < 0 || haloYCellIdx >= ny) return;

                            int haloCellIdx = haloXCellIdx + haloYCellIdx * nx;
                            unsigned haloStartingIdx = xyFaceNeighborList(haloCellIdx);
                            unsigned haloCount = xyFaceNeighborList(haloCellIdx+1) - haloStartingIdx;

            			    if (haloCount==0) return;
                            // DEBUG output
                            // if (rank == 0) std::cerr << haloCount << std::endl;

                            Vector_t<T, Dim> tempFsr = 0.0;
                            const int internalIdx = localStartingIdx + ii;

                            
                            for(unsigned jj = 0; jj < haloCount; ++jj){
                                const int haloIdx = displacement + 4 * (jj + haloStartingIdx);

                                double rsq_ij = 0;
                                Vector_t<T, Dim> dist_ij;
                                for(int d = 0; d < Dim; ++d){
                                    dist_ij[d] = R_host(internalIdx)[d] - recvBuffer_m[haloIdx + d];
                                    rsq_ij += dist_ij[d] * dist_ij[d];
                                }

                                double r_ij = Kokkos::sqrt(rsq_ij);
                                if (r_ij >= rcut_m) continue;

                                tempFsr += recvBuffer_m[haloIdx + 3] * ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                                
                            }
                            Kokkos::atomic_sub(&F_sr(internalIdx), tempFsr);
                            
                        });
                    }
                );
            }
        );
	Kokkos::fence();
        
	this->haloE_m = F_sr;
    }


    void pre_run() override {
        Inform m("Pre Run");

        for (unsigned i = 0; i < Dim; ++i) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        // this->alpha_m = 2./this->rcut_m;
        double box_length = this->boxlen_m;
        this->rmin_m = -box_length/2.;
        this->rmax_m = box_length/2.;
        this->origin_m = rmin_m;
        this->isAllPeriodic_m = true;

        this->hr_m = box_length/(double)(this->nr_m[0]);

        std::cerr << "hr: " << this->hr_m << std::endl;
        
        // initialize time stuff
        this->it_m = 0;
        this->time_m = 0.;

        // initialize field container
        this->setFieldContainer(
            std::make_shared<FieldContainer_t>(
                this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, 
                this->domain_m, this->origin_m, this->isAllPeriodic_m
            )
        );

        // Set Particle Container (to P3MParticleContainer)
        this->setParticleContainer(
            std::make_shared<ParticleContainer_t>(
                this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()
            )
        );

	    std::cerr << "Device Space: " << Device::name() << std::endl;
	    std::cerr << "Host Space: " << Host::name() << std::endl;

    
        this->fcontainer_m->initializeFields("P3M");

        Kokkos::View<int[14*3], Device> offset_device("offset_device");
        Kokkos::View<int[14*3], Host> offset("offset");

        int offset_arr[14][3] = {{ 1, 1, 1}, { 0, 1, 1}, {-1, 1, 1},
            { 1, 0, 1}, { 0, 0, 1}, {-1, 0, 1},
            { 1,-1, 1}, { 0,-1, 1}, {-1,-1, 1},
            { 1, 1, 0}, { 0, 1, 0}, {-1, 1, 0},
            { 1, 0, 0}, { 0, 0, 0}};

        Kokkos::parallel_for("Fill offset array", Kokkos::RangePolicy<Host>(0, 14*3),
            KOKKOS_LAMBDA(const int& ii){
		        const int i = ii / 3;
		        const int j = ii % 3;
                offset(3 * i + j) = offset_arr[i][j];
            }
        );

        Kokkos::deep_copy(offset_device, offset);
        this->pcontainer_m->setOffset(offset_device);

        // initialize solver
        ippl::ParameterList sp;
        sp.add("output_type", P3MSolver_t<T, Dim>::GRAD);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        this->setFieldSolver(
            std::make_shared<P3MSolver_t<T, Dim>>(
                this->fcontainer_m->getE(), this->fcontainer_m->getRho(), sp, this->alpha_m
            )
        );

        // setupMPI();

        double initTimerStart = MPI_Wtime();
        initializeParticles();
        double initTimerEnd = MPI_Wtime();
        std::cout << "Particle Initialization Time: " << initTimerEnd - initTimerStart << std::endl;

        double nlSetupStart = MPI_Wtime();
        initializeNeighborList();
        Kokkos::fence();
        double nlSetupEnd = MPI_Wtime();
        std::cout << "NL Build Time: "<< nlSetupEnd - nlSetupStart << std::endl;

        double MPITimerStart = MPI_Wtime();
        setupMPI();
        double MPITimerEnd = MPI_Wtime();
        std::cout << "MPI Setup Time: " << MPITimerEnd - MPITimerStart << std::endl;

        double start = MPI_Wtime();
        particleExchange();
        double end = MPI_Wtime();
        std::cout << "Particle Exchange Time: " << (end - start) << std::endl;

        this->fcontainer_m->getRho() = 0.0;

        double PMTimerStart = MPI_Wtime();
        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();
        double PMTimerEnd = MPI_Wtime();
        
        std::cout << "Field Solver Time: " << PMTimerEnd - PMTimerStart << std::endl;

        double PPTimerStart = MPI_Wtime();
	    this->par2par();
        double PPTimerEnd = MPI_Wtime();
        std::cout << "PP Interaction Time: " << PPTimerEnd - PPTimerStart << std::endl;

    	haloEnergyUpdate();

    	this->focusingF_m *= this->computeAvgSpaceChargeForces();
	    
        // this->pcontainer_m->update();

        std::cerr << "Pre Run finished" << std::endl;
    }

    void computeBeamStatistics() {
        // std::cerr << "Start Computing Beam Statistics" << std::endl;

        auto R = this->pcontainer_m->R.getView();
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto P = this->pcontainer_m->P.getView();
        double beamRad = this->beamRad_m;

        Vector_t<double, 12> stats = 0.0;

        Kokkos::parallel_reduce("compute sigma x", nLoc,
            KOKKOS_LAMBDA(const size_type i, ippl::Vector<double, 12>& sum){
                sum[0] += R(i)[0] * R(i)[0];
                sum[1] += P(i)[0] * P(i)[0];
                sum[2] += R(i)[0] * P(i)[0];
                sum[3] += R(i)[1] * R(i)[1];
                sum[4] += P(i)[1] * P(i)[1];
                sum[5] += R(i)[1] * P(i)[1];
                sum[6] += R(i)[2] * R(i)[2];
                sum[7] += P(i)[2] * P(i)[2];
                sum[8] += R(i)[2] * P(i)[2];
        		sum[9] += R(i)[0];
        		sum[10] += R(i)[1];
        		sum[11] += R(i)[2];
            }, stats
        );

        // double global_xsq = 0.0;
        // double global_psq = 0.0;
        // double global_xpsq = 0.0;
        ippl::Vector<double, 12> global_stats = 0.0;

        ippl::Comm->reduce(&stats[0], &global_stats[0], 12, std::plus<double>(), 0);
    
        global_stats /= this->totalP_m;

    	double sigma_x = Kokkos::sqrt(global_stats[0] - global_stats[9] * global_stats[9]);
    	double sigma_y = Kokkos::sqrt(global_stats[3] - global_stats[10] * global_stats[10]);
    	double sigma_z = Kokkos::sqrt(global_stats[6] - global_stats[11] * global_stats[11]);
        double emit_x = Kokkos::sqrt(global_stats[0] * global_stats[1] - global_stats[2] * global_stats[2]);
        double emit_y = Kokkos::sqrt(global_stats[3] * global_stats[4] - global_stats[5] * global_stats[5]);
        double emit_z = Kokkos::sqrt(global_stats[6] * global_stats[7] - global_stats[8] * global_stats[8]);
        // double beta = avg_xsq / emit_x;
        // double sigma_x = Kokkos::sqrt(avg_xsq);

    	auto rank = ippl::Comm->rank();	
    	if (rank == 0) {
                std::cerr << "Beam Statistics: " << std::endl;
                std::cerr << "RMS Beam Size: " << sigma_x <<  " , " << sigma_y << " , " << sigma_z << std::endl;
                std::cerr << "RMS Emittance: " << emit_x << " , " << emit_y << " , " << emit_z << std::endl;
    	}
    }

    void dump() {
        computeBeamStatistics();
	
    }

    void initializeParticles() {
        Inform m("Initialize Particles");

        int commSize = ippl::Comm->size();
        int rank = ippl::Comm->rank();

        static IpplTimings::TimerRef ITimer = IpplTimings::getTimer("initializeParticles");
        static IpplTimings::TimerRef CTimer = IpplTimings::getTimer("particleCreation");
        static IpplTimings::TimerRef GTimer = IpplTimings::getTimer("generateCoordinates");
        static IpplTimings::TimerRef UTimer = IpplTimings::getTimer("updateToRank");

        IpplTimings::startTimer(ITimer);
        
        unsigned np = this->totalP_m;
        unsigned nloc = np / commSize;
        
        this->Q_m = np;
        
	    // make sure all particles are accounted for
        //if(rank == commSize-1){
        //    nloc = np - (commSize-1)*nloc;
        //}

        // initialize on one rank
        if(rank == 0) {
            nloc = np;
        } else {
            nloc = 0;
        }

        IpplTimings::startTimer(CTimer);
        this->pcontainer_m->create(nloc);
        IpplTimings::stopTimer(CTimer);
        
	
        auto P = this->pcontainer_m->P.getView();
        auto R = this->pcontainer_m->R.getView();
        auto Q = this->pcontainer_m->Q.getView();

        double beamRad = this->beamRad_m;

        Kokkos::fence();
	
	    // make sure this runs on the host, device does not work yet
        Kokkos::Random_XorShift64_Pool<Device> rand_pool((size_type)(42 + 24 * rank));

        IpplTimings::startTimer(GTimer);
        Kokkos::parallel_for("initialize particles", Kokkos::RangePolicy<Device>(0, nloc),
            KOKKOS_LAMBDA(const size_t index) {
                Vector_t<T, Dim> x(0.0);

                auto generator = rand_pool.get_state();
                
                // obtain random numbers
                double u = generator.drand();
                for(int i = 0; i < Dim; ++i){
                    x[i] = generator.normal(0.0, 1.0);
                }

                rand_pool.free_state(generator);

                // calculate position
                T normsq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                Vector_t<T, Dim> pos = beamRad * (Kokkos::pow(u, 1./3.) / Kokkos::sqrt(normsq)) * x;

                for(int d = 0; d < Dim; ++d){
                    P(index)[d] = 0;		// initialize with zero momentum
		            R(index)[d] = pos[d];
                }
                Q(index) = 1;
            }
        );

        // we need to wait for all other ranks to have finished the particle initialization
        // before we can update them to their corresponding rank
        Kokkos::fence();
        ippl::Comm->barrier();
	
        IpplTimings::stopTimer(GTimer);
        
        IpplTimings::startTimer(UTimer);
        this->pcontainer_m->update();
        IpplTimings::stopTimer(UTimer);
        
        ippl::Comm->barrier();
        IpplTimings::stopTimer(ITimer);
	
	// debug output, can be ignored
        // std::cerr << this->pcontainer_m->getLocalNum() << std::endl;
    }

    /**
     * @brief Initializes a neighbor list to be used in PP interaction calculation
    */
    void initializeNeighborList() {
        Inform m("Initialize Neighbor List");

        // get communicator size and rank
        int commSize = ippl::Comm->size();
        int rank = ippl::Comm->rank();

        // get other relevant information
        size_type nLoc = this->pcontainer_m->getLocalNum();
        view_type R = this->pcontainer_m->R.getView();
        view_type P = this->pcontainer_m->P.getView();
        view_type E = this->pcontainer_m->E.getView();
        
	    // get local domain extend
        auto hLocalRegions = this->pcontainer_m->getLayout().getRegionLayout().gethLocalRegions();
        // std::cout << hLocalRegions(rank) << std::endl;
        
        // calculate chaining meshwidth and number of mesh cells
        double hCM[3], l_extend[3], r_extend[3];
        unsigned nCells[3], totalCells = 1;
        for (int d = 0; d < Dim; ++d){
            l_extend[d] = hLocalRegions(rank)[d].min();
            r_extend[d] = hLocalRegions(rank)[d].max();
            double length = hLocalRegions(rank)[d].length();

            nCells[d] = floor(length / this->rcut_m);
            this->nCells_m[d] = nCells[d];
            totalCells *= nCells[d];
            hCM[d] = length / nCells[d];
        }
	
        // allocate required (temporary) Kokkos views
        Kokkos::View<unsigned *, Device> cellIndex("cellIndex", nLoc);
        Kokkos::View<unsigned *, Device> cellParticleCount("cellParticleCount", totalCells);
        Kokkos::View<unsigned *, Device> cellStartingIdx("cellStartingIdx", totalCells+1);
        Kokkos::View<unsigned *, Device> cellCurrentIdx("cellCurrentIdx", totalCells+1);
        Kokkos::View<ippl::Vector<double, 3> *, Device> tempP("tempMom", nLoc);
        Kokkos::View<ippl::Vector<double, 3> *, Device> tempR("tempPos", nLoc);
        Kokkos::View<ippl::Vector<double, 3> *, Device> tempE("tempEn", nLoc);
	
        // calculate cell index for each particle
        Kokkos::parallel_for("CalcCellIndices", Kokkos::RangePolicy<Device>(0, nLoc), 
	    KOKKOS_LAMBDA(const int i) {
            	unsigned x_Idx = floor((R(i)[0] - l_extend[0]) / hCM[0]);
            	unsigned y_Idx = floor((R(i)[1] - l_extend[1]) / hCM[1]);
            	unsigned z_Idx = floor((R(i)[2] - l_extend[2]) / hCM[2]);

            	unsigned locCMeshIdx = x_Idx * nCells[1] * nCells[2] + y_Idx * nCells[2] + z_Idx;
            	assert(locCMeshIdx < totalCells && "Invalid Grid Position");
                // if (locCMeshIdx >= totalCells) locCMeshIdx = totalCells-1;

            	Kokkos::atomic_increment(&cellParticleCount(locCMeshIdx));
            	cellIndex(i) = locCMeshIdx;
        });

        Kokkos::fence();
 	
        // compute starting indices for each cell
	    Kokkos::parallel_scan(Kokkos::RangePolicy<Device>(0, totalCells),
	        KOKKOS_LAMBDA(const int i, unsigned& localSum, bool isFinal){
		        if(isFinal) cellStartingIdx(i) = localSum;
	            localSum += cellParticleCount(i);
	        }
	    );

        Kokkos::fence();

        // std::cerr << "last cell: " << cellStartingIdx(totalCells-1) << std::endl;
        // std::cerr << "nLoc: " << nLoc << std::endl;
    
        Kokkos::parallel_for("Set last position", Kokkos::RangePolicy<Device>(totalCells, totalCells+1),
            KOKKOS_LAMBDA(const int i){
                cellStartingIdx(i) = nLoc;
            }
        );

        Kokkos::fence();

        Kokkos::deep_copy(cellCurrentIdx, cellStartingIdx);

        Kokkos::fence();
	
        // Build temp views
        Kokkos::parallel_for("Build view", Kokkos::RangePolicy<Device>(0, nLoc), 
            KOKKOS_LAMBDA(const size_type& i){
                unsigned cellNumber = cellIndex(i);
                assert(cellNumber < totalCells && "Invalid Cell Number");
                size_type newIdx = Kokkos::atomic_fetch_add(&cellCurrentIdx(cellNumber), 1u);
                assert(newIdx < nLoc && "Invalid Index");
                tempR(newIdx) = R(i);
                tempP(newIdx) = P(i);
                tempE(newIdx) = E(i);
        });

        Kokkos::fence();
	
        // move data from Temp view into main view, there should be a better way to do this
        Kokkos::parallel_for("Copy Data", Kokkos::RangePolicy<Device>(0, nLoc),
            KOKKOS_LAMBDA(const size_type i){
                R(i) = tempR(i);
                P(i) = tempP(i);
                E(i) = tempE(i);
            }
        );
        
        this->pcontainer_m->setNL(cellStartingIdx);
    }

    void pre_step() override {
        Inform m("pre step");
    }

    void post_step() override {
        Inform m("post step");

	dump();

        this->time_m += this->dt_m;
        this->it_m++;
 }

    void grid2par() override {
        gatherCIC();
    }

    void par2grid() override {
        scatterCIC();
    }

    void par2par() override {

        // get particle data
        auto R = this->pcontainer_m->R.getView();
        auto E = this->pcontainer_m->E.getView();
        auto P = this->pcontainer_m->P.getView();
        auto offset = this->pcontainer_m->getOffset();
        auto Q = this->pcontainer_m->Q.getView();

        // get simulation specific data
        auto rcut = this->rcut_m;
        auto alpha = this->alpha_m;
        auto epsilon = this->epsilon_m;
        const double ke = 2.532638e8;

        // get neighbor mesh data
        auto cellStartingIdx = this->pcontainer_m->getNL();
        size_type totalCells = cellStartingIdx.size() - 1;
        auto nCells = this->nCells_m;
        int xCells = nCells[0];
        int yCells = nCells[1];
        int zCells = nCells[2];

        assert(totalCells == xCells * yCells * zCells && "Invalid number of cells");

        Kokkos::View<unsigned[1], Device> counter("counter");
        using team_t = typename Kokkos::TeamPolicy<Device>::member_type;
        
        // calculate interaction force
        Kokkos::parallel_for("Particle-Particle", Kokkos::TeamPolicy<Device>(totalCells, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const team_t& team){
                const size_type cellIdx = team.league_rank();

                // calculate cellIdx in each dimension
                int xIdx = cellIdx / (yCells * zCells);
                int yIdx = (cellIdx % (yCells * zCells)) / zCells;
                int zIdx = cellIdx % zCells;

                // get number of particles in current cell
                const size_type start = cellStartingIdx(cellIdx);
                const size_type end = cellStartingIdx(cellIdx+1);
                const size_type nParticles = end - start;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 14),
                    [&](const int& neighborIdx){

                        // get offset for neighbor cell
                        const int offsetX = offset(neighborIdx * 3 + 0);
                        const int offsetY = offset(neighborIdx * 3 + 1);
                        const int offsetZ = offset(neighborIdx * 3 + 2);
                    
                        // check if neighbor is within domain
                        if ((xIdx + offsetX < 0) || (xIdx + offsetX >= xCells) ||
                            (yIdx + offsetY < 0) || (yIdx + offsetY >= yCells) ||
                            (zIdx + offsetZ < 0) || (zIdx + offsetZ >= zCells)) {
                            return;
                        }

                        // get number of particles in neighbor cell
                        const size_type neighborCellIdx = (xIdx + offsetX) * yCells * zCells + (yIdx + offsetY) * zCells + (zIdx + offsetZ);
                        const size_type neighborStart = cellStartingIdx(neighborCellIdx);
                        const size_type neighborEnd = cellStartingIdx(neighborCellIdx+1);
                        const size_type nNeighborParticles = neighborEnd - neighborStart;

                        
                        auto threadVectorMDRange = 
                            Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, team_t>(team, nParticles, nNeighborParticles);
	 			
                        Kokkos::parallel_for(threadVectorMDRange, 
                            [&](const int& i, const int& j){
                                const size_type ii = start + i;
                                const size_type jj = neighborStart + j;
                                if (((cellIdx == neighborCellIdx) && ii >= jj)) return;

                                double rsq_ij = 0.0;
                                Vector_t<T, Dim> dist_ij = R(ii) - R(jj);
                                for (int d = 0; d < Dim; ++d) {
                                    rsq_ij += dist_ij[d] * dist_ij[d];
                                }

                                double r_ij = Kokkos::sqrt(rsq_ij);
		
				                if  (r_ij >= rcut) return;


                                // calculate and apply force
                                Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
				                Kokkos::atomic_sub(&E(ii), F_ij * Q(jj));
                                Kokkos::atomic_add(&E(jj), F_ij * Q(ii));
                            }
                        ); 
                    }
                );
            }
        );

        // Kokkos::fence();
        // ippl::Comm->barrier();

        // std::cerr << "Particle-Particle Interaction finished" << std::endl;
    }


    void advance() override {
        LeapFrogStep();
    }

    void haloEnergyUpdate(){
        
        auto nLoc = this->pcontainer_m->getLocalNum();
        Kokkos::View<ippl::Vector<double, 3> *, Device> E_halo("Halo Energy Device View", nLoc);
        Kokkos::deep_copy(E_halo, haloE_m);
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        Kokkos::parallel_for("update Energy by halo interaction values", nLoc,
            KOKKOS_LAMBDA(const size_type& i){
                pc->E(i) += E_halo(i);
            }
        );
    }

    void LeapFrogStep() {
        
        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

	auto rank = ippl::Comm->rank();

        pc->R = pc->R + dt * pc->P;

        pc->update();

        this->initializeNeighborList();

    	Kokkos::fence();

        particleExchange();

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

        this->par2par();

        haloEnergyUpdate();

        this->applyConstantFocusing();

        pc->P = pc->P - dt * pc->E;

        if(rank== 0) std::cerr << "LeapFrog Step " << this->it_m << " Finished." << std::endl;

    }

    double computeAvgSpaceChargeForces() {
	
	auto rank = ippl::Comm->rank();

        auto totalP = this->totalP_m;
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto E = this->pcontainer_m->E.getView();
        Vector_t<T, Dim> avgE = 0.0;

        Kokkos::parallel_reduce("compute average space charge forces", nLoc, 
            KOKKOS_LAMBDA(const size_type i, Vector_t<T, Dim>& sum){
                sum[0] += Kokkos::abs(E(i)[0]);
                sum[1] += Kokkos::abs(E(i)[1]);
                sum[2] += Kokkos::abs(E(i)[2]);
            }, avgE
        );

        // std::cerr << "Average Space Charge Forces: " << avgE << std::endl;

        Vector_t<double, Dim> globE = 0.0;

        ippl::Comm->reduce(&avgE[0], &globE[0], 3, std::plus<double>(), 0);
        if (rank == 0) std::cerr << "Average Space Charge Forces: " << globE << std::endl; 
        globE /= totalP;

        double focusingf = 0.0;
        for (unsigned d = 0; d < Dim; ++d) {
            focusingf += globE[d] * globE[d];
        }

        MPI_Bcast(&focusingf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return std::sqrt(focusingf);
    }

    void applyConstantFocusing() {
        view_type E = this->pcontainer_m->E.getView();
        view_type R = this->pcontainer_m->R.getView();

        double beamRad = this->beamRad_m;
        double focusStrength = this->focusingF_m;
        auto nLoc = this->pcontainer_m->getLocalNum();

	auto rank = ippl::Comm->rank();

        if (rank == 0) std::cerr << "Focusing Force " << focusStrength << std::endl;
        
	    Kokkos::parallel_for("apply constant focusing", nLoc,
            KOKKOS_LAMBDA(const size_type& i){
                Vector_t<T, Dim> F = focusStrength * (R(i) / beamRad);
                E(i) += F;
            }
        );
    }

    void gatherCIC(){
        gather( this->pcontainer_m->E, 
                this->fcontainer_m->getE(), 
                this->pcontainer_m->R
        );
    }

    void scatterCIC(){
        Inform m("scatter ");
        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double> *q = &this->pcontainer_m->Q;
        typename Base::particle_position_type *R = &this->pcontainer_m->R;
        Field_t<Dim> *rho               = &this->fcontainer_m->getRho();
        double Q                        = this->Q_m;
        Vector_t<double, Dim> rmin	= this->rmin_m;
        Vector_t<double, Dim> rmax	= this->rmax_m;
        Vector_t<double, Dim> hr        = this->hr_m;

        scatter(*q, *rho, *R);
        double relError = std::fabs((Q-(*rho).sum())/Q);

        // std::cerr << "Relative Error: " << relError << std::endl;

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 1e-10) {
                m << "Time step: " << it_m << endl;
                m << "Total particles in the sim. " << totalP_m << " "
                  << "after update: " << TotalParticles << endl;
                m << "Rel. error in charge conservation: " << relError << endl;
                ippl::Comm->abort();
            }
	    }   

	    double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)          = (*rho) / cellVolume;

        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i;
        double size = 1;
        for (unsigned d = 0; d < Dim; d++) {
            size *= rmax[d] - rmin[d];
        }
        *rho = *rho - (Q / size);
        
    }
};

#endif
