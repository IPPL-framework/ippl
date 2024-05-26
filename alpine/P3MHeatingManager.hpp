#ifndef IPPL_P3M_HEATING_MANAGER_HPP
#define IPPL_P3M_HEATING_MANAGER_HPP

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
const double ke = 2.532638e8;

/**
 * @class P3M3DHeatingManager
 * @brief A class that runs P3M simulation for Disorder induced Heating processes
 * 
 * @tparam T the data dype for simulation variables
 * @tparam Dim the dimensionality of the simulation
*/
template <typename T, unsigned Dim>
class P3M3DHeatingManager 
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
    
public:
    P3M3DHeatingManager(size_type totalP_, int nt_, double dt_, Vector_t<int, Dim>& nr_, double rcut_, double alpha_, double beamRad_, double focusingF_) 
        : ippl::P3M3DManager<T, Dim, FieldContainer<T, Dim> >() 
        , totalP_m(totalP_), nt_m(nt_), dt_m(dt_), nr_m(nr_), rcut_m(rcut_), alpha_m(alpha_), solver_m("P3M"), beamRad_m(beamRad_), focusingF_m(focusingF_)
        {}

    ~P3M3DHeatingManager(){}

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

public: 
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    void initializeFromCSV(std::string filename){

        std::ifstream file(filename);

        double q = -1;

        if(!file.is_open()){
            std::cerr << "Could not open file" << std::endl;
            return;
        }

        auto np = this->totalP_m;
        std::string line;
        this->pcontainer_m->create(np);

        // auto P = this->pcontainer_m->P.getView();
        auto R = this->pcontainer_m->R.getView();
        auto Q = this->pcontainer_m->Q.getView();
        auto P = this->pcontainer_m->P.getView();

        for(size_type i = 0; i < np+1; ++i){
            // if (i == 0) continue;
            std::getline(file, line);
            std::stringstream ss(line);
            std::string token;
            for(int d = 0; d < Dim; ++d){
                std::getline(ss, token, ',');
                if (i == 0) continue;
                R(i-1)[d] = std::stod(token);
                P(i-1)[d] = 0.0;
                // std::cout << token << std::endl;
                
            }
            Q(i-1) = q;
        }
        this->Q_m = np * q;
        this->pcontainer_m->update();

    }

    void pre_run() override {
        Inform m("Pre Run");

        for (unsigned i = 0; i < Dim; ++i) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        // this->alpha_m = 2./this->rcut_m;
        double box_length = 0.01;
        this->rmin_m = -box_length/2.;
        this->rmax_m = box_length/2.;
        this->origin_m = rmin_m;

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

        // probably not needed
        // this->fsolver_m->initSolver();

        // initialize particle positions and momenta
        initializeParticles();
        // initializeFromCSV("/home/timo/ETH/Thesis/ippl-build-scripts/ippl/build_openmp/test/p3m/particle_positions.csv");

        computeRMSBeamSize();

        // intialize Neighbor List
        initializeNeighborList();

        this->fcontainer_m->getRho() = 0.0;

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

	    // this->pcontainer_m->E = -1.0 * this->pcontainer_m->E;

	    this->par2par();

	    this->focusingF_m *= this->computeAvgSpaceChargeForces();
	    
        this->pcontainer_m->update();

        std::cerr << "Pre Run finished" << endl;
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
        if(rank == commSize-1){
            nloc = np - (commSize-1)*nloc;
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
        std::cerr << this->pcontainer_m->getLocalNum() << std::endl;
    }


    void computeRMSBeamSize(){
        auto R = this->pcontainer_m->R.getView();
        auto nLoc = this->pcontainer_m->getLocalNum();


        ippl::Vector<double, 6> averages(0.0);
        Kokkos::parallel_reduce("compute RMS beam size", nLoc,
            KOKKOS_LAMBDA(const size_type& i, ippl::Vector<double, 6>& sum){
                sum[0] += R(i)[0];
                sum[1] += R(i)[1];
                sum[2] += R(i)[2];
                sum[3] += R(i)[0] * R(i)[0];
                sum[4] += R(i)[1] * R(i)[1];
                sum[5] += R(i)[2] * R(i)[2];
            }, averages
        );
        ippl::Vector<double, 6> glob(0.0);
        ippl::Comm->reduce(&averages[0], &glob[0], 6, std::plus<double>(), 0);

        auto totalP = this->totalP_m;

        glob /= totalP;

        double rms_x = sqrt(glob[3] - glob[0] * glob[0]);
        double rms_y = sqrt(glob[4] - glob[1] * glob[1]);
        double rms_z = sqrt(glob[5] - glob[2] * glob[2]);


        std::cerr << "Beam Center: (" << glob[0] << ", " << glob[1] << ", " << glob[2] << ")" << std::endl;
        std::cerr << "RMS Beam Size: (" << rms_x << ", " << rms_y << ", " << rms_z << ")" << std::endl;
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

        if(commSize == 1){
            this->pcontainer_m->setNL(cellStartingIdx);
            return;
        }

        /* Ghost NL Build - Halo exchange
         * 1. Figure out where neighbors are located relative to rank
         * 2. Build Send Buffer
         * 3. Send / Recieve Particles
         */
	
	    // get FieldLayout and list of neighboring domains
        auto FL = this->fcontainer_m->getFL();
	
        // get host mirror of particle view
        view_type::HostMirror R_host = Kokkos::create_mirror_view(this->pcontainer_m->R.getView());
            
        auto comm = FL.comm; 
        
        Kokkos::View<unsigned *, Host> host_cellStartingIdx("host_cellStartingIdx", totalCells+1);
        Kokkos::deep_copy(host_cellStartingIdx, cellStartingIdx);

        Kokkos::View<unsigned *, Host> host_cellParticleCount("host_cellParticleCount", totalCells);
        Kokkos::deep_copy(host_cellParticleCount, cellParticleCount);
        
        bool neighbors[commSize];	

        unsigned totalNeighbors = 0;
        // unsigned neighborcubes = 0;
        for (int recvRank = 0; recvRank < commSize; ++recvRank) {
            if (recvRank != rank) {
                // 0: no overlap; 1: left from domain; 2: right from domain
                int overlapInDim[3];
                int equalInDim[3];

                // these are the starting and end indices of the cell range in each dim
                // Note that currently, the cellRange is flattened
                int cellStartIdx[3];
                int cellEndIdx[3];
                
                // this tells us over how many surface cells we need to iterate
                int numSurfaceCells = 1;

                // 0: no overlap; 1: face; 2: edge; 3: corner
                int overlapType = 0;
                int equalType = 0;
                
                for(unsigned d = 0; d < Dim; ++d){

                    // checks for overlap in Dimension d and assigns
                    // 0: when there is no overlap
                    // 1: for an overlap at the lower domain extend
                    // 2: for an overlap at the upper domain extend
                    overlapInDim[d] = (l_extend[d] < hLocalRegions(recvRank)[d].max() && l_extend[d] > hLocalRegions(recvRank)[d].min())
                            + 2 * (r_extend[d] > hLocalRegions(recvRank)[d].min() && r_extend[d] < hLocalRegions(recvRank)[d].max());

                    equalInDim[d] = (l_extend[d] == hLocalRegions(recvRank)[d].min())
                        + 2 * (r_extend[d] == hLocalRegions(recvRank)[d].max()); 
                    
                    overlapType += (overlapInDim[d] > 0);
                    equalType += equalInDim[d] > 0;
                    
                    // if there is no overlap in a certain dimension, we want to iterate from 0 to nCells
                    // if there is an overlap, its index is fixed at either 0 or nCells[d]-1
                    cellStartIdx[d] = (overlapInDim[d] + !overlapInDim[d] - 1) * (nCells[d]-1);
                    cellEndIdx[d] = (overlapInDim[d] ? (cellStartIdx[d]+1) : nCells[d]);
                    
                    // this is either 1 or nCells per Dimension
                    numSurfaceCells *= (cellEndIdx[d] - cellStartIdx[d]);
                }
                overlapType = (overlapType + equalType == Dim);
                
                int nParticlesToSend = 0;

                if(overlapType + equalType == Dim) {
                    neighbors[recvRank] = true;
                    totalNeighbors++;
                } else {
                    neighbors[recvRank] = false;
                    continue;
                }

                if(overlapType+equalType == 3 && nLoc > 0) {
                    
                    // find out how many particles to send
                    // (we may merge with create sendbuf, by using a vector instead of an array)
                    for(int xCellIdx = cellStartIdx[0]; xCellIdx < cellEndIdx[0]; ++xCellIdx){
                        for(int yCellIdx = cellStartIdx[1]; yCellIdx < cellEndIdx[1]; ++yCellIdx){
                            for(int zCellIdx = cellStartIdx[2]; zCellIdx < cellEndIdx[2]; ++zCellIdx){
                                unsigned CellIdx = xCellIdx * nCells[1] * nCells[2] + yCellIdx * nCells[2] + zCellIdx;
                                nParticlesToSend += host_cellParticleCount(CellIdx);
                            }
                        }
                    }
            
                    if (nParticlesToSend > 0){
                        double sendBuf[nParticlesToSend * 3];

                        // build send buffer
                        size_type sendBufIdx = 0;
                        for(int xCellIdx = cellStartIdx[0]; xCellIdx < cellEndIdx[0]; ++xCellIdx){
                            for(int yCellIdx = cellStartIdx[1]; yCellIdx < cellEndIdx[1]; ++yCellIdx){
                                for(int zCellIdx = cellStartIdx[2]; zCellIdx < cellEndIdx[2]; ++zCellIdx){
                                    unsigned CellIdx = xCellIdx * nCells[1] * nCells[2] + yCellIdx * nCells[2] + zCellIdx;
                                    size_type start = host_cellStartingIdx(CellIdx);
                                    size_type end = host_cellStartingIdx(CellIdx+1);
                        
                                    // loop over all particles in a cell
                                    for(size_type i = start; i < end; ++i){
                                        for(int d = 0; d < Dim; ++d){
                                            // assert(sendBufIdx < nParticlesToSend && "too many particles");
                                            sendBuf[3*sendBufIdx + d] = R_host(i)[d];
                                        }
                                        ++sendBufIdx;
                                    }
                                }
                            }
                        }
                        // make sure we send as many particles as expected
                        assert((sendBufIdx == nParticlesToSend) && "sendBuf invalid");
                        
                        MPI_Request request;
                        MPI_Isend(sendBuf, 3*nParticlesToSend, MPI_DOUBLE, recvRank, recvRank, ippl::Comm->getCommunicator(), &request); 
                    
                        //std::cerr << nParticlesToSend << " Particles from Rank " << rank << " to " << recvRank << std::endl;
                    } else {
                        // send dummy message, remove later
                        double dummy = 0;
                        MPI_Request request;
                        MPI_Isend(&dummy, 1, MPI_DOUBLE, recvRank, recvRank, ippl::Comm->getCommunicator(), &request);
                    }
                }
            }
        }   
        neighbors[rank] = false;
        
        // set neighbor list after initialization
        this->pcontainer_m->setNL(cellStartingIdx);
        this->pcontainer_m->setNeighbors(neighbors);
        
        if(totalNeighbors > 0){
            double *recvBuffers[totalNeighbors];
            int senderCount = 0;	
            // recieve Messages
            for(int sender = 0; sender < commSize; ++sender){
                if (neighbors[sender]){
                    // MPI stuff to facilitate exchange
                    MPI_Status status;
                    int count;

                    // Probe for message and get number of doubles
                    MPI_Probe(sender, /*rank*/ MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_DOUBLE, &count);

                    // allocate buffer and recieve
                    double recvBuf[count];
                    recvBuffers[senderCount] = recvBuf;
                    ++senderCount;
                    MPI_Recv(recvBuf, count, MPI_DOUBLE, sender,/*rank*/ MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    //std::cerr << "Rank " << rank << " recieved " << count/3 << " paricles from " << sender << std::endl;
                }
            }
        }
        std::cerr << "Rank " << rank << " is done :) " << std::endl;
    }

    void pre_step() override {
        Inform m("pre step");
        /* TODO */
    }

    void post_step() override {
        Inform m("post step");

        this->time_m += this->dt_m;
        this->it_m++;

        // add print every later
        this->dump();
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
                    [=](const int& neighborIdx){

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
                            [=](const int& i, const int& j){
                                const size_type ii = start + i;
                                const size_type jj = neighborStart + j;
                                if (((cellIdx == neighborCellIdx) && ii >= jj)) return;
                                const double ke = 2.532638e8;

                                double rsq_ij = 0.0;
                                Vector_t<T, Dim> dist_ij = R(ii) - R(jj);
                                for (int d = 0; d < Dim; ++d) {
                                    rsq_ij += dist_ij[d] * dist_ij[d];
                                }

                                double r_ij = Kokkos::sqrt(rsq_ij);
				                if  (r_ij >= rcut) return;
		                        // r_ij += !isWithinCutoff; // prevent didvide by zero
				                // rsq_ij += !isWithinCutoff;
                                // Kokkos::atomic_add(&counter(0), isWithinCutoff);


                                // calculate and apply force
                                Vector_t<T, Dim> F_ij =  ke * (dist_ij/r_ij) * ((2.0 * alpha * Kokkos::exp(-alpha * alpha * rsq_ij))/ (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                                // Vector_t<T, Dim> F_ij = 0;
				                Kokkos::atomic_sub(&E(ii), F_ij * Q(jj));
                                Kokkos::atomic_add(&E(jj), F_ij * Q(ii));
                            }
                        );
                    }
                );
            });

            Kokkos::fence();
            ippl::Comm->barrier();

	        // auto host_counter = Kokkos::create_mirror_view(counter);

            // std::cerr << "Number PP interactions: " << host_counter(0) << std::endl;

            std::cerr << "Particle-Particle Interaction finished" << std::endl;
    }


    void advance() override {
        LeapFrogStep();
    }

    double calcKineticEnergy() {
        view_type P = this->pcontainer_m->P.getView();
        auto nLoc = this->pcontainer_m->getLocalNum();

        double localEnergy = 0.0;
        double globalEnergy = 0.0;  
        
	    Kokkos::parallel_reduce("calc kinetic energy", nLoc,
            KOKKOS_LAMBDA(const size_type& i, double& sum){
                sum += 0.5 * (P)(i).dot((P)(i));
            }, localEnergy
        );
        Kokkos::fence();

        // gather local kinetic energy from other ranks
        ippl::Comm->reduce(localEnergy, globalEnergy, 1, std::plus<double>());
        ippl::Comm->barrier();

        return globalEnergy;
    }

    void compute_temperature() {
        Vector_t<double, 3> locAvgVel = 0.0;
        Vector_t<double, 3> globAvgVel = 0.0;
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto P = this->pcontainer_m->P.getView();

        Kokkos::parallel_reduce("compute average velocity", Kokkos::RangePolicy<Device>(0, nLoc),
            KOKKOS_LAMBDA(const size_type i, ippl::Vector<double, 3>& sum){
                sum += P(i);
            }, locAvgVel
        );
        ippl::Comm->reduce(&locAvgVel[0], &globAvgVel[0], 3, std::plus<double>(), 0);

        ippl::Comm->barrier();

        auto totalP = this->totalP_m;

        globAvgVel /= totalP;
        std::cerr << "Average Velocity: " << globAvgVel << std::endl;
	
        ippl::Vector<double, 3> localTemperature = 0.0;
        ippl::Vector<double, 3> globalTemperature = 0.0;

        Kokkos::parallel_reduce("compute temperature", nLoc,
            KOKKOS_LAMBDA(const size_type i, ippl::Vector<double, 3>& sum){
                sum += (P(i)-globAvgVel(0)) * (P(i)-globAvgVel(0));
            }, localTemperature
        );

        ippl::Comm->reduce(&localTemperature[0], &globalTemperature[0], 3, std::plus<double>(), 0);

        globalTemperature /= totalP;

        std::cerr << "Temperature: " << globalTemperature << std::endl;
	
    }

    void computeBeamStatistics() {
        std::cerr << "Start Computing Beam Statistics" << std::endl;

        auto R = this->pcontainer_m->R.getView();
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto P = this->pcontainer_m->P.getView();
        double beamRad = this->beamRad_m;

        Kokkos::View<ippl::Vector<double, 3>[1], Device> stats("stats");

        Kokkos::parallel_reduce("compute sigma x", nLoc,
            KOKKOS_LAMBDA(const size_type i, ippl::Vector<double, 3>& sum){
                sum[0] += R(i)[0] * R(i)[0];
                sum[1] += P(i)[0] * P(i)[0];
                sum[2] += R(i)[0] * P(i)[0];
            }, stats(0)
        );

        double global_xsq = 0.0;
        double global_psq = 0.0;
        double global_xpsq = 0.0;

        // there must be a better way to do this
        ippl::Comm->reduce(stats(0)[0], global_xsq, 1, std::plus<double>());
        ippl::Comm->reduce(stats(0)[1], global_psq, 1, std::plus<double>());
        ippl::Comm->reduce(stats(0)[2], global_xpsq, 1, std::plus<double>());
        ippl::Comm->barrier();

        double avg_xsq = global_xsq / this->totalP_m;
        double avg_psq = global_psq / this->totalP_m;
        double avg_xpsq = global_xpsq / this->totalP_m;

        double emit_x = Kokkos::sqrt(avg_xsq * avg_psq - avg_xpsq * avg_xpsq);
        // double beta = avg_xsq / emit_x;
        double sigma_x = Kokkos::sqrt(avg_xsq);
        std::cerr << "Beam Statistics: " << std::endl;
        std::cerr << "Sigma x: " << sigma_x << std::endl;
        std::cerr << "Emittance x: " << emit_x << std::endl;
    }

    void dump() override {
        Inform m("Dump");

        std::cerr << "Dumping data" << std::endl;

        double E_kin = calcKineticEnergy();
        // double E_pot = calcPotentialEnergy();
        // std::cerr << "Dumping data, Energy: " << E_kin + E_pot << std::endl;
        std::cerr << "Dumping data, Kinetic Energy: " << E_kin << std::endl;
        // std::cerr << "Dumping data, Potential Energy: " << E_pot << std::endl;
        // std::cerr << "Dumping data, Gamma eq: " << E_kin/E_pot << std::endl;

        int it = this->it_m;
	    auto host_R = Kokkos::create_mirror_view(this->pcontainer_m->R.getView());
        // DEBUG output
        std::ofstream outputFile("out/particle_positions_" + std::to_string(it) + ".csv");
        if (outputFile.is_open()) {
            // auto R = this->pcontainer_m->R.getView();
            for (size_type i = 0; i < this->pcontainer_m->getLocalNum(); ++i) {
                for (unsigned d = 0; d < Dim; ++d) {
                    outputFile << host_R(i)[d];
                    if (d < Dim - 1) outputFile << ",";
                }
                outputFile << std::endl;
            }
            outputFile.close();
        } else {
            std::cerr << "Unable to open file" << std::endl;
        }
        // computeBeamStatistics();
        compute_temperature();
        computeRMSBeamSize();
    }

    void LeapFrogStep() {
        
        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // pc->P = pc->P - 0.5 * dt * pc->E;

        pc->R = pc->R + dt * pc->P;

        pc->update();

        this->initializeNeighborList();

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

        // pc->E = -1.0 * pc->E;

        this->par2par();

        this->applyConstantFocusing();

        pc->P = pc->P - dt * pc->E;

        std::cerr << "LeapFrog Step " << this->it_m << " Finished." << std::endl;

    }

    double computeAvgSpaceChargeForces() {
        auto totalP = this->totalP_m;
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto E = this->pcontainer_m->E.getView();
        // Vector_t<T, Dim> locAvgE = 0.0;
        Vector_t<T, Dim> avgE = 0.0;

        Kokkos::parallel_reduce("compute average space charge forces", nLoc, 
            KOKKOS_LAMBDA(const size_type i, Vector_t<T, Dim>& sum){
                sum[0] += Kokkos::abs(E(i)[0]);
                sum[1] += Kokkos::abs(E(i)[1]);
                sum[2] += Kokkos::abs(E(i)[2]);
            }, avgE
        );

        // std::cerr << "Total Local Space Charge Forces: " << avgE << std::endl;

        // avgE /= totalP;

        std::cerr << "Average Space Charge Forces: " << avgE << std::endl;

        // double focusingf = 0.0;
        // for (unsigned d = 0; d < Dim; ++d) {
        //     focusingf += avgE[d] * avgE[d];
        // }

        // std::cerr << "Focusing Force: " << focusingf << std::endl;

        Vector_t<double, Dim> globE = 0.0;

        ippl::Comm->reduce(&avgE[0], &globE[0], 3, std::plus<double>(), 0);
        
        globE /= totalP;

        double focusingf = 0.0;
        for (unsigned d = 0; d < Dim; ++d) {
            focusingf += globE[d] * globE[d];
        }
        // std::cerr << "Focusing Force: " << focusingf << std::endl;
        // ippl::Comm->barrier();
        return std::sqrt(focusingf);
    }

    void applyConstantFocusing() {
        view_type E = this->pcontainer_m->E.getView();
        view_type R = this->pcontainer_m->R.getView();

        double beamRad = this->beamRad_m;
        double focusStrength = this->focusingF_m;
        auto nLoc = this->pcontainer_m->getLocalNum();

        std::cerr << "Focusing Force " << focusStrength << std::endl;
        
	Kokkos::parallel_for("apply constant focusing", nLoc,
            KOKKOS_LAMBDA(const size_type& i){
                Vector_t<T, Dim> F = focusStrength * (R(i) / beamRad);
                Kokkos::atomic_add(&E(i), F);
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

        std::cerr << "Relative Error: " << relError << std::endl;

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
