#ifndef IPPL_P3M_HEATING_MANAGER_HPP
#define IPPL_P3M_HEATING_MANAGER_HPP

// includes
#include <memory>
#include <string>
#include <iostream>
#include <Kokkos_Random.hpp>

// Alpine Headers
// #include "../alpine/LoadBalancer.hpp"
#include "../alpine/FieldContainer.hpp"

// P3M Headers
#include "Manager/P3M3DManager.h"
#include "PoissonSolvers/P3MSolver.h"
#include "P3MParticleContainer.hpp"

// Distribution functions
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/UniformDistribution.h"

// Required Datatypes
template <typename T = double, unsigned Dim = 3>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::P3MSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T = double, unsigned Dim=3, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, 3>, 1>::view_type;


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
    P3M3DHeatingManager(size_type totalP_, int nt_, double dt_, Vector_t<int, Dim>& nr_, double rcut_, double beamRad_, double focusingF_) 
        : ippl::P3M3DManager<T, Dim, FieldContainer<T, Dim> >() 
        , totalP_m(totalP_), nt_m(nt_), dt_m(dt_), nr_m(nr_), rcut_m(rcut_), solver_m("P3M"), beamRad_m(beamRad_), focusingF_m(focusingF_)
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

    // virtual void dump() {} // not defined in P3M3DManager, do later

    void pre_run() override {
        Inform m("Pre Run");

        for (unsigned i = 0; i < Dim; ++i) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->alpha_m = 6400;
        double box_length = 0.01;
        this->rmin_m = -box_length/2.;
        this->rmax_m = box_length/2.;
        this->origin_m = rmin_m;

        this->hr_m = box_length/(double)(this->nr_m[1]);
        
        // time stuff
        //this->dt_m = 2.15623e-13;
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

    
        this->fcontainer_m->initializeFields("P3M");

        // initialize solver
        ippl::ParameterList sp;
        sp.add("output_type", P3MSolver_t<T, Dim>::GRAD);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", false);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);

        this->setFieldSolver(
            std::make_shared<P3MSolver_t<T, Dim>>(
                this->fcontainer_m->getE(), this->fcontainer_m->getRho(), sp
            )
        );

        // initialize particle positions and momenta
        initializeParticles();

        // intialize Neighbor List
        initializeNeighborList();
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
        
        if(rank == commSize-1){
            nloc = np - (commSize-1)*nloc;
        }

        unsigned start = rank * nloc;

        // do domain decomp?

        // get Position and momentum view
        view_type* P = &(this->pcontainer_m->P.getView());
        view_type* R = &(this->pcontainer_m->R.getView());
        double beamRad = this->beamRad_m;

        IpplTimings::startTimer(CTimer);
        this->pcontainer_m->create(nloc);
        IpplTimings::stopTimer(CTimer);

        Kokkos::fence();

        Kokkos::Random_XorShift64_Pool<> rand_pool((size_type)(42 + 24 * rank));

        IpplTimings::startTimer(GTimer);
        Kokkos::parallel_for(nloc,
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
                    (*P)(index)[d] = 0;         // initialize with zero momentum
                    (*R)(index)[d] = pos[d];
                }
            }
        );

        // we need to wait for all other ranks to have finished the particle initialization
        // before we can update them to their corresponding rank
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(GTimer);

        // send particles to corresponding processes
        IpplTimings::startTimer(UTimer);
        this->pcontainer_m->update();
        IpplTimings::stopTimer(UTimer);

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
        view_type *R = &(this->pcontainer_m->R.getView());
        view_type *P = &(this->pcontainer_m->P.getView());
        auto ID = &(this->pcontainer_m->ID.getView());

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
            totalCells *= nCells[d];
            hCM[d] = length / nCells[d];
        }

        // allocate required (temporary) Kokkos views
        Kokkos::View<unsigned *> cellIndex("cellIndex", nLoc);
        Kokkos::View<size_type *> cellParticleCount("cellParticleCount", totalCells);
        Kokkos::View<unsigned *> cellStartingIdx("cellStartingIdx", totalCells+1);
        Kokkos::View<unsigned *> cellCurrentIdx("cellCurrentIdx", totalCells+1);
        Kokkos::View<size_type*> tempID("tempID", nLoc);
        view_type tempR("tempPos", nLoc);
        // view_type P_temp("tempMomenta", nLoc); // required for update

        // calculate cell index for each particle
        Kokkos::parallel_for(nLoc, KOKKOS_LAMBDA(const int i) {
            unsigned x_Idx = floor(((*R)(i)[0] - l_extend[0]) / hCM[0]);
            unsigned y_Idx = floor(((*R)(i)[1] - l_extend[1]) / hCM[1]);
            unsigned z_Idx = floor(((*R)(i)[2] - l_extend[2]) / hCM[2]);

            unsigned locCMeshIdx = x_Idx * nCells[1] * nCells[2] + y_Idx * nCells[2] + z_Idx;
            assert(locCMeshIdx < totalCells && "Invalid Grid Position");

            cellParticleCount(locCMeshIdx)++;
            cellIndex(i) = locCMeshIdx;
        });

        Kokkos::fence();

        // compute starting indices for each cell
        Kokkos::parallel_scan("Calculate Starting Indices", totalCells,
            KOKKOS_LAMBDA(const int i, unsigned& localSum, bool isFinal){
                if(isFinal) cellStartingIdx(i) = localSum;
                localSum += cellParticleCount(i);
            }
        );
        cellStartingIdx(totalCells) = nLoc;

        Kokkos::fence();

        Kokkos::deep_copy(cellCurrentIdx, cellStartingIdx);

        Kokkos::fence();

        // Build temp views
        Kokkos::parallel_for(nLoc, KOKKOS_LAMBDA(const size_type i){
            unsigned cellNumber = cellIndex(i);
            size_type newIdx = Kokkos::atomic_fetch_add(&cellCurrentIdx(cellNumber), 1u);
            tempR(newIdx) = (*R)(i);
            tempID(newIdx) = (*ID)(i);
        });

        Kokkos::fence();

        // Reorder particles, replace with copy later?
        Kokkos::parallel_for(nLoc, KOKKOS_LAMBDA(const size_type i){
            (*R)(i) = tempR(i);
            (*ID)(i) = tempID(i);
        });

        /* Ghost NL Build - Halo exchange
         * 1. Figure out who to send particles to
         * 2. Build Send Buffer
         * 3. Recieve Particles
         */

        auto FL = this->fcontainer_m->getFL();
        auto neighbors = FL.getNeighbors();

        // TODO

    }

    void pre_step() override {
        Inform m("pre step");
        /* TODO */
        // initialize neighborlist?
    }

    void post_step() override {
        Inform m("post step");
        /* TODO */
        this->time_m += this->dt_m;
        this->it_m++;

        // print_every + dump?
    }

    void grid2par() override {
        gatherCIC();
    }

    void par2grid() override {
        scatterCIC();
    }

    void par2par() override {
        /* TODO */
    }


    void advance() override {
        LeapFrogStep();
    }

    void LeapFrogStep() {
        
        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        /* TODO */
    }

    void applyConstantFocusing() {
        /* TODO */
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
        double Q                        = Q_m;
        Vector_t<double, Dim> rmin	= rmin_m;
        Vector_t<double, Dim> rmax	= rmax_m;
        Vector_t<double, Dim> hr        = hr_m;

        scatter(*q, *rho, *R);
        double relError = std::fabs((Q-(*rho).sum())/Q);

        m << relError << endl;

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

        // rho = rho_e - rho_i (only if periodic BCs)
        double size = 1;
        for (unsigned d = 0; d < Dim; d++) {
            size *= rmax[d] - rmin[d];
        }
        *rho = *rho - (Q / size);
        
    }
};

#endif
