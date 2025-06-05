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
#include "P3M3DManager.h"
#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"
#include "P3MParticleContainer.hpp"

// Distribution functions
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/UniformDistribution.h"

#include "datatypes.h"

/**
 * @class P3M3DBenchManager
 * @brief A class that benchmarks the P3M Method
 * 
 * @tparam T the data dype for simulation variables
 * @tparam Dim the dimensionality of the simulation
*/
template <typename T, unsigned Dim>
class P3M3DBenchManager 
    : public P3M3DManager<T, Dim, FieldContainer<T, Dim>> {
public:
    using ParticleContainer_t = P3MParticleContainer<T, Dim>;
    using Base = P3M3DManager<T, Dim, FieldContainer<T, Dim> >;
    using FieldContainer_t = FieldContainer<T, Dim>;

protected:

    size_type totalP_m;         // Total number of particles
    int nt_m;                   // Total number of time steps
    T dt_m;                // Time step size
    Vector_t<int, Dim> nr_m;    // Domain granularity
    T rcut_m;              // Interaction cutoff radius
    std::string solver_m;       // solver is P3MSolver
    T beamRad_m;           // beam radius
    T focusingF_m;         // constant focusing force
    T boxlen_m;            // box length
    
public:
    P3M3DBenchManager(size_type totalP_, int nt_, T dt_, Vector_t<int, Dim>& nr_, T rcut_, T alpha_, T beamRad_, T focusingF_, T boxlen_)
        : P3M3DManager<T, Dim, FieldContainer<T, Dim> >()
        , totalP_m(totalP_), nt_m(nt_), dt_m(dt_), nr_m(nr_), rcut_m(rcut_), solver_m("P3M"), beamRad_m(beamRad_), focusingF_m(focusingF_), boxlen_m(boxlen_), alpha_m(alpha_)
        {
        }

    ~P3M3DBenchManager(){}

protected:
    T time_m;                  // Simulation time
    T it_m;                    // Iteration counter
    T alpha_m;                 // Green's function splitting parameter
    T epsilon_m;               // Regularization for PP interaction
    Vector_t<T, Dim> rmin_m;   // minimum domain extend
    Vector_t<T, Dim> rmax_m;   // maximum domain extend
    Vector_t<T, Dim> hr_m;     // PM Meshwidth
    Vector_t<int, Dim> nCells_m;    // Number of cells in each dimension
    T Q_m;                     // Particle Charge
    Vector_t<T, Dim> origin_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;    // Domain as index range
    std::array<bool, Dim> decomp_m; // Domain Decomposition
    T rhoNorm_m;               // Rho norm, required for scatterCIC

public: 
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    T getTime() const { return time_m; }

    void setTime(T time_) { time_m = time_; }

    void pre_run() override {
        Inform m("Pre Run");

        for (unsigned i = 0; i < Dim; ++i) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        // this->alpha_m = 2./this->rcut_m;
        T box_length = this->boxlen_m;
        this->rmin_m = -box_length/2.;
        this->rmax_m = box_length/2.;
        this->origin_m = rmin_m;
        this->isAllPeriodic_m = true;

        this->hr_m = box_length/(T)(this->nr_m[0]);

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
                this->fcontainer_m->getMesh(), this->fcontainer_m->getFL(), this->rcut_m
            )
        );

	    // std::cerr << "Device Space: " << Device::name() << std::endl;
	    // std::cerr << "Host Space: " << Host::name() << std::endl;

    
        this->fcontainer_m->initializeFields("P3M");

        // initialize solver
        ippl::ParameterList sp;
        sp.add("output_type", P3MSolver_t<T, Dim>::GRAD);
        sp.add("use_heffte_defaults", false);
        sp.add("use_pencils", true);
        sp.add("use_reorder", false);
        sp.add("use_gpu_aware", true);
        sp.add("comm", ippl::p2p_pl);
        sp.add("r2c_direction", 0);
        sp.add("alpha", this->alpha_m);
        sp.add("force_constant", static_cast<T>(2.532638e8)); // ke

        this->setFieldSolver(
            std::make_shared<P3MSolver_t<T, Dim>>(
                this->fcontainer_m->getE(), this->fcontainer_m->getRho(), sp
            )
        );

        ippl::ParameterList ppInteractionParams;
        ppInteractionParams.add("rcut", this->rcut_m);
        ppInteractionParams.add("alpha", this->alpha_m);
        ppInteractionParams.add("force_constant", static_cast<T>(2.532638e8)); // ke

        this->setInteractionSolver(
            std::make_shared<typename Base::PPInteraction>(
                *this->pcontainer_m, this->pcontainer_m->E, this->pcontainer_m->R, this->pcontainer_m->Q, ppInteractionParams
                )
            );

        T initTimerStart = MPI_Wtime();
        initializeParticles();
        T initTimerEnd = MPI_Wtime();
        std::cout << "Particle Initialization Time: " << initTimerEnd - initTimerStart << std::endl;

        this->fcontainer_m->getRho() = 0.0;

        T PMTimerStart = MPI_Wtime();
        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();
        T PMTimerEnd = MPI_Wtime();
        
        std::cout << "Field Solver Time: " << PMTimerEnd - PMTimerStart << std::endl;

        T PPTimerStart = MPI_Wtime();
	    this->isolver_m->solve();
        T PPTimerEnd = MPI_Wtime();
        std::cout << "PP Interaction Time: " << PPTimerEnd - PPTimerStart << std::endl;

	    // this->focusingF_m *= this->computeAvgSpaceChargeForces();
	    
        // this->pcontainer_m->update();

        std::cerr << "Pre Run finished" << std::endl;
    }
        
    void dump() {
        // return 0;
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
        
        auto hLocalRegions = this->pcontainer_m->getLayout().getRegionLayout().gethLocalRegions();
        Vector_t<T, Dim> domainMin, domainLength;
        
        for(unsigned d = 0; d < Dim; ++d){
            domainMin[d] = hLocalRegions(rank)[d].min();
            domainLength[d] = hLocalRegions(rank)[d].length();
        }

        Kokkos::fence();
	
	    // make sure this runs on the host, device does not work yet
        Kokkos::Random_XorShift64_Pool rand_pool((size_type)(42 + 24 * rank));

        IpplTimings::startTimer(GTimer);
        Kokkos::parallel_for("initialize particles", nloc,
            KOKKOS_LAMBDA(const size_t index) {
                Vector_t<T, Dim> x(0.0);

                auto generator = rand_pool.get_state();
                
                // obtain random numbers
                Vector_t<T, Dim> u;
                for(unsigned d = 0; d < Dim; ++d){
                    u[d] = generator.drand();
                }

                // for(int i = 0; i < Dim; ++i){
                //     x[i] = generator.normal(0.0, 1.0);
                // }

                rand_pool.free_state(generator);

                // calculate position
                // T normsq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                // if (sign < 0.5) u = -u;
                Vector_t<T, Dim> pos = domainMin + domainLength * u;

                for(unsigned d = 0; d < Dim; ++d){
                    P(index)[d] = 0;		// initialize with zero momentum
		            R(index)[d] = pos[d];
                }
                Q(index) = 1.0;
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


    void pre_step() override {
        Inform m("pre step");
    }

    void post_step() override {
        Inform m("post step");

        this->time_m += this->dt_m;
        this->it_m++;
    }

    void grid2par() override {
        gatherCIC();
    }

    void par2grid() override {
        scatterCIC();
    }


    void advance() override {
        LeapFrogStep();
    }

    void LeapFrogStep() {
        
        T dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        pc->R = pc->R + dt * pc->P;

        pc->update();

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

        this->isolver_m->solve();

        // this->applyConstantFocusing();

        pc->P = pc->P - dt * pc->E;

        std::cerr << "LeapFrog Step " << this->it_m << " Finished." << std::endl;

    }

    T computeAvgSpaceChargeForces() {
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

        std::cerr << "Average Space Charge Forces: " << avgE << std::endl;

        Vector_t<T, Dim> globE = 0.0;

        ippl::Comm->reduce(&avgE[0], &globE[0], 3, std::plus<T>(), 0);
        
        globE /= totalP;

        T focusingf = 0.0;
        for (unsigned d = 0; d < Dim; ++d) {
            focusingf += globE[d] * globE[d];
        }

        return std::sqrt(focusingf);
    }

    void applyConstantFocusing() {
        auto E = this->pcontainer_m->E.getView();
        auto R = this->pcontainer_m->R.getView();

        T beamRad = this->beamRad_m;
        T focusStrength = this->focusingF_m;
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

        ippl::ParticleAttrib<T> *q = &this->pcontainer_m->Q;
        typename ParticleContainer_t::particle_position_type *R = &this->pcontainer_m->R;
        Field_t<Dim> *rho               = &this->fcontainer_m->getRho();
        T Q                        = this->Q_m;
        Vector_t<T, Dim> rmin	= this->rmin_m;
        Vector_t<T, Dim> rmax	= this->rmax_m;
        Vector_t<T, Dim> hr        = this->hr_m;

        scatter(*q, *rho, *R);
        T relError = std::fabs((Q-(*rho).sum())/Q);

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

	    T cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<T>());
        (*rho)          = (*rho) / cellVolume;

        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i;
        T size = 1;
        for (unsigned d = 0; d < Dim; d++) {
            size *= rmax[d] - rmin[d];
        }
        *rho = *rho - (Q / size);
        
    }
};

#endif
