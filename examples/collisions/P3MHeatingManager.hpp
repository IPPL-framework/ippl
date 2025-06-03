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
#include "P3M3DManager.h"
#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"
#include "P3MParticleContainer.hpp"

// Distribution functions
#include "Random/Distribution.h"
#include "Random/NormalDistribution.h"
#include "Random/UniformDistribution.h"

// Required Datatypes
template <typename T, unsigned Dim>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::FFTTruncatedGreenPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

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
    : public P3M3DManager<T, Dim, FieldContainer<T, Dim>> {
public:

    using ParticleContainer_t = P3MParticleContainer<T, Dim>;
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
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
        : P3M3DManager<T, Dim, FieldContainer<T, Dim> >()
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
                this->fcontainer_m->getMesh(), this->fcontainer_m->getFL(), this->rcut_m
            )
        );

	    std::cerr << "Device Space: " << Device::name() << std::endl;
	    std::cerr << "Host Space: " << Host::name() << std::endl;

    
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


        this->fcontainer_m->getRho() = 0.0;

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

	    // this->pcontainer_m->E = -1.0 * this->pcontainer_m->E;

	    this->isolver_m->solve();

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


    double computeRMSBeamSize(){
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

        return rms_x;
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

        // l2 norm
        double temperature = Kokkos::sqrt(globalTemperature[0] * globalTemperature[0] 
                            + globalTemperature[1] * globalTemperature[1] 
                            + globalTemperature[2] * globalTemperature[2]);

        std::cerr << "L2-Norm of Temperature: " << temperature << std::endl;
        // return temperature[0];
	
    }

    void computeBeamStatistics() {
        std::cerr << "Start Computing Beam Statistics" << std::endl;

        auto R = this->pcontainer_m->R.getView();
        auto nLoc = this->pcontainer_m->getLocalNum();
        auto P = this->pcontainer_m->P.getView();
        double beamRad = this->beamRad_m;

        Vector_t<double, 9> stats = 0.0;

        Kokkos::parallel_reduce("compute sigma x", nLoc,
            KOKKOS_LAMBDA(const size_type i, ippl::Vector<double, 9>& sum){
                sum[0] += R(i)[0] * R(i)[0];
                sum[1] += P(i)[0] * P(i)[0];
                sum[2] += R(i)[0] * P(i)[0];
                sum[3] += R(i)[1] * R(i)[1];
                sum[4] += P(i)[1] * P(i)[1];
                sum[5] += R(i)[1] * P(i)[1];
                sum[6] += R(i)[2] * R(i)[2];
                sum[7] += P(i)[2] * P(i)[2];
                sum[8] += R(i)[2] * P(i)[2];
            }, stats
        );

        // double global_xsq = 0.0;
        // double global_psq = 0.0;
        // double global_xpsq = 0.0;
        ippl::Vector<double, 9> global_stats = 0.0;

        // there must be a better way to do this
        ippl::Comm->reduce(&stats[0], &global_stats[0], 9, std::plus<double>(), 0);
    

        // double avg_xsq = global_xsq / this->totalP_m;
        // double avg_psq = global_psq / this->totalP_m;
        // double avg_xpsq = global_xpsq / this->totalP_m;
        global_stats /= this->totalP_m;

        double emit_x = Kokkos::sqrt(global_stats[0] * global_stats[1] - global_stats[2] * global_stats[2]);
        double emit_y = Kokkos::sqrt(global_stats[3] * global_stats[4] - global_stats[5] * global_stats[5]);
        double emit_z = Kokkos::sqrt(global_stats[6] * global_stats[7] - global_stats[8] * global_stats[8]);
        // double beta = avg_xsq / emit_x;
        // double sigma_x = Kokkos::sqrt(avg_xsq);
        // std::cerr << "Beam Statistics: " << std::endl;
        // std::cerr << "Sigma x: " << sigma_x << std::endl;
        std::cerr << "(Normalized) RMS Emittance: " << emit_x << " , " << emit_y << " , " << emit_z << std::endl;
    }

    void dump() override {
        Inform m("Dump");

        std::cerr << "Dumping data" << std::endl;

        double E_kin = calcKineticEnergy();
        std::cerr << "Dumping data, Kinetic Energy: " << E_kin << std::endl;

        // DEBUG output
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

        computeBeamStatistics();
        compute_temperature();
        computeRMSBeamSize();
    }

    void LeapFrogStep() {
        
        double dt                               = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        pc->R = pc->R + dt * pc->P;

        pc->update();

        this->par2grid();

        this->fsolver_m->solve();

        this->grid2par();

        this->isolver_m->solve();

        this->applyConstantFocusing();

        pc->P = pc->P - dt * pc->E;

        std::cerr << "LeapFrog Step " << this->it_m << " Finished." << std::endl;

    }

    double computeAvgSpaceChargeForces() {
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

        Vector_t<double, Dim> globE = 0.0;

        ippl::Comm->reduce(&avgE[0], &globE[0], 3, std::plus<double>(), 0);
        
        globE /= totalP;

        double focusingf = 0.0;
        for (unsigned d = 0; d < Dim; ++d) {
            focusingf += globE[d] * globE[d];
        }

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
