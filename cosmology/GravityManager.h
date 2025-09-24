#ifndef IPPL_GRAVITY_MANAGER_H
#define IPPL_GRAVITY_MANAGER_H

#include <memory>

#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "GravityParticleContainer.hpp"
#include "Manager/BaseManager.h"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

#include "mc-4-Initializer/InputParser.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

/**
 * @brief Manages the gravity simulation, including particles, fields, and load balancing.
 *
 * @tparam T Type of the particle attribute.
 * @tparam Dim Dimensionality of the configuration space.
 */
template <typename T, unsigned Dim>
class GravityManager : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>,
                                               FieldContainer<T, Dim>, LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base                = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

public:
    /**
     * @brief Constructor for GravityManager.
     *
     * @param totalP_ Total number of particles.
     * @param nt_ Number of time steps.
     * @param nr_ Number of grid points in each dimension.
     * @param lbt_ Load balance threshold.
     * @param solver_ Solver type.
     * @param stepMethod_ Time stepping method type.
     * @param par_ Inputfile parser 
     */
    GravityManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                   std::string& solver_, std::string& stepMethod_, initializer::InputParser par_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalancer<T, Dim>>()
	, parser_m(par_)
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_) {}

    /**
     * @brief Destructor for GravityManager.
     */
    ~GravityManager() {}

public:
    /**
     * @brief Folder for initial conditions.
     */
    std::string folder;

    /**
     * @brief Access to the input file with constants and simulation parameters.
     */
    initializer::InputParser parser_m;

  
    /**
     * @brief Get the total number of particles.
     *
     * @return Total number of particles.
     */
    size_type getTotalP() const { return totalP_m; }

    /**
     * @brief Set the total number of particles.
     *
     * @param totalP_ Total number of particles.
     */
    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    /**
     * @brief Get the number of time steps.
     *
     * @return Number of time steps.
     */
    int getNt() const { return nt_m; }

    /**
     * @brief Set the number of time steps.
     *
     * @param nt_ Number of time steps.
     */
    void setNt(int nt_) { nt_m = nt_; }

    /**
     * @brief Get the solver type.
     *
     * @return Solver type.
     */
    const std::string& getSolver() const { return solver_m; }

    /**
     * @brief Set the solver type.
     *
     * @param solver_ Solver type.
     */
    void setSolver(const std::string& solver_) { solver_m = solver_; }

    /**
     * @brief Get the load balance threshold.
     *
     * @return Load balance threshold.
     */
    double getLoadBalanceThreshold() const { return lbt_m; }

    /**
     * @brief Set the load balance threshold.
     *
     * @param lbt_ Load balance threshold.
     */
    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    /**
     * @brief Get the step method type.
     *
     * @return Step method type.
     */
    const std::string& getStepMethod() const { return stepMethod_m; }

    /**
     * @brief Set the step method type.
     *
     * @param stepMethod_ Step method type.
     */
    void setStepMethod(const std::string& stepMethod_) { stepMethod_m = stepMethod_; }

    /**
     * @brief Get the number of grid points in each dimension.
     *
     * @return Vector type of size Dim with the number of grid points in each dimension
     */
    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    /**
     * @brief Set the number of grid points in each dimension.
     *
     * @param nr_ Vector type of size Dim with the number of grid points in each dimension
     */
    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    /**
     * @brief Get the current simulation time.
     *
     * @return Current simulation time.
     */
    double getTime() const { return time_m; }

    /**
     * @brief Set the current simulation time.
     *
     * @param time_ Current simulation time.
     */
    void setTime(double time_) { time_m = time_; }

    /**
     * @brief Set the initial conditions folder.
     *
     * @param ic_folder Initial conditions folder, where the ICs that will be used for the
     * simulation are stored.
     */
    void setIC(std::string ic_folder) { folder = ic_folder; }

    // Calculation methods

    /**
     * @brief Calculate the time given a scaling factor.
     *
     * @param a Scaling factor.
     * @return Calculated time.
     */
    double calculateTime(double a) {
        return this->t_L
               * asinh(sqrt(pow(a, 3) * this->O_L
                            / this->O_m));  // inverse function of calculateScaling
    }

    /**
     * @brief Calculate the scaling factor given a time.
     *
     * @param t Time.
     * @return Calculated scaling factor.
     */
    double calculateScaling(double t) {
        return pow(this->O_m / this->O_L, 1. / 3.)
               * pow(sinh(t / this->t_L), 2. / 3.);  // https://arxiv.org/pdf/0803.0982.pdf (p. 6)
    }

    /**
     * @brief Calculate the Hubble parameter given a scaling factor.
     *
     * @param a Scaling factor.
     * @return Calculated Hubble parameter.
     */
    double calculateHubble(double a) {
        return this->Hubble0 * sqrt(this->O_m / pow(a, 3) + this->O_L);
    }

    /**
     * @brief Initialize the simulation time and related parameters.
     */
    void InitialiseTime() {
        Inform mes("Inititalise: ");
	parser_m.getByName("Omega_m", this->O_m);
	parser_m.getByName("Omega_L", this->O_L);
	// this->O_m      = 0.3;       // \todo need to from input file
        // this->O_L      = 0.7;       // \todo need to from input file
        this->t_L      = 2 / (3 * this->Hubble0 * sqrt(this->O_L));
        this->a_m      = 1 / (1 + this->z_m);
        this->Dloga    = 1. / (this->nt_m) * log((1 + this->z_m) / (1 + this->z_f));
        this->time_m   = this->calculateTime(this->a_m);
        this->Hubble_m = this->calculateHubble(this->a_m);  // Hubble parameter at starting time
        this->dt_m     = this->Dloga / this->Hubble_m;
        this->rho_crit0 =
            3 * this->Hubble0 * this->Hubble0 / (8 * M_PI * this->G);  // critical density today

        // Print initial parameters
        mes << "time: " << this->time_m << ", timestep: " << this->dt_m << endl;
        mes << "Dloga: " << this->Dloga << endl;
        mes << "z: " << this->z_m << ", scaling factor: " << this->a_m << endl;
        mes << "H0: " << this->Hubble0 << ", H_initial: " << this->Hubble_m << endl;
        mes << "critical density (today): " << this->rho_crit0 << endl;
    }

    /**
     * @brief Virtual method to dump data (default does nothing).
     */
    virtual void dump(){/* default does nothing */};

    // Step methods

    /**
     * @brief Pre-step method called before each simulation step.
     */
    void pre_step() override {
      //        Inform mes("Pre-step");
      //  mes << "Done" << endl;
    }

    /**
     * @brief Post-step method called after each simulation step.
     */
    void post_step() override {
        Inform mes("Post-step:");
        // Update time
        this->it_m++;
        this->a_m      = this->a_m * exp(this->Dloga);
        this->time_m   = calculateTime(this->a_m);
        this->z_m      = 1 / this->a_m - 1;
        this->Hubble_m = this->calculateHubble(this->a_m);
        // write solution to output file
        this->dump();

        // dynamic time step
        this->dt_m = this->Dloga / this->Hubble_m;

        mes << "Step: " << this->it_m;
        mes << " comological time: " << this->time_m << ", dt: " << this->dt_m << ", z: " << this->z_m
            << ", a: " << this->a_m << endl;
    }

    // Grid to particle and particle to grid methods

    /**
     * @brief Transfer data from grid to particles.
     */
    void grid2par() override { gatherCIC(); }

    /**
     * @brief Gather data using Cloud-In-Cell (CIC) method.
     */
    void gatherCIC() {
        gather(this->pcontainer_m->F, this->fcontainer_m->getF(), this->pcontainer_m->R);
    }

    /**
     * @brief Transfer data from particles to grid.
     */
    void par2grid() override { scatterCIC(); }

    /**
     * @brief Scatter particles using Cloud-In-Cell (CIC) method.
     */
    void scatterCIC() {
        Inform mes("scatter ");

        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<float>* m           = &this->pcontainer_m->m;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        Vector_t<double, Dim> rmin               = rmin_m;
        Vector_t<double, Dim> rmax               = rmax_m;
        Vector_t<double, Dim> hr                 = hr_m;

        scatter(*m, *rho, *R);
        double relError = std::fabs((M_m - (*rho).sum()) / M_m);

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 10.*Kokkos::Experimental::epsilon_v<float> ) {
                mes << "Time step: " << it_m << endl;
                mes << "Total particles in the sim. " << totalP_m << " "
                    << "after update: " << TotalParticles << endl;
                mes << "Rel. error in mass conservation: " << relError << endl;
                ippl::Comm->abort();
            }
        }

        // Convert mass assignment to actual mass density
        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)            = (*rho) / cellVolume;
        rhoNorm_m         = norm(*rho);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (M_m / size);
        }
    }

protected:
    size_type totalP_m;        ///< Total number of particles.
    int nt_m;                  ///< Number of time steps.
    Vector_t<int, Dim> nr_m;   ///< Number of grid points in each dimension.
    double lbt_m;              ///< Load balance threshold.
    std::string solver_m;      ///< Solver type.
    std::string stepMethod_m;  ///< Time stepping method type.

    double time_m;                   ///< Current simulation time. [s]
    double dt_m;                     ///< Time step size. [s]
    double a_m;                      ///< Scaling factor. [1]
    double Dloga;                    ///< Logarithmic increment of the scaling factor. [1]
    double Hubble_m;                 ///< Hubble constant at the current time [s^-1].
    double Hubble0;                  ///< Hubble constant today (73.8 km/sec/Mpc).
    double G;                        ///< Gravitational constant. [kpc^3/(Msun s^2)]
    double rho_crit0;                ///< Critical density today. [Msun/kpc^3]
    float  O_m;                      ///< Matter density parameter. [1]
    float  O_L;                      ///< Dark energy density parameter. [1]
    double t_L;                      ///< Characteristic time scale. [s]
    double z_m;                      ///< Initial redshift. [1]
    double z_f;                      ///< Final redshift.   [1]
    int it_m;                        ///< Current iteration number. [1]
    Vector_t<double, Dim> rmin_m;    ///< Minimum comoving coordinates [kpc/h].
    Vector_t<double, Dim> rmax_m;    ///< Maximum comoving coordinates [kpc/h].
    Vector_t<double, Dim> hr_m;      ///< Grid spacing in each dimension. [kpc/h]
    double M_m;                      ///< Total mass. [Msun]
    Vector_t<double, Dim> origin_m;  ///< Origin of the coordinate system. [kpc/h]
    bool isAllPeriodic_m;            ///< Flag indicating if all boundaries are periodic. [bool]
    bool isFirstRepartition_m;       ///< Flag indicating if this is the first repartition. [bool]
    ippl::NDIndex<Dim> domain_m;     ///< Domain index. [1]
    std::array<bool, Dim> decomp_m;  ///< Decomposition flags for each dimension. [bool]
    double rhoNorm_m;                ///< Normalized density. [1]
};
#endif
