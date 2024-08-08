#ifndef IPPL_GRAVITY_MANAGER_H
#define IPPL_GRAVITY_MANAGER_H

#include <memory>
#include <cmath>
#include <string>
#include <array>
#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "GravityParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

/**
 * @brief Manages the gravity simulation, including particles, fields, and load balancing.
 * 
 * @tparam T Type of the particle attribute.
 * @tparam Dim Dimensionality of the particle space.
 */
template <typename T, unsigned Dim>
class GravityManager : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t = FieldSolver<T, Dim>;
    using LoadBalancer_t = LoadBalancer<T, Dim>;
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    /**
     * @brief Constructor for GravityManager.
     * 
     * @param totalP_ Total number of particles.
     * @param nt_ Number of time steps.
     * @param nr_ Number of grid points in each dimension.
     * @param lbt_ Load balance threshold.
     * @param solver_ Solver type.
     * @param stepMethod_ Step method type.
     */
    GravityManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_, std::string& solver_, std::string& stepMethod_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>()
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

    /**
     * @brief Folder for initial conditions.
     */
    std::string folder;

    // Getters and Setters

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
     * @return Number of grid points in each dimension.
     */
    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    /**
     * @brief Set the number of grid points in each dimension.
     * 
     * @param nr_ Number of grid points in each dimension.
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
     * @param ic_folder Initial conditions folder.
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
        return this->t_L * asinh(sqrt(pow(a, 3) * this->O_L / this->O_m)); // inverse function of calculateScaling
    }

    /**
     * @brief Calculate the scaling factor given a time.
     * 
     * @param t Time.
     * @return Calculated scaling factor.
     */
    double calculateScaling(double t) {
        return pow(this->O_m / this->O_L, 1. / 3.) * pow(sinh(t / this->t_L), 2. / 3.); // https://arxiv.org/pdf/0803.0982.pdf (p. 6)
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
        this->O_m = 0.3;
        this->O_L = 0.7;
        this->t_L = 2 / (3 * this->Hubble0 * sqrt(this->O_L));
        this->a_m = 1 / (1 + this->z_m);
        this->Dloga = 1. / (this->nt_m) * log((1 + this->z_m) / (1 + this->z_f));

        this->time_m = this->calculateTime(this->a_m);
        this->Hubble_m = this->calculateHubble(this->a_m); // Hubble parameter at starting time
        this->dt_m = this->Dloga / this->Hubble_m;

        this->rho_crit0 = 3 * this->Hubble0 * this->Hubble0 / (8 * M_PI * this->G); // critical density today

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
    virtual void dump() { /* default does nothing */ }

    // Step methods

    /**
     * @brief Pre-step method called before each simulation step.
     */
    void pre_step() override {
        Inform mes("Pre-step");
        mes << "Done" << endl;
    }

    /**
     * @brief Post-step method called after each simulation step.
     */
    void post_step() override {
        Inform mes("Post-step:");
        this->it_m++;
        this->a_m = this->a_m * exp(this->Dloga);
        this->time_m = calculateTime(this->a_m);
        this->z_m = 1 / this->a_m - 1;
        this->Hubble_m = this->calculateHubble(this->a_m);
        this->dump();

        // Dynamic time step
        this->dt_m = this->Dloga / this->Hubble_m;

        mes << "Finished time step: " << this->it_m << endl;
        mes << " time: " << this->time_m << ", timestep: " << this->dt_m << ", z: " << this->z_m << ", a: " << this->a_m << endl;
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
        mes << "starting ..." << endl;
        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* m = &this->pcontainer_m->m;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho = &this->fcontainer_m->getRho();
        Vector_t<double, Dim> rmin = rmin_m;
        Vector_t<double, Dim> rmax = rmax_m;
        Vector_t<double, Dim> hr = hr_m;

        scatter(*m, *rho, *R);
        double relError = std::fabs((M_m - (*rho).sum()) / M_m);
        mes << "relative error: " << relError << endl;

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 1e-10) {
                mes << "Time step: " << it_m << endl;
                mes << "Total particles in the sim. " << totalP_m << " "
                    << "after update: " << TotalParticles << endl;
                mes << "Rel. error in mass conservation: " << relError << endl;
                ippl::Comm->abort();
            }
        }

        // Convert mass assignment to actual mass density
        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho) = (*rho) / cellVolume;
        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (M_m / size);
        }
    }

    /**
     * @brief Get the minimum coordinates of the domain.
     * 
     * @return const Vector_t<double, Dim>& Minimum coordinates.
     */
    const Vector_t<double, Dim>& getRmin() const { return rmin_m; }

    /**
     * @brief Set the minimum coordinates of the domain.
     * 
     * @param rmin_ Minimum coordinates.
     */
    void setRmin(const Vector_t<double, Dim>& rmin_) { rmin_m = rmin_; }

    /**
     * @brief Get the maximum coordinates of the domain.
     * 
     * @return const Vector_t<double, Dim>& Maximum coordinates.
     */
    const Vector_t<double, Dim>& getRmax() const { return rmax_m; }

    /**
     * @brief Set the maximum coordinates of the domain.
     * 
     * @param rmax_ Maximum coordinates.
     */
    void setRmax(const Vector_t<double, Dim>& rmax_) { rmax_m = rmax_; }

    /**
     * @brief Get the grid spacing.
     * 
     * @return const Vector_t<double, Dim>& Grid spacing.
     */
    const Vector_t<double, Dim>& getHr() const { return hr_m; }

    /**
     * @brief Set the grid spacing.
     * 
     * @param hr_ Grid spacing.
     */
    void setHr(const Vector_t<double, Dim>& hr_) { hr_m = hr_; }

    /**
     * @brief Get the total mass.
     * 
     * @return double Total mass.
     */
    double getM() const { return M_m; }

    /**
     * @brief Set the total mass.
     * 
     * @param M_ Total mass.
     */
    void setM(double M_) { M_m = M_; }

    /**
     * @brief Get the origin coordinates.
     * 
     * @return const Vector_t<double, Dim>& Origin coordinates.
     */
    const Vector_t<double, Dim>& getOrigin() const { return origin_m; }

    /**
     * @brief Set the origin coordinates.
     * 
     * @param origin_ Origin coordinates.
     */
    void setOrigin(const Vector_t<double, Dim>& origin_) { origin_m = origin_; }

    /**
     * @brief Check if all boundary conditions are periodic.
     * 
     * @return bool True if all boundary conditions are periodic, false otherwise.
     */
    bool getIsAllPeriodic() const { return isAllPeriodic_m; }

    /**
     * @brief Set if all boundary conditions are periodic.
     * 
     * @param isAllPeriodic_ True if all boundary conditions are periodic, false otherwise.
     */
    void setIsAllPeriodic(bool isAllPeriodic_) { isAllPeriodic_m = isAllPeriodic_; }

    /**
     * @brief Check if this is the first repartition.
     * 
     * @return bool True if this is the first repartition, false otherwise.
     */
    bool getIsFirstRepartition() const { return isFirstRepartition_m; }

    /**
     * @brief Set if this is the first repartition.
     * 
     * @param isFirstRepartition_ True if this is the first repartition, false otherwise.
     */
    void setIsFirstRepartition(bool isFirstRepartition_) { isFirstRepartition_m = isFirstRepartition_; }

    /**
     * @brief Get the domain index.
     * 
     * @return const ippl::NDIndex<Dim>& Domain index.
     */
    const ippl::NDIndex<Dim>& getDomain() const { return domain_m; }

    /**
     * @brief Set the domain index.
     * 
     * @param domain_ Domain index.
     */
    void setDomain(const ippl::NDIndex<Dim>& domain_) { domain_m = domain_; }

    /**
     * @brief Get the decomposition array.
     * 
     * @return const std::array<bool, Dim>& Decomposition array.
     */
    const std::array<bool, Dim>& getDecomp() const { return decomp_m; }

    /**
     * @brief Set the decomposition array.
     * 
     * @param decomp_ Decomposition array.
     */
    void setDecomp(const std::array<bool, Dim>& decomp_) { decomp_m = decomp_; }

    /**
     * @brief Get the norm of the density field.
     * 
     * @return double Norm of the density field.
     */
    double getRhoNorm() const { return rhoNorm_m; }

    /**
     * @brief Set the norm of the density field.
     * 
     * @param rhoNorm_ Norm of the density field.
     */
    void setRhoNorm(double rhoNorm_) { rhoNorm_m = rhoNorm_; }


    /**
     * @brief Set the current simulation time.
     * 
     * @param time_ Current simulation time.
     */
    void setTime(double time_) { time_m = time_; }

    /**
     * @brief Get the time step.
     * 
     * @return double Time step.
     */
    double getDt() const { return dt_m; }

    /**
     * @brief Set the time step.
     * 
     * @param dt_ Time step.
     */
    void setDt(double dt_) { dt_m = dt_; }

    /**
     * @brief Get the scaling factor.
     * 
     * @return double Scaling factor.
     */
    double getA() const { return a_m; }

    /**
     * @brief Set the scaling factor.
     * 
     * @param a_ Scaling factor.
     */
    void setA(double a_) { a_m = a_; }

    /**
     * @brief Get the logarithmic time step.
     * 
     * @return double Logarithmic time step.
     */
    double getDloga() const { return Dloga; }

    /**
     * @brief Set the logarithmic time step.
     * 
     * @param Dloga_ Logarithmic time step.
     */
    void setDloga(double Dloga_) { Dloga = Dloga_; }

    /**
     * @brief Get the Hubble constant.
     * 
     * @return double Hubble constant.
     */
    double getHubble() const { return Hubble_m; }

    /**
     * @brief Set the Hubble constant.
     * 
     * @param Hubble_ Hubble constant.
     */
    void setHubble(double Hubble_) { Hubble_m = Hubble_; }

    /**
     * @brief Get the Hubble constant at present time.
     * 
     * @return double Hubble constant at present time.
     */
    double getHubble0() const { return Hubble0; }

    /**
     * @brief Set the Hubble constant at present time.
     * 
     * @param Hubble0_ Hubble constant at present time.
     */
    void setHubble0(double Hubble0_) { Hubble0 = Hubble0_; }

    /**
     * @brief Get the gravitational constant.
     * 
     * @return double Gravitational constant.
     */
    double getG() const { return G; }

    /**
     * @brief Set the gravitational constant.
     * 
     * @param G_ Gravitational constant.
     */
    void setG(double G_) { G = G_; }

    /**
     * @brief Get the critical density at present time.
     * 
     * @return double Critical density at present time.
     */
    double getRhoCrit0() const { return rho_crit0; }

    /**
     * @brief Set the critical density at present time.
     * 
     * @param rho_crit0_ Critical density at present time.
     */
    void setRhoCrit0(double rho_crit0_) { rho_crit0 = rho_crit0_; }

    /**
     * @brief Get the matter density parameter.
     * 
     * @return double Matter density parameter.
     */
    double getOm() const { return O_m; }

    /**
     * @brief Set the matter density parameter.
     * 
     * @param O_m_ Matter density parameter.
     */
    void setOm(double O_m_) { O_m = O_m_; }

    /**
     * @brief Get the dark energy density parameter.
     * 
     * @return double Dark energy density parameter.
     */
    double getOL() const { return O_L; }

    /**
     * @brief Set the dark energy density parameter.
     * 
     * @param O_L_ Dark energy density parameter.
     */
    void setOL(double O_L_) { O_L = O_L_; }

    /**
     * @brief Get the lookback time.
     * 
     * @return double Lookback time.
     */
    double getTL() const { return t_L; }

    /**
     * @brief Set the lookback time.
     * 
     * @param t_L_ Lookback time.
     */
    void setTL(double t_L_) { t_L = t_L_; }

    /**
     * @brief Get the initial redshift.
     * 
     * @return double Initial redshift.
     */
    double getZm() const { return z_m; }

    /**
     * @brief Set the initial redshift.
     * 
     * @param z_m_ Initial redshift.
     */
    void setZm(double z_m_) { z_m = z_m_; }

    /**
     * @brief Get the final redshift.
     * 
     * @return double Final redshift.
     */
    double getZf() const { return z_f; }

    /**
     * @brief Set the final redshift.
     * 
     * @param z_f_ Final redshift.
     */
    void setZf(double z_f_) { z_f = z_f_; }
        
    /**
     * @brief Get the initial redshift (z_m).
     * 
     * @return The initial redshift.
     */
    double getZm() const { return z_m; }

    /**
     * @brief Set the initial redshift (z_m).
     * 
     * @param z_m_ The initial redshift to set.
     */
    void setZm(double z_m_) { z_m = z_m_; }

    /**
     * @brief Get the final redshift (z_f).
     * 
     * @return The final redshift.
     */
    double getZf() const { return z_f; }

    /**
     * @brief Set the final redshift (z_f).
     * 
     * @param z_f_ The final redshift to set.
     */
    void setZf(double z_f_) { z_f = z_f_; }

    /**
     * @brief Get the iteration count (it_m).
     * 
     * @return The iteration count.
     */
    int getIt() const { return it_m; }

    /**
     * @brief Set the iteration count (it_m).
     * 
     * @param it_ The iteration count to set.
     */
    void setIt(int it_) { it_m = it_; }

    /**
     * @brief Get the folder path.
     * 
     * @return The folder path as a string.
     */
    const std::string& getFolder() const { return folder; }

    /**
     * @brief Set the folder path.
     * 
     * @param folder_ The folder path to set.
     */
    void setFolder(const std::string& folder_) { folder = folder_; }

protected:
    size_type totalP_m; ///< Total number of particles.
    int nt_m; ///< Number of time steps.
    Vector_t<int, Dim> nr_m; ///< Number of particles in each dimension.
    double lbt_m; ///< Lookback time.
    std::string solver_m; ///< Solver method.
    std::string stepMethod_m; ///< Step method.

    double time_m; ///< Current time.
    double dt_m; ///< Time step.
    double a_m; ///< Scaling factor.
    double Dloga; ///< Logarithmic scale factor increment.
    double Hubble_m; ///< Hubble constant [s^-1].
    double Hubble0; ///< Hubble constant at present time (73.8 km/sec/Mpc).
    double G; ///< Gravitational constant.
    double rho_crit0; ///< Critical density at present time.
    double O_m; ///< Matter density parameter.
    double O_L; ///< Dark energy density parameter.
    double t_L; ///< Lookback time.
    double z_m; ///< Initial redshift.
    double z_f; ///< Final redshift.
    int it_m; ///< Iteration count.
    Vector_t<double, Dim> rmin_m; ///< Minimum comoving coordinates [kpc/h].
    Vector_t<double, Dim> rmax_m; ///< Maximum comoving coordinates [kpc/h].
    Vector_t<double, Dim> hr_m; ///< Grid spacing in comoving coordinates.
    double M_m; ///< Mass of the particles.
    Vector_t<double, Dim> origin_m; ///< Origin of the coordinate system.
    bool isAllPeriodic_m; ///< Flag indicating if all boundaries are periodic.
    bool isFirstRepartition_m; ///< Flag indicating if this is the first repartition.
    ippl::NDIndex<Dim> domain_m; ///< Domain index.
    std::array<bool, Dim> decomp_m; ///< Decomposition flags for each dimension.
    double rhoNorm_m; ///< Normalized density.
    };

    #endif // IPPL_GRAVITY_MANAGER_H