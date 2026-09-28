/**
 * @file FreeElectronLaserManager.h
 * @brief Boosted-frame FEL PIC orchestration, particle coupling and diagnostics.
 * @ingroup fel_runtime
 */
#ifndef IPPL_FREE_ELECTRON_LASER_MANAGER_H
#define IPPL_FREE_ELECTRON_LASER_MANAGER_H

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include "Manager/BaseManager.h"

#include "Interpolation/CurrentDeposition.hpp"

#include "Config.h"
#include "FELFieldContainer.hpp"
#include "FELParticleContainer.hpp"
#include "LorentzTransform.h"
#include "MithraBunch.h"
#include "Undulator.h"
#include "datatypes.h"
#include "units.h"

#ifdef IPPL_ENABLE_CATALYST
#include "Stream/InSitu/CatalystAdaptor.h"
#endif

/**
 * @brief Coordinate the active collocated-potential FEL simulation on all MPI ranks.
 * @ingroup fel_runtime
 * @tparam T Floating-point scalar; the shipped executable uses double.
 * @tparam Dim Mesh dimension; this manager's vectors/kernels require Dim = 3.
 *
 * Call pre_run() once, then inherited run(getNt()). Each iteration deposits the
 * preceding particle displacement, advances the active nonstandard FDTD solver,
 * optionally snapshots fields with Catalyst, pushes particles in three Boris
 * substeps, removes escaped particles and migrates survivors. post_step() then
 * advances the manager clock and writes diagnostics. The solver evolves
 * collocated four-potentials and reconstructs E/B; it is not a Yee E/B update.
 *
 * The boost is along z with gamma_frame = max(1, gamma_bunch/sqrt(1+K^2/2)).
 * State uses the normalized c=1 units of units.h in the boosted frame. External
 * undulator fields are evaluated in the laboratory and transformed for the push.
 * Radiation diagnostics transform E/B back, but retain their implemented sample
 * positions/time convention; they are not a general laboratory detector model.
 *
 * Kokkos kernels operate on rank-local particle/field views; IPPL exchanges halos
 * and migrates registered particle attributes. MPI reductions produce diagnostic
 * scalars for rank-zero output. No dielectric response or initial equilibrium
 * self-field solve is provided. Charge continuity, boundary loss, numerical
 * dispersion and diagnostic normalization require separate physics validation.
 */
template <typename T, unsigned Dim>
class FreeElectronLaserManager : public ippl::BaseManager {
public:
    /// @brief Particle storage and spatial ownership used by this manager.
    using ParticleContainer_t = FELParticleContainer<T, Dim>;
    /// @brief Cartesian field storage used for E, B and the four-current.
    using FieldContainer_t    = FELFieldContainer<T, Dim>;
    /// @brief Active solver alias from datatypes.h (nonstandard FDTD with Mur boundaries).
    using FDTDSolver_t        = ::FDTDSolver_t<T, Dim>;
    /// @brief Particle-base alias retained for application-level compatibility.
    using Base                = ippl::ParticleBase<PLayout_t<T, Dim>>;

    /**
     * @brief Retain configuration and construct the boost/undulator models.
     * @param cfg Parsed configuration already converted to normalized code units.
     * Mesh, fields and particles are allocated later by pre_run().
     */
    FreeElectronLaserManager(config cfg)
        : m_config(cfg)
        , totalP_m(cfg.num_particles)
        , nt_m(0)
        , time_m(0.0)
        , dt_m(0.0)
        , it_m(0)
        , frame_gamma_m(std::max(
              T(1), cfg.bunch_gamma
                        / std::sqrt(1 + cfg.undulator_K * cfg.undulator_K * T(0.5))))
        , uparams_m(cfg.undulator_K, cfg.undulator_period, cfg.undulator_length)
        , frame_m(ippl::UniaxialLorentzframe<T, 2>::from_gamma(frame_gamma_m))
        , undulator_m(uparams_m, 2.0 * cfg.sigma_position[2] * frame_gamma_m * frame_gamma_m) {}

    /// @brief Release owned simulation handles; the executable finalizes Catalyst/MPI.
    ~FreeElectronLaserManager() = default;

protected:
    config m_config; ///< Local configuration copy; pre_run() changes z extent and duration for the boost.

    size_type totalP_m;        ///< Requested particle count, not the live count after generation/loss.
    int nt_m;                 ///< Number of field steps derived as ceil(boosted duration/dt).
    Vector_t<int, Dim> nr_m;   ///< Global owned-cell counts, excluding halos.

    double time_m; ///< Boosted-frame manager time in normalized units.
    double dt_m;   ///< Field time step obtained from the active solver; currently h_z for c=1.
    int it_m;      ///< Completed field-step count, also used for the diagnostic ring buffer.

    Vector_t<double, Dim> rmin_m;   ///< Lower physical box corner in boosted-frame code units.
    Vector_t<double, Dim> rmax_m;   ///< Upper physical box corner in boosted-frame code units.
    Vector_t<double, Dim> hr_m;     ///< Uniform grid spacings in boosted-frame code length units.
    Vector_t<double, Dim> origin_m; ///< Mesh origin; pre_run() centers the box around zero.
    ippl::NDIndex<Dim> domain_m;    ///< Global index box with nr_m owned cells.
    std::array<bool, Dim> decomp_m; ///< MPI split directions; active setup enables only z.
    bool isAllPeriodic_m;          ///< False for the active open-boundary FEL layout.

    std::shared_ptr<FieldContainer_t> fcontainer_m;       ///< Shared mesh/E/B/J storage, set during pre_run().
    std::shared_ptr<ParticleContainer_t> pcontainer_m;     ///< Shared particle storage, set during pre_run().
    std::shared_ptr<FDTDSolver_t> solver_m;                ///< Solver owning potential histories and referencing E/B/J.

    int nsubsteps_m = 3;  ///< Boris sub-steps per FDTD step (reference behaviour).

    T frame_gamma_m;                            ///< Lorentz factor of the co-moving frame.
    ippl::undulator_parameters<T> uparams_m;    ///< Undulator parameters.
    ippl::UniaxialLorentzframe<T, 2> frame_m;   ///< Boost into the co-moving frame (z-axis).
    ippl::Undulator<T> undulator_m;             ///< Static undulator field model.

    // --- narrow-band (resonant) radiation power diagnostic state ---
    // The MITHRA-inspired diagnostic uses a rolling single-frequency DFT rather
    // than broadband flux. Its sampling convention is described at the method;
    // matching MITHRA results is a separate validation question.
    bool rp_init_m    = false;     ///< whether rp_fdt_m has been allocated yet
    int rp_Nf_m       = 0;         ///< DFT window length [time steps] (~3 resonant cycles)
    double rp_omega_m = 0.0;       ///< resonant angular frequency [1/unit_time], c = 1
    Kokkos::View<T****> rp_fdt_m;  ///< ring buffer [Nf][nx][ny][4] = (Ex,Ey,Bx,By)_lab

public:
#ifdef IPPL_ENABLE_CATALYST
    /// @brief Optional in-situ adapter; initialized here and finalized by the executable.
    ippl::CatalystAdaptor cat_viz{std::string{"FreeElectronLaser"}};
#endif

    /// @brief Return the requested particle-count metadata, not the current live count.
    size_type getTotalP() const { return totalP_m; }
    /// @brief Set count metadata only; particle storage and generator configuration are unchanged.
    /// @param totalP_ Replacement requested-count metadata.
    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    /// @brief Return the field-step count derived during pre_run().
    int getNt() const { return nt_m; }
    /// @brief Set the step-count metadata subsequently read by the run caller.
    /// @param nt_ Replacement number of field steps.
    void setNt(int nt_) { nt_m = nt_; }

    /// @brief Return global grid-count metadata.
    const Vector_t<int, Dim>& getNr() const { return nr_m; }
    /// @brief Set grid-count metadata only; this does not rebuild the mesh or solver.
    /// @param nr_ Replacement global cell counts.
    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    /// @brief Return manager time in boosted-frame code units.
    double getTime() const { return time_m; }
    /// @brief Set the manager clock only; fields, particle state and iteration are unchanged.
    /// @param time_ Replacement time in boosted-frame code units.
    void setTime(double time_) { time_m = time_; }

    /// @brief Return shared particle storage, null before setup unless explicitly assigned.
    std::shared_ptr<ParticleContainer_t> getParticleContainer() { return pcontainer_m; }
    /// @brief Assign particle storage without migrating or initializing its contents.
    /// @param pcontainer Container consistent with the current mesh/layout.
    void setParticleContainer(std::shared_ptr<ParticleContainer_t> pcontainer) {
        pcontainer_m = pcontainer;
    }

    /// @brief Return shared field storage, null before setup unless explicitly assigned.
    std::shared_ptr<FieldContainer_t> getFieldContainer() { return fcontainer_m; }
    /// @brief Assign field storage; existing solver references are not rebound.
    /// @param fcontainer Container consistent with the particles and solver.
    void setFieldContainer(std::shared_ptr<FieldContainer_t> fcontainer) {
        fcontainer_m = fcontainer;
    }

    /// @brief Return the active field-solver handle.
    std::shared_ptr<FDTDSolver_t> getFieldSolver() { return solver_m; }
    /// @brief Assign the solver handle without updating time-step metadata.
    /// @param solver Solver already bound to the intended E/B/J fields.
    void setFieldSolver(std::shared_ptr<FDTDSolver_t> solver) { solver_m = solver; }

    /// @brief Log entry into a step; this hook performs no numerical preparation.
    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    /// @brief Increment time/iteration, run all diagnostics and log completion.
    void post_step() override {
        this->time_m += this->dt_m;
        this->it_m++;
        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

    /// @brief Active particle-to-grid wrapper invoking depositCurrent().
    void par2grid() { depositCurrent(); }

    /// @brief Standalone gather wrapper; advance() instead gathers inside each push substep.
    void grid2par() { gatherFields(); }

    /**
     * @brief Reset J, deposit the previous displacement and sum halo contributions.
     *
     * assemble_current_collocated() splits each local R_nm1-to-R trajectory and
     * deposits midpoint CIC current weights into J[1..Dim]. Optional CIC charge
     * density is added to J[0] when space_charge is enabled; otherwise J[0] stays
     * zero. Kokkos atomic additions scatter into rank-local storage, then IPPL
     * accumulateHalo() sums shared contributions across ranks.
     *
     * The displacement is from the previous complete field step, not each Boris
     * substep separately. Exact discrete charge continuity is not established by
     * using trajectory segments alone and must be checked with the solver stencil.
     */
    void depositCurrent() {
        using value_type = typename SourceField_t<T, Dim>::value_type;
        this->fcontainer_m->getJ() = value_type(0);

        auto policy = Kokkos::RangePolicy<>(0, this->pcontainer_m->getLocalNum());
        ippl::assemble_current_collocated(this->fcontainer_m->getMesh(), this->pcontainer_m->Q,
                                          this->pcontainer_m->R_nm1, this->pcontainer_m->R,
                                          this->fcontainer_m->getJ(), policy, (T)this->dt_m);

        if (this->m_config.space_charge) {
            depositChargeDensity();
        }

        this->fcontainer_m->getJ().accumulateHalo();
    }

    /**
     * @brief Fill E/B halos and interpolate their current grid values to local R.
     *
     * Particle gather kernels use the existing field centering. The false gather
     * argument overwrites each gathered attribute rather than adding to it.
     * IPPL gather also fills the field halo internally, so these explicit fills
     * are additional exchanges in the current implementation.
     * The active push uses equivalent gathers repeatedly at substep positions.
     */
    void gatherFields() {
        this->fcontainer_m->getE().fillHalo();
        this->fcontainer_m->getB().fillHalo();
        this->pcontainer_m->E_gather.gather(this->fcontainer_m->getE(), this->pcontainer_m->R,
                                            false);
        this->pcontainer_m->B_gather.gather(this->fcontainer_m->getB(), this->pcontainer_m->R,
                                            false);
    }

    /**
     * @brief Advance local particles through one field step with a relativistic Boris push.
     * @tparam External Device-callable external-field functor captured by value.
     * @param external_field Callable (position, time) returning a Kokkos pair of
     * boosted-frame E and B three-vectors in code units.
     *
     * Saves R into R_nm1, explicitly fills E/B halos, and uses nsubsteps_m substeps.
     * Each IPPL gather performs its own halo fill as well.
     * Grid fields remain at the same time level but are regathered at each new
     * position. The external field is reevaluated at time_m + substep*dt/nsubsteps.
     * Kokkos updates dimensionless gamma*beta and R, with fences between stages.
     * Particles outside/on the box faces are removed, then update() migrates the
     * survivors and all registered attributes to their owning MPI ranks.
     *
     * @pre Containers and dt_m have been initialized. Substep trajectories must
     * remain within the local interpolation support until the final migration.
     * Particle removal is an open-boundary loss, not a closed-system conservation
     * operation; removed trajectories are absent from the next current deposit.
     */
    template <class External>
    void push(External external_field) {
        auto pc = this->pcontainer_m;
        auto fc = this->fcontainer_m;

        // Remember the pre-push position; it is the start point for next step's
        // current deposition (R_nm1 -> R).
        Kokkos::deep_copy(pc->R_nm1.getView(), pc->R.getView());

        fc->getE().fillHalo();
        fc->getB().fillHalo();
        Kokkos::fence();

        const T dt       = (T)this->dt_m;
        const int nsub   = nsubsteps_m;
        const T bunch_dt = dt / nsub;
        const T time     = (T)this->time_m;

        for (int bts = 0; bts < nsub; ++bts) {
            pc->E_gather.gather(fc->getE(), pc->R, false);
            pc->B_gather.gather(fc->getB(), pc->R, false);
            Kokkos::fence();

            auto gbview = pc->gamma_beta.getView();
            auto eview  = pc->E_gather.getView();
            auto bview  = pc->B_gather.getView();
            auto qview  = pc->Q.getView();
            auto mview  = pc->mass.getView();
            auto rview  = pc->R.getView();

            Kokkos::parallel_for(
                pc->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
                    const ippl::Vector<T, 3> pgammabeta = gbview(i);
                    ippl::Vector<T, 3> E_grid           = eview(i);
                    ippl::Vector<T, 3> B_grid           = bview(i);
                    ippl::Vector<T, 3> bunchpos         = rview(i);

                    Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> external_eb =
                        external_field(bunchpos, time + bunch_dt * bts);

                    ippl::Vector<ippl::Vector<T, 3>, 2> EB{
                        ippl::Vector<T, 3>(E_grid + external_eb.first),
                        ippl::Vector<T, 3>(B_grid + external_eb.second)};

                    const T charge = qview(i);
                    const T mass   = mview(i);

                    const ippl::Vector<T, 3> t1 =
                        pgammabeta + charge * bunch_dt * EB[0] / (T(2) * mass);
                    const T alpha =
                        charge * bunch_dt / (T(2) * mass * Kokkos::sqrt(1 + t1.dot(t1)));
                    const ippl::Vector<T, 3> t2 = t1 + alpha * ippl::cross(t1, EB[1]);
                    const ippl::Vector<T, 3> t3 =
                        t1
                        + ippl::cross(t2, T(2) * alpha
                                              * (EB[1] / (1.0 + alpha * alpha * (EB[1].dot(EB[1])))));
                    const ippl::Vector<T, 3> ngammabeta =
                        t3 + charge * bunch_dt * EB[0] / (T(2) * mass);

                    rview(i) = rview(i)
                               + bunch_dt * ngammabeta
                                     / (Kokkos::sqrt(T(1.0) + ngammabeta.dot(ngammabeta)));
                    gbview(i) = ngammabeta;
                });
            Kokkos::fence();
        }

        destroyOutOfBounds();
        pc->update();
    }

    /**
     * @brief Collectively set up the boosted mesh, fields, solver and particle bunch.
     *
     * Rank zero creates the output directory and broadcasts success. The copied
     * configuration is then changed in place: z extent is multiplied by the
     * frame gamma, and duration divided by it. The centered mesh is nonperiodic
     * and decomposed only along z. The active stencil requires
     * (h_z/h_x)^2 + (h_z/h_y)^2 < 1; failure aborts the MPI communicator.
     *
     * Allocates E/B/J and particles, constructs the solver, reads its dt=h_z,
     * initializes/migrates the bunch, initializes optional Catalyst registries,
     * and writes initial diagnostics. Solver potential histories start at zero;
     * no electrostatic startup solve is performed.
     * @pre IPPL/MPI/Kokkos are initialized and all ranks call this method once.
     * Repeated calls would apply the frame rescaling again.
     * @throws IpplException If the output directory cannot be created.
     */
    void pre_run() override {
        Inform m("Pre Run");

        int outputDirectoryReady = 1;
        if (ippl::Comm->rank() == 0) {
            std::error_code error;
            std::filesystem::create_directories(this->m_config.output_path, error);
            if (error) {
                outputDirectoryReady = 0;
                m << "Unable to create output directory '" << this->m_config.output_path
                  << "': " << error.message() << endl;
            }
        }
        MPI_Bcast(&outputDirectoryReady, 1, MPI_INT, 0, ippl::Comm->getCommunicator());
        if (!outputDirectoryReady) {
            throw IpplException("FreeElectronLaserManager::pre_run",
                                "Unable to create the configured output directory");
        }

        // The longitudinal box and the simulated time are measured in the
        // co-moving frame: stretch z and shorten the time accordingly.
        this->m_config.extents[2] *= frame_gamma_m;
        this->m_config.total_time /= frame_gamma_m;

        for (unsigned i = 0; i < Dim; i++) {
            this->nr_m[i]     = this->m_config.resolution[i];
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        // Open boundaries (absorbing FDTD); decompose along z only.
        this->decomp_m.fill(false);
        this->decomp_m[Dim - 1] = true;
        this->isAllPeriodic_m   = false;

        for (unsigned d = 0; d < Dim; d++) {
            this->hr_m[d]     = this->m_config.extents[d] / this->m_config.resolution[d];
            this->origin_m[d] = -this->m_config.extents[d] * 0.5;
            this->rmin_m[d]   = this->origin_m[d];
            this->rmax_m[d]   = this->origin_m[d] + this->m_config.extents[d];
        }

        // Mesh-aspect condition required by the active nonstandard FDTD stencil.
        const double rzx = this->hr_m[Dim - 1] / this->hr_m[0];
        const double rzy = this->hr_m[Dim - 1] / this->hr_m[1];
        if (rzx * rzx + rzy * rzy >= 1.0) {
            m << "Dispersion relation not satisfiable" << endl;
            ippl::Comm->abort();
        }

        m << "Discretization:" << endl
          << "nt " << "(derived)" << " Np= " << this->totalP_m << " grid = " << this->nr_m
          << endl;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m,
            this->origin_m, this->isAllPeriodic_m));

        this->fcontainer_m->initializeFields();

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        // The FDTD solver derives its own time step from the mesh (CFL); we read
        // it back to drive the particle push and diagnostics.
        this->setFieldSolver(std::make_shared<FDTDSolver_t>(this->fcontainer_m->getJ(),
                                                            this->fcontainer_m->getE(),
                                                            this->fcontainer_m->getB()));
        this->dt_m   = this->solver_m->getDt();
        this->nt_m   = (int)std::ceil(this->m_config.total_time / this->dt_m);
        this->it_m   = 0;
        this->time_m = 0.0;

        m << "dt = " << this->dt_m << " nt = " << this->nt_m << endl;

        initializeParticles();

#ifdef IPPL_ENABLE_CATALYST
        auto runtime_vis_registry = ippl::MakeVisRegistryRuntimePtr(
            "Particles", this->pcontainer_m,
            "EField",    this->fcontainer_m->getE(),
            "Bfield",    this->fcontainer_m->getB()
        );
        auto runtime_steer_registry = ippl::MakeVisRegistryRuntimePtr();
        cat_viz.Initialize(runtime_vis_registry, runtime_steer_registry);
#endif

        this->dump();

        m << "Done" << endl;
    }

    /**
     * @brief Generate the bunch on rank zero, center it and distribute by position.
     *
     * The MITHRA-style host generator/boost copies initial positions and momenta
     * into Kokkos storage. R and R_nm1 initially coincide, so the first trajectory
     * current is zero. Rank zero assigns equal Q and mass from configured totals
     * divided by the actual generated count, which can differ from totalP_m.
     * A global mean position is subtracted from both position arrays; the final
     * particle update migrates all registered attributes to their owning ranks.
     * @pre Particle storage, field layout and frame have been initialized on all ranks.
     * A nonempty bunch is required by the charge/mass and centroid divisions.
     */
    void initializeParticles() {
        Inform m("Initialize Particles");

        BunchInitialize<T> mithra = generate_mithra_config(this->m_config, frame_m);

        // The MITHRA generator produces the whole bunch on rank 0; the first
        // particle update (below) scatters it across ranks by position.
        if (ippl::Comm->rank() == 0) {
            size_t actualP = initialize_bunch_mithra(*this->pcontainer_m, mithra, frame_gamma_m);
            this->pcontainer_m->Q    = this->m_config.charge / actualP;
            this->pcontainer_m->mass = this->m_config.mass / actualP;
        } else {
            this->pcontainer_m->create(0);
        }

        // Center the bunch on the origin.
        {
            auto rview   = this->pcontainer_m->R.getView();
            auto rm1view = this->pcontainer_m->R_nm1.getView();
            ippl::Vector<T, 3> meanpos =
                this->pcontainer_m->R.sum() * (1.0 / this->pcontainer_m->getTotalNum());
            Kokkos::parallel_for(
                this->pcontainer_m->getLocalNum(), KOKKOS_LAMBDA(const size_t i) {
                    rview(i) -= meanpos;
                    rm1view(i) -= meanpos;
                });
            Kokkos::fence();
        }

        // Distribute particles to their owning ranks before the first step.
        this->pcontainer_m->update();

        m << "particles created and initial conditions assigned" << endl;
    }

    /**
     * @brief Deposit, evolve fields, optionally visualize, then push/migrate particles.
     *
     * This is the numerical body called by BaseManager::run(). The active solver
     * updates potentials, shifts their time levels and reconstructs E/B. Optional
     * Catalyst execution occurs immediately afterwards, before particle push and
     * before post_step() increments the reported manager time/iteration.
     * The push evaluates the static undulator in lab coordinates and transforms
     * its E/B into the boosted frame; no separate grid2par() call is made here.
     */
    void advance() override {
        // 1. Deposit the current produced by last step's motion (R_nm1 -> R).
        this->par2grid();

        // 2. Advance the electromagnetic field one FDTD step.
        this->solver_m->solve();

#ifdef IPPL_ENABLE_CATALYST
        // Execute immediately after the solver has recomputed E/B. The adaptor
        // makes its host snapshots synchronously during this call.
        cat_viz.Execute(this->it_m, this->time_m);
#endif

        // 3. Push particles with the self-consistent field plus the undulator
        //    field transformed into the co-moving frame.
        auto und = undulator_m;
        auto lb  = frame_m;
        this->push(KOKKOS_LAMBDA(ippl::Vector<T, 3> pos, T time) {
            lb.primedToUnprimed(pos, time);
            auto eb = und(pos);
            return lb.transform_EB(eb);
        });
    }

    /**
     * @brief Run broadband power, single-frequency power and bunch/field diagnostics.
     * Called collectively once at initialization and after every completed step.
     * Output is appended by rank zero; the methods also use MPI reductions/barriers.
     */
    void dump() {
        dumpRadiation();
        dumpRadiationBanded();
        dumpFELDiagnostics();
    }

    /**
     * @brief Integrate signed longitudinal E-cross-B flux on a downstream interior plane.
     *
     * A Kokkos reduction transforms stored E/B to the lab frame and selects view
     * indices satisfying k + local_first_z = N_z - 3, with one halo layer assumed.
     * Multiplication by transverse cell area and the units.h power-density factor
     * yields watts; MPI sums to rank zero and appends radiation_Nranks.csv.
     *
     * This is broadband signed flux of the stored field, without background,
     * near-field or outgoing-wave separation. The output distance label comes
     * from transforming z'=extents[2], not the selected plane's physical coordinate
     * in the centered box. It must not be read as an exact laboratory monitor
     * position. Boundary reflection and normalization need independent checks.
     */
    void dumpRadiation() {
        auto fc    = this->fcontainer_m;
        auto eview = fc->getE().getView();
        auto bview = fc->getB().getView();
        auto ldom  = fc->getFL().getLocalNDIndex();
        auto lb    = frame_m;

        const uint32_t nz = (uint32_t)this->nr_m[Dim - 1];

        double radiation = 0.0;
        Kokkos::parallel_reduce(
            ippl::getRangePolicy(eview, 1),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& ref) {
                Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> buncheb{eview(i, j, k),
                                                                            bview(i, j, k)};
                auto eblab    = lb.inverse_transform_EB(buncheb);
                uint32_t kg   = (uint32_t)(k + ldom.first()[2]);
                if (kg == nz - 3) {
                    ref += ippl::cross(eblab.first, eblab.second)[2];
                }
            },
            radiation);

        double power_local =
            radiation
            * double(unit_powerdensity_in_watt_per_square_meter * unit_length_in_meters
                     * unit_length_in_meters)
            * this->hr_m[0] * this->hr_m[1];
        double power_global = 0.0;
        MPI_Reduce(&power_local, &power_global, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            // Diagnostic distance label from z'=Lz; not the selected plane position.
            ippl::Vector<T, 3> pos{0, 0, (T)this->m_config.extents[2]};
            lb.primedToUnprimed(pos, (T)this->time_m);

            std::stringstream fname;
            fname << this->m_config.output_path << "radiation_" << ippl::Comm->size() << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "labframe_z, radiated_power_W" << endl;
            }
            csvout << pos[2] * unit_length_in_meters << " " << power_global << endl;
        }
        ippl::Comm->barrier();
    }

    /**
     * @brief Append a MITHRA-inspired single-frequency downstream power estimate.
     *
     * Lazily allocates a rank-local Kokkos ring buffer of lab-transformed Ex, Ey,
     * Bx and By. The same interior-plane index convention as dumpRadiation() is
     * used. The chosen wavelength is undulator_period/frame_gamma; the buffer
     * spans approximately three cycles, with omega=2*pi/lambda in code units.
     * A Kokkos DFT forms the real E-cross-conjugate-B product, normalized by 2/Nf^2,
     * transverse area and the power-density conversion. MPI sums to rank zero,
     * which appends radiation_band_Nranks.csv and the same distance label.
     *
     * Samples use boosted-frame dt even though field values are transformed to
     * the lab frame. The DFT indexes ring slots directly, and early output includes
     * the initially unfilled window. Frequency interpretation, ring-window behavior
     * and absolute power therefore require validation; this is not a broadband
     * spectrum or a general lab-frame fixed-detector time series.
     */
    void dumpRadiationBanded() {
        auto fc    = this->fcontainer_m;
        auto eview = fc->getE().getView();
        auto bview = fc->getB().getView();
        auto ldom  = fc->getFL().getLocalNDIndex();
        auto lb    = frame_m;

        const uint32_t nz = (uint32_t)this->nr_m[Dim - 1];
        const int extx    = (int)eview.extent(0);
        const int exty    = (int)eview.extent(1);
        const int extz    = (int)eview.extent(2);

        if (!rp_init_m) {
            const double lambda_rad = this->m_config.undulator_period / frame_gamma_m;
            rp_omega_m = 2.0 * M_PI / lambda_rad;  // [1/unit_time], c = 1
            rp_Nf_m    = std::max(1, (int)std::lround(3.0 * lambda_rad / this->dt_m));
            rp_fdt_m   = Kokkos::View<T****>("FEL banded field buffer", rp_Nf_m, extx, exty, 4);
            rp_init_m  = true;
        }

        const int Nf = rp_Nf_m;
        const int m  = ((this->it_m % Nf) + Nf) % Nf;  // ring slot for this step

        const int kview = (int)(nz - 3) - ldom.first()[2];
        const bool owns = (kview >= 1 && kview < extz - 1);

        auto fdt = rp_fdt_m;

        // 1. Store this step's lab-frame transverse fields into ring slot m.
        if (owns) {
            Kokkos::parallel_for(
                "FEL banded store",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {extx - 1, exty - 1}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> buncheb{
                        eview(i, j, kview), bview(i, j, kview)};
                    auto eblab    = lb.inverse_transform_EB(buncheb);
                    fdt(m, i, j, 0) = eblab.first[0];   // Ex_lab
                    fdt(m, i, j, 1) = eblab.first[1];   // Ey_lab
                    fdt(m, i, j, 2) = eblab.second[0];  // Bx_lab
                    fdt(m, i, j, 3) = eblab.second[1];  // By_lab
                });
        }

        // 2. Single-frequency DFT over the window, summed over the exit plane.
        double power_local = 0.0;
        if (owns) {
            const double omega = rp_omega_m;
            const double dt    = this->dt_m;
            Kokkos::parallel_reduce(
                "FEL banded DFT",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {extx - 1, exty - 1}),
                KOKKOS_LAMBDA(const int i, const int j, double& ref) {
                    double ex_r = 0, ex_i = 0, ey_r = 0, ey_i = 0;
                    double bx_r = 0, bx_i = 0, by_r = 0, by_i = 0;
                    for (int mm = 0; mm < Nf; ++mm) {
                        const double ph = omega * mm * dt;
                        const double cp = Kokkos::cos(ph);
                        const double sp = Kokkos::sin(ph);
                        const double ex = fdt(mm, i, j, 0);
                        const double ey = fdt(mm, i, j, 1);
                        const double bx = fdt(mm, i, j, 2);
                        const double by = fdt(mm, i, j, 3);
                        ex_r += ex * cp;  ex_i += ex * sp;
                        ey_r += ey * cp;  ey_i += ey * sp;
                        bx_r += bx * cp;  bx_i -= bx * sp;
                        by_r += by * cp;  by_i -= by * sp;
                    }
                    ref += (ex_r * by_r - ex_i * by_i) - (ey_r * bx_r - ey_i * bx_i);
                },
                power_local);
        }

        // Convert the windowed sum to a cycle-averaged power in Watts.
        power_local *= (2.0 / (double(Nf) * double(Nf)))
                       * double(unit_powerdensity_in_watt_per_square_meter * unit_length_in_meters
                                * unit_length_in_meters)
                       * this->hr_m[0] * this->hr_m[1];

        double power_global = 0.0;
        MPI_Reduce(&power_local, &power_global, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            ippl::Vector<T, 3> pos{0, 0, (T)this->m_config.extents[2]};
            lb.primedToUnprimed(pos, (T)this->time_m);

            std::stringstream fname;
            fname << this->m_config.output_path << "radiation_band_" << ippl::Comm->size()
                  << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "labframe_z, banded_power_W" << endl;
            }
            csvout << pos[2] * unit_length_in_meters << " " << power_global << endl;
        }
        ippl::Comm->barrier();
    }

    /**
     * @brief Reduce live-particle and grid-field statistics, then append feldiag_Nranks.csv.
     *
     * Bunching is the unweighted particle average of exp(i*k*z), with boosted-frame
     * z and wavelength undulator_period/(2*frame_gamma). Field diagnostics are
     * max|E| and 0.5*sum(E.E+B.B)*cellVolume over owned cells, in code units.
     * These stored grid fields exclude the separately applied undulator field.
     * The energy is not converted to joules and is not a particle-plus-field
     * conservation balance in this driven, open system.
     *
     * Also records live count, mean z, centered rms z, sqrt(mean(x^2+y^2)) and
     * mean gamma_beta_z. The transverse rms is about the origin, not a recentered
     * transverse centroid. Positions remain boosted-frame code lengths; only the
     * shared longitudinal distance label is converted to lab metres.
     * Kokkos local reductions feed MPI reductions; rank zero writes the CSV.
     */
    void dumpFELDiagnostics() {
        auto pc = this->pcontainer_m;
        auto fc = this->fcontainer_m;

        // --- micro-bunching factor at the resonant wavelength ---
        const double lambda_star = this->m_config.undulator_period / (2.0 * frame_gamma_m);
        const double kstar       = 2.0 * M_PI / lambda_star;
        auto rview               = pc->R.getView();
        double sumcos = 0.0, sumsin = 0.0;
        Kokkos::parallel_reduce(
            "FEL bunching", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t i, double& c, double& s) {
                const double phase = kstar * rview(i)[2];
                c += Kokkos::cos(phase);
                s += Kokkos::sin(phase);
            },
            sumcos, sumsin);

        double gsumcos = 0.0, gsumsin = 0.0;
        ippl::Comm->reduce(sumcos, gsumcos, 1, std::plus<double>());
        ippl::Comm->reduce(sumsin, gsumsin, 1, std::plus<double>());
        size_type nLocal = pc->getLocalNum(), nGlobal = 0;
        ippl::Comm->reduce(nLocal, nGlobal, 1, std::plus<size_type>());
        double bunching = (nGlobal > 0)
                              ? std::sqrt(gsumcos * gsumcos + gsumsin * gsumsin) / (double)nGlobal
                              : 0.0;

        // --- peak |E| and total EM field energy over the domain ---
        const int nghost = fc->getE().getNghost();
        auto Eview       = fc->getE().getView();
        auto Bview       = fc->getB().getView();
        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localE2 = 0.0, localEmax = 0.0;
        ippl::parallel_reduce(
            "FEL field stats", ippl::getRangePolicy(Eview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& Emax) {
                ippl::Vector<T, 3> E = ippl::apply(Eview, args);
                ippl::Vector<T, 3> B = ippl::apply(Bview, args);
                E2 += E.dot(E) + B.dot(B);
                double en = Kokkos::sqrt(E.dot(E));
                if (en > Emax) {
                    Emax = en;
                }
            },
            Kokkos::Sum<double>(localE2), Kokkos::Max<double>(localEmax));

        double globalE2 = 0.0, globalEmax = 0.0;
        ippl::Comm->reduce(localE2, globalE2, 1, std::plus<double>());
        ippl::Comm->reduce(localEmax, globalEmax, 1, std::greater<double>());
        double cellVolume = 1.0;
        for (unsigned d = 0; d < Dim; ++d) {
            cellVolume *= this->hr_m[d];
        }
        double fieldEnergy = 0.5 * globalE2 * cellVolume;

        // --- bunch centroid, RMS size, and mean longitudinal momentum, to see
        //     HOW particles leave the domain (all in the co-moving frame, units
        //     of unit_length). Box half-widths: z in [origin_z, origin_z+L_z],
        //     transverse +-extents/2. ---
        auto gbview = pc->gamma_beta.getView();
        double sz = 0.0, sz2 = 0.0, sperp2 = 0.0, sgbz = 0.0;
        Kokkos::parallel_reduce(
            "FEL bunch shape", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t i, double& az, double& az2, double& aperp2, double& agbz) {
                const double x = rview(i)[0], y = rview(i)[1], z = rview(i)[2];
                az += z;
                az2 += z * z;
                aperp2 += x * x + y * y;
                agbz += gbview(i)[2];
            },
            sz, sz2, sperp2, sgbz);
        double gsz = 0.0, gsz2 = 0.0, gsperp2 = 0.0, gsgbz = 0.0;
        ippl::Comm->reduce(sz, gsz, 1, std::plus<double>());
        ippl::Comm->reduce(sz2, gsz2, 1, std::plus<double>());
        ippl::Comm->reduce(sperp2, gsperp2, 1, std::plus<double>());
        ippl::Comm->reduce(sgbz, gsgbz, 1, std::plus<double>());
        double cz = 0.0, rmsz = 0.0, rmsperp = 0.0, meangbz = 0.0;
        if (nGlobal > 0) {
            cz      = gsz / nGlobal;
            rmsz    = std::sqrt(std::max(0.0, gsz2 / nGlobal - cz * cz));
            rmsperp = std::sqrt(gsperp2 / nGlobal);
            meangbz = gsgbz / nGlobal;
        }

        Inform m("FELDiag");
        m << "t=" << this->time_m << " bunching=" << bunching << " max|E|=" << globalEmax
          << " fieldEnergy=" << fieldEnergy << " Np=" << nGlobal << " cz=" << cz
          << " rms_z=" << rmsz << " rms_perp=" << rmsperp << " mean_gbz=" << meangbz << endl;

        if (ippl::Comm->rank() == 0) {
            ippl::Vector<T, 3> pos{0, 0, (T)this->m_config.extents[2]};
            frame_m.primedToUnprimed(pos, (T)this->time_m);

            std::stringstream fname;
            fname << this->m_config.output_path << "feldiag_" << ippl::Comm->size() << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "labframe_z, bunching, max_E, field_energy, num_particles, "
                          "centroid_z, rms_z, rms_perp, mean_gbz"
                       << endl;
            }
            csvout << pos[2] * unit_length_in_meters << " " << bunching << " " << globalEmax << " "
                   << fieldEnergy << " " << nGlobal << " " << cz << " " << rmsz << " " << rmsperp
                   << " " << meangbz << endl;
        }
        ippl::Comm->barrier();
    }


public:
    // NOTE: depositChargeDensity and destroyOutOfBounds must be public, not
    // protected/private, because they contain KOKKOS_LAMBDA expressions that
    // expand to extended __host__ __device__ lambdas under CUDA (nvcc). NVCC
    // forbids such lambdas inside protected or private member functions.
    // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-restrictions
    /**
     * @brief Add current-position CIC charge density to component zero of J.
     *
     * Each local particle contributes Q/cellVolume to its surrounding 2^Dim cell
     * centres with linear weights, using Kokkos atomic additions. The half-cell
     * shift matches collocated deposition/gather centering. This method fences the
     * kernel but neither zeros J nor accumulates halos; depositCurrent() performs
     * those steps and calls this only when space_charge is enabled.
     * @pre Particle interpolation support must lie in valid local/halo storage.
     * Depositing rho does not itself impose Gauss's law or solve an initial
     * Coulomb field. Public visibility also permits CUDA extended lambdas.
     */
    void depositChargeDensity() {
        auto pc             = this->pcontainer_m;
        auto fc             = this->fcontainer_m;
        auto view           = fc->getJ().getView();
        auto qview          = pc->Q.getView();
        auto rview          = pc->R.getView();
        const auto origin   = fc->getMesh().getOrigin();
        const auto h        = fc->getMesh().getMeshSpacing();
        const auto ldom     = fc->getFL().getLocalNDIndex();
        const int nghost    = fc->getJ().getNghost();
        T volume            = T(1);
        for (unsigned d = 0; d < Dim; ++d)
            volume *= h[d];

        Kokkos::parallel_for(
            "FEL deposit charge density", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t p) {
                const ippl::Vector<T, Dim> pos = rview(p);
                const T value                  = qview(p) / volume;

                Kokkos::Array<size_t, Dim> cellIdx;
                Kokkos::Array<T, Dim> xi;
                for (unsigned d = 0; d < Dim; ++d) {
                    // Half-cell shift to match the Cell-centered field and the
                    // gather (see assemble_current_collocated).
                    const T gridpos = (pos[d] - origin[d]) / h[d] - T(0.5);
                    cellIdx[d]      = static_cast<size_t>(Kokkos::floor(gridpos));
                    xi[d]           = gridpos - T(cellIdx[d]);
                }
                for (unsigned corner = 0; corner < (1u << Dim); ++corner) {
                    size_t idx[Dim];
                    T weight = T(1);
                    for (unsigned d = 0; d < Dim; ++d) {
                        const unsigned offset = (corner >> d) & 1u;
                        weight *= offset ? xi[d] : (T(1) - xi[d]);
                        idx[d] = cellIdx[d] - ldom.first()[d] + nghost + offset;
                    }
                    Kokkos::atomic_add(&(ippl::apply(view, idx)[0]), value * weight);
                }
            });
        Kokkos::fence();
    }

    /**
     * @brief Mark and remove particles outside or exactly on any physical box face.
     *
     * A Kokkos boolean mask and reduction identify local losses; destroy() compacts
     * registered attributes. No wrapping, reflection or escaping-current deposit
     * is performed here. The caller push() subsequently migrates surviving
     * particles with update(). Public visibility also permits CUDA extended lambdas.
     */
    void destroyOutOfBounds() {
        auto pc           = this->pcontainer_m;
        auto rview        = pc->R.getView();
        const auto origin = this->fcontainer_m->getMesh().getOrigin();
        ippl::Vector<T, Dim> extent;
        for (unsigned d = 0; d < Dim; ++d)
            extent[d] = this->nr_m[d] * this->hr_m[d];

        Kokkos::View<bool*> invalid("OOB particles", pc->getLocalNum());
        size_type invalid_count = 0;
        Kokkos::parallel_reduce(
            pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t i, size_type& ref) {
                bool out_of_bounds             = false;
                const ippl::Vector<T, Dim> ppos = rview(i);
                for (unsigned d = 0; d < Dim; ++d) {
                    out_of_bounds |= (ppos[d] <= origin[d]);
                    out_of_bounds |= (ppos[d] >= origin[d] + extent[d]);
                }
                invalid(i) = out_of_bounds;
                ref += out_of_bounds;
            },
            invalid_count);
        Kokkos::fence();
        pc->destroy(invalid, invalid_count);
        Kokkos::fence();
    }
};

#endif
