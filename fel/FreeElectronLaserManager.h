#ifndef IPPL_FREE_ELECTRON_LASER_MANAGER_H
#define IPPL_FREE_ELECTRON_LASER_MANAGER_H

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>

#include "Manager/BaseManager.h"

#include "Interpolation/CurrentDeposition.hpp"

#include "Config.h"
#include "FELFieldContainer.hpp"
#include "FELParticleContainer.hpp"
#ifdef IPPL_HDF5
#include "HDF5Writer.h"
#else
#include "VideoWriter.h"
#endif
#include "LorentzTransform.h"
#include "MithraBunch.h"
#include "Undulator.h"
#include "datatypes.h"
#include "units.h"

// FEL simulation manager.
//
// An electromagnetic (FDTD) PIC manager for the Free Electron Laser: it owns the
// field/particle containers and the Maxwell solver, provides the generic
// particle<->grid operations (deposit / gather / relativistic Boris push), and
// implements the FEL-specific run loop and diagnostics.
//
// It works in a Lorentz frame co-moving with the electron bunch (gamma_frame =
// gamma_bunch / sqrt(1 + K^2/2)); the static undulator field is transformed into
// this frame each step and added to the self-consistent FDTD field during the
// particle push. Radiation is transformed back to the lab frame for output.
template <typename T, unsigned Dim>
class FreeElectronLaserManager : public ippl::BaseManager {
public:
    using ParticleContainer_t = FELParticleContainer<T, Dim>;
    using FieldContainer_t    = FELFieldContainer<T, Dim>;
    using FDTDSolver_t        = ::FDTDSolver_t<T, Dim>;
    using Base                = ippl::ParticleBase<PLayout_t<T, Dim>>;

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

    ~FreeElectronLaserManager() { output_m.close(); }

protected:
    config m_config;

    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;

    double time_m;
    double dt_m;
    int it_m;

    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> origin_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;

    std::shared_ptr<FieldContainer_t> fcontainer_m;
    std::shared_ptr<ParticleContainer_t> pcontainer_m;
    std::shared_ptr<FDTDSolver_t> solver_m;

    int nsubsteps_m = 3;  ///< Boris sub-steps per FDTD step (reference behaviour).

    T frame_gamma_m;                            ///< Lorentz factor of the co-moving frame.
    ippl::undulator_parameters<T> uparams_m;    ///< Undulator parameters.
    ippl::UniaxialLorentzframe<T, 2> frame_m;   ///< Boost into the co-moving frame (z-axis).
    ippl::Undulator<T> undulator_m;             ///< Static undulator field model.
#ifdef IPPL_HDF5
    FELHDF5Writer<T, Dim> output_m;             ///< Optional HDF5 visualization output.
#else
    FELVideoWriter<T, Dim> output_m;            ///< Optional ffmpeg Poynting-flux video.
#endif

    // --- narrow-band (resonant) radiation power diagnostic state ---
    // MITHRA reports the FEL output power as a sliding-window single-frequency
    // DFT of the exit-plane fields at the resonant wavelength, not the total
    // broadband Poynting flux that dumpRadiation() integrates. These hold the
    // rolling time-domain sample buffer and the derived window length / angular
    // frequency; they are allocated lazily on the first diagnostic call.
    bool rp_init_m    = false;     ///< whether rp_fdt_m has been allocated yet
    int rp_Nf_m       = 0;         ///< DFT window length [time steps] (~3 resonant cycles)
    double rp_omega_m = 0.0;       ///< resonant angular frequency [1/unit_time], c = 1
    Kokkos::View<T****> rp_fdt_m;  ///< ring buffer [Nf][nx][ny][4] = (Ex,Ey,Bx,By)_lab

public:
    size_type getTotalP() const { return totalP_m; }
    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }
    void setNt(int nt_) { nt_m = nt_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }
    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }
    void setTime(double time_) { time_m = time_; }

    std::shared_ptr<ParticleContainer_t> getParticleContainer() { return pcontainer_m; }
    void setParticleContainer(std::shared_ptr<ParticleContainer_t> pcontainer) {
        pcontainer_m = pcontainer;
    }

    std::shared_ptr<FieldContainer_t> getFieldContainer() { return fcontainer_m; }
    void setFieldContainer(std::shared_ptr<FieldContainer_t> fcontainer) {
        fcontainer_m = fcontainer;
    }

    std::shared_ptr<FDTDSolver_t> getFieldSolver() { return solver_m; }
    void setFieldSolver(std::shared_ptr<FDTDSolver_t> solver) { solver_m = solver; }

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
        this->time_m += this->dt_m;
        this->it_m++;
        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

    // Particle -> grid: charge-conserving current deposition into the four-current
    // source field J ([1..Dim]); optionally the charge density into J[0].
    void par2grid() { depositCurrent(); }

    // Grid -> particle: interpolate E and B to the particle positions.
    void grid2par() { gatherFields(); }

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

    void gatherFields() {
        this->fcontainer_m->getE().fillHalo();
        this->fcontainer_m->getB().fillHalo();
        this->pcontainer_m->E_gather.gather(this->fcontainer_m->getE(), this->pcontainer_m->R,
                                            false);
        this->pcontainer_m->B_gather.gather(this->fcontainer_m->getB(), this->pcontainer_m->R,
                                            false);
    }

    // Relativistic Boris push with an externally supplied (E, B) field, performed
    // in sub-steps over one FDTD time step. external_field(pos, time) must return
    // a Kokkos::pair{E, B}. Faithful to the original NSFDSolverWithParticles
    // particle update.
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

    void pre_run() override {
        Inform m("Pre Run");

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

        // Courant / dispersion condition for the standard FDTD stencil.
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

        // Open rank-0 visualization output when periodic output was requested.
        output_m.open(this->m_config);

        this->dump();

        m << "Done" << endl;
    }

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

    void advance() override {
        // 1. Deposit the current produced by last step's motion (R_nm1 -> R).
        this->par2grid();

        // 2. Advance the electromagnetic field one FDTD step.
        this->solver_m->solve();

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

    void dump() {
        // These CSV files are rank-0 scalar diagnostics, complementary to the
        // visualization frames: they store globally reduced time-series values
        // every step, while HDF5/movie output stores frame data on output_rhythm.
        dumpRadiation();
        dumpRadiationBanded();
        dumpFELDiagnostics();

        // Emit visualization output on the configured rhythm. writeFrame is collective,
        // so every rank must reach it under the same condition.
        if (this->m_config.output_rhythm != 0
            && (this->it_m % (int)this->m_config.output_rhythm) == 0) {
#ifdef IPPL_HDF5
            output_m.writeFrame(this->it_m, this->time_m, *this->fcontainer_m,
                                *this->pcontainer_m, frame_m, this->m_config, this->nr_m);
#else
            output_m.writeFrame(this->it_m, *this->fcontainer_m, *this->pcontainer_m, frame_m,
                                this->m_config, this->nr_m);
#endif
        }
    }

    // Radiated power leaving the downstream end of the domain, transformed back
    // to the lab frame and integrated over the transverse exit plane.
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
        // Sum the exit-plane contribution across all z-domain ranks; rank 0 is
        // the only writer, so the CSV is a single global time series.
        MPI_Reduce(&power_local, &power_global, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            // Lab-frame longitudinal position the exit plane maps to at this time.
            ippl::Vector<T, 3> pos{0, 0, (T)this->m_config.extents[2]};
            lb.primedToUnprimed(pos, (T)this->time_m);

            std::stringstream fname;
            fname << this->m_config.output_path << "radiation_" << ippl::Comm->size() << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (std::fabs(this->time_m) < 1e-14) {
                csvout << "labframe_z_m,radiated_power_W" << endl;
            }
            csvout << pos[2] * unit_length_in_meters << "," << power_global << endl;
        }
        ippl::Comm->barrier();
    }

    // Narrow-band radiated power at the FEL resonance, computed the way MITHRA's
    // powerSample() does.
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
        // Sum the resonant-band exit-plane contribution across ranks before
        // writing one global diagnostic row.
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
                csvout << "labframe_z_m,banded_power_W" << endl;
            }
            csvout << pos[2] * unit_length_in_meters << "," << power_global << endl;
        }
        ippl::Comm->barrier();
    }

    // FEL gain diagnostics, written against the same lab-frame distance axis as
    // the radiation curve so they can be overlaid:
    //   * bunching   : micro-bunching factor |<exp(i k* z)>| at the resonant
    //                  wavelength lambda* = undulator_period / (2 gamma_frame).
    //                  Exponential growth of this is the signature of FEL gain;
    //                  if it stays at the ~1% seed there is no gain.
    //   * max|E|     : peak electric-field magnitude in the domain.
    //   * fieldEnergy: total EM field energy (0.5 * sum(E.E + B.B) * cellVolume).
    //   * N          : live particle count (catches runaway out-of-bounds loss).
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

        // All particle and field diagnostics below are reduced over the full
        // MPI decomposition before rank 0 writes the CSV row.
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
                csvout << "labframe_z_m,bunching,max_E,field_energy,num_particles,"
                          "centroid_z,rms_z,rms_perp,mean_gbz"
                       << endl;
            }
            csvout << pos[2] * unit_length_in_meters << "," << bunching << "," << globalEmax
                   << "," << fieldEnergy << "," << nGlobal << "," << cz << "," << rmsz << ","
                   << rmsperp << "," << meangbz << endl;
        }
        ippl::Comm->barrier();
    }

protected:
    // Deposit charge density (CIC) into component [0] of the four-current field,
    // needed only when space charge is enabled. Component [1..Dim] is filled by
    // assemble_current_collocated.
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

                Kokkos::Array<int, Dim> cellIdx;
                Kokkos::Array<int, Dim> localBase;
                Kokkos::Array<T, Dim> xi;
                bool inLocalAllocation = true;
                for (unsigned d = 0; d < Dim; ++d) {
                    // Half-cell shift to match the Cell-centered field and the
                    // gather (see assemble_current_collocated).
                    const T gridpos = (pos[d] - origin[d]) / h[d] - T(0.5);
                    cellIdx[d]      = static_cast<int>(Kokkos::floor(gridpos));
                    xi[d]           = gridpos - T(cellIdx[d]);
                    localBase[d]    = cellIdx[d] - ldom.first()[d] + nghost;

                    inLocalAllocation &= (localBase[d] >= 0);
                    inLocalAllocation &= (localBase[d] + 1 < static_cast<int>(view.extent(d)));
                }
                if (!inLocalAllocation) {
                    return;
                }
                for (unsigned corner = 0; corner < (1u << Dim); ++corner) {
                    size_t idx[Dim];
                    T weight = T(1);
                    for (unsigned d = 0; d < Dim; ++d) {
                        const unsigned offset = (corner >> d) & 1u;
                        weight *= offset ? xi[d] : (T(1) - xi[d]);
                        idx[d] = static_cast<size_t>(localBase[d] + offset);
                    }
                    Kokkos::atomic_add(&(ippl::apply(view, idx)[0]), value * weight);
                }
            });
        Kokkos::fence();
    }

    // Mark particles that have left the physical domain and remove them (open BC).
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
