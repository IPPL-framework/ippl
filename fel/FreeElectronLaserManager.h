#ifndef IPPL_FREE_ELECTRON_LASER_MANAGER_H
#define IPPL_FREE_ELECTRON_LASER_MANAGER_H

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>

#include "FELManager.h"
#include "LorentzTransform.h"
#include "MithraBunch.h"
#include "Undulator.h"
#include "VideoWriter.h"
#include "units.h"

// Concrete FEL simulation manager.
//
// Plays the role LandauDampingManager plays in alpine. It works in a Lorentz
// frame co-moving with the electron bunch (gamma_frame = gamma_bunch /
// sqrt(1 + K^2/2)); the static undulator field is transformed into this frame
// each step and added to the self-consistent FDTD field during the particle
// push. Radiation is transformed back to the lab frame for output.
template <typename T, unsigned Dim>
class FreeElectronLaserManager : public FELManager<T, Dim> {
public:
    using ParticleContainer_t = typename FELManager<T, Dim>::ParticleContainer_t;
    using FieldContainer_t    = typename FELManager<T, Dim>::FieldContainer_t;
    using FDTDSolver_t        = typename FELManager<T, Dim>::FDTDSolver_t;

    FreeElectronLaserManager(config cfg)
        : FELManager<T, Dim>(cfg)
        , frame_gamma_m(std::max(
              T(1), cfg.bunch_gamma
                        / std::sqrt(1 + cfg.undulator_K * cfg.undulator_K * T(0.5))))
        , uparams_m(cfg.undulator_K, cfg.undulator_period, cfg.undulator_length)
        , frame_m(ippl::UniaxialLorentzframe<T, 2>::from_gamma(frame_gamma_m))
        , undulator_m(uparams_m, 2.0 * cfg.sigma_position[2] * frame_gamma_m * frame_gamma_m) {}

    ~FreeElectronLaserManager() { video_m.close(); }

private:
    T frame_gamma_m;                            ///< Lorentz factor of the co-moving frame.
    ippl::undulator_parameters<T> uparams_m;    ///< Undulator parameters.
    ippl::UniaxialLorentzframe<T, 2> frame_m;   ///< Boost into the co-moving frame (z-axis).
    ippl::Undulator<T> undulator_m;             ///< Static undulator field model.
    FELVideoWriter<T, Dim> video_m;             ///< Optional ffmpeg Poynting-flux video.

public:
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

        // Open the ffmpeg pipe (rank 0, only if periodic output was requested).
        video_m.open(this->m_config);

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

    void dump() override {
        dumpRadiation();

        // Emit a video frame on the configured rhythm. writeFrame is collective,
        // so every rank must reach it under the same condition.
        if (this->m_config.output_rhythm != 0
            && (this->it_m % (int)this->m_config.output_rhythm) == 0) {
            video_m.writeFrame(this->it_m, *this->fcontainer_m, *this->pcontainer_m, frame_m,
                               this->m_config, this->nr_m);
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
                csvout << "labframe_z, radiated_power_W" << endl;
            }
            csvout << pos[2] * unit_length_in_meters << " " << power_global << endl;
        }
        ippl::Comm->barrier();
    }
};

#endif
