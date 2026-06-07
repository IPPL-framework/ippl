#ifndef IPPL_FEL_MANAGER_H
#define IPPL_FEL_MANAGER_H

#include <memory>

#include "Manager/BaseManager.h"

#include "Interpolation/CurrentDeposition.hpp"

#include "Config.h"
#include "FELFieldContainer.hpp"
#include "FELParticleContainer.hpp"
#include "datatypes.h"

// Base manager for the FEL simulation.
//
// Plays the role that AlpineManager plays in the alpine module, but for an
// electromagnetic (FDTD) PIC scheme: it owns the field/particle containers and
// the Maxwell solver, and provides the generic particle<->grid operations.
//
// It deliberately inherits ippl::BaseManager rather than ippl::PicManager:
// PicManager exists to coordinate a FieldSolverBase and a load balancer, both
// of which the FEL module drops (the FDTD solver is not a FieldSolverBase, and
// there is no load balancing). We keep alpine's surface (par2grid / grid2par,
// container getters/setters) so the concrete manager reads like
// LandauDampingManager.
template <typename T, unsigned Dim>
class FELManager : public ippl::BaseManager {
public:
    using ParticleContainer_t = FELParticleContainer<T, Dim>;
    using FieldContainer_t    = FELFieldContainer<T, Dim>;
    using FDTDSolver_t        = ::FDTDSolver_t<T, Dim>;
    using Base                = ippl::ParticleBase<PLayout_t<T, Dim>>;

    FELManager(config cfg)
        : m_config(cfg)
        , totalP_m(cfg.num_particles)
        , nt_m(0)
        , time_m(0.0)
        , dt_m(0.0)
        , it_m(0) {}

    virtual ~FELManager() {}

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

    virtual void dump() { /* default does nothing */ }

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

protected:
    int nsubsteps_m = 3;  ///< Boris sub-steps per FDTD step (reference behaviour).

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

                Kokkos::Array<size_t, Dim> cellIdx;
                Kokkos::Array<T, Dim> xi;
                for (unsigned d = 0; d < Dim; ++d) {
                    const T gridpos = (pos[d] - origin[d]) / h[d];
                    cellIdx[d]      = static_cast<size_t>(gridpos);
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
