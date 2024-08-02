#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <memory>

#include "BaseDistributionFunction.hpp"
#include "BaseParticleDistribution.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "SimulationParameters.hpp"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class VortexInCellManagerBase
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalanceStrategy<T, Dim>,
                              FieldSolver<T, Dim, FieldContainer<T, Dim>>> {
public:
    SimulationParameters<T, Dim> params;

    VortexInCellManagerBase(SimulationParameters<T, Dim> params_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalanceStrategy<T, Dim>,
                           FieldSolver<T, Dim, FieldContainer<T, Dim>>>()
        , params(params_) {}

    ~VortexInCellManagerBase() {}

public:
    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
        this->params.it += 1;
        dump();
    }

    virtual void dump() = 0;
};

template <typename T, unsigned Dim>
class VortexInCellManager : public VortexInCellManagerBase<T, Dim> {
public:
    VortexInCellManager(SimulationParameters<T, Dim> params_)
        : VortexInCellManagerBase<T, Dim>(params_) {}

    ~VortexInCellManager() {}
};

template <typename T>
class VortexInCellManager<T, 2> : public VortexInCellManagerBase<T, 2> {
public:
    VortexInCellManager(SimulationParameters<T, 2> params_)
        : VortexInCellManagerBase<T, 2>(params_) {
        this->setFieldContainer(std::make_shared<FieldContainer<T, 2>>(this->params));

        this->setParticleContainer(std::make_shared<ParticleContainer<T, 2>>(
            this->getFieldContainer()->getMesh(), this->getFieldContainer()->getFL()));

        this->setFieldSolver(std::make_shared<FieldSolver<T, 2, FieldContainer<T, 2>>>());

        this->fsolver_m->initSolver(this->fcontainer_m);

        initParticles();

        this->pcontainer_m->initDump();

        // First step of the Euler time integration
        par2grid();
        this->fsolver_m->solve(this->fcontainer_m);
        updateFields();
        grid2par();

        std::shared_ptr<ParticleContainer<T, 2>> pc = this->getParticleContainer();

        pc->R_old = pc->R;
        pc->R     = pc->R_old + pc->P * this->params.dt;
        pc->update();
    }

    void initParticles() {
        std::shared_ptr<ParticleContainer<T, 2>> pc = this->getParticleContainer();

        Circle<T, Dim> circ(10.0);

        Vector_t<T, Dim> center = 0.5 * (this->params.rmax - this->params.rmin);
        ShiftTransformation<T, Dim> shift_to_center(-center);

        circ.applyTransformation(shift_to_center);
        FilteredDistribution<T, Dim> filteredDist(circ, this->params.rmin, this->params.rmax,
                                                  new GridPlacement<T, Dim>(this->params.nr));

        this->params.np = filteredDist.getNumParticles();

        view_type particle_view = filteredDist.getParticles();

        pc->create(this->params.np);

        std::cout << this->params.np << std::endl;

        for (int i = 0; i < 5; i++) {
            Circle<T, Dim> added_circle((i + 1) * 0.5);
            added_circle.applyTransformation(shift_to_center);
            circ += added_circle;
        }

        Kokkos::parallel_for(
            "AddParticles", filteredDist.getNumParticles(), KOKKOS_LAMBDA(const int& i) {
                pc->R(i)     = particle_view(i);
                pc->omega(i) = circ.evaluate(pc->R(i));
            });

        Kokkos::fence();
    }

    ~VortexInCellManager() {}

    void par2grid() override {
        std::shared_ptr<FieldContainer<T, 2>> fc    = this->getFieldContainer();
        std::shared_ptr<ParticleContainer<T, 2>> pc = this->getParticleContainer();

        fc->getOmegaField() = 0.0;
        scatter(pc->omega, fc->getOmegaField(), pc->R);
    }

    void grid2par() override {
        std::shared_ptr<FieldContainer<T, 2>> fc    = this->getFieldContainer();
        std::shared_ptr<ParticleContainer<T, 2>> pc = this->getParticleContainer();

        pc->P = 0.0;
        gather(pc->P, fc->getUField(), pc->R);
    }

    void updateFields() {
        std::shared_ptr<FieldContainer<T, 2>> fc = this->getFieldContainer();

        VField_t<T, 2> u_field = fc->getUField();
        u_field                = 0.0;

        const int nghost = u_field.getNghost();
        auto view        = u_field.getView();

        auto omega_view = fc->getOmegaField().getView();
        fc->getOmegaField().fillHalo();

        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j) {
                view(i, j) = {
                    (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * this->params.hr(1)),
                    -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * this->params.hr(0))};
            });
    }

    void advance() override { LeapFrogStep(); }

    void LeapFrogStep() {
        static IpplTimings::TimerRef PTimer      = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer      = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef SolveTimer  = IpplTimings::getTimer("solve");
        // static IpplTimings::TimerRef ETimer           = IpplTimings::getTimer("energy");

        std::shared_ptr<ParticleContainer<T, 2>> pc = this->getParticleContainer();

        par2grid();

        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->solve(this->fcontainer_m);
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        this->updateFields();
        IpplTimings::stopTimer(PTimer);

        grid2par();

        // drift
        IpplTimings::startTimer(RTimer);
        typename ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>::particle_position_type
            R_old_temp = pc->R_old;

        pc->R_old = pc->R;
        pc->R     = R_old_temp + 2 * pc->P * this->params.dt;
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        /*
        IpplTimings::startTimer(ETimer);
        this->computeEnergy();
        IpplTimings::stopTimer(ETimer);
      */
    }

    void dump() override { this->pcontainer_m->dump(this->params.it); }
};

template <typename T>
class VortexInCellManager<T, 3> : public VortexInCellManagerBase<T, 3> {
public:
    VortexInCellManager(SimulationParameters<T, 3> params_)
        : VortexInCellManagerBase<T, 3>(params_) {
        this->setFieldContainer(std::make_shared<FieldContainer<T, 3>>(this->params));

        this->setParticleContainer(std::make_shared<ParticleContainer<T, 3>>(
            this->getFieldContainer()->getMesh(), this->getFieldContainer()->getFL()));

        this->setFieldSolver(std::make_shared<FieldSolver<T, 3, FieldContainer<T, 3>>>());
        this->fsolver_m->initSolver(this->fcontainer_m);

        initParticles();
        this->pcontainer_m->initDump();

        par2grid();
        this->fsolver_m->solve(this->fcontainer_m);
        updateFields();
        grid2par();
    }

    void initParticles() {
        std::cout << "initialize particles 3d" << std::endl;

        std::shared_ptr<ParticleContainer<T, Dim>> pc = this->getParticleContainer();
        Circle<T, Dim> circ(10.0);
        FilteredDistribution<T, Dim> filteredDist(circ, this->params.rmin, this->params.rmax,
                                                  new GridPlacement<T, Dim>(this->params.nr));
        this->params.np = filteredDist.getNumParticles();

        view_type particle_view = filteredDist.getParticles();

        pc->create(this->params.np);

        std::cout << this->params.np << std::endl;

        Kokkos::parallel_for(
            "AddParticles", this->params.np, KOKKOS_LAMBDA(const int& i) {
                pc->R(i)       = particle_view(i);
                pc->omega_x(i) = 0;
                pc->omega_y(i) = 0;
                pc->omega_z(i) = circ.evaluate(pc->R(i));
            });

        Kokkos::fence();
    }

    ~VortexInCellManager() {}

    void par2grid() override {
        std::cout << "3dim par to grid" << std::endl;
        std::shared_ptr<FieldContainer<T, 3>> fc    = this->getFieldContainer();
        std::shared_ptr<ParticleContainer<T, 3>> pc = this->getParticleContainer();

        fc->getOmegaFieldx() = 0.0;
        fc->getOmegaFieldy() = 0.0;
        fc->getOmegaFieldz() = 0.0;

        scatter(pc->omega_x, fc->getOmegaFieldx(), pc->R);
        scatter(pc->omega_y, fc->getOmegaFieldy(), pc->R);
        scatter(pc->omega_z, fc->getOmegaFieldz(), pc->R);
    }

    void grid2par() override {
        std::shared_ptr<FieldContainer<T, 3>> fc    = this->getFieldContainer();
        std::shared_ptr<ParticleContainer<T, 3>> pc = this->getParticleContainer();

        pc->P = 0.0;
        gather(pc->P, fc->getUField(), pc->R);
    }

    void updateFields() {
        std::shared_ptr<FieldContainer<T, 3>> fc = this->getFieldContainer();

        VField_t<T, 3> u_field = fc->getUField();
        u_field                = 0.0;

        const int nghost = u_field.getNghost();
        auto view        = u_field.getView();

        auto omega_view_x = fc->getOmegaFieldx().getView();
        fc->getOmegaFieldx().fillHalo();

        auto omega_view_y = fc->getOmegaFieldy().getView();
        fc->getOmegaFieldy().fillHalo();

        auto omega_view_z = fc->getOmegaFieldz().getView();
        fc->getOmegaFieldz().fillHalo();

        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                view(i, j, k) = {// ux
                                 (omega_view_z(i, j + 1, k) - omega_view_z(i, j - 1, k))
                                         / (2 * this->params.hr(1))
                                     - (omega_view_y(i, j, k + 1) - omega_view_y(i, j, k - 1))
                                           / (2 * this->params.hr(2)),
                                 // uy
                                 (omega_view_x(i, j, k + 1) - omega_view_x(i, j, k - 1))
                                         / (2 * this->params.hr(2))
                                     - (omega_view_z(i + 1, j, k) - omega_view_z(i - 1, j, k))
                                           / (2 * this->params.hr(0)),
                                 // uz
                                 (omega_view_y(i + 1, j, k) - omega_view_y(i - 1, j, k))
                                     / (2 * this->params.hr(0)),
                                 -(omega_view_z(i, j + 1, k) - omega_view_z(i, j - 1, k))
                                     / (2 * this->params.hr(1))};
            });
    }

    void advance() override { LeapFrogStep(); }

    void LeapFrogStep() {
        static IpplTimings::TimerRef PTimer      = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer      = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef SolveTimer  = IpplTimings::getTimer("solve");
        // static IpplTimings::TimerRef ETimer           = IpplTimings::getTimer("energy");

        std::shared_ptr<ParticleContainer<T, 3>> pc = this->getParticleContainer();

        par2grid();

        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->solve(this->fcontainer_m);
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        this->updateFields();
        IpplTimings::stopTimer(PTimer);

        grid2par();

        // drift
        IpplTimings::startTimer(RTimer);
        typename ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>::particle_position_type
            R_old_temp = pc->R_old;

        pc->R_old = pc->R;
        pc->R     = R_old_temp + 2 * pc->P * this->params.dt;
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        /*
        IpplTimings::startTimer(ETimer);
        this->computeEnergy();
        IpplTimings::stopTimer(ETimer);
      */
    }

    void dump() override { this->pcontainer_m->dump(this->params.it); }
};

#endif
