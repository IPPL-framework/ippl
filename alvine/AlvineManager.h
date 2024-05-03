#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "ParticleFieldStrategy.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "Manager/FieldSolverBase.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AlvineManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainerBase,
                              LoadBalanceStrategy, FieldSolverStrategy<FieldContainerBase>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using particle_field_strategy_type = typename std::shared_ptr<ParticleFieldStrategy<FieldContainerBase, ParticleContainer_t>>;

protected:
    unsigned nt_m;
    unsigned it_m;
    unsigned np_m;
    Vector_t<int, Dim> nr_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::string solver_m;
    double lbt_m;
    particle_field_strategy_type particle_field_strategy_m;

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, std::string& solver_, double lbt_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainerBase, LoadBalanceStrategy, FieldSolverStrategy<FieldContainerBase> >() 
        , nt_m(nt_)
        , nr_m(nr_)
        , solver_m(solver_)
        , lbt_m(lbt_)
        , particle_field_strategy_m(nullptr) {}

    ~AlvineManager(){}

protected:
    double time_m;
    double dt_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> hr_m;

public:

    void setParticleFieldStrategy(particle_field_strategy_type particle_field_strategy) { particle_field_strategy_m = particle_field_strategy; }

    double getTime() { return time_m; }

    void setTime(double time_) { time_m = time_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
      this->time_m += this->dt_m;
      this->it_m++;

      this->dump();
    }

    void grid2par() override { 
        if ( particle_field_strategy_m ) {
            particle_field_strategy_m->grid2par(this->fcontainer_m, this->pcontainer_m);
        } else {
            throw std::runtime_error("Particle-Field strategy not defined");
        }
    }

    void par2grid() override { 
        if ( particle_field_strategy_m ) {
            particle_field_strategy_m->par2grid(this->fcontainer_m, this->pcontainer_m);
        } else {
            throw std::runtime_error("Particle-Field strategy not defined");
        }

    }

    void updateFields() {
        if ( particle_field_strategy_m ) {
            particle_field_strategy_m->updateFields(this->fcontainer_m);
        } else {
            throw std::runtime_error("Particle-Field strategy not defined");
        }

    }

};
#endif
