#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AlvineManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
protected:
    unsigned nt_m;
    unsigned it_m;
    unsigned np_m;
    Vector_t<int, Dim> nr_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>() 
        , nt_m(nt_)
        , nr_m(nr_) {}

    ~AlvineManager(){}

protected:
    double time_m;
    double dt_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> hr_m;

public:

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

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
    }

    void par2grid() override { scatterCIC(); }

    void scatterCIC() {
      scatter(this->pcontainer_m->omega_m, this->fcontainer_m->getOmega_field(), this->pcontainer_m->R);
    }
};
#endif
