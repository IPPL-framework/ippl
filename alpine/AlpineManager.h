#ifndef IPPL_ALPINE_MANAGER_H
#define IPPL_ALPINE_MANAGER_H

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
#include "FEM/FEMInterpolate.hpp"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AlpineManager : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>,
                                              FieldContainer<T, Dim>, LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base                = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

protected:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    std::string stepMethod_m;
    std::vector<std::string> preconditioner_params_m;
public:
    AlpineManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                  std::string& solver_, std::string& stepMethod_,
                  std::vector<std::string> preconditioner_params = {})
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_)
        , preconditioner_params_m(preconditioner_params) {}
    ~AlpineManager() {}

protected:
    double time_m;
    double dt_m;
    int it_m;
    Vector_t<double, Dim> kw_m;
    double alpha_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> hr_m;
    double Q_m;
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    bool isFirstRepartition_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
    double rhoNorm_m;

public:
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const std::string& getSolver() const { return solver_m; }

    void setSolver(const std::string& solver_) { solver_m = solver_; }

    double getLoadBalanceThreshold() const { return lbt_m; }

    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    const std::string& getStepMethod() const { return stepMethod_m; }

    void setStepMethod(const std::string& stepMethod_) { stepMethod_m = stepMethod_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    std::vector<std::string> getPreconditionerParams() const { return preconditioner_params_m; };

    virtual void dump(){/* default does nothing */};

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }

    void post_step() override {
        // Update time
        this->time_m += this->dt_m;
        this->it_m++;
        // write solution to output file
        this->dump();

        Inform m("Post-step:");
        m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
    }

    void grid2par() override { 
        if ((getSolver() == "FEM") || (getSolver() == "FEM_PRECON")) {
            gatherFEM();
        } else {
            gatherCIC();
        }
    }

    void gatherCIC() {
        gather(this->pcontainer_m->E, this->fcontainer_m->getE(), this->pcontainer_m->R);
    }

    void gatherFEM() {
        using exec_space = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        size_type localParticles = this->pcontainer_m->getLocalNum();
        policy_type iteration_policy(0, localParticles);

        // Since the interpolation is agnostic to preconditioning, we can keep 
        // this hard-coded without any ifs (FEM or FEM_PRECON).
        auto& space = (std::get<FEMSolver_t<T, Dim>>(this->fsolver_m->getSolver())).getSpace();

        interpolate_grad_to_diracs(this->pcontainer_m->E, this->fcontainer_m->getPhi(),
                                   this->pcontainer_m->R, space, iteration_policy);
    }

    void par2grid() override {
        if (getSolver() == "FEM") {
            scatterFEM();
        } else {
            scatterCIC();
        }
    }

    void scatterCIC() {
        Inform m("scatter ");

        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* q          = &this->pcontainer_m->q;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        double Q                                 = Q_m;

        scatter(*q, *rho, *R);

        double relError = std::fabs((Q - (*rho).sum()) / Q);
        m << relError << endl;

        checkChargeConservation(relError, m);

        getDensity(rho);
    }

    void scatterFEM() {
        Inform m("scatter ");

        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double>* q          = &this->pcontainer_m->q;
        typename Base::particle_position_type* R = &this->pcontainer_m->R;
        Field_t<Dim>* rho                        = &this->fcontainer_m->getRho();
        double Q                                 = Q_m;
        size_type localParticles                 = this->pcontainer_m->getLocalNum();

        using exec_space = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        policy_type iteration_policy(0, localParticles);

        // Since the interpolation is agnostic to preconditioning, we can keep 
        // this hard-coded without any ifs (FEM or FEM_PRECON).
        auto& space = (std::get<FEMSolver_t<T, Dim>>(this->fsolver_m->getSolver())).getSpace();

        assemble_rhs_from_particles(*q, *rho, *R, space, iteration_policy);

        double relError = std::fabs((Q - (*rho).sum()) / Q);
        m << relError << endl;

        double num = 1e-14;
        checkChargeConservation(num, m);

        getDensity(rho);
    }

    void checkChargeConservation(double& relError, Inform& m) {
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
    }

    void getDensity(Field_t<Dim>* rho) {
        Vector_t<double, Dim> rmin               = rmin_m;
        Vector_t<double, Dim> rmax               = rmax_m;
        Vector_t<double, Dim> hr                 = this->hr_m;
        double Q                                 = Q_m;

        if ((this->fsolver_m->getStype() != "FEM") || (this->fsolver_m->getStype() != "FEM_PRECON")) {
            double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
            (*rho)            = (*rho) / cellVolume;
        }

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (Q / size);
        }
    }
};
#endif
