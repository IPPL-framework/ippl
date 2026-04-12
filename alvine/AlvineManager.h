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
    Vector_t<int, Dim> nr_m;
    unsigned np_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::string solver_m;
    int dump_freq_m;

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_, int dump_freq_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>() 
        , nt_m(nt_)
        , nr_m(nr_)
        , np_m(np_)
        , solver_m(solver_)
	, dump_freq_m(dump_freq_) {}

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
    }

    void post_step() override {
      Inform m("Step: ");
      this->time_m += this->dt_m;
      this->it_m++;

      if(this->it_m % dump_freq_m == 0) {
      	this->dump();
      }
      m << this->it_m << " Done" << endl;
    }

    void grid2par() override { 
	gatherCIC(); 
    }

    void gatherCIC() {
      this->pcontainer_m->P = 0.0;
      gather(this->pcontainer_m->P, this->fcontainer_m->getUField(), this->pcontainer_m->R);
    }

    void par2grid() override {
	scatterCIC(); 
    }

    void computeVelocityField() {

      VField_t<T, Dim> u_field = this->fcontainer_m->getUField();
      u_field = 0.0;

      if constexpr (Dim == 2) {
        const int nghost = u_field.getNghost();
        auto view = u_field.getView();

        auto omega_view = this->fcontainer_m->getOmegaField().getView();
        this->fcontainer_m->getOmegaField().fillHalo();

        Vector_t<double, Dim> hr = hr_m;
        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j) {
                view(i, j) = {
                    (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * hr(1)), 
                    -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * hr(0))
                };

            });
      } else if constexpr (Dim == 3) {
        //TODO compute velocity field in 3D, this should be a simple curl operation (one line)
      }
    }

double computeParticleCirculation() {
    double gamma_local = 0.0;

    auto omega_view = this->pcontainer_m->omega.getView();
    auto nlocal = this->pcontainer_m->getLocalNum();

    Kokkos::parallel_reduce(
        "particle_circulation",
        nlocal,
        KOKKOS_LAMBDA(const int i, double& lsum) {
            lsum += omega_view(i);
        },
        gamma_local
    );

    double gamma_global = 0.0;
    ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

    return gamma_global;
}


double computeGridCirculation() {
    double gamma_local = 0.0;

    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega_view = omegaField.getView();

    const double dA = hr_m[0] * hr_m[1];
    const int nghost = omegaField.getNghost();

    Kokkos::parallel_reduce(
        "grid_circulation",
        ippl::getRangePolicy(omega_view, nghost),
        KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
            lsum += omega_view(i, j);
        },
        gamma_local
    );

    gamma_local *= dA;

    double gamma_global = 0.0;
    ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

    return gamma_global;
}

void checkCirculationConservation(double relError, Inform& m) {
    size_type TotalParticles = 0;
    size_type localParticles = this->pcontainer_m->getLocalNum();

    ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

    if (ippl::Comm->rank() == 0) {
        if (TotalParticles != np_m || relError > 1e-12) {
            m << "Time step: " << it_m << endl;
            m << "Total particles expected: " << np_m
              << " after update: " << TotalParticles << endl;
            m << "Rel. error in circulation conservation: " << relError << endl;
            ippl::Comm->abort();
        }
    }
}


void scatterCIC() {
    Inform m("scatter ");

    this->fcontainer_m->getOmegaField() = 0.0;

    if constexpr (Dim == 2) {
        // Scatter particle strengths to grid
        scatter(this->pcontainer_m->omega,
                this->fcontainer_m->getOmegaField(),
                this->pcontainer_m->R);

        // Convert deposited circulation to vorticity density
        this->fcontainer_m->getOmegaField() =
            this->fcontainer_m->getOmegaField() / (hr_m[0] * hr_m[1]);

        // Conservation check
        double gammaParticles = computeParticleCirculation();
        double gammaGrid      = computeGridCirculation();

        double relError = std::fabs((gammaParticles - gammaGrid) /
                                    std::max(std::fabs(gammaParticles), 1e-30));

        m << "particle circulation = " << gammaParticles
          << ", grid circulation = " << gammaGrid
          << ", relError = " << relError << endl;

        checkCirculationConservation(relError, m);

    } else if constexpr (Dim == 3) {
        // TODO 3D version
    }
}
};
#endif
