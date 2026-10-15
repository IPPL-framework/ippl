#ifndef IPPL_LANDAU_DAMPING_NEUROP_MANAGER_H
#define IPPL_LANDAU_DAMPING_NEUROP_MANAGER_H

#include <numeric>
#include <pybind11/stl.h>

#include "Interoperability/InferenceBridge.h"
#include "Interoperability/IpplDLPack.h"
#include "LandauDampingManager.h"

namespace py = pybind11;

template <typename T, unsigned Dim>
class LandauDampingNeurOpManager : public LandauDampingManager<T, Dim> {
private:
    ippl::Interoperability::InferenceBridge* bridge_m;

public:
    using Manager_t = LandauDampingManager<T, Dim>;

    LandauDampingNeurOpManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                               std::string& solver_, std::string& stepMethod_,
                               std::vector<std::string>& preconditioner_params_,
                               ippl::Interoperability::InferenceBridge* bridge_)
        : Manager_t(totalP_, nt_, nr_, lbt_, solver_, stepMethod_, preconditioner_params_)
        , bridge_m(bridge_) {}

    void initializeParticlesPD() {
        Inform m("Initialize Particles");

        using DistR_t =
            ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for (unsigned int i = 0; i < Dim; i++) {
            parR[i * 2]     = this->alpha_m;
            parR[i * 2 + 1] = this->kw_m[i];
        }
        DistR_t distR(parR);

        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        const size_type totalP = this->totalP_m;
        const size_type nranks = (size_type)ippl::Comm->size();
        size_type nlocal       = totalP / nranks;
        if ((size_type)ippl::Comm->rank() < (totalP - nlocal * nranks)) {
            ++nlocal;  // distribute the rest over the lowest ranks
        }

        int seed = 42;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        // local copies, since all four InverseTransformSampling parameters are
        // non-const lvalue references, so member/temporary arguments will not bind.
        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> rmax = this->rmax_m;

        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   DistR_t>;
        // this overload (no RegionLayout) samples the whole domain on every rank and
        // generates exactly nlocal particles
        samplingR_t samplingR(distR, rmax, rmin, nlocal);

        this->pcontainer_m->create(nlocal);

        view_type R = this->pcontainer_m->R.getView();
        samplingR.generate(R, rand_pool64);

        view_type P = this->pcontainer_m->P.getView();
        double mu[Dim];
        double sd[Dim];
        for (unsigned int i = 0; i < Dim; i++) {
            mu[i] = 0.0;
            sd[i] = 1.0;
        }
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m / totalP;

        // no migration so no pc->update()
        m << "particles created and initial conditions assigned " << endl;
    }

    void pre_run() override {
        Inform m("Pre Run");

        const double pi = Kokkos::numbers::pi_v<T>;

        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);

        this->kw_m    = 0.5;
        this->alpha_m = 0.05;
        this->rmin_m  = 0.0;
        this->rmax_m  = 2 * pi / this->kw_m;
        this->hr_m    = this->rmax_m / this->nr_m;

        // Q = -\int\int f dx dv
        this->Q_m =
            std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1., std::multiplies<double>());
        this->origin_m = this->rmin_m;
        this->dt_m   = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m   = 0;
        this->time_m = 0.0;
        this->isAllPeriodic_m = true;

        m << "Discretization:" << endl
          << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->setFieldContainer(std::make_shared<typename Manager_t::FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<typename Manager_t::ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL(), false));

        initializeParticlesPD();

        py::dict params;
        params["dim"]            = Dim;
        params["totalP"]         = this->totalP_m;
        params["alpha"]          = this->alpha_m;
        params["dt"]             = this->dt_m;
        params["Q_total"]        = this->Q_m;
        params["q_per_particle"] = this->Q_m / static_cast<double>(this->totalP_m);

        std::vector<double> L(Dim), kw(Dim);
        for (unsigned d = 0; d < Dim; ++d) {
            L[d]  = this->rmax_m[d] - this->rmin_m[d];
            kw[d] = this->kw_m[d];
        }
        params["L"]  = L;
        params["kw"] = kw;

        bridge_m->call("init", ippl::Comm->rank(), ippl::Comm->size(), Kokkos::device_id(), params);

        inferField();
        this->dump();

        m << "Done" << endl;
    }

    void advance() override {
        if (this->stepMethod_m != "LeapFrog")
            throw IpplException("advance()", "Step method not set/Not implemented");

        double dt = this->dt_m;
        auto pc   = this->pcontainer_m;

        pc->P = pc->P - 0.5 * dt * pc->E;  // kick
        pc->R = pc->R + dt * pc->P;        // drift

        //  Apply the periodic BC only to keep nlocal constant for the whole run.
        auto& layout = pc->getLayout();
        layout.applyBC(pc->R, layout.getRegionLayout().getDomain());

        inferField();  // replaced par2grid, pic solver, then grid2par

        pc->P = pc->P - 0.5 * dt * pc->E;
    }

    void inferField() {
        static IpplTimings::TimerRef inferTimer = IpplTimings::getTimer("inference");

        auto pc = this->pcontainer_m;
        Kokkos::fence();

        py::capsule rCap = ippl::Interoperability::attribToDLPackVec(
            pc->R, ippl::Interoperability::ReadOnly::Yes,
            ippl::Interoperability::DLPackVersion::Legacy);

        py::capsule eCap = ippl::Interoperability::attribToDLPackVec(
            pc->E, ippl::Interoperability::ReadOnly::No,
            ippl::Interoperability::DLPackVersion::Legacy);

        IpplTimings::startTimer(inferTimer);
        bridge_m->call("infer", rCap, eCap, pc->getLocalNum());
        Kokkos::fence();
        IpplTimings::stopTimer(inferTimer);
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);

        LandauDampingManager<T, Dim>::dumpLandau();

        IpplTimings::stopTimer(dumpDataTimer);
    }
};
#endif