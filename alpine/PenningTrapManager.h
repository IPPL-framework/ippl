#ifndef IPPL_PENNING_TRAP_MANAGER_H
#define IPPL_PENNING_TRAP_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "AlpineManager.h"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class PenningTrapManager : public AlpineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    PenningTrapManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : AlpineManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~PenningTrapManager(){}

private:
    Vector_t<double, Dim> length_m;
    double Bext_m;
    unsigned int nrMax_m;
    double dxFinest_m;
    double alpha_m;
    double DrInv_m;

public:

    void pre_run() override {
        Inform m("Pre Run");
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }
        this->decomp_m.fill(true);

        this->rmin_m = 0;
        this->rmax_m = 20;

        length_m = this->rmax_m - this->rmin_m;
        this->hr_m     = length_m / this->nr_m;

        this->Q_m      = -1562.5;
        Bext_m   = 5.0;
        this->origin_m = this->rmin_m;

        nrMax_m    = 2048;  // Max grid size in our studies
        dxFinest_m = length_m[0] / nrMax_m;
        this->dt_m = 0.5 * dxFinest_m;  // size of timestep

        this->it_m   = 0;
        this->time_m = 0.0;

        this->alpha_m = -0.5 * this->dt_m;
        DrInv_m = 1.0 / (1 + (std::pow((this->alpha_m * Bext_m), 2)));

        m << "Discretization:" << endl << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer( std::make_shared<FieldContainer_t>( this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m, this->isAllPeriodic_m) );

        this->setParticleContainer( std::make_shared<ParticleContainer_t>( this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()) );

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(), &this->fcontainer_m->getPhi()) );

        this->fsolver_m->initSolver();

        this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        initializeParticles();

        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);

        this->fcontainer_m->getRho() = 0.0;

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);

        this->par2grid();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->dump();

        m << "Done";
    }

    void initializeParticles(){
        Inform m("Initialize Particles");

        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();
        Vector_t<double, Dim> mu, sd;
        for (unsigned d = 0; d < Dim; d++) {
            mu[d] = 0.5 * length_m[d] + this->origin_m[d];
        }
        sd[0] = 0.15 * length_m[0];
        sd[1] = 0.05 * length_m[1];
        sd[2] = 0.20 * length_m[2];

        using DistR_t = ippl::random::NormalDistribution<double, Dim>;
        double parR[2 * Dim];
        for(unsigned int i=0; i<Dim; i++){
            parR[i * 2   ]  = mu[i];
            parR[i * 2 + 1] = sd[i];
        }
        DistR_t distR(parR);

        Vector_t<double, Dim> hr     = this->hr_m;
        Vector_t<double, Dim> origin = this->origin_m;
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            this->isFirstRepartition_m     = true;
            const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
            const int nghost               = this->fcontainer_m->getRho().getNghost();
            auto rhoview                   = this->fcontainer_m->getRho().getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.getFullPdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
            IpplTimings::stopTimer(domainDecomposition);
        }

	static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh);
        size_type totalP = this->totalP_m;
        int seed           = 42;
        using size_type    = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   DistR_t>;
        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> rmax = this->rmax_m;
        samplingR_t samplingR(distR, rmax, rmin, rlayout, totalP);
        size_type nlocal = samplingR.getLocalSamplesNum();

        this->pcontainer_m->create(nlocal);

        view_type* R = &(this->pcontainer_m->R.getView());
        samplingR.generate(*R, rand_pool64);

        view_type* P = &(this->pcontainer_m->P.getView());

        double muP[Dim] = {0.0, 0.0, 0.0};
        double sdP[Dim] = {1.0, 1.0, 1.0};
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, muP, sdP));

        Kokkos::fence();
        ippl::Comm->barrier();

        IpplTimings::stopTimer(particleCreation);

        this->pcontainer_m->q = this->Q_m / this->totalP_m;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
	else{
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");

        double alpha = this->alpha_m;
        double Bext = this->Bext_m;
        double DrInv = this->DrInv_m;
        double V0  = 30 * this->length_m[2];
        Vector_t<double, Dim> length = this->length_m;
        Vector_t<double, Dim> origin = this->origin_m;
        double dt = this->dt_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc = this->fcontainer_m;

        IpplTimings::startTimer(PTimer);
        auto Rview = pc->R.getView();
        auto Pview = pc->P.getView();
        auto Eview = pc->E.getView();
        Kokkos::parallel_for(
               "Kick1", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
                double Eext_x = -(Rview(j)[0] - origin[0] - 0.5 * length[0])
                                * (V0 / (2 * Kokkos::pow(length[2], 2)));
                double Eext_y = -(Rview(j)[1] - origin[1] - 0.5 * length[1])
                                * (V0 / (2 * Kokkos::pow(length[2], 2)));
                double Eext_z = (Rview(j)[2] - origin[2] - 0.5 * length[2])
                                * (V0 / (Kokkos::pow(length[2], 2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];

                Pview(j)[0] += alpha * (Eext_x + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eext_y - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eext_z;
        });
        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(PTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + dt * pc->P;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        size_type totalP = this->totalP_m;
        int it = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
            IpplTimings::startTimer(domainDecomposition);
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL = &fc->getFL();
            this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        // scatter the charge onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        this->grid2par();

        IpplTimings::startTimer(PTimer);
        auto R2view = pc->R.getView();
        auto P2view = pc->P.getView();
        auto E2view = pc->E.getView();
        Kokkos::parallel_for(
           "Kick2", pc->getLocalNum(), KOKKOS_LAMBDA(const size_t j) {
           double Eext_x = -(R2view(j)[0] - origin[0] - 0.5 * length[0])
                         * (V0 / (2 * Kokkos::pow(length[2], 2)));
           double Eext_y = -(R2view(j)[1] - origin[1] - 0.5 * length[1])
                          * (V0 / (2 * Kokkos::pow(length[2], 2)));
           double Eext_z = (R2view(j)[2] - origin[2] - 0.5 * length[2])
                           * (V0 / (Kokkos::pow(length[2], 2)));

           Eext_x += E2view(j)[0];
           Eext_y += E2view(j)[1];
           Eext_z += E2view(j)[2];

           P2view(j)[0] = DrInv * (P2view(j)[0] + alpha * (Eext_x + P2view(j)[1] * Bext + alpha * Bext * Eext_y));
           P2view(j)[1] = DrInv * (P2view(j)[1] + alpha * (Eext_y - P2view(j)[0] * Bext - alpha * Bext * Eext_x));
           P2view(j)[2] += alpha * Eext_z;
        });
        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(PTimer);
    }

    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);
        dumpData();
        IpplTimings::stopTimer(dumpDataTimer);
    }

    void dumpData() {
        auto Pview                   = this->pcontainer_m->P.getView();
        double kinEnergy             = 0.0;
        double potEnergy             = 0.0;
        this->fcontainer_m->getRho() = dot(this->fcontainer_m->getE(), this->fcontainer_m->getE());
        potEnergy = 0.5 * this->hr_m[0] * this->hr_m[1] * this->hr_m[2] * this->fcontainer_m->getRho().sum();

        Kokkos::parallel_reduce(
            "Particle Kinetic Energy", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(kinEnergy));

        kinEnergy *= 0.5;
        double gkinEnergy = 0.0;

        ippl::Comm->reduce(kinEnergy, gkinEnergy, 1, std::plus<double>());

        const int nghostE = this->fcontainer_m->getE().getNghost();
        auto Eview        = this->fcontainer_m->getE().getView();
        Vector_t<T, Dim> normE;

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        for (unsigned d = 0; d < Dim; ++d) {
            T temp = 0.0;
            ippl::parallel_reduce(
                "Vector E reduce", ippl::getRangePolicy(Eview, nghostE),
                KOKKOS_LAMBDA(const index_array_type& args, T& valL) {
                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    T myVal = std::pow(ippl::apply(Eview, args)[d], 2);
                    valL += myVal;
                },
                Kokkos::Sum<T>(temp));
            Kokkos::fence();
            T globaltemp          = 0.0;
            ippl::Comm->reduce(temp, globaltemp, 1, std::plus<double>());

            normE[d] = std::sqrt(globaltemp);
            ippl::Comm->barrier();
        }

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/ParticleField_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(10);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if ( std::fabs(this->time_m) < 1e-14 ) {
                csvout << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2";
                for (unsigned d = 0; d < Dim; d++) {
                    csvout << ", E" << static_cast<char>((Dim <= 3 ? 'x' : '1') + d) << "_norm2";
                }
                csvout << endl;
            }

            csvout << this->time_m << " " << potEnergy << " " << gkinEnergy << " "
                   << potEnergy + gkinEnergy << " " << this->rhoNorm_m << " ";
            for (unsigned d = 0; d < Dim; d++) {
                csvout << normE[d] << " ";
            }
            csvout << endl;
            csvout.flush();
        }
        ippl::Comm->barrier();
    }
};
#endif
