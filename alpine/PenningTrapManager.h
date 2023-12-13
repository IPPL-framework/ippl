#ifndef IPPL_PENNING_TRAP_MANAGER_H
#define IPPL_PENNING_TRAP_MANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

const char* TestName = "PenningTrap";

class PenningTrapManager
    : public ippl::PicManager<double, 3, ParticleContainer<double, 3>, FieldContainer<double, 3>,
                              LoadBalancer<double, 3>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
private:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    std::string stepMethod_m;
public:
    PenningTrapManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_, std::string& solver_, std::string& stepMethod_)
        : ippl::PicManager<double, 3, ParticleContainer<double, 3>, FieldContainer<double, 3>, LoadBalancer<double, 3>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_){}
    ~PenningTrapManager(){}
private:
    double time_m;
    double dt_m;
    int it_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> length_m;
    Vector_t<double, Dim> hr_m;
    double Q_m;
    double Bext_m;
    Vector_t<double, Dim> origin_m;
    unsigned int nrMax_m;
    double dxFinest_m;
    bool isAllPeriodic_m;
    bool isFirstRepartition_m;
    double alpha_m;
    double DrInv_m;
    double rhoNorm_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;

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

    void setStepMethod(const std::string& step_method_) { stepMethod_m = step_method_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }
    void post_step() override {
        // Update time
        time_m += dt_m;
        it_m++;
        // wrtie solution to output file
        dump();

        Inform m("Post-step:");
        m << "Finished time step: " << it_m << " time: " << time_m << endl;
    }
    void pre_run() override {
        Inform m("Pre Run");
        for (unsigned i = 0; i < Dim; i++) {
            domain_m[i] = ippl::Index(nr_m[i]);
        }
        decomp_m.fill(true);

        rmin_m = 0;
        rmax_m = 20;

        length_m = rmax_m - rmin_m;
        hr_m     = length_m / nr_m;

        Q_m      = -1562.5;
        Bext_m   = 5.0;
        origin_m = rmin_m;

        nrMax_m    = 2048;  // Max grid size in our studies
        dxFinest_m = length_m[0] / nrMax_m;
        dt_m       = 0.5 * dxFinest_m;  // size of timestep

        it_m = 0;
        time_m = 0.0;

        alpha_m = -0.5 * dt_m;
        DrInv_m = 1.0 / (1 + (std::pow((alpha_m * Bext_m), 2)));

        m << "Discretization:" << endl << "nt " << nt_m << " Np= " << totalP_m << " grid = " << nr_m << endl;

        isAllPeriodic_m = true;

        std::shared_ptr<Mesh_t<Dim>> mesh = std::make_shared<Mesh_t<Dim>>(domain_m, hr_m, origin_m);

        std::shared_ptr<FieldLayout_t<Dim>> FL = std::make_shared<FieldLayout_t<Dim>>(MPI_COMM_WORLD, domain_m, decomp_m, isAllPeriodic_m);

        std::shared_ptr<PLayout_t<T, Dim>> PL = std::make_shared<PLayout_t<T, Dim>>(*FL, *mesh);

        setParticleContainer( std::make_shared<ParticleContainer_t>(PL) );

        setFieldContainer( std::make_shared<FieldContainer_t>(hr_m, rmin_m, rmax_m, decomp_m) );

        fcontainer_m->initializeFields(mesh, FL, solver_m);

        setFieldSolver( std::make_shared<FieldSolver_t>(solver_m, &fcontainer_m->getRho(), &fcontainer_m->getE(), &fcontainer_m->getPhi()) );

        fsolver_m->initSolver();

        setLoadBalancer( std::make_shared<LoadBalancer_t>( lbt_m, fcontainer_m, pcontainer_m, fsolver_m) );

        initializeParticles(mesh, FL);

        fcontainer_m->getRho() = 0.0;

        fsolver_m->runSolver();

        par2grid();

        fsolver_m->runSolver();

        grid2par();

        dump();

        m << "Done";
    }

    void initializeParticles(std::shared_ptr<Mesh_t<Dim>> mesh, std::shared_ptr<FieldLayout_t<Dim>> FL){
        Inform m("Initialize Particles");

        Vector_t<double, Dim> mu, sd;
        for (unsigned d = 0; d < Dim; d++) {
            mu[d] = 0.5 * length_m[d] + origin_m[d];
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

        Vector_t<double, Dim> hr     = hr_m;
        Vector_t<double, Dim> origin = origin_m;
        if ((lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            isFirstRepartition_m           = true;
            const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
            const int nghost               = fcontainer_m->getRho().getNghost();
            auto rhoview                   = fcontainer_m->getRho().getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = distR.getFullPdf(xvec);
                });

            Kokkos::fence();

            loadbalancer_m->initializeORB(FL.get(), mesh.get());
            loadbalancer_m->repartition(FL.get(), mesh.get(), isFirstRepartition_m);
        }

        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh);
        size_type totalP = totalP_m;
        int seed           = 42;
        using size_type    = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        using samplingR_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   DistR_t>;
        Vector_t<double, Dim> rmin = rmin_m;
        Vector_t<double, Dim> rmax = rmax_m;
        samplingR_t samplingR(distR, rmax, rmin, rlayout, totalP);
        size_type nlocal = samplingR.getLocalSamplesNum();

        pcontainer_m->create(nlocal);

        view_type* R = &(pcontainer_m->R.getView());
        samplingR.generate(*R, rand_pool64);

        view_type* P = &(pcontainer_m->P.getView());

        double muP[Dim] = {0.0, 0.0, 0.0};
        double sdP[Dim] = {1.0, 1.0, 1.0};
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, muP, sdP));

        Kokkos::fence();
        ippl::Comm->barrier();

        pcontainer_m->q = Q_m / totalP_m;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
        if (stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        Inform m("LeapFrog");

        double alpha = this->alpha_m;
        double Bext = this->Bext_m;
        double DrInv = this->DrInv_m;
        double V0  = 30 * this->length_m[2];
        Vector_t<double, Dim> length = this->length_m;
        Vector_t<double, Dim> origin = origin_m;
        double dt = dt_m;
        std::shared_ptr<ParticleContainer_t> pc = pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc = fcontainer_m;

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

        // drift
        pc->R = pc->R + dt * pc->P;

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        size_type totalP = totalP_m;
        int it = it_m;
        bool isFirstRepartition = false;
        if (loadbalancer_m->balance(totalP, it + 1)) {
            auto* mesh = &fc->getRho().get_mesh();
            auto* FL = &fc->getFL();
            loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
        }

        // scatter the charge onto the underlying grid
        par2grid();

        // Field solve
        fsolver_m->runSolver();

        // gather E field
        grid2par();

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
    }

    void par2grid() override { scatterCIC(); }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        Base::particle_position_type *Ep = &pcontainer_m->E;
        Base::particle_position_type *R = &pcontainer_m->R;
        VField_t<T, Dim> *Ef             = &fcontainer_m->getE();
        gather(*Ep, *Ef, *R);
    }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->getRho() = 0.0;

        using Base                      = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        ippl::ParticleAttrib<double> *q = &pcontainer_m->q;
        Base::particle_position_type *R = &pcontainer_m->R;
        Field_t<Dim> *rho               = &fcontainer_m->getRho();
        double Q                        = Q_m;
        Vector_t<double, Dim> rmin      = rmin_m;
        Vector_t<double, Dim> rmax      = rmax_m;
        Vector_t<double, Dim> hr        = hr_m;

        scatter(*q, *rho, *R);
        m << std::fabs((Q - (*rho).sum()) / Q) << endl;

        size_type TotalParticles = 0;
        size_type localParticles = pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)          = (*rho) / cellVolume;

        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (Q / size);
        }
    }

    void dump() { dumpData(); }

    void dumpData() {
        auto Pview                   = pcontainer_m->P.getView();
        double kinEnergy             = 0.0;
        double potEnergy             = 0.0;
        fcontainer_m->getRho()       = dot(fcontainer_m->getE(), fcontainer_m->getE());
        potEnergy = 0.5 * hr_m[0] * hr_m[1] * hr_m[2] * fcontainer_m->getRho().sum();

        Kokkos::parallel_reduce(
            "Particle Kinetic Energy", pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const int i, double& valL) {
                double myVal = dot(Pview(i), Pview(i)).apply();
                valL += myVal;
            },
            Kokkos::Sum<double>(kinEnergy));

        kinEnergy *= 0.5;
        double gkinEnergy = 0.0;

        ippl::Comm->reduce(kinEnergy, gkinEnergy, 1, std::plus<double>());

        const int nghostE = fcontainer_m->getE().getNghost();
        auto Eview        = fcontainer_m->getE().getView();
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

            if ( std::fabs(time_m) < 1e-14 ) {
                csvout << "time, Potential energy, Kinetic energy, Total energy, Rho_norm2";
                for (unsigned d = 0; d < Dim; d++) {
                    csvout << ", E" << static_cast<char>((Dim <= 3 ? 'x' : '1') + d) << "_norm2";
                }
                csvout << endl;
            }

            csvout << time_m << " " << potEnergy << " " << gkinEnergy << " "
                   << potEnergy + gkinEnergy << " " << rhoNorm_m << " ";
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
