#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

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

const char* TestName = "LandauDamping";

// define functions used in sampling particles
struct CustomDistributionFunctions {
  struct CDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params_p) const {
           return x + (params_p[d * 2 + 0] / params_p[d * 2 + 1]) * Kokkos::sin(params_p[d * 2 + 1] * x);
       }
  };

  struct PDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params_p) const {
           return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
       }
  };

  struct Estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params_p) const {
            return u + params_p[d] * 0.;
	}
  };
};

class LandauDampingManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
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
    LandauDampingManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_, std::string& solver_, std::string& stepMethod_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_){}
    ~LandauDampingManager(){}

private:
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
        kw_m    = 0.5;
        alpha_m = 0.05;
        rmin_m  = 0.0;
        rmax_m  = 2 * pi / kw_m;

        hr_m = rmax_m / nr_m;
        // Q = -\int\int f dx dv
        Q_m = std::reduce(rmax_m.begin(), rmax_m.end(), -1., std::multiplies<double>());
        origin_m = rmin_m;
        dt_m     = std::min(.05, 0.5 * *std::min_element(hr_m.begin(), hr_m.end()));
        it_m     = 0;
        time_m   = 0.0;

        m << "Discretization:" << endl
          << "nt " << nt_m << " Np= " << totalP_m << " grid = " << nr_m << endl;

        std::shared_ptr<Mesh_t<Dim>> mesh = std::make_shared<Mesh_t<Dim>>(domain_m, hr_m, origin_m);

        isAllPeriodic_m = true;

        std::shared_ptr<FieldLayout_t<Dim>> FL = std::make_shared<FieldLayout_t<Dim>>(MPI_COMM_WORLD, domain_m, decomp_m, isAllPeriodic_m);

        std::shared_ptr<PLayout_t<T, Dim>> PL = std::make_shared<PLayout_t<T, Dim>>(*FL, *mesh);

        if (solver_m == "OPEN") {
            throw IpplException("LandauDamping", "Open boundaries solver incompatible with this simulation!");
        }

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

        using DistR_t = ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
        double parR[2 * Dim];
        for(unsigned int i=0; i<Dim; i++){
            parR[i * 2   ]  = alpha_m;
            parR[i * 2 + 1] = kw_m[i];
        }
        DistR_t distR(parR);

        Vector_t<double, Dim> kw     = kw_m;
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
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>( *FL, *mesh );

        // unsigned int
        size_type totalP = totalP_m;
        int seed           = 42;
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

        double mu[Dim];
        double sd[Dim];
        for(unsigned int i=0; i<Dim; i++){
            mu[i] = 0.0;
            sd[i] = 1.0;
        }
        Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, mu, sd));
        Kokkos::fence();
        ippl::Comm->barrier();

        pcontainer_m->q = Q_m/totalP;
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

        double dt                               = dt_m;
        std::shared_ptr<ParticleContainer_t> pc = pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = fcontainer_m;

        pc->P = pc->P - 0.5 * dt * pc->E;

        // drift
        pc->R = pc->R + dt * pc->P;

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        size_type totalP        = totalP_m;
        int it                  = it_m;
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

        // kick
        pc->P = pc->P - 0.5 * dt * pc->E;
    }

    void par2grid() override { scatterCIC(); }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        Base::particle_position_type *Ep = &pcontainer_m->E;
        Base::particle_position_type *R  = &pcontainer_m->R;
        VField_t<T, Dim> *Ef             = &fcontainer_m->getE();
        gather(*Ep, *Ef, *R);
    }

    void scatterCIC() {
        Inform m("scatter ");
        fcontainer_m->getRho() = 0.0;

        using Base                        = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        ippl::ParticleAttrib<double> *q = &pcontainer_m->q;
        Base::particle_position_type *R = &pcontainer_m->R;
        Field_t<Dim> *rho               = &fcontainer_m->getRho();
        double Q                        = Q_m;
        Vector_t<double, Dim> rmin      = rmin_m;
        Vector_t<double, Dim> rmax      = rmax_m;
        Vector_t<double, Dim> hr        = hr_m;

        scatter(*q, *rho, *R);
        double relError = std::fabs((Q-(*rho).sum())/Q);

        m << relError << endl;

        size_type TotalParticles = 0;
        size_type localParticles = pcontainer_m->getLocalNum();

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

        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)          = (*rho) / cellVolume;

        // rho = rho_e - rho_i (only if periodic BCs)
        if (fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (Q / size);
        }
    }

    void dump() { dumpLandau(fcontainer_m->getE().getView()); }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = fcontainer_m->getE().getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Eview, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(Eview, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(fcontainer_m->getHr().begin(), fcontainer_m->getHr().end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if ( std::fabs(time_m) < 1e-14 ) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
