#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <memory>
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "LandauDampingManager.h"
#include "Random/InverseTransformSampling_ND.h" 
 
const char* TestName = "LandauDamping";
 
 // define functions used in sampling particles
struct custom_cdf{
    KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, const double *params) const {
          return x + (params[d*Dim+0] / params[d*Dim+1]) * Kokkos::sin(params[d*Dim+1] * x);
    }
};
struct custom_pdf{
    KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params) const {
          return  1.0 + params[d*Dim+0] * Kokkos::cos(params[d*Dim+1] * x);
    }
};
struct custom_estimate{
    KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params) const {
          return u + params[d]*0.;
    }
};

KOKKOS_FUNCTION
double PDF3D(const Vector_t<double, Dim>& xvec, const double& alpha, const Vector_t<double, Dim>& kw,
           const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * Kokkos::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

class LandauDampingManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>> {
public:
    double loadbalancethreshold_m;
    double time_m;
    LandauDampingManager()
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>>(),totalP(0), nt(0), lbt(0), dt(0),  step_method("LeapFrog"){
    }
    Vector_t<int, Dim> nr;
    size_type totalP;
    int nt;
    double lbt;
    double dt;
    int it;
    std::string step_method;
 public:
     using ParticleContainer_t = ParticleContainer<T, Dim>;
     using FieldContainer_t = FieldContainer<T, Dim>;
     using FieldSolver_t= FieldSolver<T, Dim>;
     using LoadBalancer_t= LoadBalancer<T, Dim>;
	
    std::string solver;
    Vector_t<double, Dim> kw;
    double alpha;
    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax;
    Vector_t<double, Dim> hr;
    double Q;
    Vector_t<double, Dim> origin;
    bool isAllPeriodic;
    bool isFirstRepartition;
private:
    ippl::NDIndex<Dim> domain;
    ippl::e_dim_tag decomp[Dim];

public:
     void pre_step() override {
        Inform m("Pre-step");
        m << "Done" << endl;
    }
    void post_step() override {
        // Update time
        this->time_m += this->dt;
        this->it ++;
        // wrtie solution to output file
        this->dumpLandau();
        
        Inform m("Post-step:");
        m << "Finished time step: " << this->it << " time: " << this->time_m << endl;
         
    }
    void pre_run() override {
         Inform m("Pre Run");
         for (unsigned i = 0; i < Dim; i++) {
            this->domain[i] = ippl::Index(this->nr[i]);
        }
        for (unsigned d = 0; d < Dim; ++d) {
            this->decomp[d] = ippl::PARALLEL;
        }
        this->kw = 0.5;
        this->alpha = 0.05;
        this->rmin = 0.0;
        this->rmax = 2 * pi / this->kw;

        this->hr = this->rmax / this->nr;
        // Q = -\int\int f dx dv
        this->Q = std::reduce(this->rmax.begin(), this->rmax.end(), -1., std::multiplies<double>());
        this->origin = this->rmin;
        this->dt              = std::min(.05, 0.5 * *std::min_element(this->hr.begin(), this->hr.end()));
        this->it = 0;
        
        m << "Discretization:" << endl
            << "nt " << this->nt << " Np= " << this->totalP << " grid = " << this->nr << endl;
        
        Mesh_t<Dim> *mesh;
	mesh = new Mesh_t<Dim>(this->domain, this->hr, this->origin);


        this->isAllPeriodic = true;
        FieldLayout_t<Dim> *FL;
	FL = new FieldLayout_t<Dim>(this->domain, this->decomp, this->isAllPeriodic);
	
	PLayout_t<T, Dim> *PL;
        PL = new PLayout_t<T, Dim>(*FL, *mesh);
        
        if (this->solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }
        
        this->pcontainer_m = std::make_shared<ParticleContainer_t>(*PL);
        this->fcontainer_m = std::make_shared<FieldContainer_t>(this->hr, this->rmin, this->rmax, this->decomp);
        this->fcontainer_m->initializeFields(*mesh, *FL);
        
        this->fsolver_m = std::make_shared<FieldSolver_t>(this->solver, this->fcontainer_m->rho_m, this->fcontainer_m->E_m);
        this->fsolver_m->initSolver();
        this->loadbalancer_m = std::make_shared<LoadBalancer_t>(this->solver, this->lbt, this->fcontainer_m->rho_m, this->fcontainer_m->E_m, *FL, this->pcontainer_m->R);
        
        this->setParticleContainer(pcontainer_m);
        this->setFieldContainer(fcontainer_m);
        this->setFieldSolver(fsolver_m);
        this->setLoadBalancer(loadbalancer_m);
        
        this ->initializeParticles();
        this->fcontainer_m->rho_m = 0.0;
        this->fsolver_m->runSolver();
        this->par2grid();
        this->fsolver_m->runSolver();
        this->grid2par();
        m << "Done";
    }
    
    void initializeParticles(){
        Inform m("Initialize Particles");

        using DistR_t = ippl::random::Distribution<double, Dim, 2*Dim, custom_pdf, custom_cdf, custom_estimate>;
        const double parR[2*Dim] = {alpha, kw[0], alpha, kw[1], alpha, kw[2]};
        DistR_t distR(parR);

        auto mesh = fcontainer_m->rho_m.get_mesh();
        auto FL = fcontainer_m->getLayout();
        if ((this->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            this->isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            const int nghost               = this->fcontainer_m->rho_m.getNghost();
            auto rhoview                   = this->fcontainer_m->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->rho_m.getFieldRangePolicy(),
                KOKKOS_CLASS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    distR.full_pdf(xvec);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition);
        }
        // Sample particle positions:
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(FL, mesh);

        int seed = 42;
        using size_type = ippl::detail::size_type;
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));
        using samplingR_t = ippl::random::sample_its<double, Dim, Kokkos::DefaultExecutionSpace, DistR_t>;
        samplingR_t samplingR(distR, rmax, rmin, rlayout, totalP);
        size_type nloc = samplingR.getLocalNum();
        this->pcontainer_m->create(nloc);
        samplingR.generate(this->pcontainer_m->R.getView(), rand_pool64);

        Kokkos::parallel_for(
            nloc, ippl::random::randn_functor<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      this->pcontainer_m->P.getView(), rand_pool64));

        Kokkos::fence();
        ippl::Comm->barrier();

        this->pcontainer_m->q = this->Q / this->totalP;
        m << "particles created and initial conditions assigned " << endl;
    }

    void advance() override {
            if (this->step_method == "LeapFrog"){
                LeapFrogStep();
            }
    }
    void LeapFrogStep(){
             // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            this->isFirstRepartition = false;
            // kick

            this->pcontainer_m->P = pcontainer_m->P - 0.5 * this->dt * pcontainer_m->E;

            // drift
            this->pcontainer_m->R = pcontainer_m->R + this->dt * pcontainer_m->P;

            // Since the particles have moved spatially update them to correct processors
            this->pcontainer_m->update();

            // Domain Decomposition
            //if (loadbalancer_m->balance(this->totalP, this->it + 1)) {
            //    auto mesh = fcontainer_m->rho_m.get_mesh();
            //    auto FL = fcontainer_m->getLayout();
            //    loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition);
            //}
            // scatter the charge onto the underlying grid
            this->par2grid();
            
            // Field solve
            this->fsolver_m->runSolver();

            // gather E field
            this->grid2par();

            // kick
            this->pcontainer_m->P = this->pcontainer_m->P - 0.5 * this->dt * this->pcontainer_m->E;
    }

    void par2grid() override {
        scatterCIC();
    }

    void grid2par() override {
        gatherCIC();
    }
    
    void gatherCIC() { gather(pcontainer_m->E, fcontainer_m->E_m, pcontainer_m->R); }
    
    void scatterCIC() {
        Inform m("scatter ");

        fcontainer_m->rho_m = 0.0;
        scatter(pcontainer_m->q, fcontainer_m->rho_m, pcontainer_m->R);

         m << std::fabs((this->Q - fcontainer_m->rho_m.sum()) / this->Q)  << endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        double cellVolume =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), 1., std::multiplies<double>());
        fcontainer_m->rho_m = fcontainer_m->rho_m / cellVolume;

        // rho = rho_e - rho_i (only if periodic BCs)
        if (fsolver_m->stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= fcontainer_m->rmax_m[d] - fcontainer_m->rmin_m[d];
            }
            fcontainer_m->rho_m = fcontainer_m->rho_m - (this->Q / size);
        }
    }

    void dumpLandau() { dumpLandau(fcontainer_m->E_m.getView()); }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = fcontainer_m->E_m.getNghost();

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
        MPI_Reduce(&localEx2, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        double fieldEnergy =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        MPI_Reduce(&localExNorm, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());

        
        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname<<"_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if (time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
