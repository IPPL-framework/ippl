#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <memory>
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "LandauDampingManager.h"
#include "Random/InverseTransformSampling.h" 
 
const char* TestName = "LandauDamping";
 
 // define functions used in sampling particles
KOKKOS_FUNCTION
double CDF(double y, double alpha, double k) {
    return y + (alpha / k) * std::sin(k * y);
}

KOKKOS_FUNCTION
double PDF(double y, double alpha, double k) {
    return  (1.0 + alpha * Kokkos::cos(k * y));
}

KOKKOS_FUNCTION
double ESTIMATE(double u, double alpha) {
    return u/(1+alpha); // maybe E[x] is good enough as the first guess
}

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
    Mesh_t<Dim> mesh;
    FieldLayout_t<Dim> FL;
    PLayout_t<T, Dim> PL;
    ippl::detail::RegionLayout<double, 3, Mesh_t<3>> rlayout;
    std::shared_ptr<ParticleContainer_t> pc;
    
public:
    void post_step() override {
        // Update time
        this->time_m += this->dt;
        // wrtie solution to output file
        this->dumpLandau();
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
        
        // Create Mesh, FieldLayout, ParticleLayout
        this->mesh = Mesh_t<Dim>(this->domain, this->hr, this->origin);
        this->isAllPeriodic = true;
        this->FL = FieldLayout_t<Dim>(this->domain, this->decomp, this->isAllPeriodic);
        this->PL = PLayout_t<T, Dim>(this->FL, this->mesh);
        
        if (this->solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }
       
        this->pcontainer_m = std::make_shared<ParticleContainer_t>(this->PL);
        this->fcontainer_m = std::make_shared<FieldContainer_t>(this->hr, this->rmin, this->rmax, this->decomp);
        this->fcontainer_m->initializeFields(this->mesh, this->FL);
        this->fsolver_m = std::make_shared<FieldSolver_t>(this->solver, this->fcontainer_m->rho_m, this->fcontainer_m->E_m);
        this->fsolver_m->initSolver();
        this->loadbalancer_m = std::make_shared<LoadBalancer_t>(this->solver, this->lbt, this->fcontainer_m->rho_m, this->fcontainer_m->E_m, this->FL, this->pcontainer_m->R);
                
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
    }
    void initializeParticles(){
        Inform m("Initialize Particles");
        if ((this->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            m << "Starting first repartition" << endl;
            this->isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = this->FL.getLocalNDIndex();
            const int nghost               = this->fcontainer_m->rho_m.getNghost();
            auto rhoview                   = this->fcontainer_m->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * this->hr + this->origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = PDF3D(xvec, this->alpha, this->kw, Dim);
                });

            Kokkos::fence();

            this->loadbalancer_m->initializeORB(this->FL, this->mesh);
            this->loadbalancer_m->repartition(this->FL, this->mesh, this->isFirstRepartition);
        }
        
        
         // Sample particle positions:
         this->rlayout = ippl::detail::RegionLayout<double, 3, Mesh_t<3>>(FL, mesh);
         using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 3, Kokkos::DefaultExecutionSpace>;
         ippl::random::Distribution<double, 3> distR;
         for(int d=0; d<3; d++){
                 double k = this->kw[d];
                 double alpha = this->alpha;
		 distR.setCdfFunction(d, [alpha, k](double y) { return CDF(y, alpha, k);});
		 distR.setPdfFunction(d, [alpha, k](double y) { return PDF(y, alpha, k);});
		 distR.setEstimationFunction(d, [alpha](double u) { return ESTIMATE(u, alpha);});
         }
         InvTransSampl_t its(this->rmin, this->rmax, rlayout, distR, this->totalP);
         unsigned int nloc = its.getLocalNum();
         this->pcontainer_m->create(nloc);
         its.generate(distR, this->pcontainer_m->R.getView(), 42 + 100 * ippl::Comm->rank());
         
         // Sample particle velocity:
         // Box-Muller method
         //Kokkos::parallel_for(
         //   nloc, ippl::random::generate_random_normal<double, Kokkos::DefaultExecutionSpace, 3>(
         //            pcontainer_m->P.getView(), 0.0, 1.0, 42 + 100 * ippl::Comm->rank()));
         
         // standard method
         Kokkos::Random_XorShift64_Pool<> rand_pool64_((size_type)(42 + 100 * ippl::Comm->rank()));
         Kokkos::parallel_for(
            nloc, ippl::random::generate_random_normal_basic<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                     this->pcontainer_m->P.getView(), rand_pool64_));
        
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
            if (loadbalancer_m->balance(this->totalP, this->it + 1)) {
                loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition);
            }

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
            fname<<"_test";
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
