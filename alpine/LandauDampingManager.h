#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <memory>
#include "Manager/BaseManager.h"

//template <typename T, unsigned Dim>
class LandauDampingManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>> {
public:
    double loadbalancethreshold_m;
    double time_m;
    LandauDampingManager()
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>>(),totalP(0), nt(0), lbs(0), dt(0) {
    }
    Vector_t<int, Dim> nr;
    size_type totalP;
    int nt;
    double lbs;
    double dt;
    int it;
 public:
    std::string solver;

    ippl::NDIndex<Dim> domain;
    ippl::e_dim_tag decomp[Dim];
    
    Vector_t<double, Dim> kw;
    double alpha;
    Vector_t<double, Dim> rmin;
     Vector_t<double, Dim> rmax;

    Vector_t<double, Dim> hr;
    double Q;
    Vector_t<double, Dim> origin;

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
    }
    /*
    void advance(Mesh_t<Dim> mesh, FieldLayout_t<Dim> FL, PLayout_t<double, Dim> PL) override {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            bool isFirstRepartition = false;
            // kick

            pcontainer_m->P = pcontainer_m->P - 0.5 * this->dt * pcontainer_m->E;

            // drift
            pcontainer_m->R = pcontainer_m->R + this->dt * pcontainer_m->P;

            // Since the particles have moved spatially update them to correct processors
            pcontainer_m->update();

            // Domain Decomposition
            if (loadbalancer_m->balance(this->totalP, this->it + 1)) {
                loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            }

            // scatter the charge onto the underlying grid
            this->par2grid();
            
            // Field solve
            fcontainer_m->runSolver();

            // gather E field
            this->grid2par();

            // kick
            pcontainer_m->P = pcontainer_m->P - 0.5 * this->dt * pcontainer_m->E;


            this->time_m += this->dt;            
            this->dumpLandau();
    }*/
        
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
