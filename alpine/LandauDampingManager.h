#ifndef IPPL_LANDAU_DAMPING_MANAGER_H
#define IPPL_LANDAU_DAMPING_MANAGER_H

#include <memory>
#include "Manager/BaseManager.h"

//template <typename T, unsigned Dim>
class LandauDampingManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>> {
public:
    double loadbalancethreshold_m;
    double time_m;
    double Q_m;
    LandauDampingManager()
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>, LoadBalancer<double, 3>>(){
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

         m << std::fabs((Q_m - fcontainer_m->rho_m.sum()) / Q_m)  << endl;

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
            fcontainer_m->rho_m = fcontainer_m->rho_m - (Q_m / size);
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
