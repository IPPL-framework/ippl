#ifndef IPPL_LOAD_BALANCER_H
#define IPPL_LOAD_BALANCER_H

#include <memory>

    template <typename T, unsigned Dim = 3>
    class LoadBalancer   : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{ // inherent from ParticleBase, cause we need getLayout() and update()
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    public:
        std::string stype_m;
        double loadbalancethreshold_m;
        Field_t<Dim> rho_m;
        VField_t<T, Dim> E_m;
        ippl::FieldLayout<Dim> fl_m;
        typename Base::particle_position_type R_m;  // particle velocity
        
        Solver_t<T, Dim> solver_m;
        unsigned int loadbalancefreq_m;
        
        LoadBalancer(std::string solver, double lbs, Field_t<Dim> &rho, VField_t<T, Dim> &E, ippl::FieldLayout<Dim>& fl, typename Base::particle_position_type &R)
           : stype_m(solver), loadbalancethreshold_m(lbs), rho_m(rho), E_m(E), fl_m(fl), R_m(R) {}
        ORB<T, Dim> orb;
    public:
        // Constructor, destructor, and other member functions as needed
        LoadBalancer() {   }

        ~LoadBalancer() {  }
        
     void updateLayout(ippl::FieldLayout<Dim>& fl, ippl::UniformCartesian<T, Dim>& mesh, bool& isFirstRepartition) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        E_m.updateLayout(fl);
        rho_m.updateLayout(fl);
        //if (stype_m == "CG") {
        //     this->phi_m.updateLayout(fl);
        //    phi_m.setFieldBC(allPeriodic);
        //}

        // Update layout with new FieldLayout
        PLayout_t<T, Dim>& layout = this->getLayout();
        layout.updateLayout(fl, mesh);
        IpplTimings::stopTimer(tupdateLayout);
        static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
        IpplTimings::startTimer(tupdatePLayout);
        if (!isFirstRepartition) {
            this->update();
        }
        IpplTimings::stopTimer(tupdatePLayout);
    }

    void initializeORB(ippl::FieldLayout<Dim>& fl, ippl::UniformCartesian<T, Dim>& mesh) {
        orb.initialize(fl, mesh, rho_m);
    }

    void repartition(ippl::FieldLayout<Dim>& fl, ippl::UniformCartesian<T, Dim>& mesh, bool& isFirstRepartition) {
        // Repartition the domains
        bool res = orb.binaryRepartition(R_m, fl, isFirstRepartition);

        if (res != true) {
            std::cout << "Could not repartition!" << std::endl;
            return;
        }
        // Update
        this->updateLayout(fl, mesh, isFirstRepartition);
        if constexpr (Dim == 2 || Dim == 3) {
            if (stype_m == "FFT") {
                std::get<FFTSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
            }
            if constexpr (Dim == 3) {
                if (stype_m == "P3M") {
                    std::get<P3MSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                } else if (stype_m == "OPEN") {
                    std::get<OpenSolver_t<T, Dim>>(solver_m).setRhs(rho_m);
                }
            }
        }
    }

    bool balance(size_type totalP, const unsigned int nstep) {
        if (ippl::Comm->size() < 2) {
            return false;
        }
        if (std::strcmp(TestName, "UniformPlasmaTest") == 0) {
            return (nstep % loadbalancefreq_m == 0);
        } else {
            int local = 0;
            std::vector<int> res(ippl::Comm->size());
            double equalPart = (double)totalP / ippl::Comm->size();
            double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
            if (dev > loadbalancethreshold_m) {
                local = 1;
            }
            MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT,
                          ippl::Comm->getCommunicator());

            for (unsigned int i = 0; i < res.size(); i++) {
                if (res[i] == 1) {
                    return true;
                }
            }
            return false;
        }
    }
    };
    
#endif
