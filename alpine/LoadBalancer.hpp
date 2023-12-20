#ifndef IPPL_LOAD_BALANCER_H
#define IPPL_LOAD_BALANCER_H

#include "ParticleContainer.hpp"
#include <memory>

template <typename T, unsigned Dim>
class LoadBalancer{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using FieldSolver_t= ippl::FieldSolverBase<T, Dim>;
    private:
        double loadbalancethreshold_m;
        Field_t<Dim>* rho_m;
        VField_t<T, Dim>* E_m;
        Field<T, Dim> *phi_m;
        std::shared_ptr<ParticleContainer<T, Dim>> pc_m;
        std::shared_ptr<FieldSolver_t> fs_m;
        unsigned int loadbalancefreq_m;
        ORB<T, Dim> orb;
    public:
        LoadBalancer(double lbs, std::shared_ptr<FieldContainer<T,Dim>> &fc, std::shared_ptr<ParticleContainer<T, Dim>> &pc, std::shared_ptr<FieldSolver_t> &fs)
           :loadbalancethreshold_m(lbs), rho_m(&fc->getRho()), E_m(&fc->getE()), phi_m(&fc->getPhi()), pc_m(pc), fs_m(fs) {}

        ~LoadBalancer() {  }

        double getLoadBalanceThreshold() const { return loadbalancethreshold_m; }
        void setLoadBalanceThreshold(double threshold) { loadbalancethreshold_m = threshold; }

        Field_t<Dim>* getRho() const { return rho_m; }
        void setRho(Field_t<Dim>* rho) { rho_m = rho; }

        VField_t<T, Dim>* getE() const { return E_m; }
        void setE(VField_t<T, Dim>* E) { E_m = E; }

        Field<T, Dim>* getPhi() { return phi_m; }
        void setPhi(Field<T, Dim>* phi) { phi_m = phi; }

        std::shared_ptr<ParticleContainer<T, Dim>> getParticleContainer() const { return pc_m; }
        void setParticleContainer(std::shared_ptr<ParticleContainer<T, Dim>> pc) { pc_m = pc; }

        std::shared_ptr<FieldSolver_t> getFieldSolver() const { return fs_m; }
        void setFieldSolver(std::shared_ptr<FieldSolver_t> fs) { fs_m = fs; }

        void updateLayout(ippl::FieldLayout<Dim>* fl, ippl::UniformCartesian<T, Dim>* mesh, bool& isFirstRepartition) {
            // Update local fields

            static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
            IpplTimings::startTimer(tupdateLayout);
            (*E_m).updateLayout(*fl);
            (*rho_m).updateLayout(*fl);

            if (fs_m->getStype() == "CG") {
                phi_m->updateLayout(*fl);
                phi_m->setFieldBC(phi_m->getFieldBC());
            }

            // Update layout with new FieldLayout
            PLayout_t<T, Dim>* layout = &pc_m->getLayout();
            (*layout).updateLayout(*fl, *mesh);
            IpplTimings::stopTimer(tupdateLayout);
            static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
            IpplTimings::startTimer(tupdatePLayout);
            if (!isFirstRepartition) {
                pc_m->update();
            }
            IpplTimings::stopTimer(tupdatePLayout);
        }

        void initializeORB(ippl::FieldLayout<Dim>* fl, ippl::UniformCartesian<T, Dim>* mesh) {
            orb.initialize(*fl, *mesh, *rho_m);
        }

        void repartition(ippl::FieldLayout<Dim>* fl, ippl::UniformCartesian<T, Dim>* mesh, bool& isFirstRepartition) {
            // Repartition the domains

            using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
            typename Base::particle_position_type *R;
            R = &pc_m->R;
            bool res = orb.binaryRepartition(*R, *fl, isFirstRepartition);
            if (res != true) {
                std::cout << "Could not repartition!" << std::endl;
                return;
            }
            // Update
            this->updateLayout(fl, mesh, isFirstRepartition);
            if constexpr (Dim == 2 || Dim == 3) {
                if (fs_m->getStype() == "FFT") {
                    std::get<FFTSolver_t<T, Dim>>(fs_m->getSolver()).setRhs(*rho_m);
                }
                if constexpr (Dim == 3) {
                    if (fs_m->getStype() == "P3M") {
                        std::get<P3MSolver_t<T, Dim>>(fs_m->getSolver()).setRhs(*rho_m);
                    } else if (fs_m->getStype() == "OPEN") {
                        std::get<OpenSolver_t<T, Dim>>(fs_m->getSolver()).setRhs(*rho_m);
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
                double dev       = std::abs((double)pc_m->getLocalNum() - equalPart) / totalP;
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
