#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>
#include "Manager/BaseManager.h"
#include "Decomposition/OrthogonalRecursiveBisection.h"
#include "Manager/FieldSolverBase.h"

 namespace ippl {

    template <typename T, unsigned Dim, class pc, class fc, class orb>
    class PicManager : public BaseManager {
    public:
        PicManager()
            : BaseManager(), fcontainer_m(nullptr), pcontainer_m(nullptr), loadbalancer_m(nullptr) {}

        virtual ~PicManager() = default;

        virtual void par2grid() = 0;

        virtual void grid2par() = 0;

        void setParticleContainer(std::shared_ptr<pc> pcontainer){
            pcontainer_m = pcontainer;
        }

        void setFieldContainer(std::shared_ptr<fc> fcontainer){
            fcontainer_m = fcontainer;
        }

        void setFieldSolver(std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver) {
            fsolver_m = fsolver;
        }

        void setLoadBalancer(std::shared_ptr<orb> loadbalancer){
            loadbalancer_m = loadbalancer;
        }

    protected:
        std::shared_ptr<fc> fcontainer_m;

        std::shared_ptr<pc> pcontainer_m;

        std::shared_ptr<orb> loadbalancer_m;

        std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver_m;

    };
}  // namespace ippl
 
#endif

