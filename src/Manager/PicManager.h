#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>

#include "Manager/BaseManager.h"

#include "Decomposition/OrthogonalRecursiveBisection.h"
//#include "Decomposition/LoadBalancer.h"
//#include "Field/Field.h"

 namespace ippl {

    template <class pc, class fc, class fs, class orb>
    class PicManager : public BaseManager {
    public:
        PicManager()
            : BaseManager(), fcontainer_m(nullptr), pcontainer_m(nullptr), loadbalancer_m(nullptr), fsolver_m(nullptr) {}

        virtual ~PicManager() = default;

        virtual void par2grid() = 0;

        virtual void grid2par() = 0;

        void setParticleContainer(std::shared_ptr<pc> pcontainer){
            pcontainer_m = pcontainer;
        }
        void setFieldContainer(std::shared_ptr<fc> fcontainer){
            fcontainer_m = fcontainer;
        }
        void setFieldSolver(std::shared_ptr<fs> fsolver){
            fsolver_m = fsolver;
        }
        
        void setLoadBalancer(std::shared_ptr<orb> loadbalancer){
            loadbalancer_m = loadbalancer;
        }

    protected:
     
        std::shared_ptr<fc> fcontainer_m;

        std::shared_ptr<pc> pcontainer_m;

        std::shared_ptr<orb> loadbalancer_m;

        std::shared_ptr<fs> fsolver_m;
    };
}  // namespace ippl
 
#endif

