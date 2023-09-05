#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>

#include "Manager/BaseManager.h"
 
 namespace ippl {

    template <class pc, class fc, class fs>
    class PicManager : public BaseManager {
    public:
        PicManager()
            : BaseManager() {}

        virtual ~PicManager() = default;

        virtual void par2grid() = 0;

        virtual void grid2par() = 0;

        void setParticleContainer(std::shared_ptr<pc> pcontainer){
            pcontainer_m = pcontainer;
        }
        void setFieldContainer(std::shared_ptr<fc> fcontainer){
            fcontainer_m = fcontainer;
        }
        void setFieldSolver(std::unique_ptr<fs> fsolver){
            fsolver_m = fsolver;
        }
    
	//void setStepper(Stepper* stepper) {
	// stepper_m = std::make_unique<Stepper>(stepper);
	//};

	// void setFieldSolver(...);

	// void setLoadBalancer(...);

    protected:
     
        std::shared_ptr<fc> fcontainer_m;

        std::shared_ptr<pc> pcontainer_m;

        //std::unique_ptr<Stepper> stepper_m;

        //std::unique_ptr<LoadBalancer> loadbalancer_m;

        std::unique_ptr<fs> fsolver_m;
    };
}  // namespace ippl
 
#endif

