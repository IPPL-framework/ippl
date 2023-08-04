#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>

#include "Manager/BaseManager.h"

namespace ippl {

    class PicManager : public Basemanager {
    public:
        PicManager()
            : BaseManager() {}

        virtual ~PicManager() = default;

        virtual void par2grid() = 0;

        virtual void grid2par() = 0;

    protected:
        std::unique_ptr<FieldsContainer> fcontainer_m;

        std::unique_ptr<ParticlesContainer> pcontainer_m;

        std::unique_ptr<Stepper> stepper_m;

        std::unique_ptr<LoadBalancer> loadbalancer_m;

        std::unique_ptr<ForceSolver> fsolver_m;
    };
}  // namespace ippl

#endif
