#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>
#include "Manager/BaseManager.h"
#include "Decomposition/OrthogonalRecursiveBisection.h"
#include "Manager/FieldSolverBase.h"

 namespace ippl {

    /**
    * @class PicManager
    * @brief A template class for managing Particle-in-Cell (PIC) simulations.
    *
    * The PicManager class is a template class that extends the functionality of the BaseManager class
    * for Particle-in-Cell simulations. It provides methods for particle-to-grid and grid-to-particle operations,
    * as well as a method for dumping simulation data.
    *
    * @tparam T The data type for simulation variables.
    * @tparam Dim The dimensionality of the simulation (e.g., 2D or 3D).
    * @tparam pc The particle container type.
    * @tparam fc The field container type.
    * @tparam orb The load balancer type.
    */
    template <typename T, unsigned Dim, class pc, class fc, class orb>
    class PicManager : public BaseManager {
    public:
        PicManager()
            : BaseManager(), fcontainer_m(nullptr), pcontainer_m(nullptr), loadbalancer_m(nullptr) {}

        virtual ~PicManager() = default;

       /**
        * @brief Particle-to-grid operation.
        *
        * Derived classes can override this method to perform particle-to-grid operations.
        * The default implementation does nothing.
        */
        virtual void par2grid() { /* default does nothing */ };

       /**
        * @brief Grid-to-particle operation.
        *
        * Derived classes can override this method to perform grid-to-particle operations.
        * The default implementation does nothing.
        */
        virtual void grid2par() { /* default does nothing */ };

       /**
        * @brief Dump simulation data.
        *
        * Derived classes can override this method to implement custom data dumping procedures.
        * The default implementation does nothing.
        */
        virtual void dump() { /* default does nothing */ };

        inline std::shared_ptr<pc> getParticleContainer() {
            return pcontainer_m;
        }

        inline void setParticleContainer(std::shared_ptr<pc> pcontainer){
            pcontainer_m = pcontainer;
        }

        inline std::shared_ptr<fc> getFieldContainer() {
            return fcontainer_m;
        }

        inline void setFieldContainer(std::shared_ptr<fc> fcontainer){
            fcontainer_m = fcontainer;
        }

        inline std::shared_ptr<ippl::FieldSolverBase<T, Dim>> getFieldSolver() {
            return fsolver_m;
        }

        inline void setFieldSolver(std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver) {
            fsolver_m = fsolver;
        }

        inline std::shared_ptr<orb> getLoadBalancer() {
            return loadbalancer_m;
        }

        inline void setLoadBalancer(std::shared_ptr<orb> loadbalancer){
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

