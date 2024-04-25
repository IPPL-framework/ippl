#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>

#include "Decomposition/OrthogonalRecursiveBisection.h"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"

namespace ippl {

    /**
     * @class PicManager
     * @brief A template class for managing Particle-in-Cell (PIC) simulations.
     *
     * The PicManager class is a template class that extends the functionality of the BaseManager
     * class for Particle-in-Cell simulations. It provides methods for particle-to-grid and
     * grid-to-particle operations, as well as a method for dumping simulation data.
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
            : BaseManager()
            , fcontainer_m(nullptr)
            , pcontainer_m(nullptr)
            , loadbalancer_m(nullptr) {}

        virtual ~PicManager() = default;

        /**
         * @brief Particle-to-grid operation.
         *
         * In a derived class, the user must override this method to perform particle-to-grid
         * operations.
         */
        virtual void par2grid() = 0;

        /**
         * @brief Grid-to-particle operation.
         *
         * In a derived class, the user must override this method to perform grid-to-particle
         * operations.
         */
        virtual void grid2par() = 0;

        std::shared_ptr<pc> getParticleContainer() { return pcontainer_m; }

        void setParticleContainer(std::shared_ptr<pc> pcontainer) { pcontainer_m = pcontainer; }

        std::shared_ptr<fc> getFieldContainer() { return fcontainer_m; }

        void setFieldContainer(std::shared_ptr<fc> fcontainer) { fcontainer_m = fcontainer; }

        std::shared_ptr<ippl::FieldSolverBase<T, Dim>> getFieldSolver() { return fsolver_m; }

        void setFieldSolver(std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver) {
            fsolver_m = fsolver;
        }

        std::shared_ptr<orb> getLoadBalancer() { return loadbalancer_m; }

        void setLoadBalancer(std::shared_ptr<orb> loadbalancer) { loadbalancer_m = loadbalancer; }

    protected:
        std::shared_ptr<fc> fcontainer_m;

        std::shared_ptr<pc> pcontainer_m;

        std::shared_ptr<orb> loadbalancer_m;

        std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver_m;
    };
}  // namespace ippl

#endif
