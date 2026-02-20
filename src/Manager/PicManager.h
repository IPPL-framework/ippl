#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>
#include <stdexcept>
#include <vector>

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
     * It supports multiple particle containers (bunches). By default, a single bunch is used,
     * preserving backward compatibility. The load balancer applies to the first (default) bunch.
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

        /**
         * @brief Get the default (first) particle container.
         *
         * Provided for backward compatibility with single-bunch simulations.
         */
        std::shared_ptr<pc> getParticleContainer() { return pcontainer_m; }


        /**
         * @brief Get a particle container by index.
         * @param i The index of the particle container (0-based).
         * @return Shared pointer to the particle container at index i.
         * @throws std::out_of_range if the index is invalid.
         */
        std::shared_ptr<pc> getParticleContainer(size_t i) {
            if (i >= pcontainers_m.size()) {
                throw std::out_of_range("PicManager::getParticleContainer: index out of range");
            }
            return pcontainers_m[i];
        }

        /**
         * @brief Set the default (first) particle container.
         *
         * Provided for backward compatibility. This sets the first bunch and also
         * updates pcontainers_m[0] to stay in sync.
         */
        void setParticleContainer(std::shared_ptr<pc> pcontainer) {
            pcontainer_m = pcontainer;
            if (pcontainers_m.empty()) {
                pcontainers_m.push_back(pcontainer);
            } else {
                pcontainers_m[0] = pcontainer;
            }
        }

        /**
         * @brief Add an additional particle container (bunch).
         * @param pcontainer Shared pointer to the particle container to add.
         * @return The index assigned to the new particle container.
         */
        size_t addParticleContainer(std::shared_ptr<pc> pcontainer) {
            pcontainers_m.push_back(pcontainer);
            // If this is the first container being added, also set pcontainer_m
            if (pcontainers_m.size() == 1) {
                pcontainer_m = pcontainer;
            }
            return pcontainers_m.size() - 1;
        }

        /**
         * @brief Get the number of particle containers (bunches).
         * @return The number of particle containers.
         */
        size_t getNumParticleContainers() const { return pcontainers_m.size(); }

        /**
         * @brief Get a const reference to the vector of all particle containers.
         * @return Const reference to the vector of shared pointers.
         */
        const std::vector<std::shared_ptr<pc>>& getParticleContainers() const {
            return pcontainers_m;
        }

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

        /**
         * @brief Default (first) particle container, kept for backward 
         * compatibility.
         */
        std::shared_ptr<pc> pcontainer_m;

        /**
         * @brief All particle containers managed by this PIC manager.
         * the first entry (index 0) is the same as the one pointed to by
         * pcontainer_m.
         */
        std::vector<std::shared_ptr<pc>> pcontainers_m;

        std::shared_ptr<orb> loadbalancer_m;

        std::shared_ptr<ippl::FieldSolverBase<T, Dim>> fsolver_m;
    };
}  // namespace ippl

#endif
