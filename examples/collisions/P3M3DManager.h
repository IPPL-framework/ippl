#ifndef IPPL_P3M3D_MANAGER
#define IPPL_P3M3D_MANAGER

#include <memory>

#include "Decomposition/OrthogonalRecursiveBisection.h"
#include "Interaction/TruncatedGreenParticleInteraction.h"
#include "Manager/BaseManager.h"
#include "Manager/FieldSolverBase.h"
#include "Manager/PicManager.h"
#include "P3MParticleContainer.hpp"
#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"

/**
 * @class P3M3DManager
 * @brief A template class for managing Particle-Particle Particle-Mesh (P3M/PPPM) simulations
 *
 * The P3M3DManager class is a template class that extends the functionality of
 * the BaseManager class for P3M simulations. It provides functionality for
 * particle-particle, particle-to-grid and grid-to-particle operations
 *
 * @tparam T the data type for simulation variables
 * @tparam Dim  The dimensionality of the simulation (here 3D only)
 * @tparam fc   The field container type
 */
template <typename T, unsigned Dim, class fc>
class P3M3DManager : public ippl::BaseManager {
    // use only the P3M Particle Container
    using pc = P3MParticleContainer<T, Dim>;

public:
    using PPInteraction =
        ippl::TruncatedGreenParticleInteraction<pc, typename pc::particle_vector_type,
                                                typename pc::particle_scalar_type>;
    P3M3DManager()
        : BaseManager() {}

    virtual ~P3M3DManager() = default;

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
     * @brief Dump Simulation Data
     *
     * In a derived class, the user must override this method to dump simulation data.
     */
    virtual void dump() = 0;

    std::shared_ptr<pc> getParticleContainer() { return pcontainer_m; }

    void setParticleContainer(std::shared_ptr<pc> pcontainer) { pcontainer_m = pcontainer; }

    std::shared_ptr<fc> getFieldContainer() { return fcontainer_m; }

    void setFieldContainer(std::shared_ptr<fc> fcontainer) { fcontainer_m = fcontainer; }

    std::shared_ptr<P3MSolver_t<T, Dim> > getFieldSolver() { return fsolver_m; }

    void setFieldSolver(std::shared_ptr<P3MSolver_t<T, Dim> > fsolver) { fsolver_m = fsolver; }

    std::shared_ptr<PPInteraction> getInteractionSolver() { return isolver_m; }

    void setInteractionSolver(std::shared_ptr<PPInteraction> isolver) { isolver_m = isolver; }

protected:
    std::shared_ptr<fc> fcontainer_m;

    std::shared_ptr<pc> pcontainer_m;

    std::shared_ptr<P3MSolver_t<T, Dim> > fsolver_m;

    std::shared_ptr<PPInteraction> isolver_m;
};

#endif
