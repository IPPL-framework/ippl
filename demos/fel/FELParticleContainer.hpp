/**
 * @file FELParticleContainer.hpp
 * @brief Registered particle attributes for the boosted-frame FEL PIC loop.
 * @ingroup fel_particles
 */
#ifndef IPPL_FEL_PARTICLE_CONTAINER_H
#define IPPL_FEL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

#include "datatypes.h"

/**
 * @brief Store FEL particle state and register it for IPPL migration/compaction.
 * @ingroup fel_particles
 * @tparam T Scalar type of particle attributes and spatial layout.
 * @tparam Dim Position-space dimension; the active FEL application uses three.
 *
 * The base class supplies current position R, particle counts and storage
 * management. The manager supplies the bunch, assigns per-particle charge and
 * mass, gathers fields, pushes particles and calls update() for MPI migration.
 * Registered attributes follow particles when their owner or local index changes.
 * Kokkos views reside in the configured memory space; this class does not require
 * host copies for normal gather/push/deposition kernels.
 *
 * In the FEL run, positions and gathered fields use the boosted frame and the
 * normalized units in units.h; gamma_beta is dimensionless momentum p/(mc).
 * This storage class does not itself transform frames or integrate trajectories.
 * Open particle boundary flags perform no wrapping/reflection: explicit removal
 * is done by FreeElectronLaserManager::destroyOutOfBounds().
 */
template <typename T, unsigned Dim = 3>
class FELParticleContainer : public ippl::ParticleBase<PLayout_t<T, Dim>> {
    /// @brief IPPL base owning current positions, counts and registered attributes.
    using Base = ippl::ParticleBase<PLayout_t<T, Dim>>;

public:
    ippl::ParticleAttrib<T> Q;                       ///< Signed charge per simulation particle, in code units.
    ippl::ParticleAttrib<T> mass;                    ///< Mass per simulation particle, in code units.
    ippl::ParticleAttrib<Vector_t<T, 3>> gamma_beta; ///< Dimensionless three-vector p/(mc) in the boosted frame.
    typename Base::particle_position_type R_nm1;     ///< Pre-push position used with R for the next current deposit.
    typename Base::particle_position_type R_np1;     ///< Registered next-position hook; unused by the active manager.
    ippl::ParticleAttrib<Vector_t<T, 3>> E_gather;   ///< Interpolated grid E only; external undulator E is added in push().
    ippl::ParticleAttrib<Vector_t<T, 3>> B_gather;   ///< Interpolated grid B only; external undulator B is added in push().

private:
    PLayout_t<T, Dim> pl_m; ///< Value-owned spatial layout binding particles to mesh/rank ownership.

public:
    /**
     * @brief Bind the layout, register attributes and select open particle flags.
     * @param mesh Cartesian mesh used to determine particle ownership/interpolation.
     * @param FL MPI field layout shared with the simulation fields.
     * No particles or bunch distribution are created by this constructor.
     */
    FELParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
    }

    /// @brief Release the container and its attribute/layout storage.
    ~FELParticleContainer() {}

    /**
     * @brief Dormant shared-pointer layout accessor, unused by the FEL run.
     * @return Declared shared-pointer result; the stored layout is a value object.
     * @warning The signature and storage representation are incompatible as
     * written. This template member is not a supported active-path accessor.
     */
    std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
    /**
     * @brief Dormant shared-pointer layout setter, unused by the FEL run.
     * @param pl Declared replacement pointer; storage is currently a value object.
     * @warning This hook requires a representation fix before use; it does not
     * implement a supported particle-repartitioning operation.
     */
    void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

    /**
     * @brief Name and register all seven FEL-specific attributes with the base.
     *
     * Called once during construction so particle creation, destruction and MPI
     * migration handle these arrays together with R. Registration also supplies
     * names to output/visualization adapters; it does not assign physical values.
     */
    void registerAttributes() {
        gamma_beta.set_name("gamma_beta");
        E_gather.set_name("electric_field");
        B_gather.set_name("magnetic_field");
        Q.set_name("charge");
        mass.set_name("mass");
        R_nm1.set_name("previous_position");
        R_np1.set_name("next_position");

        this->addAttribute(gamma_beta);
        this->addAttribute(E_gather);
        this->addAttribute(B_gather);
        this->addAttribute(Q);
        this->addAttribute(mass);
        this->addAttribute(R_nm1);
        this->addAttribute(R_np1);
    }

    /// @brief Select the no-wrapping particle flags used by manager-owned removal.
    void setupBCs() { setBCAllOpen(); }

private:
    /// @brief Set ippl::NO on every particle face; no particles are removed here.
    void setBCAllOpen() { this->setParticleBC(ippl::NO); }
};

#endif
