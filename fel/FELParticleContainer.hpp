#ifndef IPPL_FEL_PARTICLE_CONTAINER_H
#define IPPL_FEL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

#include "datatypes.h"

// Define the FELParticleContainer class.
//
// Mirrors alpine/ParticleContainer.hpp but carries the attributes required by a
// relativistic electromagnetic push:
//   * Q, mass        : per-particle charge and mass.
//   * gamma_beta      : relativistic momentum (gamma * beta).
//   * R_nm1, R_np1    : positions at the previous / next step. R_nm1 is the
//                       start point for the charge-conserving current
//                       deposition (X0 -> X1 = R_nm1 -> R).
//   * E_gather, B_gather : E and B interpolated to the particle positions.
//
// Boundary conditions are NO (open): particles leaving the domain are removed
// explicitly by the manager rather than wrapped or reflected.
template <typename T, unsigned Dim = 3>
class FELParticleContainer : public ippl::ParticleBase<PLayout_t<T, Dim>> {
    using Base = ippl::ParticleBase<PLayout_t<T, Dim>>;

public:
    ippl::ParticleAttrib<T> Q;                       // charge
    ippl::ParticleAttrib<T> mass;                    // mass
    ippl::ParticleAttrib<Vector_t<T, 3>> gamma_beta; // relativistic momentum (gamma*beta)
    typename Base::particle_position_type R_nm1;     // position at previous step
    typename Base::particle_position_type R_np1;     // position at next step
    ippl::ParticleAttrib<Vector_t<T, 3>> E_gather;   // E field at particle position
    ippl::ParticleAttrib<Vector_t<T, 3>> B_gather;   // B field at particle position

private:
    PLayout_t<T, Dim> pl_m;

public:
    FELParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
    }

    ~FELParticleContainer() {}

    std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
    void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

    void registerAttributes() {
        this->addAttribute(Q);
        this->addAttribute(mass);
        this->addAttribute(gamma_beta);
        this->addAttribute(R_nm1);
        this->addAttribute(R_np1);
        this->addAttribute(E_gather);
        this->addAttribute(B_gather);
    }

    void setupBCs() { setBCAllOpen(); }

private:
    void setBCAllOpen() { this->setParticleBC(ippl::NO); }
};

#endif
