#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    public:
        typename Base::particle_position_type R_old;
        typename Base::particle_position_type P;  

    private:
        PLayout_t<T, Dim> pl_m;
    public:
        ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
            this->initialize(pl_m);
            registerAttributes();
            setupBCs();
        }

        virtual ~ParticleContainer() = default;

        std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
        void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

        virtual void initDump() = 0;

        virtual void dump(double it) = 0;

	      void setupBCs() { setBCAllPeriodic(); }

    private:
        void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
        void registerAttributes() {
            this->addAttribute(P);
            this->addAttribute(R_old);
        }
};

template <typename T>
class ParticleContainer<T, 2> : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ParticleContainer<T, 2>;

public:
    ippl::ParticleAttrib<T> omega;

    ~ParticleContainer() {};

    ParticleContainer(Mesh_t<2>& mesh, FieldLayout_t<2>& FL) : Base(mesh, FL) {
        this->registerAttributes();

    }

private:
    void registerAttributes() override {
        Base::registerAttributes();
        this->addAttribute(omega);
    }
};

#endif
