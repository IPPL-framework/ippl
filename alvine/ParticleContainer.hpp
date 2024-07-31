#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim>
class ParticleContainerBase : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

public:
    typename Base::particle_position_type R_old;
    typename Base::particle_position_type P;

private:
    PLayout_t<T, Dim> pl_m;

public:
    ParticleContainerBase(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);

        setupBCs();
        registerAttributes();
    }

    virtual ~ParticleContainerBase() = default;

    PLayout_t<T, Dim>& getPL() { return pl_m; }
    void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

    virtual void initDump() = 0;

    virtual void dump(double it) = 0;

    void setupBCs() { setBCAllPeriodic(); }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
    void registerAttributes() {
        this->addAttribute(P);
        this->addAttribute(R_old);
        std::cout << "general register attributes" << std::endl;
    }
};

template <typename T, unsigned Dim>
class ParticleContainer : public ParticleContainerBase<T, Dim> {
public:
    ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL) {}
};

template <typename T>
class ParticleContainer<T, 2> : public ParticleContainerBase<T, 2> {
public:
    ippl::ParticleAttrib<T> omega;

    ~ParticleContainer() {};

    ParticleContainer(Mesh_t<2>& mesh, FieldLayout_t<2>& FL)
        : ParticleContainerBase<T, Dim>(mesh, FL) {
        registerAttributes();
    }

    void initDump() override {
        Inform csvout(NULL, "particles.csv", Inform::OVERWRITE);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        csvout << "time,index,pos_x,pos_y,vorticity" << endl;
    }

    void dump(double it) override {
        Inform csvout(NULL, "particles.csv", Inform::APPEND);

        for (unsigned i = 0; i < this->getTotalNum(); i++) {
            csvout << it << "," << i;
            for (unsigned d = 0; d < Dim; d++) {
                csvout << "," << this->R(i)[d];
            }
            csvout << "," << this->omega(i) << endl;
        }
    }

private:
    void registerAttributes() {
        this->addAttribute(omega);
        std::cout << "two dim register attributes" << std::endl;
    }
};

template <typename T>
class ParticleContainer<T, 3> : public ParticleContainerBase<T, 3> {
public:
    ippl::ParticleAttrib<T> omega_x;
    ippl::ParticleAttrib<T> omega_y;
    ippl::ParticleAttrib<T> omega_z;

    ~ParticleContainer() {};

    ParticleContainer(Mesh_t<3>& mesh, FieldLayout_t<3>& FL)
        : ParticleContainerBase<T, Dim>(mesh, FL) {
        registerAttributes();
    }

    void initDump() override {
        Inform csvout(NULL, "particles.csv", Inform::OVERWRITE);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);
        csvout << "time,index,pos_x,pos_y,pos_z,vor_x,vor_y,vor_z" << endl;
    }

    ippl::ParticleAttrib<Vector<T, 1>> get_R(int d) {
        ippl::ParticleAttrib<Vector<T, 1>> R_k;
        const auto pcount = this->getTotalNum();
        R_k.realloc(pcount);

        Kokkos::parallel_for(
            "Copy phase space", pcount,
            KOKKOS_CLASS_LAMBDA(const size_t i) { R_k(i) = this->R(i)[d]; });
        return R_k;
    }

    void dump(double it) override {
        Inform csvout(NULL, "particles.csv", Inform::APPEND);

        for (unsigned i = 0; i < this->getTotalNum(); i++) {
            csvout << it << "," << i;
            for (unsigned d = 0; d < Dim; d++) {
                csvout << "," << this->R(i)[d];
            }
            csvout << "," << this->omega_x(i);
            csvout << "," << this->omega_y(i);
            csvout << "," << this->omega_z(i) << endl;
        }
    }

private:
    void registerAttributes() {
        this->addAttribute(omega_x);
        this->addAttribute(omega_y);
        this->addAttribute(omega_z);
        std::cout << "three dim register attributes" << std::endl;
    }
};

#endif
