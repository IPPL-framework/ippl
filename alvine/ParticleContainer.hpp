#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

class ParticleContainerBase {
  public:

    virtual void initDump() = 0;

    virtual void dump(double it) = 0;

    virtual ~ParticleContainerBase() = default;
};

// Define the ParticlesContainer class
template <typename T, unsigned Dim>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>, public ParticleContainerBase {
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


	      void setupBCs() { setBCAllPeriodic(); }

    private:
        void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
        void registerAttributes() {
            this->addAttribute(P);
            this->addAttribute(R_old);
        }
};

template <typename T>
class TwoDimParticleContainer : public ParticleContainer<T, 2> {
    public:
        using omega_type = ippl::ParticleAttrib<T>;

        omega_type omega;

        TwoDimParticleContainer(Mesh_t<2>& mesh, FieldLayout_t<2>& FL)
            : ParticleContainer<T, 2>(mesh, FL) {
            this->addAttribute(omega);
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
};

template <typename T>
class ThreeDimParticleContainer : public ParticleContainer<T, 3> {
    public:
        using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
        using omega_type = Base::particle_position_type;
         

        omega_type omega;
        std::array<ippl::ParticleAttrib<T>, 3> omega_split;
        

        ThreeDimParticleContainer(Mesh_t<3>& mesh, FieldLayout_t<3>& FL)
            : ParticleContainer<T, 3>(mesh, FL) {
            this->addAttribute(omega);

            for (const auto& omega_i : omega_split) {
                this->addAttribute(omega_i);
            }
        }


        void initDump() override {
            Inform csvout(NULL, "particles.csv", Inform::OVERWRITE);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            csvout << "time,index,pos_x,pos_y,pos_z,vorticity_x,vorticity_y,vorticity_z" << endl;
        }

        void dump(double it) override {
            Inform csvout(NULL, "particles.csv", Inform::APPEND);
            
            for (unsigned i = 0; i < this->getTotalNum(); i++) {
              csvout << it << "," << i; 
              for (unsigned d = 0; d < Dim; d++) {
                csvout << "," << this->R(i)[d];
              }
              for (unsigned d = 0; d < Dim; d++) {
                csvout << "," << this->omega(i)[d];
              }
              csvout << endl;
            }
        }
};
#endif
