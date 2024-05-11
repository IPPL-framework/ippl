#ifndef PARTICLE_FIELD_STRATEGY_H
#define PARTICLE_FIELD_STRATEGY_H

#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"

class ParticleFieldStrategy {
  public:

    virtual ~ParticleFieldStrategy() = default;

    virtual void par2grid(std::shared_ptr<FieldContainerBase> fcontainer, std::shared_ptr<ParticleContainerBase> pcontainer) = 0;

    virtual void updateFields(std::shared_ptr<FieldContainerBase> fcontainer) = 0;

    virtual void grid2par(std::shared_ptr<FieldContainerBase> fcontainer, std::shared_ptr<ParticleContainerBase> pcontainer) = 0;
};

template<typename T>
class TwoDimParticleFieldStrategy : public ParticleFieldStrategy {
  public:
    void par2grid(std::shared_ptr<FieldContainerBase> fcontainer, std::shared_ptr<ParticleContainerBase> pcontainer) override {

        std::shared_ptr<TwoDimParticleContainer<T>> pc = std::dynamic_pointer_cast<TwoDimParticleContainer<T>>(pcontainer);
        std::shared_ptr<TwoDimFieldContainer<T>> fc = std::dynamic_pointer_cast<TwoDimFieldContainer<T>>(fcontainer);

        fc->getOmegaField() = 0.0;
        scatter(pc->omega, fc->getOmegaField(), pc->R);
    }

    void updateFields(std::shared_ptr<FieldContainerBase> fcontainer) override {

        std::shared_ptr<TwoDimFieldContainer<T>> fc = std::dynamic_pointer_cast<TwoDimFieldContainer<T>>(fcontainer);

        VField_t<T, 2> u_field = fc->getUField();
        u_field = 0.0;

        const int nghost = u_field.getNghost();
        auto view = u_field.getView();

        auto omega_view = fc->getOmegaField().getView();
        fc->getOmegaField().fillHalo();

        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j) {
                view(i, j) = {
                         (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * fc->getHr()(1)), 
                        -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * fc->getHr()(0))
                        };

            });
    }

    void grid2par(std::shared_ptr<FieldContainerBase> fcontainer, std::shared_ptr<ParticleContainerBase> pcontainer) override {

        std::shared_ptr<TwoDimParticleContainer<T>> pc = std::dynamic_pointer_cast<TwoDimParticleContainer<T>>(pcontainer);
        std::shared_ptr<TwoDimFieldContainer<T>> fc = std::dynamic_pointer_cast<TwoDimFieldContainer<T>>(fcontainer);

        pc->P = 0.0;
        gather(pc->P, fc->getUField(), pc->R);
    }

};

template<typename T>
class ThreeDimParticleFieldStrategy : public ParticleFieldStrategy {
  public:
    void par2grid([[maybe_unused]] std::shared_ptr<FieldContainerBase> fc, [[maybe_unused]] std::shared_ptr<ParticleContainerBase> pc) override { 
         
    }

    void updateFields([[maybe_unused]] std::shared_ptr<FieldContainerBase> fc) override { }

    void grid2par([[maybe_unused]] std::shared_ptr<FieldContainerBase> fc,[[maybe_unused]] std::shared_ptr<ParticleContainerBase> pc) override { }

};


#endif
