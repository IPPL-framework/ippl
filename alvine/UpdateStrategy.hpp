#ifndef UPDATE_MANAGER_H
#define UPDATE_MANAGER_H

#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"

template <class fc, class pc>  
class UpdateStrategy {
  public:

    virtual ~UpdateStrategy() = default;

    virtual void par2grid(std::shared_ptr<fc> fcontainer, std::shared_ptr<pc> pcontainer);

    virtual void updateFields(std::shared_ptr<fc> fcontainer);

    virtual void grid2par(std::shared_ptr<fc> fcontainer, std::shared_ptr<pc> pcontainer);
};

template<typename T>
class TwoDimUpdateStrategy : public UpdateStrategy<FieldContainer<T, 2>, ParticleContainer<T, 2>> {
  public:
    void par2grid(std::shared_ptr<FieldContainer<T, 2>> fc, std::shared_ptr<ParticleContainer<T, 2>> pc) override {

        fc->getOmegaField() = 0.0;
        scatter(pc->omega, fc->getOmegaField(), pc->R);
    }

    void updateFields(std::shared_ptr<FieldContainer<T, 2>> fc) override {

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

    void grid2par(std::shared_ptr<FieldContainer<T, 2>> fc, std::shared_ptr<ParticleContainer<T, 2>> pc) override {
        pc->P = 0.0;
        gather(pc->P, fc->getUField(), pc->R);
    }

};

template<typename T>
class ThreeDimUpdateStrategy : public UpdateStrategy<FieldContainer<T, 3>, ParticleContainer<T, 3>> {
  public:
    void par2grid(std::shared_ptr<FieldContainer<T, 3>> fc, std::shared_ptr<ParticleContainer<T, 3>> pc) override { }

    void updateFields(std::shared_ptr<FieldContainer<T, 3>> fc) override { }

    void grid2par(std::shared_ptr<FieldContainer<T, 3>> fc, std::shared_ptr<ParticleContainer<T, 3>> pc) override { }

};


#endif
