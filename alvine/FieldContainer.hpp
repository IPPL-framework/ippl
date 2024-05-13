#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/FieldSolverBase.h"
#include "SimulationParameters.hpp"

template <typename T, unsigned Dim>
class FieldContainer {
public:
    FieldContainer(Mesh_t<Dim>, FieldLayout_t<Dim> fl) {}

    virtual ~FieldContainer() = default;
};

template <typename T>
class FieldContainer<T, 2> {
    Field<T, 2> omega_field;
    VField_t<T, 2> u_field;

    Mesh_t<Dim> mesh_m;
    FieldLayout_t<Dim> fl_m;

public:
    FieldContainer(SimulationParameters<T, 2> params)
        : mesh_m(params.domain, params.hr, params.origin)
        , fl_m(MPI_COMM_WORLD, params.domain, params.decomp, true) {
        omega_field.initialize(mesh_m, fl_m);
        u_field.initialize(mesh_m, fl_m);
    }

    Field<T, 2>& getOmegaField() { return omega_field; }
    void setOmegaField(Field<T, 2>& omega_field_) { this->omega_field = omega_field_; }

    VField_t<T, 2>& getUField() { return u_field; }
    void setUField(VField_t<T, 2>& u_field_) { this->u_field = u_field_; }

    Mesh_t<Dim>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<Dim>& mesh) { mesh_m = mesh; }

    FieldLayout_t<Dim>& getFL() { return fl_m; }
    void setFL(std::shared_ptr<FieldLayout_t<Dim>>& fl) { fl_m = fl; }

    ~FieldContainer() = default;
};

template <typename T>
class FieldContainer<T, 3> {
public:
    FieldContainer(Mesh_t<3> mesh, FieldLayout_t<3> fl) {}

    ~FieldContainer() = default;
};
#endif
