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

    Mesh_t<2> mesh_m;
    FieldLayout_t<2> fl_m;

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

    Mesh_t<2>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<2>& mesh) { mesh_m = mesh; }

    FieldLayout_t<2>& getFL() { return fl_m; }
    void setFL(std::shared_ptr<FieldLayout_t<2>>& fl) { fl_m = fl; }

    ~FieldContainer() = default;
};

template <typename T>
class FieldContainer<T, 3> {
    Field<T, 3> omega_field;

    Field<T, 1> omega_field_x;
    Field<T, 1> omega_field_y;
    Field<T, 1> omega_field_z;

    VField_t<T, 3> u_field;

    VField_t<T, 1> u_field_x;
    VField_t<T, 1> u_field_y;
    VField_t<T, 1> u_field_z;

    Mesh_t<3> mesh_m;

    Mesh_t<1> mesh_m_x;
    Mesh_t<1> mesh_m_y;
    Mesh_t<1> mesh_m_z;

    FieldLayout_t<3> fl_m;

    FieldLayout_t<1> fl_x;
    FieldLayout_t<1> fl_y;
    FieldLayout_t<1> fl_z;

public:
    FieldContainer(SimulationParameters<T, 3> params)
        : mesh_m(params.domain, params.hr, params.origin)
        , fl_m(MPI_COMM_WORLD, params.domain, params.decomp, true) {

        std::cout << params.domain[0]<< ", " << params.hr[0] << ", " << params.origin[0] << std::endl;
        mesh_m_x.initialize(params.domain[0], params.hr[0], params.origin[0]);
        mesh_m_y.initialize(params.domain[1], params.hr[1], params.origin[1]);
        mesh_m_z.initialize(params.domain[2], params.hr[2], params.origin[2]);

        fl_x.initialize(params.domain[0], {params.decomp[0]}, true);
        fl_y.initialize(params.domain[1], {params.decomp[1]}, true);
        fl_z.initialize(params.domain[2], {params.decomp[2]}, true);

        omega_field.initialize(mesh_m, fl_m);

        omega_field_x.initialize(mesh_m_x, fl_x);
        omega_field_y.initialize(mesh_m_y, fl_y);
        omega_field_z.initialize(mesh_m_z, fl_z);

        u_field.initialize(mesh_m, fl_m);

        u_field_x.initialize(mesh_m_x, fl_x);
        u_field_y.initialize(mesh_m_y, fl_y);
        u_field_z.initialize(mesh_m_z, fl_z);
    }

    Field<T, 3>& getOmegaField() { return omega_field; }
    void setOmegaField(Field<T, 3>& omega_field_) { this->omega_field = omega_field_; }

    Field<T, 1>& getOmegaFieldx() { return omega_field_x; }
    Field<T, 1>& getOmegaFieldy() { return omega_field_y; }
    Field<T, 1>& getOmegaFieldz() { return omega_field_z; }

    VField_t<T, 3>& getUField() { return u_field; }
    void setUField(VField_t<T, 3>& u_field_) { this->u_field = u_field_; }

    Mesh_t<Dim>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<Dim>& mesh) { mesh_m = mesh; }

    FieldLayout_t<Dim>& getFL() { return fl_m; }
    void setFL(std::shared_ptr<FieldLayout_t<Dim>>& fl) { fl_m = fl; }

    ~FieldContainer() = default;
};
#endif
