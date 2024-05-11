#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/FieldSolverBase.h"

template <typename T, unsigned Dim>
class FieldContainer {
public:
    FieldContainer(Mesh_t<Dim>, FieldLayout_t<Dim> fl) {}

    virtual ~FieldContainer() = default;
};

template <typename T>
class FieldContainer<T, 2> {
    Field<T, Dim> omega_field;

public: 
    FieldContainer(Mesh_t<2> mesh, FieldLayout_t<2> fl) {
        omega_field.initialize(mesh, fl);
    }

    ~FieldContainer() = default;
};

template <typename T>
class FieldContainer<T, 3> {

public: 
    FieldContainer(Mesh_t<3> mesh, FieldLayout_t<3> fl) {
    }

    ~FieldContainer() = default;
};
#endif
