#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the FieldsContainer class
template <typename T, unsigned Dim = 3>
class FieldContainer{
  using vorticity_field_type = std::conditional<Dim == 2, Field<T, Dim>, VField_t<T, Dim>>::type;


public:
    FieldContainer(Vector_t<T, Dim>& hr, Vector_t<T, Dim>& rmin,
                   Vector_t<T, Dim>& rmax, std::array<bool, Dim> decomp,
                   ippl::NDIndex<Dim> domain, Vector_t<T, Dim> origin,
                   bool isAllPeriodic)
        : hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , decomp_m(decomp)
        , mesh_m(domain, hr, origin)
        , fl_m(MPI_COMM_WORLD, domain, decomp, isAllPeriodic) {}

    ~FieldContainer(){}

private:
    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    std::array<bool, Dim> decomp_m;

    vorticity_field_type A_field_m;
    vorticity_field_type omega_field_m;
    VField_t<T, Dim> u_field_m;

    Mesh_t<Dim> mesh_m;
    FieldLayout_t<Dim> fl_m;

public:

    vorticity_field_type& getA_field() { return A_field_m; }
    void setA_field(vorticity_field_type& A_field) { A_field_m = A_field; }

    vorticity_field_type& getOmega_field() { return omega_field_m; }
    void setOmega_field(vorticity_field_type& omega_field) { omega_field_m = omega_field; }

    VField_t<T, Dim>& getU_field() { return u_field_m; }
    void setOmega_field(VField_t<T, Dim>& u_field) { u_field_m = u_field; }

    Vector_t<double, Dim>& getHr() { return hr_m; }
    void setHr(const Vector_t<double, Dim>& hr) { hr_m = hr; }

    Vector_t<double, Dim>& getRMin() { return rmin_m; }
    void setRMin(const Vector_t<double, Dim>& rmin) { rmin_m = rmin; }

    Vector_t<double, Dim>& getRMax() { return rmax_m; }
    void setRMax(const Vector_t<double, Dim>& rmax) { rmax_m = rmax; }

    std::array<bool, Dim> getDecomp() { return decomp_m; }
    void setDecomp(std::array<bool, Dim> decomp) { decomp_m = decomp; }

    Mesh_t<Dim>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<Dim> & mesh) { mesh_m = mesh; }

    FieldLayout_t<Dim>& getFL() { return fl_m; }
    void setFL(std::shared_ptr<FieldLayout_t<Dim>>& fl) { fl_m = fl; }

    void initializeFields() {
        A_field_m.initialize(mesh_m, fl_m);
        omega_field_m.initialize(mesh_m, fl_m);
        u_field_m.initialize(mesh_m, fl_m);
    }
};

#endif
