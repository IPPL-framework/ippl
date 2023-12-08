#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the FieldsContainer class
template <typename T, unsigned Dim = 3>
class FieldContainer{
public:
    FieldContainer(Vector_t<double, Dim>& hr, Vector_t<double, Dim>& rmin,
                   Vector_t<double, Dim>& rmax, std::array<bool, Dim> decomp)
        : hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , decomp_m(decomp) {}

    ~FieldContainer(){}

private:
    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    std::array<bool, Dim> decomp_m;
    VField_t<T, Dim> E_m;
    Field_t<Dim> rho_m;
    Field<T, Dim> phi_m;
    std::shared_ptr<Mesh_t<Dim>> mesh_m;
    std::shared_ptr<FieldLayout_t<Dim>> fl_m;

public:
    inline VField_t<T, Dim>& getE() { return E_m; }
    inline void setE(VField_t<T, Dim>& E) { E_m = E; }

    inline Field_t<Dim>& getRho() { return rho_m; }
    inline void setRho(Field_t<Dim>& rho) { rho_m = rho; }

    inline Field<T, Dim>& getPhi() { return phi_m; }
    inline void setPhi(Field<T, Dim>& phi) { phi_m = phi; }

    inline Vector_t<double, Dim>& getHr() { return hr_m; }
    inline void setHr(const Vector_t<double, Dim>& hr) { hr_m = hr; }

    inline Vector_t<double, Dim>& getRMin() { return rmin_m; }
    inline void setRMin(const Vector_t<double, Dim>& rmin) { rmin_m = rmin; }

    inline Vector_t<double, Dim>& getRMax() { return rmax_m; }
    inline void setRMax(const Vector_t<double, Dim>& rmax) { rmax_m = rmax; }

    inline std::array<bool, Dim> getDecomp() { return decomp_m; }
    inline void setDecomp(std::array<bool, Dim> decomp) { decomp_m = decomp; }

    inline Mesh_t<Dim>& getMesh() { return *mesh_m; }
    inline void setMesh(std::shared_ptr<Mesh_t<Dim>>& mesh) { mesh_m = mesh; }

    inline FieldLayout_t<Dim>& getFL() { return *fl_m; }
    inline void setFL(std::shared_ptr<FieldLayout_t<Dim>>& fl) { fl_m = fl; }

    void initializeFields(std::shared_ptr<Mesh_t<Dim>> mesh, std::shared_ptr<FieldLayout_t<Dim>> fl, std::string stype_m = "") {
        E_m.initialize(*mesh, *fl);
        rho_m.initialize(*mesh, *fl);
        if (stype_m == "CG") {
            phi_m.initialize(*mesh, *fl);
        }
        fl_m = fl;
        mesh_m = mesh;
    }
};

#endif
