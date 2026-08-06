#ifndef IPPL_FEL_FIELD_CONTAINER_H
#define IPPL_FEL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

#include "datatypes.h"

// Define the FELFieldContainer class.
//
// Mirrors alpine/FieldContainer.hpp but holds the fields needed for the
// electromagnetic FEL simulation: the electric field E, the magnetic field B
// (both three-component vector fields), and the four-current source J
// ([0] = charge density, [1..Dim] = current density) that drives the FDTD
// solver. Unlike the alpine container the field layout is not periodic; the
// decomposition flags select which axes are split across MPI ranks.
template <typename T, unsigned Dim = 3>
class FELFieldContainer {
public:
    FELFieldContainer(Vector_t<T, Dim>& hr, Vector_t<T, Dim>& rmin, Vector_t<T, Dim>& rmax,
                      std::array<bool, Dim> decomp, ippl::NDIndex<Dim> domain,
                      Vector_t<T, Dim> origin, bool isAllPeriodic)
        : hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , decomp_m(decomp)
        , mesh_m(domain, hr, origin)
        , fl_m(MPI_COMM_WORLD, domain, decomp, isAllPeriodic) {}

    ~FELFieldContainer() {}

private:
    Vector_t<T, Dim> hr_m;
    Vector_t<T, Dim> rmin_m;
    Vector_t<T, Dim> rmax_m;
    std::array<bool, Dim> decomp_m;
    VField_t<T, Dim> E_m;
    VField_t<T, Dim> B_m;
    SourceField_t<T, Dim> J_m;
    Mesh_t<Dim> mesh_m;
    FieldLayout_t<Dim> fl_m;

public:
    VField_t<T, Dim>& getE() { return E_m; }
    void setE(VField_t<T, Dim>& E) { E_m = E; }

    VField_t<T, Dim>& getB() { return B_m; }
    void setB(VField_t<T, Dim>& B) { B_m = B; }

    SourceField_t<T, Dim>& getJ() { return J_m; }
    void setJ(SourceField_t<T, Dim>& J) { J_m = J; }

    Vector_t<T, Dim>& getHr() { return hr_m; }
    void setHr(const Vector_t<T, Dim>& hr) { hr_m = hr; }

    Vector_t<T, Dim>& getRMin() { return rmin_m; }
    void setRMin(const Vector_t<T, Dim>& rmin) { rmin_m = rmin; }

    Vector_t<T, Dim>& getRMax() { return rmax_m; }
    void setRMax(const Vector_t<T, Dim>& rmax) { rmax_m = rmax; }

    std::array<bool, Dim> getDecomp() { return decomp_m; }
    void setDecomp(std::array<bool, Dim> decomp) { decomp_m = decomp; }

    Mesh_t<Dim>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<Dim>& mesh) { mesh_m = mesh; }

    FieldLayout_t<Dim>& getFL() { return fl_m; }
    void setFL(FieldLayout_t<Dim>& fl) { fl_m = fl; }

    void initializeFields() {
        E_m.initialize(mesh_m, fl_m);
        B_m.initialize(mesh_m, fl_m);
        J_m.initialize(mesh_m, fl_m);
    }
};

#endif
