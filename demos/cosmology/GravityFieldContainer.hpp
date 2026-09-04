#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

/**
 * @brief A container class that manages various fields, mesh, and field layouts
 *
 * @tparam T Data type used for the fields (e.g., double)
 * @tparam Dim Dimensionality of the field (default is 3)
 */
template <typename T, unsigned Dim = 3>
class FieldContainer {
public:
    /**
     * @brief Constructor for the FieldContainer class
     *
     * @param hr Grid spacing vector
     * @param rmin Starting Points of the Domain
     * @param rmax Ending Points of the Domain
     * @param decomp Decomposition array indicating which dimensions are decomposed MPI ranks
     * @param domain The domain over which the mesh is defined
     * @param origin The origin of the mesh
     * @param isAllPeriodic Flag indicating periodic boundary conditions in all dimension
     */
    FieldContainer(Vector_t<T, Dim>& hr, Vector_t<T, Dim>& rmin, Vector_t<T, Dim>& rmax,
                   std::array<bool, Dim> decomp, ippl::NDIndex<Dim> domain, Vector_t<T, Dim> origin,
                   bool isAllPeriodic)
        : hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , decomp_m(decomp)
        , mesh_m(domain, hr, origin)
        , fl_m(MPI_COMM_WORLD, domain, decomp, isAllPeriodic) {}

    /**
     * @brief Destructor for the FieldContainer class
     */
    ~FieldContainer() {}

    ///< Get/Set the force field F
    VField_t<T, Dim>& getF() { return F_m; }
    void setF(VField_t<T, Dim>& F) { F_m = F; }

    ///< Get/Set the mass density rho
    Field_t<Dim>& getRho() { return rho_m; }
    void setRho(Field_t<Dim>& rho) { rho_m = rho; }

    ///< Get/Set potential phi
    Field<T, Dim>& getPhi() { return phi_m; }
    void setPhi(Field<T, Dim>& phi) { phi_m = phi; }

    ///< Get/Set the grid spacing vector hr
    Vector_t<double, Dim>& getHr() { return hr_m; }
    void setHr(const Vector_t<double, Dim>& hr) { hr_m = hr; }

    ///< Get/Set starting points of the domain
    Vector_t<double, Dim>& getRMin() { return rmin_m; }
    void setRMin(const Vector_t<double, Dim>& rmin) { rmin_m = rmin; }

    ///< Get/Set ending points of the domain
    Vector_t<double, Dim>& getRMax() { return rmax_m; }
    void setRMax(const Vector_t<double, Dim>& rmax) { rmax_m = rmax; }

    ///< Get/Set the decomposition array
    std::array<bool, Dim> getDecomp() { return decomp_m; }
    void setDecomp(std::array<bool, Dim> decomp) { decomp_m = decomp; }

    ///< Get/Set the mesh
    Mesh_t<Dim>& getMesh() { return mesh_m; }
    void setMesh(Mesh_t<Dim>& mesh) { mesh_m = mesh; }

    ///< Get/Set the field layout
    FieldLayout_t<Dim>& getFL() { return fl_m; }
    void setFL(std::shared_ptr<FieldLayout_t<Dim>>& fl) { fl_m = fl; }

    /**
     * @brief Initialize the fields based on the provided type
     *
     * @param stype_m String indicating the solver type (e.g., "CG" for the conjugate gradient
     * solver)
     */
    void initializeFields(std::string stype_m = "") {
        F_m.initialize(mesh_m, fl_m);
        rho_m.initialize(mesh_m, fl_m);
        if (stype_m == "CG") {
            phi_m.initialize(mesh_m, fl_m);
        }
    }

private:
    // Member variables
    Vector_t<double, Dim> hr_m;      ///< Grid spacing vector
    Vector_t<double, Dim> rmin_m;    ///< Start points of domain
    Vector_t<double, Dim> rmax_m;    ///< End points of domain
    std::array<bool, Dim> decomp_m;  ///< Decomposition array for each dimension
    VField_t<T, Dim> F_m;            ///< Force field
    Field_t<Dim> rho_m;              ///< Mass density
    Field<T, Dim> phi_m;             ///< Potential
    Mesh_t<Dim> mesh_m;              ///< Mesh associated with the fields
    FieldLayout_t<Dim> fl_m;         ///< Field layout
};

#endif
