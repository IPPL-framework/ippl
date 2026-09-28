/**
 * @file FELFieldContainer.hpp
 * @brief Distributed collocated E, B and four-current storage used by FEL.
 * @ingroup fel_fields
 */
#ifndef IPPL_FEL_FIELD_CONTAINER_H
#define IPPL_FEL_FIELD_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

#include "datatypes.h"

/**
 * @brief Own the Cartesian mesh, MPI layout and three FEL field arrays.
 * @ingroup fel_fields
 * @tparam T Scalar type of field components and supplied geometry metadata.
 * @tparam Dim Mesh dimension; the FEL manager uses three dimensions.
 *
 * E and B each have three components; J stores (rho, Jx, Jy, Jz) in 3D.
 * All components use the cell centering selected in datatypes.h. IPPL owns
 * rank-local Kokkos storage, including halos. This container supplies storage;
 * deposition, communication and evolution are driven by the manager/solver.
 * The active solver owns the three four-potential histories separately.
 *
 * FEL supplies boosted-frame geometry and normalized units from units.h.
 * The container itself performs no frame or unit conversion and is also reused
 * by mesh-only ChDR code with SI geometry. Its constructor uses MPI_COMM_WORLD;
 * nonperiodic, z-only decomposition is a manager choice, not hard-coded here.
 * No dielectric constitutive coefficients or absorbing update are stored here.
 */
template <typename T, unsigned Dim = 3>
class FELFieldContainer {
public:
    /**
     * @brief Construct mesh and layout; call initializeFields() to allocate arrays.
     * @param hr Uniform spacing in each coordinate direction.
     * @param rmin Lower physical bounds retained as metadata.
     * @param rmax Upper physical bounds retained as metadata.
     * @param decomp Directions eligible for MPI decomposition.
     * @param domain Global index domain of the mesh.
     * @param origin Physical origin used to construct the Cartesian mesh.
     * @param isAllPeriodic Periodicity flag passed to the field layout.
     */
    FELFieldContainer(Vector_t<T, Dim>& hr, Vector_t<T, Dim>& rmin, Vector_t<T, Dim>& rmax,
                      std::array<bool, Dim> decomp, ippl::NDIndex<Dim> domain,
                      Vector_t<T, Dim> origin, bool isAllPeriodic)
        : hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , decomp_m(decomp)
        , mesh_m(domain, hr, origin)
        , fl_m(MPI_COMM_WORLD, domain, decomp, isAllPeriodic) {}

    /// @brief Release owned mesh/layout objects and field handles.
    ~FELFieldContainer() {}

private:
    Vector_t<T, Dim> hr_m;          ///< Cached uniform spacing in caller-supplied length units.
    Vector_t<T, Dim> rmin_m;        ///< Cached lower physical bounds.
    Vector_t<T, Dim> rmax_m;        ///< Cached upper physical bounds.
    std::array<bool, Dim> decomp_m; ///< Cached eligible decomposition directions.
    VField_t<T, Dim> E_m;          ///< Reconstructed electric field, including local halos.
    VField_t<T, Dim> B_m;          ///< Reconstructed magnetic field, including local halos.
    SourceField_t<T, Dim> J_m;     ///< Deposited charge/current source, reset before each deposit.
    Mesh_t<Dim> mesh_m;            ///< Uniform Cartesian geometry shared by the fields.
    FieldLayout_t<Dim> fl_m;       ///< MPI ownership layout on MPI_COMM_WORLD.

public:
    /// @brief Return mutable electric-field storage; no halo exchange is performed.
    VField_t<T, Dim>& getE() { return E_m; }
    /// @brief Assign the electric field using IPPL field-assignment semantics.
    /// @param E Compatible field to assign; no unit or frame conversion is applied.
    void setE(VField_t<T, Dim>& E) { E_m = E; }

    /// @brief Return mutable magnetic-field storage; no halo exchange is performed.
    VField_t<T, Dim>& getB() { return B_m; }
    /// @brief Assign the magnetic field using IPPL field-assignment semantics.
    /// @param B Compatible field to assign; no unit or frame conversion is applied.
    void setB(VField_t<T, Dim>& B) { B_m = B; }

    /// @brief Return mutable four-current storage (charge density in component zero).
    SourceField_t<T, Dim>& getJ() { return J_m; }
    /// @brief Assign the source field using IPPL field-assignment semantics.
    /// @param J Compatible charge/current field to assign.
    void setJ(SourceField_t<T, Dim>& J) { J_m = J; }

    /// @brief Return the cached spacing, not a direct view into mesh geometry.
    Vector_t<T, Dim>& getHr() { return hr_m; }
    /// @brief Update cached spacing only; existing mesh/fields are not rebuilt.
    /// @param hr Replacement spacing metadata.
    void setHr(const Vector_t<T, Dim>& hr) { hr_m = hr; }

    /// @brief Return cached lower physical bounds.
    Vector_t<T, Dim>& getRMin() { return rmin_m; }
    /// @brief Update cached lower bounds only; existing mesh/fields are unchanged.
    /// @param rmin Replacement lower-bound metadata.
    void setRMin(const Vector_t<T, Dim>& rmin) { rmin_m = rmin; }

    /// @brief Return cached upper physical bounds.
    Vector_t<T, Dim>& getRMax() { return rmax_m; }
    /// @brief Update cached upper bounds only; existing mesh/fields are unchanged.
    /// @param rmax Replacement upper-bound metadata.
    void setRMax(const Vector_t<T, Dim>& rmax) { rmax_m = rmax; }

    /// @brief Return cached decomposition flags by value.
    std::array<bool, Dim> getDecomp() { return decomp_m; }
    /// @brief Update cached flags only; this does not repartition the layout.
    /// @param decomp Replacement decomposition metadata.
    void setDecomp(std::array<bool, Dim> decomp) { decomp_m = decomp; }

    /// @brief Return the mesh object used by deposition, interpolation and fields.
    Mesh_t<Dim>& getMesh() { return mesh_m; }
    /// @brief Assign mesh geometry without reallocating already initialized fields.
    /// @param mesh Compatible replacement mesh; caller must preserve field consistency.
    void setMesh(Mesh_t<Dim>& mesh) { mesh_m = mesh; }

    /// @brief Return the layout describing each MPI rank's owned global indices.
    FieldLayout_t<Dim>& getFL() { return fl_m; }
    /// @brief Assign the layout object without migrating particles or resizing fields.
    /// @param fl Compatible replacement layout; this is not an adaptive-mesh operation.
    void setFL(FieldLayout_t<Dim>& fl) { fl_m = fl; }

    /**
     * @brief Initialize E, B and J against the owned mesh and layout.
     *
     * IPPL allocates rank-local Kokkos arrays with its default halo width and
     * installs default no-operation face conditions. This method does not fill
     * halos, deposit a source or configure the solver's absorbing boundaries.
     * It is an initialization hook, not an explicit field-reset operation.
     */
    void initializeFields() {
        E_m.initialize(mesh_m, fl_m);
        B_m.initialize(mesh_m, fl_m);
        J_m.initialize(mesh_m, fl_m);
    }
};

#endif
