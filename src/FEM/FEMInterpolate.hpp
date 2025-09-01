#ifndef IPPL_FEMINTERPOLATE_H
#define IPPL_FEMINTERPOLATE_H


namespace ippl {

    /** 
    * @brief Mapping from global position to element ND index and
    * reference coordinates (xi ∈ [0,1)^Dim) on a UniformCartesian mesh.
    *
    * Assumes the input x is strictly inside the computational domain so that
    * for each dimension d: 0 ≤ (x[d]-origin[d])/h[d] < nr[d]-1.
    */
    template <typename T, unsigned Dim, class Mesh>
    KOKKOS_INLINE_FUNCTION void 
    locate_element_nd_and_xi(const Mesh& mesh,
        const ippl::Vector<T,Dim>& x,
        ippl::Vector<size_t,Dim>& e_nd,
        ippl::Vector<T,Dim>& xi) {

        const auto nr = mesh.getGridsize(); // vertices per axis
        const auto h = mesh.getMeshSpacing();
        const auto org = mesh.getOrigin();


        for (unsigned d = 0; d < Dim; ++d) {
            const T s = (x[d] - org[d]) / h[d]; // To cell units
            const size_t e = static_cast<size_t>(std::floor(s));
            e_nd[d] = e;
            xi[d] = s - static_cast<T>(e);
        }
    }

} // namespace ippl
#endif
