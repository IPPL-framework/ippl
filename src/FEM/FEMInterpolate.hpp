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



    /**
     * @brief Assemble a P1 FEM load vector (RHS) from particle attributes.
     *
     * For each particle position x, locate the owning element (ND index e_nd) and
     * reference coordinate xi. Deposit the particle attribute value into the
     * element's nodal DOFs using P1 Lagrange shape functions evaluated at xi.
     *
     * @tparam AttribIn   Particle attribute type with getView()(p) -> scalar
     * @tparam Field      ippl::Field with rank=Dim nodal coefficients (RHS)
     * @tparam PosAttrib  Particle position attribute with getView()(p) -> Vector<T,Dim>
     * @tparam Space      Lagrange space providing element/DOF/topology queries
     * @tparam policy_type Kokkos execution policy (defaults to Field::execution_space)
     */
    template <typename AttribIn, typename Field, typename PosAttrib, typename Space,
        typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void assemble_rhs_from_particles(const AttribIn& attrib, Field& f,
                                             const PosAttrib& pp, const Space& space,
                                             policy_type iteration_policy)
    {
        constexpr unsigned Dim = Field::dim;
        using T          = typename Field::value_type;
        using view_type  = typename Field::view_type;
        using mesh_type  = typename Field::Mesh_t;

        static IpplTimings::TimerRef t = IpplTimings::getTimer("assemble_rhs_from_particles(P1)");

        IpplTimings::startTimer(t);

        view_type view = f.getView();
        
        // Mesh / layout (for locating + indexing into the field view)
        const mesh_type& mesh = f.get_mesh();

        const ippl::FieldLayout<Dim>& layout = f.getLayout();
        const ippl::NDIndex<Dim>&     lDom   = layout.getLocalNDIndex();
        const int                     nghost = f.getNghost();

        // Particle attribute/device views
        auto d_attr = attrib.getView();  // scalar weight per particle (e.g. charge)
        auto d_pos  = pp.getView();      // positions (Vector<T,Dim>) per particle

        Kokkos::parallel_for("assemble_rhs_from_particles_P1", iteration_policy,
            KOKKOS_LAMBDA(const size_t p) {
                const Vector<T,Dim> x   = d_pos(p);
                const T val = d_attr(p);  

                Vector<size_t,Dim> e_nd;
                Vector<T,Dim>      xi;
                locate_element_nd_and_xi<T,Dim>(mesh, x, e_nd, xi);
                // Convert to the element's linear index
                const size_t e_lin = space.getElementIndex(e_nd);

                // Vertex indices and DOFs for this element
                const auto verts = space.getElementMeshVertexIndices(e_nd);
                const auto dofs  = space.getGlobalDOFIndices(e_lin);

                // Deposit into each vertex/DOF
                for (size_t a = 0; a < dofs.dim; ++a) {
                    const size_t local = space.getLocalDOFIndex(e_lin, dofs[a]); 
                    const T w = space.evaluateRefElementShapeFunction(local, xi);

                    const auto v_nd = space.getMeshVertexNDIndex(verts[a]); // ND coords (global, vertex-centered)
                    ippl::Vector<size_t,Dim> I;                             // indices into view

                    for (unsigned d = 0; d < Dim; ++d) {
                        I[d] = static_cast<size_t>(v_nd[d] - lDom.first()[d] + nghost);
                    }
                    if constexpr (Dim == 1) {
                        Kokkos::atomic_add(&view(I[0]), val * w);
                    } else if constexpr (Dim == 2) {
                        Kokkos::atomic_add(&view(I[0], I[1]), val * w);
                    } else if constexpr (Dim == 3) {
                        Kokkos::atomic_add(&view(I[0], I[1], I[2]), val * w);
                    }
                }
            }
        );

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);

    }

    /**
     * @brief Interpolate a P1 FEM field to particle positions.
     *
     * For each particle position x, locate the owning element (ND index e_nd) and
     * reference coordinate xi. Evaluate P1 Lagrange shape functions at xi to
     * combine nodal coefficients and write u(x) to the particle attribute.
     *
     * @tparam AttribOut   Particle attribute type with getView()(p) -> scalar
     * @tparam Field       ippl::Field with rank=Dim nodal coefficients
     * @tparam PosAttrib   Particle position attribute with getView()(p) -> Vector<T,Dim>
     * @tparam Space       Lagrange space providing element/DOF/topology queries
     * @tparam policy_type Kokkos execution policy (defaults to Field::execution_space)
     */
    template <typename AttribOut, typename Field, typename PosAttrib, typename Space,
              typename policy_type = Kokkos::RangePolicy<typename Field::execution_space>>
    inline void interpolate_to_diracs(AttribOut& attrib_out,
                                               const Field& coeffs,
                                               const PosAttrib& pp,
                                               const Space& space,
                                               policy_type iteration_policy)
    {
        constexpr unsigned Dim = Field::dim;
        using T                 = typename AttribOut::value_type;
        using field_value_type  = typename Field::value_type;
        using view_type         = typename Field::view_type;
        using mesh_type         = typename Field::Mesh_t;

        static IpplTimings::TimerRef timer =
            IpplTimings::getTimer("interpolate_field_to_particles(P1)");
        IpplTimings::startTimer(timer);

        view_type view     = coeffs.getView();
        const mesh_type& M = coeffs.get_mesh();


        const ippl::FieldLayout<Dim>& layout = coeffs.getLayout();
        const ippl::NDIndex<Dim>&     lDom   = layout.getLocalNDIndex();
        const int                     nghost = coeffs.getNghost();

        // Particle device views
        auto d_pos = pp.getView();
        auto d_out = attrib_out.getView();

        // Small device helper to read a rank-Dim view at ND indices
        auto read_view = KOKKOS_LAMBDA(const view_type& v,
                const ippl::Vector<size_t,Dim>& I) -> field_value_type {

            if constexpr (Dim == 1) return v(I[0]);
            if constexpr (Dim == 2) return v(I[0], I[1]);
            if constexpr (Dim == 3) return v(I[0], I[1], I[2]);
        };


        Kokkos::parallel_for("interpolate_to_diracs_P1", iteration_policy,
                KOKKOS_LAMBDA(const size_t p) {

            const Vector<T,Dim> x   = d_pos(p);

            ippl::Vector<size_t,Dim> e_nd;
            ippl::Vector<T,Dim>      xi;
            locate_element_nd_and_xi<T,Dim>(M, x, e_nd, xi);
            const size_t e_lin = space.getElementIndex(e_nd);

            const auto verts = space.getElementMeshVertexIndices(e_nd);
            const auto dofs  = space.getGlobalDOFIndices(e_lin);

            field_value_type up = field_value_type(0);

            for (size_t a = 0; a < dofs.dim; ++a) {
                const size_t local = space.getLocalDOFIndex(e_lin, dofs[a]);
                const field_value_type w = space.evaluateRefElementShapeFunction(local, xi);

                const auto v_nd = space.getMeshVertexNDIndex(verts[a]);
                ippl::Vector<size_t,Dim> I;
                for (unsigned d = 0; d < Dim; ++d) {
                    I[d] = static_cast<size_t>(v_nd[d] - lDom.first()[d] + nghost);
                }

                up += read_view(view, I) * w;
            }
            d_out(p) = static_cast<T>(up);
        });
    }
} // namespace ippl
#endif
