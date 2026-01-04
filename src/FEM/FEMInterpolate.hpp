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
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void 
    locate_element_nd_and_xi(const Vector<T, Dim>& hr,
        const Vector<T, Dim>& origin,
        const Vector<T, Dim>& x,
        Vector<size_t, Dim>& e_nd,
        Vector<T, Dim>& xi) {

        for (unsigned d = 0; d < Dim; ++d) {
            const T s = (x[d] - origin[d]) / hr[d]; // To cell units
            const size_t e = static_cast<size_t>(Kokkos::floor(s));
            e_nd[d] = e;
            xi[d] = s - static_cast<T>(e);
        }
    }


    template<class View, class IVec, std::size_t... Is>
    KOKKOS_INLINE_FUNCTION
    auto view_ptr_impl(View& v, const IVec& I, std::index_sequence<Is...>)
      -> decltype(&v(I[Is]...)) {
      return &v(I[Is]...);
    }

    template<int D, class View, class IVec>
    KOKKOS_INLINE_FUNCTION
    auto view_ptr(View& v, const IVec& I)
      -> decltype(view_ptr_impl(v, I, std::make_index_sequence<D>{})) {
      return view_ptr_impl(v, I, std::make_index_sequence<D>{});
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

        const auto hr = mesh.getMeshSpacing();
        const auto origin = mesh.getOrigin();

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = f.getNghost();

        // Particle attribute/device views
        auto d_attr = attrib.getView();  // scalar weight per particle (e.g. charge)
        auto d_pos  = pp.getView();      // positions (Vector<T,Dim>) per particle

        // make device copy of space
        auto device_space = space.getDeviceMirror();

        Kokkos::parallel_for("assemble_rhs_from_particles_P1", iteration_policy,
            KOKKOS_LAMBDA(const size_t p) {
                const Vector<T, Dim> x = d_pos(p);
                const T val = d_attr(p);  

                Vector<size_t, Dim> e_nd;
                Vector<T, Dim> xi;

                locate_element_nd_and_xi<T, Dim>(hr, origin, x, e_nd, xi);

                // DOFs for this element
                const auto dofs = device_space.getGlobalDOFIndices(e_nd);

                // Deposit into each vertex/DOF
                for (size_t a = 0; a < dofs.dim; ++a) {
                    const size_t local = device_space.getLocalDOFIndex(e_nd, dofs[a]); 
                    const T w = device_space.evaluateRefElementShapeFunction(local, xi);

                    // ND coords (global, vertex-centered)
                    const auto v_nd = device_space.getMeshVertexNDIndex(dofs[a]);
                    Vector<size_t, Dim> I; // indices into view

                    for (unsigned d = 0; d < Dim; ++d) {
                        I[d] = static_cast<size_t>(v_nd[d] - lDom.first()[d] + nghost);
                    }
                    Kokkos::atomic_add(view_ptr<Dim>(view, I), val * w);
                }
            }
        );

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);

    }

    template<class View, class IVec, std::size_t... Is>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) view_ref_impl(View& v, const IVec& I, std::index_sequence<Is...>) {
        return v(I[Is]...);
    }

    template<int D, class View, class IVec>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) view_ref(View& v, const IVec& I) {
        return view_ref_impl(v, I, std::make_index_sequence<D>{});
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

        view_type view = coeffs.getView();
        const mesh_type& mesh = coeffs.get_mesh();

        const auto hr = mesh.getMeshSpacing();
        const auto origin = mesh.getOrigin();

        const FieldLayout<Dim>& layout = coeffs.getLayout();
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = coeffs.getNghost();

        // Particle device views
        auto d_pos = pp.getView();
        auto d_out = attrib_out.getView();

        // make device copy of space
        auto device_space = space.getDeviceMirror();
        Kokkos::parallel_for("interpolate_to_diracs_P1", iteration_policy,
                KOKKOS_LAMBDA(const size_t p) {

            const Vector<T, Dim> x = d_pos(p);

            Vector<size_t, Dim> e_nd;
            Vector<T, Dim> xi;
            locate_element_nd_and_xi<T, Dim>(hr, origin, x, e_nd, xi);

            const auto dofs = device_space.getGlobalDOFIndices(e_nd);

            field_value_type up = field_value_type(0);

            for (size_t a = 0; a < dofs.dim; ++a) {
                const size_t local = device_space.getLocalDOFIndex(e_nd, dofs[a]);
                const field_value_type w = device_space.evaluateRefElementShapeFunction(local, xi);

                const auto v_nd = device_space.getMeshVertexNDIndex(dofs[a]);
                Vector<size_t, Dim> I;
                for (unsigned d = 0; d < Dim; ++d) {
                    I[d] = static_cast<size_t>(v_nd[d] - lDom.first()[d] + nghost);
                }

                up += view_ref<Dim>(view, I) * w;
            }
            d_out(p) = static_cast<T>(up);
        });
    }

} // namespace ippl
#endif
