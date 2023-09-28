//
// File FieldOperations
//   Differential operators for fields
//

namespace ippl {
    /*!
     * User interface of gradient
     * @param u field
     */
    template <typename Field>
    detail::meta_grad<Field> grad(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type   = typename Field::Mesh_t;
        using vector_type = typename mesh_type::vector_type;

        mesh_type& mesh = u.get_mesh();
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_grad<Field>(u, vectors);
    }

    /*!
     * User interface of divergence in three dimensions.
     * @param u field
     */
    template <typename Field>
    detail::meta_div<Field> div(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type   = typename Field::Mesh_t;
        using vector_type = typename mesh_type::vector_type;

        mesh_type& mesh = u.get_mesh();
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_div<Field>(u, vectors);
    }

    /*!
     * User interface of Laplacian
     * @param u field
     */
    template <typename Field>
    detail::meta_laplace<Field> laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / std::pow(mesh.getMeshSpacing(d), 2);
        }
        return detail::meta_laplace<Field>(u, hvector);
    }

    /*!
     * User interface of curl in three dimensions.
     * @param u field
     */
    template <typename Field>
    detail::meta_curl<Field> curl(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type xvector(0);
        xvector[0] = 1.0;
        typename mesh_type::vector_type yvector(0);
        yvector[1] = 1.0;
        typename mesh_type::vector_type zvector(0);
        zvector[2] = 1.0;
        typename mesh_type::vector_type hvector(0);
        hvector = mesh.getMeshSpacing();
        return detail::meta_curl<Field>(u, xvector, yvector, zvector, hvector);
    }

    /*!
     * User interface of Hessian in three dimensions.
     * @param u field
     */
    template <typename Field>
    detail::meta_hess<Field> hess(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type   = typename Field::Mesh_t;
        using vector_type = typename mesh_type::vector_type;

        mesh_type& mesh = u.get_mesh();
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 1;
        }
        auto hvector = mesh.getMeshSpacing();

        return detail::meta_hess<Field>(u, vectors, hvector);
    }
}  // namespace ippl
