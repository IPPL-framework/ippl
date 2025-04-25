//
// Helper functions used in PoissonCG.h
//

#ifndef IPPL_LAPLACE_HELPERS_H
#define IPPL_LAPLACE_HELPERS_H
namespace ippl {
    namespace detail {
        // Implements the poisson matrix acting on a d dimensional field
        template <typename E>
        struct meta_poisson : public Expression<meta_poisson<E>, sizeof(E)> {
            constexpr static unsigned dim = E::dim;

            KOKKOS_FUNCTION
            meta_poisson(const E& u)
                : u_m(u) {}

            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T                = typename E::Mesh_t::value_type;
                T res                  = 0;
                index_type coords[dim] = {args...};
                auto&& center          = apply(u_m, coords);
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};

                    coords[d] -= 1;
                    auto&& left = apply(u_m, coords);

                    coords[d] += 2;
                    auto&& right = apply(u_m, coords);
                    res += (2.0 * center - left - right);
                }
                return res;
            }

        private:
            const E u_m;
        };

        template <typename E>
        struct meta_lower_laplace
            : public Expression<meta_lower_laplace<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)
                                    + 2 * sizeof(typename E::Layout_t::NDIndex_t)
                                    + sizeof(unsigned)> {
            constexpr static unsigned dim = E::dim;
            using value_type              = typename E::value_type;

            KOKKOS_FUNCTION
            meta_lower_laplace(const E& u, const typename E::Mesh_t::vector_type& hvector,
                               unsigned nghosts, const typename E::Layout_t::NDIndex_t& ldom,
                               const typename E::Layout_t::NDIndex_t& domain)
                : u_m(u)
                , hvector_m(hvector)
                , nghosts_m(nghosts)
                , ldom_m(ldom)
                , domain_m(domain) {}

            /*
             * n-dimensional lower triangular Laplacian
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;
                T res            = 0;

                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim]       = {args...};
                    const int global_index       = coords[d] + ldom_m[d].first() - nghosts_m;
                    const int size               = domain_m.length()[d];
                    const bool not_left_boundary = (global_index != 0);
                    const bool right_boundary    = (global_index == size - 1);

                    coords[d] -= 1;
                    auto&& left = apply(u_m, coords);

                    coords[d] += 2;
                    auto&& right = apply(u_m, coords);

                    // not_left_boundary and right_boundary are boolean values
                    // Because of periodic boundary conditions we need to add this boolean mask to
                    // obtain the lower triangular part of the Laplace Operator
                    res += hvector_m[d] * (not_left_boundary * left + right_boundary * right);
                }
                return res;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using Layout_t    = typename E::Layout_t;
            using vector_type = typename Mesh_t::vector_type;
            using domain_type = typename Layout_t::NDIndex_t;
            const E u_m;
            const vector_type hvector_m;
            const unsigned nghosts_m;
            const domain_type ldom_m;
            const domain_type domain_m;
        };

        template <typename E>
        struct meta_upper_laplace
            : public Expression<meta_upper_laplace<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)
                                    + 2 * sizeof(typename E::Layout_t::NDIndex_t)
                                    + sizeof(unsigned)> {
            constexpr static unsigned dim = E::dim;
            using value_type              = typename E::value_type;

            KOKKOS_FUNCTION
            meta_upper_laplace(const E& u, const typename E::Mesh_t::vector_type& hvector,
                               unsigned nghosts, const typename E::Layout_t::NDIndex_t& ldom,
                               const typename E::Layout_t::NDIndex_t& domain)
                : u_m(u)
                , hvector_m(hvector)
                , nghosts_m(nghosts)
                , ldom_m(ldom)
                , domain_m(domain) {}

            /*
             * n-dimensional upper triangular Laplacian
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;
                T res            = 0;

                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim]        = {args...};
                    const int global_index        = coords[d] + ldom_m[d].first() - nghosts_m;
                    const int size                = domain_m.length()[d];
                    const bool left_boundary      = (global_index == 0);
                    const bool not_right_boundary = (global_index != size - 1);

                    coords[d] -= 1;
                    auto&& left = apply(u_m, coords);

                    coords[d] += 2;
                    auto&& right = apply(u_m, coords);

                    // left_boundary and not_right_boundary are boolean values
                    // Because of periodic boundary conditions we need to add this boolean mask to
                    // obtain the upper triangular part of the Laplace Operator
                    res += hvector_m[d] * (left_boundary * left + not_right_boundary * right);
                }
                return res;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using Layout_t    = typename E::Layout_t;
            using vector_type = typename Mesh_t::vector_type;
            using domain_type = typename Layout_t::NDIndex_t;
            const E u_m;
            const vector_type hvector_m;
            const unsigned nghosts_m;
            const domain_type ldom_m;
            const domain_type domain_m;
        };

        template <typename E>
        struct meta_upper_and_lower_laplace
            : public Expression<meta_upper_and_lower_laplace<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)> {
            constexpr static unsigned dim = E::dim;
            using value_type              = typename E::value_type;

            KOKKOS_FUNCTION
            meta_upper_and_lower_laplace(const E& u, const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , hvector_m(hvector) {}

            /*
             * n-dimensional upper+lower triangular Laplacian
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;
                T res            = 0;
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};
                    coords[d] -= 1;
                    auto&& left = apply(u_m, coords);
                    coords[d] += 2;
                    auto&& right = apply(u_m, coords);
                    res += hvector_m[d] * (left + right);
                }
                return res;
            }

        private:
            using vector_type = typename E::Mesh_t::vector_type;
            const E u_m;
            const vector_type hvector_m;
        };
    }  // namespace detail

    /*!
     * User interface of poisson
     * @param u field
     */
    template <typename Field>
    detail::meta_poisson<Field> poisson(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        return detail::meta_poisson<Field>(u);
    }

    /*!
     * User interface of lower triangular Laplacian
     * @param u field
     */
    template <typename Field>
    detail::meta_lower_laplace<Field> lower_laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        return lower_laplace_no_comm(u);
    }

    /*!
     * User interface of lower triangular Laplacian without exchange of halo cells
     * @param u field
     */
    template <typename Field>
    detail::meta_lower_laplace<Field> lower_laplace_no_comm(Field& u) {
        constexpr unsigned Dim = Field::dim;

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / Kokkos::pow(mesh.getMeshSpacing(d), 2);
        }
        const auto& layout = u.getLayout();
        unsigned nghosts   = u.getNghost();
        const auto& ldom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        return detail::meta_lower_laplace<Field>(u, hvector, nghosts, ldom, domain);
    }

    /*!
     * User interface of upper triangular Laplacian
     * @param u field
     */
    template <typename Field>
    detail::meta_upper_laplace<Field> upper_laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        return upper_laplace_no_comm(u);
    }

    /*!
     * User interface of upper triangular Laplacian without exchange of halo cells
     * @param u field
     */
    template <typename Field>
    detail::meta_upper_laplace<Field> upper_laplace_no_comm(Field& u) {
        constexpr unsigned Dim = Field::dim;

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / Kokkos::pow(mesh.getMeshSpacing(d), 2);
        }
        const auto& layout = u.getLayout();
        unsigned nghosts   = u.getNghost();
        const auto& ldom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        return detail::meta_upper_laplace<Field>(u, hvector, nghosts, ldom, domain);
    }

    /*!
     * User interface of upper+lower triangular Laplacian
     * @param u field
     */
    template <typename Field>
    detail::meta_upper_and_lower_laplace<Field> upper_and_lower_laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        return upper_and_lower_laplace_no_comm(u);
    }

    /*!
     * User interface of upper+lower triangular Laplacian without exchange of halo cells
     * @param u field
     */
    template <typename Field>
    detail::meta_upper_and_lower_laplace<Field> upper_and_lower_laplace_no_comm(Field& u) {
        constexpr unsigned Dim = Field::dim;

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / Kokkos::pow(mesh.getMeshSpacing(d), 2);
        }
        return detail::meta_upper_and_lower_laplace<Field>(u, hvector);
    }

    /*!
     * Returns the factor by which to multiply the u field to get
     * the inverse of the diagonal of the Laplacian.
     *
     * @param u field
     */

    template <typename Field>
    double negative_inverse_diagonal_laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;
        using mesh_type        = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        double sum    = 0.0;
        double factor = 1.0;
        typename mesh_type::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; ++d) {
            hvector[d] = Kokkos::pow(mesh.getMeshSpacing(d), 2);
            sum += hvector[d] * Kokkos::pow(mesh.getMeshSpacing((d + 1) % Dim), 2);
            factor *= hvector[d];
        }

        return 0.5 * (factor / sum);
    }

    template <typename Field>
    double diagonal_laplace(Field& u) {
        constexpr unsigned Dim = Field::dim;
        using mesh_type        = typename Field::Mesh_t;
        mesh_type& mesh        = u.get_mesh();
        double sum             = 0.0;
        for (unsigned d = 0; d < Dim; ++d) {
            sum += 1 / (Kokkos::pow(mesh.getMeshSpacing(d), 2));
        }

        // u = - 2.0 * sum * u;
        return -2.0 * sum;
    }

    template <typename Field>
    void mult(Field& u, const double c) {
        using view_type = Field::view_type;

        view_type view = u.getView();

        Kokkos::parallel_for(
            "Field_mult_const", u.getFieldRangePolicy(),
            KOKKOS_LAMBDA(int i, int j, int k) { view(i, j, k) *= c; });
        return;
    }
}  // namespace ippl
#endif  // IPPL_LAPLACE_HELPERS_H
