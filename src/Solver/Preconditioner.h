//
// Preconditioners for various operators.
//

#ifndef IPPL_PRECONDITIONER_H
#define IPPL_PRECONDITIONER_H

#include "Expression/IpplOperations.h" // get the function apply()

namespace ippl{
    namespace detail {
        /*!
        * Meta function of Laplacian Preconditioner
         */
        template <typename E>
        struct meta_laplace_jacobian_preconditioner
            : public Expression<meta_laplace_jacobian_preconditioner<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)> {
            constexpr static unsigned dim = E::dim;

            KOKKOS_FUNCTION
            meta_laplace_jacobian_preconditioner(const E& u, const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , hvector_m(hvector) {}

            /*
             * n-dimensional Laplacian preconditioner
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;
                T res            = 0;
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};
                    auto&& center          = apply(u_m, coords);
                    res += 1. / hvector_m[d] * (0.5 * center);
                }
                return res;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            const vector_type hvector_m;
        };
    }// namespace detail


    /*!
     * User interface of Laplacian_preconditioner
     * @param u field
     */
    template <typename Field>
    detail::meta_laplace_jacobian_preconditioner<Field> laplace_jacobian_preconditioner(Field& u) {
        constexpr unsigned Dim = Field::dim;

        u.fillHalo();
        BConds<Field, Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        using mesh_type = typename Field::Mesh_t;
        mesh_type& mesh = u.get_mesh();
        typename mesh_type::vector_type hvector(0);

        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / std::pow(mesh.getMeshSpacing(d), 2); //Jacobian Preconditioner
        }
        return detail::meta_laplace_jacobian_preconditioner<Field>(u, hvector);
    }
} //namespace ippl

#endif  // IPPL_PRECONDITIONER_H
