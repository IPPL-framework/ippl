#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"

namespace ippl {

    template <typename T, unsigned Dim,
              class M=UniformCartesian<double, Dim>,
              class C=typename M::DefaultCentering >
    class Field : public detail::FieldExpression<Field<T, Dim, M, C>>
                , public BareField<T, Dim>
    {
    public:
        using Mesh_t      = M;
        using Centering_t = C;
        using Layout_t    = FieldLayout<Dim>;
        using BareField_t = BareField<T, Dim>;
        using view_type   = typename BareField_t::view_type;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the Field methods to check that the Field has
        // been properly initialized.
        Field();

        virtual ~Field() = default;

        // Constructors including a Mesh object as argument:
        Field(Mesh_t&, Layout_t&);

        // Initialize the Field, also specifying a mesh
        void initialize(Mesh_t&, Layout_t&);

        // Access to the mesh
        Mesh_t& get_mesh() const { return *mesh_m; }

        // Assignment from constants and other arrays.
        using BareField<T, Dim>::operator=;

        Field(const Field&) = default;

        template <typename E>
        Field& operator=(const detail::FieldExpression<E>& expr);

    private:
        //   // The boundary conditions.
        //   bcond_container Bc;

        // The Mesh object, and a flag indicating if we constructed it
        Mesh_t* mesh_m;
    };
}

#include "Field/Kokkos_Field.hpp"


namespace ippl {

    namespace detail{
        /*
         * Gradient
         */
        template <typename E>
        struct field_meta_grad : public FieldExpression<field_meta_grad<E>> {
            field_meta_grad(const E& u) : u_m(u) {
                Mesh_t& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(Mesh_t::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(Mesh_t::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }

            }

            auto operator()() const {
                if constexpr (Mesh_t::Dimension == 1) {
                    return grad(u_m, xvector_m);
                }

                if constexpr (Mesh_t::Dimension == 2) {
                    return grad(u_m, xvector_m, yvector_m);
                }

                if constexpr (Mesh_t::Dimension == 3) {
                    return grad(u_m, xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            using Mesh_t = typename E::Mesh_t;
            const E& u_m;
            typename Mesh_t::vector_type xvector_m;
            typename Mesh_t::vector_type yvector_m;
            typename Mesh_t::vector_type zvector_m;
        };
    }

    template <typename E,
              typename = std::enable_if<detail::isFieldExpression<E>::value>>
    detail::field_meta_grad<E> grad(const E& u) {
        return detail::field_meta_grad<E>(*static_cast<const E*>(&u));
    }

    namespace detail {
        /*
         * Divergence
         */
        template <typename E>
        struct field_meta_div : public FieldExpression<field_meta_div<E>> {
            field_meta_div(const E& u) : u_m(u) {
                Mesh_t& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(Mesh_t::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(Mesh_t::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }
            }

            auto operator()() const {
                if constexpr (Mesh_t::Dimension == 1) {
                    return div(u_m, xvector_m);
                }

                if constexpr (Mesh_t::Dimension == 2) {
                    return div(u_m, xvector_m, yvector_m);
                }

                if constexpr (Mesh_t::Dimension == 3) {
                    return div(u_m, xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            using Mesh_t = typename E::Mesh_t;
            const E& u_m;
            typename Mesh_t::vector_type xvector_m;
            typename Mesh_t::vector_type yvector_m;
            typename Mesh_t::vector_type zvector_m;
        };
    }

    template <typename E,
              typename = std::enable_if<detail::isFieldExpression<E>::value>>
    detail::field_meta_div<E> div(const E& u) {
        return detail::field_meta_div<E>(*static_cast<const E*>(&u));
    }

    namespace detail {
        /*
         * Laplacian
         */
        template <typename E>
        struct field_meta_laplace : public FieldExpression<field_meta_laplace<E>> {
            field_meta_laplace(const E& u) : u_m(u) {
                Mesh_t& mesh = u.get_mesh();

                hvector_m[0] = 1.0 / std::pow(mesh.getMeshSpacing(0),2);

                if constexpr(Mesh_t::Dimension > 1) {
                    hvector_m[1] = 1.0 / std::pow(mesh.getMeshSpacing(1),2);
                }

                if constexpr(Mesh_t::Dimension == 3) {
                    hvector_m[2] = 1.0 / std::pow(mesh.getMeshSpacing(2),2);
                }
            }

            auto operator()() const {
                    return laplace(u_m, hvector_m);
            }

        private:
            using Mesh_t = typename E::Mesh_t;
            const E& u_m;
            typename Mesh_t::vector_type hvector_m;
        };
    }

    template <typename E,
              typename = std::enable_if<detail::isFieldExpression<E>::value>>
    detail::field_meta_laplace<E> laplace(const E& u) {
        return detail::field_meta_laplace<E>(*static_cast<const E*>(&u));
    }
}

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
