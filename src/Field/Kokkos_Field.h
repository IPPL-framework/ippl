#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"

namespace ippl {

    template<typename T, unsigned Dim,
            class M=UniformCartesian<double, Dim>,
            class C=typename M::DefaultCentering >
    class Field //: public FieldExpression<Field<T, Dim, M, C>>
                : public BareField<T, Dim> {

    public:
        typedef M Mesh_t;
        typedef C Centering_t;
        typedef FieldLayout<Dim> Layout_t;

        typedef typename BareField<T, Dim>::LField_t LField_t;

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

//         using BareField<T, Dim>::operator[];

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
        template<typename T, unsigned Dim, class M, class C>
        struct field_meta_grad : public FieldExpression<field_meta_grad<T, Dim, M, C>> {
            field_meta_grad(const Field<T, Dim, M, C>& u) : u_m(u) {
                M& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(M::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(M::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }

            }

            auto operator[](size_t i) const {
                if constexpr (M::Dimension == 1) {
                    return grad(u_m[i], xvector_m);
                }

                if constexpr (M::Dimension == 2) {
                    return grad(u_m[i], xvector_m, yvector_m);
                }

                if constexpr (M::Dimension == 3) {
                    return grad(u_m[i], xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            const Field<T, Dim, M, C>& u_m;
            typename M::vector_type xvector_m;
            typename M::vector_type yvector_m;
            typename M::vector_type zvector_m;
        };
    }

    template<typename T, unsigned Dim, class M, class C>
    detail::field_meta_grad<T, Dim, M, C> grad(const Field<T, Dim, M, C>& u) {
        return detail::field_meta_grad<T, Dim, M, C>(u);
    }

    namespace detail {
        /*
         * Divergence
         */
        template<typename T, unsigned Dim, class M, class C>
        struct field_meta_div : public FieldExpression<field_meta_div<T, Dim, M, C>> {
            field_meta_div(const Field<T, Dim, M, C>& u) : u_m(u) {
                M& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(M::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(M::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }
            }

            auto operator[](size_t i) const {
                if constexpr (M::Dimension == 1) {
                    return div(u_m[i], xvector_m);
                }

                if constexpr (M::Dimension == 2) {
                    return div(u_m[i], xvector_m, yvector_m);
                }

                if constexpr (M::Dimension == 3) {
                    return div(u_m[i], xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            const Field<T, Dim, M, C>& u_m;
            typename M::vector_type xvector_m;
            typename M::vector_type yvector_m;
            typename M::vector_type zvector_m;
        };
    }

    template<typename T, unsigned Dim, class M, class C>
    detail::field_meta_div<T, Dim, M, C> div(const Field<T, Dim, M, C>& u) {
        return detail::field_meta_div<T, Dim, M, C>(u);
    }

    namespace detail {
        /*
         * Laplacian
         */
        template<typename T, unsigned Dim, class M, class C>
        struct field_meta_laplace : public FieldExpression<field_meta_laplace<T, Dim, M, C>> {
            field_meta_laplace(const Field<T, Dim, M, C>& u) : u_m(u) {
                M& mesh = u.get_mesh();

                hvector_m[0] = 1.0 / std::pow(mesh.getMeshSpacing(0),2);

                if constexpr(M::Dimension > 1) {
                    hvector_m[1] = 1.0 / std::pow(mesh.getMeshSpacing(1),2);
                }

                if constexpr(M::Dimension == 3) {
                    hvector_m[2] = 1.0 / std::pow(mesh.getMeshSpacing(2),2);
                }
            }

            auto operator[](size_t i) const {
                    return laplace(u_m[i], hvector_m);
            }

        private:
            const Field<T, Dim, M, C>& u_m;
            typename M::vector_type hvector_m;
        };
    }

    template<typename T, unsigned Dim, class M, class C>
    detail::field_meta_laplace<T, Dim, M, C> laplace(const Field<T, Dim, M, C>& u) {
        return detail::field_meta_laplace<T, Dim, M, C>(u);
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
