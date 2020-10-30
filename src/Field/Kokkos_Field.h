#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"
#include "Field/BCond.h"

namespace ippl {

    template<typename T, unsigned Dim,
            class Mesh = UniformCartesian<double, Dim>,
            class Cell = typename Mesh::DefaultCentering >
    class Field //: public FieldExpression<Field<T, Dim, Mesh, Centering>>
                : public BareField<T, Dim> {

    public:
        typedef Mesh Mesh_t;
        typedef Cell Centering_t;
        typedef FieldLayout<Dim> Layout_t;
        typedef BConds<T, Dim, Mesh, Cell> bc_container;
        typedef typename BareField<T, Dim>::LField_t LField_t;

        /*!
         * Default constructor, which should be used only if the user calls the
         * 'initialize' function before doing anything else.  There are no special
         * checks in the rest of the Field methods to check that the Field has
         * been properly initialized
         */
        Field();

        virtual ~Field() = default;

        /*!
         * Constructor which includes a Mesh and Layout object as argument
         * @param m mesh
         * @param l field layout
         */
        Field(Mesh_t& m, Layout_t& l);

        /*!
         * Constructor which includes a Mesh and Layout object as argument
         * @param m mesh
         * @param l field layout
         * @param bc boundary conditions at each face
         */
        Field(Mesh_t& m, Layout_t& l, const bc_container& bc);

        /*!
         * Initialize the Field, specifying a mesh and a field layout
         * @param m mesh
         * @param l field layout
         */
        void initialize(Mesh_t& m, Layout_t& l);


        /*!
         * Initialize the Field, specifying a mesh and a field layout
         * @param m mesh
         * @param l field layout
         * @param bc boundary conditions at each face
         */
        void initialize(Mesh_t& m, Layout_t& l, const bc_container& bc);

        // Access to the mesh
        Mesh_t& get_mesh() const { return *mesh_m; }

        // Assignment from constants and other arrays.
        using BareField<T, Dim>::operator=;

    private:
        //! Boundary conditions.
        bc_container bc_m;

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
        template<typename T, unsigned Dim, class Mesh, class Cell>
        struct field_meta_grad : public FieldExpression<field_meta_grad<T, Dim, Mesh, Cell>> {
            field_meta_grad(const Field<T, Dim, Mesh, Cell>& u) : u_m(u) {
                Mesh& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(Mesh::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(Mesh::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }

            }

            auto operator[](size_t i) const {
                if constexpr (Mesh::Dimension == 1) {
                    return grad(u_m[i], xvector_m);
                }

                if constexpr (Mesh::Dimension == 2) {
                    return grad(u_m[i], xvector_m, yvector_m);
                }

                if constexpr (Mesh::Dimension == 3) {
                    return grad(u_m[i], xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            const Field<T, Dim, Mesh, Cell>& u_m;
            typename Mesh::vector_type xvector_m;
            typename Mesh::vector_type yvector_m;
            typename Mesh::vector_type zvector_m;
        };
    }

    template<typename T, unsigned Dim, class Mesh, class Cell>
    detail::field_meta_grad<T, Dim, Mesh, Cell> grad(const Field<T, Dim, Mesh, Cell>& u) {
        return detail::field_meta_grad<T, Dim, Mesh, Cell>(u);
    }

    namespace detail {
        /*
         * Divergence
         */
        template<typename T, unsigned Dim, class Mesh, class Cell>
        struct field_meta_div : public FieldExpression<field_meta_div<T, Dim, Mesh, Cell>> {
            field_meta_div(const Field<T, Dim, Mesh, Cell>& u) : u_m(u) {
                Mesh& mesh = u.get_mesh();

                xvector_m[0] = 0.5 / mesh.getMeshSpacing(0);

                if constexpr(Mesh::Dimension > 1) {
                    xvector_m[1] = 0.0;
                    yvector_m[0] = 0.0;
                    yvector_m[1] = 0.5 / mesh.getMeshSpacing(1);
                }

                if constexpr(Mesh::Dimension == 3) {
                    xvector_m[2] = 0.0;
                    yvector_m[2] = 0.0;
                    zvector_m[0] = 0.0;
                    zvector_m[1] = 0.0;
                    zvector_m[2] = 0.5 / mesh.getMeshSpacing(2);
                }
            }

            auto operator[](size_t i) const {
                if constexpr (Mesh::Dimension == 1) {
                    return div(u_m[i], xvector_m);
                }

                if constexpr (Mesh::Dimension == 2) {
                    return div(u_m[i], xvector_m, yvector_m);
                }

                if constexpr (Mesh::Dimension == 3) {
                    return div(u_m[i], xvector_m, yvector_m, zvector_m);
                }
            }

        private:
            const Field<T, Dim, Mesh, Cell>& u_m;
            typename Mesh::vector_type xvector_m;
            typename Mesh::vector_type yvector_m;
            typename Mesh::vector_type zvector_m;
        };
    }

    template<typename T, unsigned Dim, class Mesh, class Cell>
    detail::field_meta_div<T, Dim, Mesh, Cell> div(const Field<T, Dim, Mesh, Cell>& u) {
        return detail::field_meta_div<T, Dim, Mesh, Cell>(u);
    }

    namespace detail {
        /*
         * Laplacian
         */
        template<typename T, unsigned Dim, class Mesh, class Cell>
        struct field_meta_laplace : public FieldExpression<field_meta_laplace<T, Dim, Mesh, Cell>> {
            field_meta_laplace(const Field<T, Dim, Mesh, Cell>& u) : u_m(u) {
                Mesh& mesh = u.get_mesh();

                hvector_m[0] = 1.0 / std::pow(mesh.getMeshSpacing(0),2);

                if constexpr(Mesh::Dimension > 1) {
                    hvector_m[1] = 1.0 / std::pow(mesh.getMeshSpacing(1),2);
                }

                if constexpr(Mesh::Dimension == 3) {
                    hvector_m[2] = 1.0 / std::pow(mesh.getMeshSpacing(2),2);
                }
            }

            auto operator[](size_t i) const {
                    return laplace(u_m[i], hvector_m);
            }

        private:
            const Field<T, Dim, Mesh, Cell>& u_m;
            typename Mesh::vector_type hvector_m;
        };
    }

    template<typename T, unsigned Dim, class Mesh, class Cell>
    detail::field_meta_laplace<T, Dim, Mesh, Cell> laplace(const Field<T, Dim, Mesh, Cell>& u) {
        return detail::field_meta_laplace<T, Dim, Mesh, Cell>(u);
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
