//
// File FieldOperations
//   Differential operators for fields
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

namespace ippl {
    /*!
     * User interface of gradient
     * @param u field
     */
    template <typename T, unsigned Dim, class Mesh, class Centering>
    detail::meta_grad<Field<T, Dim, Mesh, Centering>> grad(Field<T, Dim, Mesh, Centering>& u) {
        u.fillHalo();
        BConds<T, Dim, Mesh, Centering>& bcField = u.getFieldBC();
        bcField.apply(u);
        Mesh& mesh        = u.get_mesh();
        using vector_type = typename Mesh::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_grad<Field<T, Dim, Mesh, Centering>>(u, vectors);
    }

    /*!
     * User interface of divergence in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class Mesh, class Centering>
    detail::meta_div<Field<T, Dim, Mesh, Centering>> div(Field<T, Dim, Mesh, Centering>& u) {
        u.fillHalo();
        BConds<T, Dim, Mesh, Centering>& bcField = u.getFieldBC();
        bcField.apply(u);
        Mesh& mesh        = u.get_mesh();
        using vector_type = typename Mesh::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_div<Field<T, Dim, Mesh, Centering>>(u, vectors);
    }

    /*!
     * User interface of Laplacian
     * @param u field
     */
    template <typename T, unsigned Dim, class Mesh, class Centering>
    detail::meta_laplace<Field<T, Dim, Mesh, Centering>> laplace(
        Field<T, Dim, Mesh, Centering>& u) {
        u.fillHalo();
        BConds<T, Dim, Mesh, Centering>& bcField = u.getFieldBC();
        bcField.apply(u);
        Mesh& mesh = u.get_mesh();
        typename Mesh::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / std::pow(mesh.getMeshSpacing(d), 2);
        }
        return detail::meta_laplace<Field<T, Dim, Mesh, Centering>>(u, hvector);
    }

    /*!
     * User interface of curl in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class Mesh, class Centering>
    detail::meta_curl<Field<T, Dim, Mesh, Centering>> curl(Field<T, Dim, Mesh, Centering>& u) {
        u.fillHalo();
        BConds<T, Dim, Mesh, Centering>& bcField = u.getFieldBC();
        bcField.apply(u);
        Mesh& mesh = u.get_mesh();
        typename Mesh::vector_type xvector(0);
        xvector[0] = 1.0;
        typename Mesh::vector_type yvector(0);
        yvector[1] = 1.0;
        typename Mesh::vector_type zvector(0);
        zvector[2] = 1.0;
        typename Mesh::vector_type hvector(0);
        hvector = mesh.getMeshSpacing();
        return detail::meta_curl<Field<T, Dim, Mesh, Centering>>(u, xvector, yvector, zvector,
                                                                 hvector);
    }

    /*!
     * User interface of Hessian in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class Mesh, class Centering>
    detail::meta_hess<Field<T, Dim, Mesh, Centering>> hess(Field<T, Dim, Mesh, Centering>& u) {
        u.fillHalo();
        BConds<T, Dim, Mesh, Centering>& bcField = u.getFieldBC();
        bcField.apply(u);
        Mesh& mesh = u.get_mesh();

        using vector_type = typename Mesh::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 1;
        }
        auto hvector = mesh.getMeshSpacing();

        return detail::meta_hess<Field<T, Dim, Mesh, Centering>>(u, vectors, hvector);
    }

    /*!
     * Hessian based on onesided differencing in three dimensions
     * Forward (`+`) or backward (`-`) differencing depending on `IdxOp`
     * @param u field
     */
    template <typename IdxOp, typename T, unsigned Dim, class M, class C>
    detail::meta_onesidedHess<IdxOp, Field<T, Dim, M, C>> onesidedHess(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T,Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        M& mesh = u.get_mesh();
        typename M::vector_type xvector(0);
        xvector[0] = 1.0;
        typename M::vector_type yvector(0);
        yvector[1] = 1.0;
        typename M::vector_type zvector(0);
        zvector[2] = 1.0;
        typename M::vector_type hvector(0);
        hvector = mesh.getMeshSpacing();
        return detail::meta_onesidedHess<IdxOp, Field<T, Dim, M, C>>(u, xvector, yvector, zvector, hvector);
    }

    /*!
     * User interface of Hessian with forward differencing of second order in three dimensions
     * @param u field
     */
    template <typename IdxOp=std::binary_function<size_t,size_t,size_t>, typename T, unsigned Dim, class M, class C>
    auto forwardHess(Field<T, Dim, M, C>& u) {
        return onesidedHess<std::plus<size_t>>(u);
    }

    /*!
     * User interface of Hessian with backward differencing of second order in three dimensions
     * @param u field
     */
    template <typename IdxOp=std::binary_function<size_t,size_t,size_t>, typename T, unsigned Dim, class M, class C>
    auto backwardHess(Field<T, Dim, M, C>& u) {
        return onesidedHess<std::minus<size_t>>(u);
    }
}  // namespace ippl
