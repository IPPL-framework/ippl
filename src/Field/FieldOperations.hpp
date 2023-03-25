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
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_grad<Field<T, Dim, M, C>> grad(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T, Dim, M>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh           = u.get_mesh();
        using vector_type = typename M::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_grad<Field<T, Dim, M, C>>(u, vectors);
    }

    /*!
     * User interface of divergence in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_div<Field<T, Dim, M, C>> div(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T, Dim, M>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh           = u.get_mesh();
        using vector_type = typename M::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 0.5 / mesh.getMeshSpacing(d);
        }
        return detail::meta_div<Field<T, Dim, M, C>>(u, vectors);
    }

    /*!
     * User interface of Laplacian
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_laplace<Field<T, Dim, M, C>> laplace(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T, Dim, M>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh = u.get_mesh();
        typename M::vector_type hvector(0);
        for (unsigned d = 0; d < Dim; d++) {
            hvector[d] = 1.0 / std::pow(mesh.getMeshSpacing(d), 2);
        }
        return detail::meta_laplace<Field<T, Dim, M, C>>(u, hvector);
    }

    /*!
     * User interface of curl in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_curl<Field<T, Dim, M, C>> curl(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T, Dim, M>& bcField = u.getFieldBC();
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
        return detail::meta_curl<Field<T, Dim, M, C>>(u, xvector, yvector, zvector, hvector);
    }

    /*!
     * User interface of Hessian in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_hess<Field<T, Dim, M, C>> hess(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T, Dim, M>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh = u.get_mesh();

        using vector_type = typename M::vector_type;
        vector_type vectors[Dim];
        for (unsigned d = 0; d < Dim; d++) {
            vectors[d]    = 0;
            vectors[d][d] = 1;
        }
        auto hvector = mesh.getMeshSpacing();

        return detail::meta_hess<Field<T, Dim, M, C>>(u, vectors, hvector);
    }
}  // namespace ippl
