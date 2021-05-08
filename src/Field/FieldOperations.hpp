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
     * User interface of gradient in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_grad<Field<T, Dim, M, C>> grad(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T,Dim>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh = u.get_mesh();
        typename M::vector_type xvector(0);
        xvector[0] = 0.5 / mesh.getMeshSpacing(0);
        typename M::vector_type yvector(0);
            yvector[1] = 0.5 / mesh.getMeshSpacing(1);
        typename M::vector_type zvector(0);
        zvector[2] = 0.5 / mesh.getMeshSpacing(2);
        return detail::meta_grad<Field<T, Dim, M, C>>(u, xvector, yvector, zvector);
    }


    /*!
     * User interface of divergence in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_div<Field<T, Dim, M, C>> div(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T,Dim>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh = u.get_mesh();
        typename M::vector_type xvector(0);
        xvector[0] = 0.5 / mesh.getMeshSpacing(0);
        typename M::vector_type yvector(0);
        yvector[1] = 0.5 / mesh.getMeshSpacing(1);
        typename M::vector_type zvector(0);
        zvector[2] = 0.5 / mesh.getMeshSpacing(2);
        return detail::meta_div<Field<T, Dim, M, C>>(u, xvector, yvector, zvector);
    }


    /*!
     * User interface of Laplacian in three dimensions.
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_laplace<Field<T, Dim, M, C>> laplace(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T,Dim>& bcField = u.getFieldBC();
        bcField.apply(u);
        M& mesh = u.get_mesh();
        typename M::vector_type hvector(0);
        hvector[0] = 1.0 / std::pow(mesh.getMeshSpacing(0), 2);
        hvector[1] = 1.0 / std::pow(mesh.getMeshSpacing(1), 2);
        hvector[2] = 1.0 / std::pow(mesh.getMeshSpacing(2), 2);
        return detail::meta_laplace<Field<T, Dim, M, C>>(u, hvector);
    }
}
