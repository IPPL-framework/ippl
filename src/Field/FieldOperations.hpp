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
     * User interface of Hessian with safe Boundary handling (one sided differencing) in three dimensions
     * @param u field
     */
    template <typename T, unsigned Dim, class M, class C>
    detail::meta_hess<Field<T, Dim, M, C>> onesidedHess(Field<T, Dim, M, C>& u) {
        u.fillHalo();
        BConds<T,Dim>& bcField = u.getFieldBC();
        bcField.apply(u);

        // Check if on physical boundary
        const auto &layout = u.getLayout();
        const int &nghost = u.getNghost();
        const auto &domain = layout.getDomain();
        const auto &lDomains = layout.getHostLocalDomains();
        int myRank = Ippl::Comm->rank();
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        bool isBoundary = (lDomains[myRank][1].max() == domain[1].max()) ||
                          (lDomains[myRank][1].min() == domain[1].min());

        if(Ippl::Comm->rank() == 0){
            std::cout << "isBoundary: " << isBoundary << std::endl;
            std::cout << "nghost: " << nghost << std::endl;
            std::cout << lDom << std::endl;
        }

        M& mesh = u.get_mesh();
        typename M::vector_type xvector(0);
        xvector[0] = 1.0;
        typename M::vector_type yvector(0);
        yvector[1] = 1.0;
        typename M::vector_type zvector(0);
        zvector[2] = 1.0;
        typename M::vector_type hvector(0);
        hvector = mesh.getMeshSpacing();
        return detail::meta_hess<Field<T, Dim, M, C>>(u, xvector, yvector, zvector, hvector);
    }
}  // namespace ippl
