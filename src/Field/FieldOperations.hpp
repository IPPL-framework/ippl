//
// File FieldOperations
//   Differential operators, norms, and a scalar product for fields
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
     * Computes the inner product of two fields
     * @param f1 first field
     * @param f2 second field
     * @return Result of f1^T f2
     */
    template <typename T, unsigned Dim>
    T innerProduct(const Field<T, Dim>& f1, const Field<T, Dim>& f2) {
        T sum = 0;
        auto view1 = f1.getView();
        auto view2 = f2.getView();
        Kokkos::parallel_reduce("Field::innerProduct(Field&, Field&)", f1.getRangePolicy(),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& val) {
                val += view1(i, j, k) * view2(i, j, k);
            },
            Kokkos::Sum<T>(sum)
        );
        T globalSum = 0;
        MPI_Datatype type = get_mpi_datatype<T>(sum);
        MPI_Allreduce(&sum, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
        return globalSum;
    }

    /*!
     * Computes the Lp-norm of a field
     * @param field field
     * @param p desired norm (default 2)
     * @return The desired norm of the field
     */
    template<typename T, unsigned Dim, class M, class C>
    T norm(const Field<T, Dim, M, C>& field, int p = 2) {
        T local = 0;
        auto view = field.getView();
        switch (p) {
        case 0:
        {
            Kokkos::parallel_reduce("Field::norm(0)", field.getRangePolicy(),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, T& val) {
                    T myVal = std::abs(view(i, j, k));
                    if (myVal > val)
                        val = myVal;
                },
                Kokkos::Max<T>(local)
            );
            T globalMax = 0;
            MPI_Datatype type = get_mpi_datatype<T>(local);
            MPI_Allreduce(&local, &globalMax, 1, type, MPI_MAX, Ippl::getComm());
            return globalMax;
        }
        case 2:
            return std::sqrt(innerProduct(field, field));
        default:
        {
            Kokkos::parallel_reduce("Field::norm(int) general", field.getRangePolicy(),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, double& val) {
                    val += std::pow(std::abs(view(i, j, k)), p);
                },
                Kokkos::Sum<T>(local)
            );
            T globalSum = 0;
            MPI_Datatype type = get_mpi_datatype<T>(local);
            MPI_Allreduce(&local, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
            return std::pow(globalSum, 1.0 / p);
        }
        }
    }


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
