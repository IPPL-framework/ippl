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

    namespace detail {

        #define DefineBaseFunctor(name)                                 \
        template <typename T, unsigned Dim>                             \
        class name : public name##Base<T, Dim> {};

        #define DefineBinaryFunctorBase(name, op)                       \
        template <typename T, unsigned Dim, typename ... Indices>       \
        class name##Base {                                              \
            using field = const Field<T, Dim>&;                         \
            using view = typename detail::ViewType<T, Dim>::view_type;  \
            view v1, v2;                                                \
                                                                        \
        public:                                                         \
            name##Base(field f1, field f2) {                            \
                v1 = f1.getView();                                      \
                v2 = f2.getView();                                      \
            }                                                           \
                                                                        \
            KOKKOS_IMPL_FUNCTION                                        \
            void operator()(Indices ... i, T& val) const {              \
                op                                                      \
            }                                                           \
        };                                                              \
        DefineBaseFunctor(name)

        #define DefineUnaryFunctorBase(name, op)                        \
        template <typename T, unsigned Dim, typename ... Indices>       \
        class name##Base {                                              \
            using field = const Field<T, Dim>&;                         \
            using view = typename detail::ViewType<T, Dim>::view_type;  \
            const int p;                                                \
            view v;                                                     \
                                                                        \
        public:                                                         \
            name##Base(field f, int p = 0) : p(p) {                     \
                v = f.getView();                                        \
            }                                                           \
                                                                        \
            KOKKOS_IMPL_FUNCTION                                        \
            void operator()(Indices ... i, T& val) const {              \
                op                                                      \
            }                                                           \
        };                                                              \
        DefineBaseFunctor(name)

        #define DefineFunctor(name, dim, indices...)                    \
        template <typename T>                                           \
        class name<T, dim> : public name##Base<T, dim, indices> {       \
        public:                                                         \
            using Base = name##Base<T, dim, indices>;                   \
            using Base::name##Base;                                     \
            using Base::operator();                                     \
        };

        #define Define3DFunctor(name) DefineFunctor(name, 3, const size_t, const size_t, const size_t)
        #define Define2DFunctor(name) DefineFunctor(name, 2, const size_t, const size_t)

        DefineBinaryFunctorBase(InnerProductFunctor, val += v1(i...) * v2(i...);)
        Define2DFunctor(InnerProductFunctor)
        Define3DFunctor(InnerProductFunctor)

        DefineUnaryFunctorBase(InfNormFunctor,
            T myVal = std::abs(v(i...));
            if (myVal > val)
                val = myVal;
        )
        Define2DFunctor(InfNormFunctor)
        Define3DFunctor(InfNormFunctor)

        DefineUnaryFunctorBase(LpNormFunctor, val += std::pow(std::abs(v(i...)), p);)
        Define2DFunctor(LpNormFunctor)
        Define3DFunctor(LpNormFunctor)

    }

    /*!
     * Computes the inner product of two fields
     * @param f1 first field
     * @param f2 second field
     * @return Result of f1^T f2
     */
    template <typename T, unsigned Dim>
    T innerProduct(const Field<T, Dim>& f1, const Field<T, Dim>& f2) {
        T sum = 0;
        detail::InnerProductFunctorBase functor = detail::InnerProductFunctor<T, Dim>(f1, f2);
        Kokkos::parallel_reduce("Field::innerProduct(Field&, Field&)", f1.getRangePolicy(),
            functor,
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
            detail::InfNormFunctorBase functor = detail::InfNormFunctor<T, Dim>(field);
            Kokkos::parallel_reduce("Field::norm(0)", field.getRangePolicy(),
                functor,
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
            detail::LpNormFunctorBase functor = detail::LpNormFunctor<T, Dim>(field, p);
            Kokkos::parallel_reduce("Field::norm(int) general", field.getRangePolicy(),
                functor,
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
