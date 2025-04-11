//
// File BareFieldOperations
//   Norms and a scalar product for fields
//

namespace ippl {
    /*!
     * Computes the inner product of two fields
     * @param f1 first field
     * @param f2 second field
     * @return Result of f1^T f2
     */
    template <typename BareField>
    typename BareField::value_type innerProduct(const BareField& f1, const BareField& f2) {
        using T                = typename BareField::value_type;
        constexpr unsigned Dim = BareField::dim;

        T sum                  = 0;
        auto layout            = f1.getLayout();
        auto view1             = f1.getView();
        auto view2             = f2.getView();
        using exec_space       = typename BareField::execution_space;
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        ippl::parallel_reduce(
            "Field::innerProduct(Field&, Field&)", f1.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                val += apply(view1, args) * apply(view2, args);
            },
            Kokkos::Sum<T>(sum));
        T globalSum = 0;
        layout.comm.allreduce(sum, globalSum, 1, std::plus<T>());
        return globalSum;
    }

    /*!
     * Computes the Lp-norm of a field
     * @param field field
     * @param p desired norm (default 2)
     * @return The desired norm of the field
     */
    template <typename BareField>
    typename BareField::value_type norm(const BareField& field, int p = 2) {
        using T                = typename BareField::value_type;
        constexpr unsigned Dim = BareField::dim;

        T local                = 0;
        auto layout            = field.getLayout();
        auto view              = field.getView();
        using exec_space       = typename BareField::execution_space;
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        switch (p) {
            case 0: {
                ippl::parallel_reduce(
                    "Field::norm(0)", field.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                        T myVal = Kokkos::abs(apply(view, args));
                        if (myVal > val)
                            val = myVal;
                    },
                    Kokkos::Max<T>(local));
                T globalMax = 0;
                layout.comm.allreduce(local, globalMax, 1, std::greater<T>());
                return globalMax;
            }
            default: {
                ippl::parallel_reduce(
                    "Field::norm(int) general", field.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                        val += std::pow(Kokkos::abs(apply(view, args)), p);
                    },
                    Kokkos::Sum<T>(local));
                T globalSum = 0;
                layout.comm.allreduce(local, globalSum, 1, std::plus<T>());
                return std::pow(globalSum, 1.0 / p);
            }
        }
    }
}  // namespace ippl
