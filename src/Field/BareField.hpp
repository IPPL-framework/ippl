//
// Class BareField
//   A BareField consists of multple LFields and represents a field.
//
#include "Ippl.h"

#include <cstdlib>
#include <map>
#include <utility>

#include "Communicate/DataTypes.h"

#include "Utility/Inform.h"
#include "Utility/IpplInfo.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class... ViewArgs>
        struct isExpression<BareField<T, Dim, ViewArgs...>> : std::true_type {};
    }  // namespace detail

    template <typename T, unsigned Dim, class... ViewArgs>
    BareField<T, Dim, ViewArgs...>::BareField()
        : nghost_m(1)
        , layout_m(nullptr) {}

    template <typename T, unsigned Dim, class... ViewArgs>
    BareField<T, Dim, ViewArgs...> BareField<T, Dim, ViewArgs...>::deepCopy() const {
        BareField<T, Dim, ViewArgs...> copy(*layout_m, nghost_m);
        Kokkos::deep_copy(copy.dview_m, dview_m);
        return copy;
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    BareField<T, Dim, ViewArgs...>::BareField(Layout_t& l, int nghost)
        : nghost_m(nghost)
        //     , owned_m(0)
        , layout_m(&l) {
        setup();
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::initialize(Layout_t& l, int nghost) {
        if (layout_m == 0) {
            layout_m = &l;
            nghost_m = nghost;
            setup();
        }
    }

    // ML
    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::updateLayout(Layout_t& l, int nghost) {
        // std::cout << "Got in BareField::updateLayout()" << std::endl;
        layout_m = &l;
        nghost_m = nghost;
        setup();
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::setup() {
        owned_m = layout_m->getLocalNDIndex();

        auto resize = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            this->resize((owned_m[Idx].length() + 2 * nghost_m)...);
        };
        resize(std::make_index_sequence<Dim>{});
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    template <typename... Args>
    void BareField<T, Dim, ViewArgs...>::resize(Args... args) {
        Kokkos::resize(dview_m, args...);
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::fillHalo() {
        if (layout_m->comm.size() > 1) {
            halo_m.fillHalo(dview_m, layout_m);
        }
        if (layout_m->isAllPeriodic_m) {
            using Op = typename detail::HaloCells<T, Dim, ViewArgs...>::assign;
            halo_m.template applyPeriodicSerialDim<Op>(dview_m, layout_m, nghost_m);
        }
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::accumulateHalo() {
        if (layout_m->comm.size() > 1) {
            halo_m.accumulateHalo(dview_m, layout_m);
        }
        if (layout_m->isAllPeriodic_m) {
            using Op = typename detail::HaloCells<T, Dim, ViewArgs...>::rhs_plus_assign;
            halo_m.template applyPeriodicSerialDim<Op>(dview_m, layout_m, nghost_m);
        }
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    BareField<T, Dim, ViewArgs...>& BareField<T, Dim, ViewArgs...>::operator=(T x) {
        using index_array_type = typename RangePolicy<Dim, execution_space>::index_array_type;
        ippl::parallel_for(
            "BareField::operator=(T)", getRangePolicy(dview_m),
            KOKKOS_CLASS_LAMBDA(const index_array_type& args) { apply(dview_m, args) = x; });
        return *this;
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    template <typename E, size_t N>
    BareField<T, Dim, ViewArgs...>& BareField<T, Dim, ViewArgs...>::operator=(
        const detail::Expression<E, N>& expr) {
        using capture_type     = detail::CapturedExpression<E, N>;
        capture_type expr_     = reinterpret_cast<const capture_type&>(expr);
        using index_array_type = typename RangePolicy<Dim, execution_space>::index_array_type;
        ippl::parallel_for(
            "BareField::operator=(const Expression&)", getRangePolicy(dview_m, nghost_m),
            KOKKOS_CLASS_LAMBDA(const index_array_type& args) {
                apply(dview_m, args) = apply(expr_, args);
            });
        return *this;
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::write(std::ostream& out) const {
        Kokkos::fence();
        detail::write<T, Dim>(dview_m, out);
    }

    template <typename T, unsigned Dim, class... ViewArgs>
    void BareField<T, Dim, ViewArgs...>::write(Inform& inf) const {
        write(inf.getDestination());
    }

#define DefineReduction(fun, name, op, MPI_Op)                                                 \
    template <typename T, unsigned Dim, class... ViewArgs>                                     \
    T BareField<T, Dim, ViewArgs...>::name(int nghost) const {                                 \
        PAssert_LE(nghost, nghost_m);                                                          \
        T temp                 = 0.0;                                                          \
        using index_array_type = typename RangePolicy<Dim, execution_space>::index_array_type; \
        ippl::parallel_reduce(                                                                 \
            "fun", getRangePolicy(dview_m, nghost_m - nghost),                                 \
            KOKKOS_CLASS_LAMBDA(const index_array_type& args, T& valL) {                       \
                T myVal = apply(dview_m, args);                                                \
                op;                                                                            \
            },                                                                                 \
            Kokkos::fun<T>(temp));                                                             \
        T globaltemp = 0.0;                                                                    \
        layout_m->comm.allreduce(temp, globaltemp, 1, MPI_Op<T>());                            \
        return globaltemp;                                                                     \
    }

    DefineReduction(Sum, sum, valL += myVal, std::plus)
    DefineReduction(Max, max, if (myVal > valL) valL = myVal, std::greater)
    DefineReduction(Min, min, if (myVal < valL) valL = myVal, std::less)
    DefineReduction(Prod, prod, valL *= myVal, std::multiplies)

}  // namespace ippl
