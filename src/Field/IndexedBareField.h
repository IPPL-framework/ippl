//
// Class IndexedBareField
//   Lightweight indexed view into a BareField.
//
#ifndef IPPL_INDEXED_BARE_FIELD_H
#define IPPL_INDEXED_BARE_FIELD_H

#include <Kokkos_Core.hpp>

#include "Expression/IpplExpressions.h"
#include "Index/NDIndex.h"
#include "Utility/ParallelDispatch.h"

namespace ippl {

    template <typename Field, unsigned Brackets = Field::dim>
    class IndexedBareField
        : public detail::Expression<IndexedBareField<Field, Brackets>,
                                    sizeof(Field) + sizeof(NDIndex<Field::dim>)> {
    public:
        using field_type      = Field;
        using value_type      = typename field_type::value_type;
        using view_type       = typename field_type::view_type;
        using execution_space = typename field_type::execution_space;
        using Domain_t        = NDIndex<field_type::dim>;

        constexpr static unsigned dim = field_type::dim;
        constexpr static size_t expression_size = sizeof(Field) + sizeof(Domain_t);

        IndexedBareField(field_type& field, const Domain_t& domain)
            : view_m(field.getView())
            , owned_m(field.getOwned())
            , domain_m(domain)
            , nghost_m(field.getNghost()) {}

        IndexedBareField(const IndexedBareField&) = default;

        IndexedBareField(const view_type& view, const Domain_t& owned, const Domain_t& domain,
                         int nghost)
            : view_m(view)
            , owned_m(owned)
            , domain_m(domain)
            , nghost_m(nghost) {}

        auto operator[](const Index& index) const {
            static_assert(Brackets < dim, "Too many Index brackets for field dimension.");
            Domain_t domain = domain_m;
            domain[Brackets] = index;
            return IndexedBareField<Field, Brackets + 1>(view_m, owned_m, domain, nghost_m);
        }

        auto operator[](int index) const {
            return operator[](Index(index, index));
        }

        template <typename... Args>
        KOKKOS_INLINE_FUNCTION value_type operator()(Args... args) const {
            static_assert(Brackets == dim, "IndexedBareField expression requires all dimensions.");
            static_assert(sizeof...(Args) == dim);
            typename RangePolicy<dim, execution_space>::index_array_type rel{args...};
            return apply(view_m, relativeToView(rel));
        }

        IndexedBareField& operator=(value_type value) {
            static_assert(Brackets == dim, "IndexedBareField assignment requires all dimensions.");
            assign(ValueAssign{value});
            return *this;
        }

        template <typename E, size_t N>
        IndexedBareField& operator=(const detail::Expression<E, N>& expr) {
            static_assert(Brackets == dim, "IndexedBareField assignment requires all dimensions.");
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            assign(ExpressionAssign<E, N>{expr_});
            return *this;
        }

        IndexedBareField& operator=(const IndexedBareField& rhs) {
            static_assert(Brackets == dim, "IndexedBareField assignment requires all dimensions.");
            const detail::Expression<IndexedBareField<Field, Brackets>, expression_size>& expr = rhs;
            return operator=(expr);
        }

        const Domain_t& getDomain() const { return domain_m; }

    private:
        view_type view_m;
        Domain_t owned_m;
        Domain_t domain_m;
        int nghost_m;

    public:
        struct ValueAssign {
            value_type value;

            template <typename Coords>
            KOKKOS_INLINE_FUNCTION value_type operator()(const Coords&) const {
                return value;
            }
        };

        template <typename E, size_t N>
        struct ExpressionAssign {
            detail::CapturedExpression<E, N> expr;

            template <typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Coords& rel) const {
                return apply(expr, rel);
            }
        };

    private:
        template <typename Coords>
        KOKKOS_INLINE_FUNCTION auto relativeToView(const Coords& rel) const {
            typename RangePolicy<dim, execution_space>::index_array_type viewCoords;
            for (unsigned d = 0; d < dim; ++d) {
                const auto global = domain_m[d].first() + rel[d] * domain_m[d].stride();
                viewCoords[d] = (global - owned_m[d].first()) / owned_m[d].stride() + nghost_m;
            }
            return viewCoords;
        }

        template <typename Coords>
        KOKKOS_INLINE_FUNCTION auto globalToRelative(const Coords& global) const {
            typename RangePolicy<dim, execution_space>::index_array_type rel;
            for (unsigned d = 0; d < dim; ++d) {
                rel[d] = (global[d] - domain_m[d].first()) / domain_m[d].stride();
            }
            return rel;
        }

        template <typename Coords>
        KOKKOS_INLINE_FUNCTION auto globalToView(const Coords& global) const {
            typename RangePolicy<dim, execution_space>::index_array_type viewCoords;
            for (unsigned d = 0; d < dim; ++d) {
                viewCoords[d] = (global[d] - owned_m[d].first()) / owned_m[d].stride() + nghost_m;
            }
            return viewCoords;
        }

    public:
        template <typename Functor>
        void assign(const Functor& functor) {
            Domain_t local = domain_m.intersect(owned_m);
            if (local.empty()) {
                return;
            }

            using range_policy_type = RangePolicy<dim, execution_space>;
            using index_type        = typename range_policy_type::index_type;
            Kokkos::Array<index_type, dim> begin, end;
            for (unsigned d = 0; d < dim; ++d) {
                begin[d] = 0;
                end[d]   = local[d].length();
            }

            ippl::parallel_for(
                "IndexedBareField::operator=", createRangePolicy<dim, execution_space>(begin, end),
                KOKKOS_CLASS_LAMBDA(const typename range_policy_type::index_array_type& args) {
                    typename range_policy_type::index_array_type global;
                    for (unsigned d = 0; d < dim; ++d) {
                        global[d] = local[d].first() + args[d] * local[d].stride();
                    }
                    const auto rel = globalToRelative(global);
                    apply(view_m, globalToView(global)) = functor(rel);
                });
        }
    };

}  // namespace ippl

#endif
