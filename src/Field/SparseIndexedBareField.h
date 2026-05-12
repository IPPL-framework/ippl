//
// Class SparseIndexedBareField
//   Sparse point view into a BareField.
//
#ifndef IPPL_SPARSE_INDEXED_BARE_FIELD_H
#define IPPL_SPARSE_INDEXED_BARE_FIELD_H

#include <Kokkos_Core.hpp>

#include "Expression/IpplExpressions.h"
#include "Index/SIndex.h"
#include "Utility/ParallelDispatch.h"

namespace ippl {

    template <typename Field>
    class SparseIndexedBareField
        : public detail::Expression<SparseIndexedBareField<Field>,
                                    sizeof(Field) + sizeof(SIndex<Field::dim>)> {
    public:
        using field_type      = Field;
        using value_type      = typename field_type::value_type;
        using view_type       = typename field_type::view_type;
        using execution_space = typename field_type::execution_space;
        using sindex_type     = SIndex<field_type::dim>;
        using point_type      = typename sindex_type::point_type;
        using points_view_type = Kokkos::View<point_type*, execution_space>;

        constexpr static unsigned dim = 1;
        constexpr static unsigned field_dim = field_type::dim;
        constexpr static size_t expression_size = sizeof(Field) + sizeof(sindex_type);

        SparseIndexedBareField(field_type& field, const sindex_type& sindex)
            : view_m(field.getView())
            , owned_m(field.getOwned())
            , points_m(sindex.template getDevicePoints<execution_space>())
            , nghost_m(field.getNghost()) {}

        SparseIndexedBareField(const SparseIndexedBareField&) = default;

        template <typename Idx>
        KOKKOS_INLINE_FUNCTION value_type operator()(Idx i) const {
            return apply(view_m, pointToView(points_m(i)));
        }

        SparseIndexedBareField& operator=(value_type value) {
            assign(ValueAssign{value});
            return *this;
        }

        template <typename E, size_t N>
        SparseIndexedBareField& operator=(const detail::Expression<E, N>& expr) {
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            assign(ExpressionAssign<E, N>{expr_});
            return *this;
        }

        SparseIndexedBareField& operator=(const SparseIndexedBareField& rhs) {
            const detail::Expression<SparseIndexedBareField, expression_size>& expr = rhs;
            return operator=(expr);
        }

        SparseIndexedBareField& operator+=(value_type value) {
            update(PlusValueAssign{value});
            return *this;
        }

        SparseIndexedBareField& operator-=(value_type value) {
            update(MinusValueAssign{value});
            return *this;
        }

        SparseIndexedBareField& operator*=(value_type value) {
            update(MultiplyValueAssign{value});
            return *this;
        }

        SparseIndexedBareField& operator/=(value_type value) {
            update(DivideValueAssign{value});
            return *this;
        }

        template <typename E, size_t N>
        SparseIndexedBareField& operator+=(const detail::Expression<E, N>& expr) {
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            update(PlusExpressionAssign<E, N>{expr_});
            return *this;
        }

        template <typename E, size_t N>
        SparseIndexedBareField& operator-=(const detail::Expression<E, N>& expr) {
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            update(MinusExpressionAssign<E, N>{expr_});
            return *this;
        }

        template <typename E, size_t N>
        SparseIndexedBareField& operator*=(const detail::Expression<E, N>& expr) {
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            update(MultiplyExpressionAssign<E, N>{expr_});
            return *this;
        }

        template <typename E, size_t N>
        SparseIndexedBareField& operator/=(const detail::Expression<E, N>& expr) {
            using capture_type = detail::CapturedExpression<E, N>;
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            update(DivideExpressionAssign<E, N>{expr_});
            return *this;
        }

    private:
        view_type view_m;
        NDIndex<field_dim> owned_m;
        points_view_type points_m;
        int nghost_m;

        struct ValueAssign {
            value_type value;

            template <typename Coords>
            KOKKOS_INLINE_FUNCTION value_type operator()(const Coords&) const {
                return value;
            }
        };

        struct PlusValueAssign {
            value_type value;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords&) const {
                return current + value;
            }
        };

        struct MinusValueAssign {
            value_type value;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords&) const {
                return current - value;
            }
        };

        struct MultiplyValueAssign {
            value_type value;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords&) const {
                return current * value;
            }
        };

        struct DivideValueAssign {
            value_type value;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords&) const {
                return current / value;
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

        template <typename E, size_t N>
        struct PlusExpressionAssign {
            detail::CapturedExpression<E, N> expr;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords& rel) const {
                return current + apply(expr, rel);
            }
        };

        template <typename E, size_t N>
        struct MinusExpressionAssign {
            detail::CapturedExpression<E, N> expr;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords& rel) const {
                return current - apply(expr, rel);
            }
        };

        template <typename E, size_t N>
        struct MultiplyExpressionAssign {
            detail::CapturedExpression<E, N> expr;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords& rel) const {
                return current * apply(expr, rel);
            }
        };

        template <typename E, size_t N>
        struct DivideExpressionAssign {
            detail::CapturedExpression<E, N> expr;

            template <typename Current, typename Coords>
            KOKKOS_INLINE_FUNCTION auto operator()(const Current& current, const Coords& rel) const {
                return current / apply(expr, rel);
            }
        };

        KOKKOS_INLINE_FUNCTION bool isLocal(const point_type& point) const {
            for (unsigned d = 0; d < field_dim; ++d) {
                if (!owned_m[d].contains(Index(point[d], point[d]))) {
                    return false;
                }
            }
            return true;
        }

        KOKKOS_INLINE_FUNCTION auto pointToView(const point_type& point) const {
            typename RangePolicy<field_dim, execution_space>::index_array_type viewCoords;
            for (unsigned d = 0; d < field_dim; ++d) {
                viewCoords[d] = (point[d] - owned_m[d].first()) / owned_m[d].stride() + nghost_m;
            }
            return viewCoords;
        }

        template <typename Functor>
        void assign(const Functor& functor) {
            ippl::parallel_for(
                "SparseIndexedBareField::operator=", getRangePolicy(points_m),
                KOKKOS_CLASS_LAMBDA(
                    const typename RangePolicy<1, execution_space>::index_array_type& args) {
                    const auto i = args[0];
                    const point_type point = points_m(i);
                    if (isLocal(point)) {
                        typename RangePolicy<1, execution_space>::index_array_type rel;
                        rel[0] = i;
                        apply(view_m, pointToView(point)) = functor(rel);
                    }
                });
        }

        template <typename Functor>
        void update(const Functor& functor) {
            ippl::parallel_for(
                "SparseIndexedBareField::operator op=", getRangePolicy(points_m),
                KOKKOS_CLASS_LAMBDA(
                    const typename RangePolicy<1, execution_space>::index_array_type& args) {
                    const auto i = args[0];
                    const point_type point = points_m(i);
                    if (isLocal(point)) {
                        typename RangePolicy<1, execution_space>::index_array_type rel;
                        rel[0] = i;
                        const auto viewCoords = pointToView(point);
                        apply(view_m, viewCoords) = functor(apply(view_m, viewCoords), rel);
                    }
                });
        }
    };

}  // namespace ippl

#endif
