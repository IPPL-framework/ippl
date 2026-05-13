//
// Class SIndex
//   Sparse set of field index points.
//
#ifndef IPPL_SINDEX_H
#define IPPL_SINDEX_H

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "FieldLayout/FieldLayout.h"
#include "Expression/IpplExpressions.h"
#include "Index/NDIndex.h"
#include "Index/SOffset.h"
#include "Types/Vector.h"
#include "Utility/ParallelDispatch.h"

namespace ippl {

    template <unsigned Dim>
    class IndexedSIndex;

    template <unsigned Dim>
    class SIndex {
    public:
        using point_type = Vector<int, Dim>;

        SIndex() = default;

        explicit SIndex(FieldLayout<Dim>& layout)
            : layout_m(&layout)
            , domain_m(layout.getDomain()) {}

        SIndex(const SIndex&) = default;

        SIndex(const SIndex& sindex, const SOffset<Dim>& offset)
            : layout_m(sindex.layout_m)
            , domain_m(sindex.domain_m)
            , points_m(sindex.points_m)
            , offset_m(sindex.offset_m + offset) {}

        void initialize(FieldLayout<Dim>& layout) {
            layout_m = &layout;
            domain_m = layout.getDomain();
            points_m.clear();
        }

        bool needInitialize() const { return layout_m == nullptr; }

        SIndex& operator=(const SIndex&) = default;

        SIndex& operator=(const SOffset<Dim>& offset) {
            clear();
            offset_m = SOffset<Dim>();
            addIndex(pointFromOffset(offset));
            return *this;
        }

        SIndex& operator=(const NDIndex<Dim>& domain) {
            clear();
            addIndex(domain);
            return *this;
        }

        template <typename E, size_t N>
        SIndex& operator=(const detail::Expression<E, N>& expr) {
            assignExpression(expr, domain_m, EvaluationMode::ViewCoordinates);
            return *this;
        }

        SIndex& operator&=(const SIndex& rhs) {
            eraseIf([&](const point_type& point) { return !rhs.hasIndex(point); });
            domain_m = domain_m.intersect(rhs.domain_m);
            return *this;
        }

        SIndex& operator&=(const SOffset<Dim>& offset) {
            const point_type point = pointFromOffset(offset);
            eraseIf([&](const point_type& existing) { return !samePoint(existing, point); });
            return *this;
        }

        SIndex& operator&=(const NDIndex<Dim>& domain) {
            eraseIf([&](const point_type& point) { return !contains(domain, effectivePoint(point)); });
            domain_m = domain_m.intersect(domain);
            return *this;
        }

        SIndex& operator|=(const SIndex& rhs) {
            domain_m = boundingUnion(domain_m, rhs.domain_m);
            for (const auto& point : rhs.points_m) {
                addIndex(point);
            }
            return *this;
        }

        SIndex& operator|=(const SOffset<Dim>& offset) {
            domain_m = boundingUnion(domain_m, singlePointDomain(pointFromOffset(offset)));
            addIndex(pointFromOffset(offset));
            return *this;
        }

        SIndex& operator|=(const NDIndex<Dim>& domain) {
            domain_m = boundingUnion(domain_m, domain);
            addIndex(domain);
            return *this;
        }

        bool addIndex(const point_type& point) {
            if (!contains(domain_m, point) || hasIndex(point)) {
                return false;
            }
            points_m.push_back(point);
            return true;
        }

        bool addIndex(const NDIndex<Dim>& point) {
            if (point.empty()) {
                return false;
            }

            bool addedAny = false;
            point_type values;
            addIndex(point, values, 0, addedAny);
            return addedAny;
        }

        bool removeIndex(const point_type& point) {
            const auto originalSize = points_m.size();
            eraseIf([&](const point_type& existing) { return samePoint(existing, point); });
            return points_m.size() != originalSize;
        }

        bool removeIndex(const SOffset<Dim>& offset) {
            return removeIndex(pointFromOffset(offset));
        }

        bool removeIndex(const NDIndex<Dim>& point) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (point[d].length() != 1) {
                    return false;
                }
            }
            return removeIndex(pointFromNDIndex(point));
        }

        void clear() { points_m.clear(); }

        std::size_t size() const { return points_m.size(); }

        bool empty() const { return points_m.empty(); }

        const std::vector<point_type>& points() const { return points_m; }

        const NDIndex<Dim>& getDomain() const { return domain_m; }

        void setDomain(const NDIndex<Dim>& domain) { domain_m = domain; }

        FieldLayout<Dim>& getFieldLayout() const { return *layout_m; }

        SOffset<Dim>& getOffset() { return offset_m; }

        const SOffset<Dim>& getOffset() const { return offset_m; }

        bool hasIndex(const point_type& point) const {
            for (const auto& existing : points_m) {
                bool same = true;
                for (unsigned d = 0; d < Dim; ++d) {
                    same = same && existing[d] == point[d];
                }
                if (same) {
                    return true;
                }
            }
            return false;
        }

        bool hasIndex(const NDIndex<Dim>& point) const {
            for (unsigned d = 0; d < Dim; ++d) {
                if (point[d].length() != 1) {
                    return false;
                }
            }
            return hasIndex(pointFromNDIndex(point));
        }

        SIndex operator()(const SOffset<Dim>& offset) const { return SIndex(*this, offset); }

        IndexedSIndex<Dim> operator[](const NDIndex<Dim>& domain) {
            return IndexedSIndex<Dim>(*this, domain);
        }

        IndexedSIndex<Dim> operator[](const Index& index) {
            NDIndex<Dim> domain = domain_m;
            domain[0] = index;
            return operator[](domain);
        }

        SIndex operator()(int i0) const {
            static_assert(Dim == 1);
            return operator()(SOffset<Dim>(i0));
        }

        SIndex operator()(int i0, int i1) const {
            static_assert(Dim == 2);
            return operator()(SOffset<Dim>(i0, i1));
        }

        SIndex operator()(int i0, int i1, int i2) const {
            static_assert(Dim == 3);
            return operator()(SOffset<Dim>(i0, i1, i2));
        }

        SIndex operator()(int i0, int i1, int i2, int i3) const {
            static_assert(Dim == 4);
            return operator()(SOffset<Dim>(i0, i1, i2, i3));
        }

        SIndex operator()(int i0, int i1, int i2, int i3, int i4) const {
            static_assert(Dim == 5);
            return operator()(SOffset<Dim>(i0, i1, i2, i3, i4));
        }

        SIndex operator()(int i0, int i1, int i2, int i3, int i4, int i5) const {
            static_assert(Dim == 6);
            return operator()(SOffset<Dim>(i0, i1, i2, i3, i4, i5));
        }

        template <typename ExecutionSpace>
        Kokkos::View<point_type*, ExecutionSpace> getDevicePoints() const {
            Kokkos::View<point_type*, ExecutionSpace> devicePoints("SIndex::points", points_m.size());
            auto hostPoints = Kokkos::create_mirror_view(devicePoints);
            for (std::size_t i = 0; i < points_m.size(); ++i) {
                hostPoints(i) = effectivePoint(points_m[i]);
            }
            Kokkos::deep_copy(devicePoints, hostPoints);
            return devicePoints;
        }

        friend SIndex operator+(const SIndex& sindex, const SOffset<Dim>& offset) {
            return SIndex(sindex, offset);
        }

        friend SIndex operator+(const SOffset<Dim>& offset, const SIndex& sindex) {
            return SIndex(sindex, offset);
        }

        friend SIndex operator-(const SIndex& sindex, const SOffset<Dim>& offset) {
            return SIndex(sindex, -offset);
        }

    private:
        FieldLayout<Dim>* layout_m = nullptr;
        NDIndex<Dim> domain_m;
        std::vector<point_type> points_m;
        SOffset<Dim> offset_m;

        enum class EvaluationMode { ViewCoordinates, RelativeCoordinates };

        static bool contains(const NDIndex<Dim>& domain, const point_type& point) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (!domain[d].contains(Index(point[d], point[d]))) {
                    return false;
                }
            }
            return true;
        }

        static bool samePoint(const point_type& lhs, const point_type& rhs) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (lhs[d] != rhs[d]) {
                    return false;
                }
            }
            return true;
        }

        static point_type pointFromOffset(const SOffset<Dim>& offset) {
            point_type point;
            for (unsigned d = 0; d < Dim; ++d) {
                point[d] = offset[d];
            }
            return point;
        }

        static point_type pointFromNDIndex(const NDIndex<Dim>& domain) {
            point_type point;
            for (unsigned d = 0; d < Dim; ++d) {
                point[d] = domain[d].first();
            }
            return point;
        }

        point_type effectivePoint(const point_type& point) const {
            point_type shifted = point;
            for (unsigned d = 0; d < Dim; ++d) {
                shifted[d] += offset_m[d];
            }
            return shifted;
        }

        template <typename Predicate>
        void eraseIf(const Predicate& predicate) {
            auto out = points_m.begin();
            for (auto in = points_m.begin(); in != points_m.end(); ++in) {
                if (!predicate(*in)) {
                    *out = *in;
                    ++out;
                }
            }
            points_m.erase(out, points_m.end());
        }

        void addIndex(const NDIndex<Dim>& domain, point_type& values, unsigned d, bool& addedAny) {
            if (d == Dim) {
                addedAny = addIndex(values) || addedAny;
                return;
            }

            for (int value = domain[d].first(); value <= domain[d].last();
                 value += domain[d].stride()) {
                values[d] = value;
                addIndex(domain, values, d + 1, addedAny);
            }
        }

        static NDIndex<Dim> boundingUnion(const NDIndex<Dim>& lhs, const NDIndex<Dim>& rhs) {
            NDIndex<Dim> result;
            for (unsigned d = 0; d < Dim; ++d) {
                result[d] = Index(std::min(lhs[d].first(), rhs[d].first()),
                                  std::max(lhs[d].last(), rhs[d].last()), lhs[d].stride());
            }
            return result;
        }

        static NDIndex<Dim> singlePointDomain(const point_type& point) {
            NDIndex<Dim> result;
            for (unsigned d = 0; d < Dim; ++d) {
                result[d] = Index(point[d], point[d]);
            }
            return result;
        }

        template <typename Coords>
        KOKKOS_INLINE_FUNCTION static typename RangePolicy<Dim, Kokkos::DefaultExecutionSpace>::index_type flatIndex(
            const Coords& coords, const Kokkos::Array<
                                      typename RangePolicy<Dim, Kokkos::DefaultExecutionSpace>::
                                          index_type,
                                      Dim>& extents) {
            typename RangePolicy<Dim, Kokkos::DefaultExecutionSpace>::index_type flat = 0;
            for (unsigned d = 0; d < Dim; ++d) {
                flat = flat * extents[d] + coords[d];
            }
            return flat;
        }

    public:
        template <typename E, size_t N>
        void assignExpression(const detail::Expression<E, N>& expr, const NDIndex<Dim>& domain,
                              EvaluationMode mode) {
            clear();
            setDomain(domain);

            NDIndex<Dim> local = domain.intersect(getFieldLayout().getLocalNDIndex());
            if (local.empty()) {
                return;
            }

            using execution_space   = Kokkos::DefaultExecutionSpace;
            using range_policy_type = RangePolicy<Dim, execution_space>;
            using index_type        = typename range_policy_type::index_type;
            using capture_type      = detail::CapturedExpression<E, N>;

            Kokkos::Array<index_type, Dim> begin, end, extents;
            index_type size = 1;
            for (unsigned d = 0; d < Dim; ++d) {
                begin[d]   = 0;
                end[d]     = local[d].length();
                extents[d] = local[d].length();
                size *= extents[d];
            }

            Kokkos::View<int*, execution_space> selected("SIndex::selected", size);
            capture_type expr_ = reinterpret_cast<const capture_type&>(expr);
            const int nghost   = 1;
            auto localCopy     = local;
            auto domainCopy    = domain;

            ippl::parallel_for(
                "SIndex::operator=", createRangePolicy<Dim, execution_space>(begin, end),
                KOKKOS_LAMBDA(const typename range_policy_type::index_array_type& args) {
                    typename range_policy_type::index_array_type exprCoords;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const auto global = localCopy[d].first() + args[d] * localCopy[d].stride();
                        if (mode == EvaluationMode::ViewCoordinates) {
                            exprCoords[d] =
                                (global - localCopy[d].first()) / localCopy[d].stride() + nghost;
                        } else {
                            exprCoords[d] =
                                (global - domainCopy[d].first()) / domainCopy[d].stride();
                        }
                    }
                    selected(flatIndex(args, extents)) = apply(expr_, exprCoords) ? 1 : 0;
                });

            auto hostSelected = Kokkos::create_mirror_view(selected);
            Kokkos::deep_copy(hostSelected, selected);

            point_type global;
            addSelected(local, extents, hostSelected, global, 0, 0);
        }

        template <typename HostView>
        void addSelected(const NDIndex<Dim>& local, const Kokkos::Array<
                                                       typename RangePolicy<
                                                           Dim, Kokkos::DefaultExecutionSpace>::
                                                           index_type,
                                                       Dim>& extents,
                         const HostView& selected, point_type& global, unsigned d,
                         typename RangePolicy<Dim, Kokkos::DefaultExecutionSpace>::index_type flat) {
            if (d == Dim) {
                if (selected(flat)) {
                    addIndex(global);
                }
                return;
            }

            for (typename RangePolicy<Dim, Kokkos::DefaultExecutionSpace>::index_type i = 0;
                 i < extents[d]; ++i) {
                global[d] = local[d].first() + static_cast<int>(i) * local[d].stride();
                addSelected(local, extents, selected, global, d + 1, flat * extents[d] + i);
            }
        }

        friend class IndexedSIndex<Dim>;
    };

    template <unsigned Dim>
    class IndexedSIndex {
    public:
        IndexedSIndex(SIndex<Dim>& sindex, const NDIndex<Dim>& domain)
            : sindex_m(sindex)
            , domain_m(domain) {}

        template <typename E, size_t N>
        IndexedSIndex& operator=(const detail::Expression<E, N>& expr) {
            sindex_m.assignExpression(expr, domain_m, SIndex<Dim>::EvaluationMode::RelativeCoordinates);
            return *this;
        }

    private:
        SIndex<Dim>& sindex_m;
        NDIndex<Dim> domain_m;
    };

    template <typename E, size_t N, typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
    auto gt(const detail::Expression<E, N>& expr, const T& value) {
        return expr > value;
    }

}  // namespace ippl

#endif
