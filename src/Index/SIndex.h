//
// Class SIndex
//   Sparse set of field index points.
//
#ifndef IPPL_SINDEX_H
#define IPPL_SINDEX_H

#include <Kokkos_Core.hpp>

#include <vector>

#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"
#include "Types/Vector.h"

namespace ippl {

    template <unsigned Dim>
    class SIndex {
    public:
        using point_type = Vector<int, Dim>;

        SIndex() = default;

        explicit SIndex(FieldLayout<Dim>& layout)
            : layout_m(&layout)
            , domain_m(layout.getDomain()) {}

        void initialize(FieldLayout<Dim>& layout) {
            layout_m = &layout;
            domain_m = layout.getDomain();
            points_m.clear();
        }

        bool needInitialize() const { return layout_m == nullptr; }

        bool addIndex(const point_type& point) {
            if (!contains(domain_m, point) || hasIndex(point)) {
                return false;
            }
            points_m.push_back(point);
            return true;
        }

        bool addIndex(const NDIndex<Dim>& point) {
            point_type values;
            for (unsigned d = 0; d < Dim; ++d) {
                if (point[d].length() != 1) {
                    return false;
                }
                values[d] = point[d].first();
            }
            return addIndex(values);
        }

        void clear() { points_m.clear(); }

        std::size_t size() const { return points_m.size(); }

        bool empty() const { return points_m.empty(); }

        const std::vector<point_type>& points() const { return points_m; }

        const NDIndex<Dim>& getDomain() const { return domain_m; }

        FieldLayout<Dim>& getFieldLayout() const { return *layout_m; }

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
            point_type values;
            for (unsigned d = 0; d < Dim; ++d) {
                if (point[d].length() != 1) {
                    return false;
                }
                values[d] = point[d].first();
            }
            return hasIndex(values);
        }

        template <typename ExecutionSpace>
        Kokkos::View<point_type*, ExecutionSpace> getDevicePoints() const {
            Kokkos::View<point_type*, ExecutionSpace> devicePoints("SIndex::points", points_m.size());
            auto hostPoints = Kokkos::create_mirror_view(devicePoints);
            for (std::size_t i = 0; i < points_m.size(); ++i) {
                hostPoints(i) = points_m[i];
            }
            Kokkos::deep_copy(devicePoints, hostPoints);
            return devicePoints;
        }

    private:
        FieldLayout<Dim>* layout_m = nullptr;
        NDIndex<Dim> domain_m;
        std::vector<point_type> points_m;

        static bool contains(const NDIndex<Dim>& domain, const point_type& point) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (!domain[d].contains(Index(point[d], point[d]))) {
                    return false;
                }
            }
            return true;
        }
    };

}  // namespace ippl

#endif
