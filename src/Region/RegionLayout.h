//
// Class RegionLayout
//   RegionLayout stores a partitioned set of NDRegion objects, to represent
//   the parallel layout of an encompassing NDRegion.  It also contains
//   functions to find the subsets of the NDRegion partitions which intersect
//   or touch a given NDRegion.  It is similar to FieldLayout, with the
//   following changes:
//   1. It uses NDRegion instead of NDIndex, so it is templated on the position
//      data type (although it can be constructed with an NDIndex and a Mesh
//      as well);
//   2. It does not contain any consideration for guard cells;
//   3. It can store not only the partitioned domain, but periodic copies of
//      the partitioned domain for use by particle periodic boundary conditions
//   4. It also keeps a list of FieldLayoutUser's, so that it can notify them
//      when the internal FieldLayout here is reparitioned or otherwise changed.
//
//   If this is constructed with a FieldLayout, it stores a pointer to it
//   so that if we must repartition the copy of the FieldLayout that
//   is stored here, we will end up repartitioning all the registered Fields.
//
#ifndef IPPL_REGION_LAYOUT_H
#define IPPL_REGION_LAYOUT_H

#include <array>

#include "Types/ViewTypes.h"

#include "Utility/TypeUtils.h"

#include "Region/NDRegion.h"

namespace ippl {
    namespace detail {

        template <typename T, unsigned Dim, class Mesh, class... Properties>
        class RegionLayout {
            template <typename... Props>
            using base_type = RegionLayout<T, Dim, Mesh, Props...>;

        public:
            using NDRegion_t       = NDRegion<T, Dim>;
            using view_type        = typename ViewType<NDRegion_t, 1, Properties...>::view_type;
            using host_mirror_type = typename view_type::host_mirror_type;

            using uniform_type = typename CreateUniformType<base_type, view_type>::type;

            // Default constructor.  To make this class actually work, the user
            // will have to later call 'changeDomain' to set the proper Domain
            // and get a new partitioning.
            RegionLayout();

            // Constructor which takes a FieldLayout and a MeshType
            // This one compares the domain of the FieldLayout and the domain of
            // the MeshType to determine the centering of the index space.
            RegionLayout(const FieldLayout<Dim>&, const Mesh&);

            ~RegionLayout() = default;

            const NDRegion_t& getDomain() const { return region_m; }

            const view_type getdLocalRegions() const;

            const host_mirror_type gethLocalRegions() const;

            void write(std::ostream& = std::cout) const;

            void changeDomain(const FieldLayout<Dim>&, const Mesh& mesh);  // previously private...

        private:
            NDRegion_t convertNDIndex(const NDIndex<Dim>&, const Mesh& mesh) const;
            void fillRegions(const FieldLayout<Dim>&, const Mesh& mesh);

            //! Offset from 'normal' Index space to 'Mesh' Index space
            std::array<int, Dim> indexOffset_m;

            //! Offset needed between centering of Index space and Mesh points
            std::array<bool, Dim> centerOffset_m;

            NDRegion_t region_m;

            //! local regions (device view)
            view_type dLocalRegions_m;

            //! local regions (host mirror view)
            host_mirror_type hLocalRegions_m;

            view_type subdomains_m;
        };

        template <typename T, unsigned Dim, class Mesh>
        std::ostream& operator<<(std::ostream&, const RegionLayout<T, Dim, Mesh>&);

    }  // namespace detail
}  // namespace ippl

#include "Region/RegionLayout.hpp"

#endif
