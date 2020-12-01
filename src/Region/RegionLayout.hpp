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
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
        template <typename T, unsigned Dim, class Mesh>
        RegionLayout<T, Dim, Mesh>::RegionLayout()
        : dLocalRegions_m("local regions (device)", 0)
        , hLocalRegions_m(Kokkos::create_mirror_view(dLocalRegions_m))
        {
            indexOffset_m.fill(0);
            centerOffset_m.fill(0);
        }


        template <typename T, unsigned Dim, class Mesh>
        RegionLayout<T, Dim, Mesh>::RegionLayout(const FieldLayout<Dim>& fl,
                                                 const Mesh& mesh)
        : RegionLayout()
        {
            changeDomain(fl, mesh);
        }



        template <typename T, unsigned Dim, class Mesh>
        void RegionLayout<T, Dim, Mesh>::changeDomain(const FieldLayout<Dim>& fl,
                                                      const Mesh& mesh)
        {
            // set our index space offset
            for (unsigned int d = 0; d < Dim; ++d) {
                indexOffset_m[d] = fl.getDomain()[d].first();
                centerOffset_m[d] = 1;
            }

            region_m = convertNDIndex(fl.getDomain(), mesh);

            fillRegions(fl, mesh);
        }


        // convert a given NDIndex into an NDRegion ... if this object was
        // constructed from a FieldLayout, this does nothing, but if we are maintaining
        // our own internal FieldLayout, we must convert from the [0,N-1] index
        // space to our own continuous NDRegion space.
        // NOTE: THIS ASSUMES THAT REGION'S HAVE first() < last() !!
        template <typename T, unsigned Dim, class Mesh>
        typename RegionLayout<T, Dim, Mesh>::NDRegion_t
        RegionLayout<T, Dim, Mesh>::convertNDIndex(const NDIndex<Dim>& ni,
                                                   const Mesh& mesh) const
        {
            // find first and last points in NDIndex and get coordinates from mesh
            NDIndex<Dim> firstPoint, lastPoint;
            for (unsigned int d = 0; d < Dim; d++) {
                int first = ni[d].first() - indexOffset_m[d];
                int last  = ni[d].last()  - indexOffset_m[d] + centerOffset_m[d];
                firstPoint[d] = Index(first, first);
                lastPoint[d] = Index(last, last);
            }

            // convert to mesh space
            Vector<T, Dim> firstCoord = mesh.getVertexPosition(firstPoint);
            Vector<T, Dim> lastCoord = mesh.getVertexPosition(lastPoint);
            NDRegion_t ndregion;
            for (unsigned int d = 0; d < Dim; d++) {
                ndregion[d] = PRegion<T>(firstCoord(d), lastCoord(d));
            }
            return ndregion;
        }


        template <typename T, unsigned Dim, class Mesh>
        void RegionLayout<T, Dim, Mesh>::fillRegions(const FieldLayout<Dim>& fl,
                                                     const Mesh& mesh)
        {
            using domain_type = typename FieldLayout<Dim>::host_mirror_type;
            const domain_type& ldomains = fl.getLocalDomains();

            Kokkos::resize(hLocalRegions_m, ldomains.size());
            Kokkos::resize(dLocalRegions_m, ldomains.size());

            using size_type = typename domain_type::size_type;
            for (size_type i = 0; i < ldomains.size(); ++i) {
                hLocalRegions_m(i) = convertNDIndex(ldomains(i), mesh);
            }

            Kokkos::deep_copy(dLocalRegions_m, hLocalRegions_m);
        }


        template <typename T, unsigned Dim, class Mesh>
        void RegionLayout<T, Dim, Mesh>::write(std::ostream& out) const
        {
            if (Ippl::Comm->rank() > 0) {
                return;
            }

            out << "Total region = " << region_m << "\n"
                << "Total number of subregions = " << hLocalRegions_m.size() << "\n";

            using size_type = typename host_mirror_type::size_type;
            for (size_type i = 0; i < hLocalRegions_m.size(); ++i) {
                out << "    subregion " << i << " " << hLocalRegions_m(i) << "\n";
            }
        }

        template <typename T, unsigned Dim, class Mesh>
        const typename RegionLayout<T, Dim, Mesh>::view_type& 
        RegionLayout<T, Dim, Mesh>::getdLocalRegions() const
        {
            return dLocalRegions_m;
        }

        template <typename T, unsigned Dim, class Mesh>
        std::ostream& operator<<(std::ostream& out, const RegionLayout<T, Dim, Mesh>& rl)
        {
            rl.write(out);
            return out;
        }
    }
}
