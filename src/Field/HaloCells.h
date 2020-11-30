//
// Class HaloCells
//   The guard / ghost cells of BareField.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_HALO_CELLS_H
#define IPPL_HALO_CELLS_H

#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"
#include "Communicate/Archive.h"
#include "FieldLayout/FieldLayout.h"
#include <array>

namespace ippl {
    namespace detail {
        template <typename T>
        struct FieldData {
            using view_type = typename detail::ViewType<T, 1>::view_type;

            void serialize(Archive<>& ar) {
                ar << buffer;
            }

            void deserialize(Archive<>& ar) {
                ar >> buffer;
            }

            view_type buffer;
        };


        struct assign {
            template <typename T>
            KOKKOS_INLINE_FUNCTION
            void operator()(T& lhs, const T& rhs) const {
                lhs = rhs;
            }
        };


        struct plus {
            template <typename T>
            KOKKOS_INLINE_FUNCTION
            void operator()(T& lhs, const T& rhs) const {
                lhs += rhs;
            }
        };


        template <typename T, unsigned Dim>
        class HaloCells
        {

        public:
            struct intersect_type {
                std::array<long, Dim> lo;
                std::array<long, Dim> hi;
            };


            enum SendOrder {
                HALO_TO_INTERNAL,
                INTERNAL_TO_HALO
            };


            using bound_type = std::array<long, 2 * Dim>;
            using bounds_type = std::array<bound_type, 2 * Dim>;
            using view_type = typename detail::ViewType<T, Dim>::view_type;
            using Layout_t  = FieldLayout<Dim>;

            HaloCells();

            void fillLocalHalo(view_type& view,
                               const T& value,
                               int nghost);

            void accumulateHalo(view_type&, const Layout_t* layout, int nghost);

            void fillHalo(view_type&, const Layout_t* layout, int nghost);


            void pack(const intersect_type& range,
                      const view_type& view,
                      FieldData<T>& fd);

            template <class Op>
            void unpack(const intersect_type& range,
                        const view_type& view,
                        FieldData<T>& fd);


        private:

            intersect_type getBounds(const NDIndex<Dim>&,
                                     const NDIndex<Dim>&,
                                     const NDIndex<Dim>&,
                                     int nghost);

//             intersect_type getHaloBounds(


            template <class Op>
            void exchangeFaces(view_type& view,
                               const Layout_t* layout,
                               int nghost,
                               SendOrder order);

            template <class Op>
            void exchangeEdges(view_type& view,
                               const Layout_t* layout,
                               int nghost,
                               SendOrder order);

            template <class Op>
            void exchangeVertices(view_type& view,
                                  const Layout_t* layout,
                                  int nghost,
                                  SendOrder order);


            auto makeSubview(const view_type&, const intersect_type&);


            /*! lower halo cells (ordering x, y, z)
             * x --> lower y-z plane
             * x --> upper y-z plane
             * y --> lower x-z plane
             * y --> upper x-z plane
             * z --> lower x-y plane
             * z --> upper x-y plane
             */
            bounds_type haloBounds_m;

            /*!
             * These subviews correspond to the internal data that is used
             * to exchange.
             */
            bounds_type internalBounds_m;
        };
    }
}

#include "Field/HaloCells.hpp"

#endif