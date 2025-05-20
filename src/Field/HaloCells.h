//
// Class HaloCells
//   The guard / ghost cells of BareField.
//
#ifndef IPPL_HALO_CELLS_H
#define IPPL_HALO_CELLS_H

#include <array>

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Communicate/Archive.h"
#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"

namespace ippl {
    namespace detail {
        /*!
         * Helper class to send / receive field data.
         */
        template <typename T, class... ViewArgs>
        struct FieldBufferData {
            using view_type    = typename detail::ViewType<T, 1, ViewArgs...>::view_type;
            using archive_type = Archive<typename view_type::memory_space>;

            void serialize(archive_type& ar, size_type nsends) { ar.serialize(buffer, nsends); }

            void deserialize(archive_type& ar, size_type nrecvs) { ar.deserialize(buffer, nrecvs); }

            view_type buffer;
        };

        /*!
         * This class provides the functionality to do field halo exchange.
         * @file HaloCells.h
         */
        template <typename T, unsigned Dim, class... ViewArgs>
        class HaloCells {
        public:
            using view_type       = typename detail::ViewType<T, Dim, ViewArgs...>::view_type;
            using Layout_t        = FieldLayout<Dim>;
            using bound_type      = typename Layout_t::bound_type;
            using databuffer_type = FieldBufferData<T, ViewArgs...>;

            enum SendOrder {
                HALO_TO_INTERNAL,
                INTERNAL_TO_HALO,
                HALO_TO_INTERNAL_NOGHOST
            };

            HaloCells();

            /*!
             * Send halo data to internal cells. This operation uses
             * assign_plus functor to assign the data.
             * @param view the original field data
             * @param layout the field layout storing the domain decomposition
             */
            void accumulateHalo(view_type& view, Layout_t* layout);

            /*!
             * Send halo data to internal cells for only the physical cells
             * along that dimension. The halo cells on the corners are not sent.
             * This operation uses assign_plus functor to assign the data.
             * @param view the original field data
             * @param layout the field layout storing the domain decomposition
             * @param nghost the number of ghost cells
             */
            void accumulateHalo_noghost(view_type& view, Layout_t* layout, int nghost);

            /*!
             * Send interal data to halo cells. This operation uses
             * assign functor to assign the data.
             * @param view the original field data
             * @param layout the field layout storing the domain decomposition
             */
            void fillHalo(view_type&, Layout_t* layout);

            /*!
             * Pack the field data to be sent into a contiguous array.
             * @param range the bounds of the subdomain to be sent
             * @param view the original view
             * @param fd the buffer to pack into
             */
            void pack(const bound_type& range, const view_type& view, databuffer_type& fd,
                      size_type& nsends);

            /*!
             * Unpack the received field data and assign it.
             * @param range the bounds of the subdomain to be received
             * @param view the original view
             * @param fd the buffer to unpack from (received data)
             * @tparam Op the data assigment operator
             */
            template <typename Op>
            void unpack(const bound_type& range, const view_type& view, databuffer_type& fd);

            /*!
             * Operator for the unpack function.
             * This operator is used in case of INTERNAL_TO_HALO.
             */
            struct assign {
                KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const { lhs = rhs; }
            };

            /*!
             * Operator for the unpack function.
             * This operator is used in case of HALO_TO_INTERNAL.
             */
            struct lhs_plus_assign {
                KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const { lhs += rhs; }
            };

            /*!
             * This operator is used in case of HALO_TO_INTERNAL for
             * all periodic BCs application in BareField.
             */
            struct rhs_plus_assign {
                KOKKOS_INLINE_FUNCTION void operator()(const T& lhs, T& rhs) const { rhs += lhs; }
            };

            /*!
             * Apply all periodic boundary conditions for the
             * serial dimensions. Used in case of both fillHalo
             * and accumulateHalo with the help of operator as
             * template parameter.
             */
            template <typename Op>
            void applyPeriodicSerialDim(view_type& view, const Layout_t* layout, const int nghost);

        private:
            /*!
             * Exchange the data of halo cells.
             * @param view is the original field data
             * @param layout the field layout storing the domain decomposition
             * @param order the data send orientation
             * @tparam Op the data assigment operator of the
             * unpack function call
             */
            template <class Op>
            void exchangeBoundaries(view_type& view, Layout_t* layout, SendOrder order, int nghost = 1);

            /*!
             * Extract the subview of the original data. This does not copy.
             * A subview points to the same memory.
             * @param view is the original field data
             * @param intersect the bounds of the intersection
             */
            auto makeSubview(const view_type& view, const bound_type& intersect);

            databuffer_type haloData_m;
        };
    }  // namespace detail
}  // namespace ippl

#include "Field/HaloCells.hpp"

#endif
