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
namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells()
        {
            static_assert(Dim < 4, "Dimension must be less than 4!");
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::resize(const view_type& view, int nghost)
        {
            fillBounds(haloBounds_m, view, nghost);

            fillBounds(internalBounds_m, view, nghost, nghost);
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::getHaloSubView(const view_type& view,
                                          unsigned int face)
        {
            return subview(haloBounds_m, view, face);
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::getInternalSubView(const view_type& view,
                                              unsigned int face)
        {
            return subview(internalBounds_m, view, face);
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillHalo(view_type& view, const T& value) {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            using Kokkos::parallel_for;

            for (unsigned int face = 0; face < 2 * Dim; ++face) {
                auto halo = getHaloSubView(view, face);

                parallel_for("HaloCells::fillHalo()",
                             mdrange_type({0, 0, 0},
                                          {halo.extent(0),
                                           halo.extent(1),
                                           halo.extent(2)}),
                             KOKKOS_CLASS_LAMBDA(const int i,
                                                 const int j,
                                                 const int k) {
                                 halo(i, j, k) = value;
                });
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeHalo(view_type& view,
                                             const Layout_t* layout,
                                             int /*nghost*/)
        {
            /* The neighbor list has length 2 * Dim. Each index
             * denotes a face. The value tells which MPI rank
             * we need to send to.
             */
            using n_type = typename Layout_t::neighbor_container_type;
            const n_type& neighbors = layout->getNeighbors();


//             using nd_view_type = Layout_t::view_type;

//             nd_view_type

            // send
            for (size_t face = 0; face < neighbors.size(); ++face) {
                if (neighbors[face].empty()) {
                    /* if we are on a physical / mesh boundary
                     */
                    continue;
                }

                auto internal = getInternalSubView(view, face);

                // pack internal data from view
                view_type buffer("buffer",
                                 internal.extent(0),
                                 internal.extent(1),
                                 internal.extent(2));

//                 for (size_t i = 0; i < Dim; ++i) {
//                     std::cout << view.stride(i) << " " << buffer.stride(i) << " " << internal.stride(i) << std::endl;
//                 }

                pack(internal, buffer);

                // send data
            }

            // receive
            for (size_t face = 0; face < neighbors.size(); ++face) {
                if (neighbors[face].empty()) {
                    /* if we are on a physical / mesh boundary
                     */
                    continue;
                }


//                 view_type buffer("buffer");

                // receive data

                // unpack received
//                 unpack(halo, buffer);
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(auto& internal, view_type& buffer) const {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

            Kokkos::parallel_for(
                "HaloCells::pack()",
                mdrange_type({0, 0, 0},
                             {internal.extent(0), internal.extent(1), internal.extent(2)}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                    buffer(i, j, k) = internal(i, j, k);
            });
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::unpack(auto& halo, view_type& buffer) const {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

            Kokkos::parallel_for(
                "HaloCells::pack()",
                mdrange_type({0, 0, 0},
                             {buffer.extent(0), buffer.extent(1), buffer.extent(2)}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                    halo(i, j, k) = buffer(i, j, k);
            });
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillBounds(bounds_type& bounds,
                                           const view_type& view,
                                           int nghost,
                                           int shift)
        {
            for (size_t i = 0; i < Dim; ++i) {
                for (size_t j = 0; j < Dim; ++j) {
                    if (i == j) {
                        // lower bound
                        bounds[2 * i][2 * j] = shift;
                        bounds[2 * i][2 * j + 1] = nghost + shift;

                        // upper bound
                        bounds[2 * i + 1][2 * j] = view.extent(j) - nghost - shift;
                        bounds[2 * i + 1][2 * j + 1] = view.extent(j) - shift;
                    } else {
                        // lower bound
                        bounds[2 * i][2 * j] = nghost;
                        bounds[2 * i][2 * j + 1] = view.extent(j) - nghost;

                        // upper bound
                        bounds[2 * i + 1][2 * j] = nghost;
                        bounds[2 * i + 1][2 * j + 1] = view.extent(j) - nghost;
                    }
                }
            }
        }

        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::subview(bounds_type& bounds, const view_type& view, unsigned int face) {
            using Kokkos::make_pair;
            return Kokkos::subview(view,
                                   make_pair(bounds[face][0], bounds[face][1]),
                                   make_pair(bounds[face][2], bounds[face][3]),
                                   make_pair(bounds[face][4], bounds[face][5]));
        }
    }
}