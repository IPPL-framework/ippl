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
        HaloCells<T, Dim>::lowerHalo(view_type& view, unsigned int dim) {
            return lower(haloBounds_m, view, dim);
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::upperHalo(view_type& view, unsigned int dim) {
            return upper(haloBounds_m, view, dim);
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::lowerInternal(view_type& view, unsigned int dim) {
            return lower(internalBounds_m, view, dim);
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::upperInternal(view_type& view, unsigned int dim) {
            return upper(internalBounds_m, view, dim);
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::fillHalo(view_type& view, const T& value) {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            using Kokkos::parallel_for;

            for (unsigned int dim = 0; dim < 3; ++dim) {
                auto lo = lowerHalo(view, dim);
                auto hi = upperHalo(view, dim);

                parallel_for("HaloCells::fillHalo()",
                             mdrange_type({0, 0, 0},
                                          {lo.extent(0),
                                           lo.extent(1),
                                           lo.extent(2)}),
                             KOKKOS_CLASS_LAMBDA(const int i,
                                                 const int j,
                                                 const int k) {
                                 lo(i, j, k) = value;
                                 hi(i, j, k) = value;
                });
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::exchangeHalo(view_type& /*view*/,
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
            for (size_t i = 0; i < neighbors.size(); ++i) {
                if (neighbors[i] < 0) {
                    /* if we are on a physical / mesh boundary
                     * --> rank number is negative, hence, we do nothing
                     */
                    continue;
                }


                // pack internal data from view
//                 view_type buffer("buffer", 10, 10, 10);

//                 pack(buffer, i);

                // send data
            }

            // receive
            for (size_t i = 0; i < neighbors.size(); ++i) {
                if (neighbors[i] < 0) {
                    /* if we are on a physical / mesh boundary
                     * --> rank number is negative, hence, we do nothing
                     */
                    continue;
                }


                // receive data

                // unpack received
//                 view_type buffer("buffer");
//                 unpack(view, buffer, nghost);

                // assign data to halo subviews
            }
        }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::pack(view_type& buffer, int index) const {
//             using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
//
//             auto& halo = halo_m[index];
//
//             Kokkos::parallel_for(
//                 "HaloCells::pack()",
//                 mdrange_type({0, 0, 0},
//                              {halo.extent(0), halo.extent(1), halo.extent(2)}),
//                 KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
//                     std::cout << i << " " << j << " " << k << std::endl;
//                     buffer(i, j, k) = halo(i, j, k);
//             });
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
        HaloCells<T, Dim>::lower(bounds_type& bounds, const view_type& view, unsigned int dim) {
            using Kokkos::make_pair;
            return Kokkos::subview(view,
                                   make_pair(bounds[2 * dim][0], bounds[2 * dim][1]),
                                   make_pair(bounds[2 * dim][2], bounds[2 * dim][3]),
                                   make_pair(bounds[2 * dim][4], bounds[2 * dim][5]));
        }


        template <typename T, unsigned Dim>
        auto
        HaloCells<T, Dim>::upper(bounds_type& bounds, const view_type& view, unsigned int dim) {
            using Kokkos::make_pair;
            return Kokkos::subview(view,
                                   make_pair(bounds[2 * dim + 1][0], bounds[2 * dim + 1][1]),
                                   make_pair(bounds[2 * dim + 1][2], bounds[2 * dim + 1][3]),
                                   make_pair(bounds[2 * dim + 1][4], bounds[2 * dim + 1][5]));
        }
    }
}