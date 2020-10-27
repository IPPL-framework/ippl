//
// Class ParticleLayout
//   Base class for all particle layout classes.
//
//   This class is used as the generic base class for all classes
//   which maintain the information on where all the particles are located
//   on a parallel machine.  It is responsible for performing particle
//   load balancing.
//
//   If more general layout information is needed, such as the global -> local
//   mapping for each particle, then derived classes must provide this info.
//
//   When particles are created or destroyed, this class is also responsible
//   for determining where particles are to be created, gathering this
//   information, and recalculating the global indices of all the particles.
//   For consistency, creation and destruction requests are cached, and then
//   performed all in one step when the update routine is called.
//
//   Derived classes must provide the following:
//     1) Specific version of update and loadBalance.  These are not virtual,
//        as this class is used as a template parameter (instead of being
//        assigned to a base class pointer).
//     2) Internal storage to maintain their specific layout mechanism
//     3) the definition of a class pair_iterator, and a function
//        void getPairlist(int, pair_iterator&, pair_iterator&) to get a
//        begin/end iterator pair to access the local neighbor of the Nth
//        local atom.  This is not a virtual function, it is a requirement of
//        the templated class for use in other parts of the code.
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
        /*
         * Functor to apply boundary conditions
         */
        template<typename T, unsigned Dim, class PT, class NDI>
        struct ApplyBC {
            PT pos_m;
            NDI nr_m;
            ParticleBConds<T, Dim> bcs_m;

            ApplyBC(PT pos, const NDI& nr, const ParticleBConds<T, Dim>& bcs) {
                pos_m = pos;
                nr_m = nr;
                bcs_m = bcs;
            }

            KOKKOS_INLINE_FUNCTION
            void operator() (const size_t i, const size_t j) const {
                pos_m(i)[j] = bcs_m.apply(pos_m(i)[j], j, nr_m);
            }
        };


        template<typename T, unsigned Dim>
        template<class PT, class NDI>
        void ParticleLayout<T, Dim>::applyBC(PT& R, const NDI& nr) {
            using mdrange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            long int len = R.extent(0);
            Kokkos::parallel_for("ParticleLayout::applyBC()",
                                 mdrange({0, 0}, {len, Dim}),
                                 ApplyBC<T, Dim, PT, NDI>(R, nr, bcs_m));
        }
    }
}
