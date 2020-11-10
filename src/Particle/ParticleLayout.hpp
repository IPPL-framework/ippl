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
        template<typename T, unsigned Dim>
        void ParticleLayout<T, Dim>::applyBC(const particle_position_type& R,
                                             const NDRegion<T, Dim>& nr)
        {
            /* loop over all faces
             * 0: lower x-face
             * 1: lower y-face
             * 2: lower z-face
             * 3: upper x-face
             * 4: upper y-face
             * 5: upper z-face
             */
            for (unsigned i = 0; i < 2 * Dim; ++i) {
                unsigned face = i % Dim;
                switch (bcs_m[i]) {
                    case BC::PERIODIC:
                        Kokkos::parallel_for("Periodic BC",
                                             R.getView().extent(0),
                                             PeriodicBC(R.getView(), nr, face));
                        break;
                    case BC::REFLECTIVE:
                        Kokkos::parallel_for("Reflective BC",
                                             R.getView().extent(0),
                                             ReflectiveBC(R.getView(), nr, face));
                        break;
                    case BC::SINK:
                        Kokkos::parallel_for("Sink BC",
                                             R.getView().extent(0),
                                             SinkBC(R.getView(), nr, face));
                        break;
                    case BC::NO:
                    default:
                        break;
                }
                Kokkos::fence();
            }
        }
    }
}
