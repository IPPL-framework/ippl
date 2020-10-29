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

#ifndef IPPL_PARTICLE_LAYOUT_H
#define IPPL_PARTICLE_LAYOUT_H

#include "Particle/ParticleBConds.h"

#include "Particle/ParticleAttrib.h"

#include <map>

namespace ippl {
    namespace detail {
        // ParticleLayout class definition.  Template parameters are the type
        // and dimension of the ParticlePos object used for the particles.
        template<typename T, unsigned Dim>
        class ParticleLayout {

        public:

            typedef T                             value_type;
            typedef std::int64_t                  index_type;
            typedef Vector<T, Dim>                vector_type;
            typedef ParticleAttrib<vector_type>   particle_position_type;
            typedef std::array<BC, 2 * Dim>       bc_container_type;


            static constexpr unsigned dim = Dim;

        public:
            ParticleLayout()
            {
                bcs_m.fill(BC::NO);
            };

            ~ParticleLayout() = default;

            template<class PBase>
            void update(PBase&) {
                //FIXME
                std::cout << "TODO" << std::endl;
            }

            /*!
             * Copy over the given boundary conditions.
             * @param bcs are the boundary conditions
             */
            void setBConds(bc_container_type bcs) {
                bcs_m = bcs;
            }

            /*!
             * Apply the given boundary conditions to the current particle positions.
             * @tparam PT is the type of particle position attribute container
             * @tparam NDI is the type of index object (NDIndex or NDRegion)
             * @param
             */
            void applyBC(const particle_position_type& R, const NDRegion<T, Dim>& nr);

        private:
            //! the list of boundary conditions for this set of particles
            bc_container_type bcs_m;
        };
    }
}

#include "Particle/ParticleLayout.hpp"

#endif
