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

namespace ippl {
    // ParticleLayout class definition.  Template parameters are the type
    // and dimension of the ParticlePos object used for the particles.
    template<typename T, unsigned Dim>
    class ParticleLayout {

    public:
//         // useful enumerations
//         enum { Dimension = Dim };	                      // dim of coord data
//         enum UpdateFlags { SWAP, BCONDS, NUMFLAGS, OPTDESTROY, ALL }; // update opts
//
        typedef T               value_type;
        typedef std::int64_t    index_type;
        typedef Vector<T, Dim>  vector_type;
//
//         // in addition, all subclasses must provide a typedef or class definition
//         // for a type 'pair_iterator', which iterates over lists of 'pair_t' objs,
//         // and typedefs for the ParticlePos_t and ParticleIndex_t types.
//
    public:
        ParticleLayout() = default;

        ~ParticleLayout() = default;

        template<class PBase>
        void update(PBase&) {
            //FIXME
            std::cout << "TODO" << std::endl;
        }
//
//         // set the flags used to indicate what to do during the update
//         void setUpdateFlag(UpdateFlags f, bool val) {
//             if (f == ALL) {
//             UpdateOptions = (~0);
//             } else {
//             unsigned int mask = (1 << f);
//             UpdateOptions = (val ? (UpdateOptions|mask) : (UpdateOptions&(~mask)));
//             }
//         }
//
//         // get the flags used to indicate what to do during the update
//         bool getUpdateFlag(UpdateFlags f) const {
//             return (f==ALL ? (UpdateOptions==(~0u)) : ((UpdateOptions & (1u << f))!=0));
//         }
//
//         // get the boundary conditions container
//         ParticleBConds<T,Dim>& getBConds() { return BoundConds; }
//
//         // copy over the given boundary conditions
//         void setBConds(const ParticleBConds<T,Dim>& bc) { BoundConds = bc; }
//
//     protected:
//         // apply the given boundary conditions to the given set of n particle
//         // positions.  This modifies the position values based on the BC type.
//         //mwerks  template<class PPT, class NDI>
//         //mwerks  void apply_bconds(unsigned, PPT&, const ParticleBConds<T,Dim>&, const NDI&);
//         /////////////////////////////////////////////////////////////////////
//         // Apply the given boundary conditions to the current particle positions.
//         // PPT is the type of particle position attribute container, and NDI is
//         // the type of index object (NDIndex or NDRegion)
//         template<class PPT, class NDI>
//         void apply_bconds(unsigned n, PPT& R,
//                             const ParticleBConds<T,Dim>& bcs,
//                             const NDI& nr) {
//
//             // apply boundary conditions to the positions
//             for (unsigned int i=0; i < n; ++i)
//             for (unsigned int j=0; j < Dim; j++)
//                 R[i][j] = bcs.apply(R[i][j], j, nr);
//         }
//
//     private:
//         // the list of boundary conditions for this set of particles
//         ParticleBConds<T,Dim> BoundConds;
//
//         // flags indicating what should be updated
//         unsigned int UpdateOptions;
    };
}

#include "Particle/Kokkos_ParticleLayout.hpp"

#endif
