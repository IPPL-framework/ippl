//
// Class ParticleBase
//   Base class for all user-defined particle classes.
//
//   ParticleBase is a container and manager for a set of particles.
//   The user must define a class derived from ParticleBase which describes
//   what specific data attributes the particle has (e.g., mass or charge).
//   Each attribute is an instance of a ParticleAttribute<T> class; ParticleBase
//   keeps a list of pointers to these attributes, and performs global
//   operations on them such as update, particle creation and destruction,
//   and inter-processor particle migration.
//
//   ParticleBase is templated on the ParticleLayout mechanism for the particles.
//   This template parameter should be a class derived from ParticleLayout.
//   ParticleLayout-derived classes maintain the info on which particles are
//   located on which processor, and performs the specific communication
//   required between processors for the particles.  The ParticleLayout is
//   templated on the type and dimension of the atom position attribute, and
//   ParticleBase uses the same types for these items as the given
//   ParticleLayout.
//
//   ParticleBase and all derived classes have the following common
//   characteristics:
//       - The spatial positions of the N particles are stored in the
//         particle_position_type variable R
//       - The global index of the N particles are stored in the
//         particle_index_type variable ID
//       - A pointer to an allocated layout class.  When you construct a
//         ParticleBase, you must provide a layout instance, and ParticleBase
//         will delete this instance when it (the ParticleBase) is deleted.
//
//   To use this class, the user defines a derived class with the same
//   structure as in this example:
//
//     class UserParticles :
//              public ParticleBase< ParticleSpatialLayout<double,3> > {
//     public:
//       // attributes for this class
//       ParticleAttribute<double> rad;  // radius
//       particle_position_type    vel;  // velocity, same storage type as R
//
//       // constructor: add attributes to base class
//       UserParticles(ParticleSpatialLayout<double,2>* L) : ParticleBase(L) {
//         addAttribute(rad);
//         addAttribute(vel);
//       }
//     };
//
//   This example defines a user class with 3D position and two extra
//   attributes: a radius rad (double), and a velocity vel (a 3D Vector).
//
//   After each 'time step' in a calculation, which is defined as a period
//   in which the particle positions may change enough to affect the global
//   layout, the user must call the 'update' routine, which will move
//   particles between processors, etc.  After the Nth call to update, a
//   load balancing routine will be called instead.  The user may set the
//   frequency of load balancing (N), or may supply a function to
//   determine if load balancing should be done or not.
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
#ifndef IPPL_PARTICLE_BASE_H
#define IPPL_PARTICLE_BASE_H

#include "Particle/ParticleLayout.h"


#include <vector>

namespace ippl {
    /*!
     * @class ParticleBase
     * @tparam PLayout the particle layout implementing an algorithm to
     * distribute the particles among MPI ranks
     */
    template <class PLayout, class... Properties>
    class ParticleBase {

    public:
        using vector_type            = typename PLayout::vector_type;
        using index_type             = typename PLayout::index_type;
        using particle_position_type = typename PLayout::particle_position_type ;
        using particle_index_type    = ParticleAttrib<index_type>;

        using Layout_t              = PLayout;
        using attribute_type        = typename detail::ParticleAttribBase<Properties...>;
        using attribute_container_t = std::vector<attribute_type*>;
        using attribute_iterator    = typename attribute_container_t::iterator;
        using bc_container_type     = typename PLayout::bc_container_type;
        using hash_type             = typename detail::ViewType<int, 1, Properties...>::view_type;

    public:
        //! view of particle positions
        particle_position_type R;

        //! view of particle IDs
        particle_index_type ID;

        /*!
         * If this constructor is used, the user must call 'initialize' with
         * a layout object in order to use this.
         */
        ParticleBase();

        /*!
         * Ctor called when layout is provided with std::shared_ptr. It
         * calls the default ctor which then calls the private ctor. The
         * layout instance is moved to this class, hence, the argument
         * is null afterwards, i.e., layout == nullptr.
         * @param layout to be moved.
         */
        ParticleBase(Layout_t& layout);


        /* cannot use '= default' since we get a
         * compiler warning otherwise:
         * warning: calling a __host__ function("std::vector< ::ippl::detail::ParticleAttribBase *, ::std::allocator<
         * ::ippl::detail::ParticleAttribBase *> > ::~vector") from a __host__ __device__ function("ippl::ParticleBase<
         * ::ippl::ParticleLayout<double, (unsigned int)3u> > ::~ParticleBase") is not allowed
         */
        ~ParticleBase() {} // = default; //{ }

        /*!
         * Initialize the particle layout. Needs to be called
         * when the ParticleBase instance is constructed with the
         * default ctor.
         */
        void initialize(Layout_t& layout);

        /*!
         * @returns processor local number of particles
         */
        size_t getLocalNum() const { return localNum_m; }

        /*!
         * Set the processor local number of particles
         * @param nLocal number of particles
         */
        void setLocalNum(size_t nLocal) { localNum_m = nLocal; }


        /*!
         * @returns particle layout
         */
        Layout_t& getLayout() { return *layout_m; }

        /*!
         * @returns particle layout
         */
        const Layout_t& getLayout() const { return *layout_m; }


        /*!
         * Set all boundary conditions
         * @param bc the boundary conditions
         */
        void setParticleBC(const bc_container_type& bcs) {
            layout_m->setParticleBC(bcs);
        }

        /*!
         * Set all boundary conditions to this BC
         * @param bc the boundary conditions
         */
        void setParticleBC(BC bc) {
            layout_m->setParticleBC(bc);
        }


        /*!
         * Add particle attribute
         * @param pa attribute to be added to ParticleBase
         */
        void addAttribute(detail::ParticleAttribBase<Properties...>& pa);


        /*!
         * Get particle attribute
         * @param i attribute number in container
         * @returns a pointer to the attribute
         */
        attribute_type* getAttribute(size_t i) {
            return attributes_m[i];
        }


        /*!
         * @returns the number of attributes
         */
        typename attribute_container_t::size_type getAttributeNum() const {
            return attributes_m.size();
        }


        /*!
         * Create nLocal processor local particles
         * @param nLocal number of local particles to be created
         */
        void create(size_t nLocal);

        /*!
         * Create a new particle with a given ID
         * @param id particle identity number
         */
        void createWithID(index_type id);

        /*!
         * Create nTotal particles globally, equally distributed among all processors
         * @param nTotal number of total particles to be created
         */
        void globalCreate(size_t nTotal);

        /*!
         * Delete particles.
         * @param
         */
        void destroy();


        /*!
         * Serialize to do MPI calls.
         * @param ar archive
         */
        void serialize(detail::Archive<Properties...>& ar);


        /*!
         * Deserialize to do MPI calls.
         * @param ar archive
         */
        void deserialize(detail::Archive<Properties...>& ar);


        /*!
         * Redistribute particles among MPI ranks.
         * This function calls the underlying particle layout
         * routine.
         */
//         template <class BufferType>
        void update();


//     protected:

        /*!
         * Fill attributes of buffer.
         * @tparam Buffer is a bunch type
         * @param buffer to send
         * @param hash function to access index.
         */
        template <class Buffer>
        void pack(Buffer& buffer, const hash_type& hash);

        /*!
         * Fill my attributes.
         * @tparam Buffer is a bunch type
         * @param buffer received
         */
        template <class Buffer>
        void unpack(Buffer& buffer);

    private:
        //! particle layout
        // cannot use std::unique_ptr due to Kokkos
        Layout_t* layout_m;

        //! processor local number of particles
        size_t localNum_m;

        //! all attributes
        attribute_container_t attributes_m;

        //! next unique particle ID
        index_type nextID_m;

        //! number of MPI ranks
        index_type numNodes_m;
    };
}

#include "Particle/ParticleBase.hpp"

#endif
