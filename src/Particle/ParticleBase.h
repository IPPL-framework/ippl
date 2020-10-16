// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_PARTICLE_BASE_H
#define IPPL_PARTICLE_BASE_H

/*
 * ParticleBase - Base class for all user-defined particle classes.
 *
 * ParticleBase is a container and manager for a set of particles.
 * The user must define a class derived from ParticleBase which describes
 * what specific data attributes the particle has (e.g., mass or charge).
 * Each attribute is an instance of a ParticleAttribute<T> class; ParticleBase
 * keeps a list of pointers to these attributes, and performs global
 * operations on them such as update, particle creation and destruction,
 * and inter-processor particle migration.
 *
 * ParticleBase is templated on the ParticleLayout mechanism for the particles.
 * This template parameter should be a class derived from ParticleLayout.
 * ParticleLayout-derived classes maintain the info on which particles are
 * located on which processor, and performs the specific communication
 * required between processors for the particles.  The ParticleLayout is
 * templated on the type and dimension of the atom position attribute, and
 * ParticleBase uses the same types for these items as the given
 * ParticleLayout.
 *
 * ParticleBase and all derived classes have the following common
 * characteristics:
 *     - The spatial positions of the N particles are stored in the
 *       ParticlePos_t variable R
 *     - The global index of the N particles are stored in the
 *       ParticleIndex_t variable ID
 *     - A pointer to an allocated layout class.  When you construct a
 *       ParticleBase, you must provide a layout instance, and ParticleBase
 *       will delete this instance when it (the ParticleBase) is deleted.
 *
 * To use this class, the user defines a derived class with the same
 * structure as in this example:
 *
 *   class UserParticles :
 *            public ParticleBase< ParticleSpatialLayout<double,2> > {
 *   public:
 *     // attributes for this class
 *     ParticleAttribute<double> rad;  // radius
 *     ParticlePos_t             vel;  // velocity, same storage type as R
 *
 *     // constructor: add attributes to base class
 *     UserParticles(ParticleSpatialLayout<double,2>* L) : ParticleBase(L) {
 *       addAttribute(rad);
 *       addAttribute(vel);
 *     }
 *   };
 *
 * This example defines a user class with 2D position and two extra
 * attributes: a radius rad (double), and a velocity vel (a 2D Vektor).
 *
 * After each 'time step' in a calculation, which is defined as a period
 * in which the particle positions may change enough to affect the global
 * layout, the user must call the 'update' routine, which will move
 * particles between processors, etc.  After the Nth call to update, a
 * load balancing routine will be called instead.  The user may set the
 * frequency of load balancing (N), or may supply a function to
 * determine if load balancing should be done or not.
 *
 * Each ParticleBase can contain zero or more 'ghost' particles, which are
 * copies of particle data collected from other nodes.  These ghost particles
 * have many of the same function calls as for the 'regular' particles, with
 * the word 'ghost' prepended.  They are not necessary; but may be used to
 * improve performance of some parallel algorithms (such as calculating
 * neighbor lists).  The actual determination of what ghost particles should
 * be stored in this object (if any) is done by the specific layout object.
 *
 * ParticleBase also contains information on the types of boundary conditions
 * to use.  ParticleBase contains a ParticleBConds object, which is an array
 * of particle boundary condition functions.  By default, these BC's are null,
 * so that nothing special happens at the boundary, but the user may set these
 * BC's by using the 'getBConds' method to access the BC container.  The BC's
 * will then be used at the next update.  In fact, the BC container is stored
 * within the ParticleLayout object, but the interface the user uses to access
 * it is via ParticleBase.
 *
 * You can create an uninitialized ParticleBase, by using the default
 * constructor.  In this case, it will not do any initialization of data
 * structures, etc., and will not contain a layout instance.  In order to use
 * the ParticleBase in any useful way in this case, use the 'initialize()'
 * method which takes as an argument the layout instance to use.
 */

#include "Particle/Kokkos_ParticleAttrib.h"
#include "Particle/Kokkos_ParticleLayout.h"


#include <vector>

namespace ippl {
    /*!
     * @class ParticleBase
     * @tparam PLayout the particle layout implementing an algorithm to
     * distribute the particles among MPI ranks
     */
    template<class PLayout>
    class ParticleBase {

    public:
        typedef typename PLayout::vector_type vector_type;
        typedef typename PLayout::index_type  index_type;
        typedef ParticleAttrib<vector_type>   particle_position_type;
        typedef ParticleAttrib<index_type>    particle_index_type;

        typedef PLayout                           Layout_t;
        typedef std::vector<ParticleAttribBase*> attribute_container_t;
        typedef attribute_container_t::iterator  attribute_iterator;

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
        ParticleBase(std::shared_ptr<PLayout>& layout);


        /* cannot use '= default' since we get a
         * compiler warning otherwise:
         * warning: calling a __host__ function("std::vector< ::ippl::ParticleAttribBase *, ::std::allocator<
         * ::ippl::ParticleAttribBase *> > ::~vector") from a __host__ __device__ function("ippl::ParticleBase<
         * ::ippl::ParticleLayout<double, (unsigned int)3u> > ::~ParticleBase") is not allowed
         */
        ~ParticleBase() { };

        /*!
         * Initialize the particle layout. Needs to be called
         * when the ParticleBase instance is constructed with the
         * default ctor.
         */
        void initialize(std::shared_ptr<PLayout>& layout);

        /*!
         * @returns total number of particles
         */
        size_t getTotalNum() const { return totalNum_m; }

        /*!
         * @returns processor local number of particles
         */
        size_t getLocalNum() const { return localNum_m; }


        /*!
         * @returns processor local number of particles that will
         * be deleted at the next destroy call.
         */
        size_t getDestroyNum() const { return destroyNum_m; }

        /*!
         * Set the total number of particles
         * @param nTotal number of particles
         */
        void setTotalNum(size_t nTotal) { totalNum_m = nTotal; }

        /*!
         * Set the processor local number of particles
         * @param nLotal number of particles
         */
        void setLocalNum(size_t nLocal) { localNum_m = nLocal; }


        /*!
         * @returns particle layout
         */
        PLayout& getLayout() { return *layout_m; }

        /*!
         * @returns particle layout
         */
        const PLayout& getLayout() const { return *layout_m; }


        /*!
         * Add particle attribute
         * @param pa attribute to be added to ParticleBase
         */
        void addAttribute(ParticleAttribBase& pa);


//         attribute_container_t::size_type
//         numAttributes() const { return attributes_m.size(); }

        // obtain the beginning and end iterators for our attribute list
//         attribute_iterator begin() { return attributes_m.begin(); }
//         attribute_iterator end()   { return attributes_m.end(); }

//
//         // Update the particle object after a timestep.  This routine will change
//         // our local, total, create particle counts properly.
//         virtual void update();
//         virtual void update(const ParticleAttrib<char>& canSwap);
//

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
//
//         // delete M particles, starting with the Ith particle.  If the last argument
//         // is true, the destroy will be done immediately, otherwise the request
//         // will be cached.
//         void destroy(size_t, size_t, bool = false);
//
//         // Actually perform the delete atoms action for all the attributes; the
//         // calls to destroy() only stored a list of what to do.  This actually
//         // does it.  This should in most cases only be called by the layout manager.
//         void performDestroy(bool updateLocalNum = false);
//
    private:
        /*!
         * Ctor called when layout == nullptr (i.e., by the default constructor)
         * which happens always since all ctors call default ctor.
         * @param layout is the particle layout
         */
        ParticleBase(std::shared_ptr<PLayout>&& layout);

    private:
        //! particle layout
        // cannot use std::unique_ptr due to Kokkos
        std::shared_ptr<PLayout> layout_m;

        //! total number of particles
        size_t totalNum_m;

        //! processor local number of particles
        size_t localNum_m;

        //! processor local particles to be deleted
        size_t destroyNum_m;

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
