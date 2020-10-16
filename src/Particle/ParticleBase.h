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

// include files
// #include "Particle/AbstractParticle.h"

#include "Particle/Kokkos_ParticleAttrib.h"
#include "Particle/Kokkos_ParticleLayout.h"

// // #include "DataSource/DataSource.h"
// // #include "DataSource/MakeDataSource.h"
// // #include "Message/Formatter.h"
#include <vector>
// #include <algorithm>  // Include algorithms
// #include <utility>
// #include <iostream>


// forward declarations
// class Inform;
// class Message;
// template <class PLayout> class ParticleBase;
// template <class PLayout>
// std::ostream& operator<<(std::ostream&, const ParticleBase<PLayout>&);
// template <class T, unsigned D> class ParticleBConds;

namespace ippl {

    // ParticleBase class definition.  Template parameter is the specific
    // ParticleLayout-derived class which determines how the particles are
    // distributed among processors.
//     template<class PLayout>

    template<class PLayout>
    class ParticleBase {
//                             public AbstractParticle<typename PLayout::Position_t, PLayout::Dimension> {

    public:
        typedef typename PLayout::vector_type vector_type;
        typedef typename PLayout::index_type  index_type;
        typedef ParticleAttrib<vector_type>   particle_position_type;
        typedef ParticleAttrib<index_type>    particle_index_type;
        // useful enums
//         enum { Dim = PLayout::Dimension };

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
         * @param layout is the particle layout
         */
        ParticleBase(PLayout *layout);

        ~ParticleBase() = default;
//
//         //
//         // Initialization methods
//         //
//
//         // For a ParticleBase that was created with the default constructor,
//         // initialize performs the same actions as are done in the non-default
//         // constructor.  If this object has already been initialized, it is
//         // an error.  For initialize, you must supply a layout instance.
//         void initialize(PLayout *);
//
//
//         //
//         // Accessor functions for this class
//         //
//
//         // return/change the total or local number of particles
        size_t getTotalNum() const { return totalNum_m; }
        size_t getLocalNum() const { return localNum_m; }


//         size_t getDestroyNum() const { return DestroyNum; }
//         size_t getGhostNum() const { return GhostNum; }
//         void setTotalNum(size_t n) { TotalNum = n; }
//         void setLocalNum(size_t n) { LocalNum = n; }
//
//         unsigned int getMinimumNumberOfParticlesPerCore() const { return MIN_NUM_PART_PER_CORE; };
//         void setMinimumNumberOfParticlesPerCore(unsigned int n) { MIN_NUM_PART_PER_CORE=n; };
//
//
//         // get the layout manager
//         PLayout& getLayout() { return *Layout; }
//         const PLayout& getLayout() const { return *Layout; }
//
//         // get or set the boundary conditions container
//         ParticleBConds<Position_t,PLayout::Dimension>& getBConds() {
//             return Layout->getBConds();
//         }
//         void setBConds(const ParticleBConds<Position_t,PLayout::Dimension>& bc) {
//             Layout->setBConds(bc);
//         }
//
//         // Return a boolean value indicating if we are on a processor which can
//         // be used for single-node particle creation and initialization
//         bool singleInitNode() const;
//
//         // get or set the flags used to indicate what to do during the update
//         bool getUpdateFlag(UpdateFlags f) const {
//             return getLayout().getUpdateFlag(f);
//         }
//         void setUpdateFlag(UpdateFlags f, bool val) {
//             getLayout().setUpdateFlag(f, val);
//         }
//
//         //
//         // attribute manipulation methods
//         //
//
//         // add a new attribute ... called by constructor of this and derived classes
        void addAttribute(ParticleAttribBase& pa);
//
//         // get a pointer to the base class for the Nth attribute
//         ParticleAttribBase&
//         getAttribute(attrib_container_t::size_type N) { return *(AttribList[N]); }
//
//         // return the number of attributes in our list
        attribute_container_t::size_type
        numAttributes() const { return attributes_m.size(); }

        // obtain the beginning and end iterators for our attribute list
        attribute_iterator begin() { return attributes_m.begin(); }
        attribute_iterator end()   { return attributes_m.end(); }
//
//         // reset the particle ID's to be globally consecutive, 0 thru TotalNum-1.
//         void resetID();
//
//         //
//         // Global operations on all attributes
//         //
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

        /*
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
//         // Put the data for M particles starting from local index I in a Message.
//         // Return the number of particles put in the Message.
//         size_t putMessage(Message&, size_t, size_t);
//         // put the data for particles on a list into a Message, given list of indices
//         // Return the number of particles put in the Message.
//         size_t putMessage(Message&, const std::vector<size_t>&);
//
//         size_t putMessage(Message&, size_t);
//
//
//         Format* getFormat();
//
//         size_t writeMsgBuffer(MsgBuffer*&, const std::vector<size_t>&);
//
//         template<class O>
//         size_t writeMsgBufferWithOffsets(MsgBuffer*&, const std::vector<size_t>&, const std::vector<O>&);
//
//         size_t readMsgBuffer(MsgBuffer *);
//         size_t readGhostMsgBuffer(MsgBuffer *, int);
//
//         // Retrieve particles from the given message and store them.
//         // Return the number of particles retrieved.
//         size_t getMessage(Message&);
//
//         size_t getSingleMessage(Message&);
//
//         // retrieve particles from the given message and store them, also
//         // signaling we are creating the given number of particles.  Return the
//         // number of particles created.
//         size_t getMessageAndCreate(Message&);
//
//         // Actually perform the delete atoms action for all the attributes; the
//         // calls to destroy() only stored a list of what to do.  This actually
//         // does it.  This should in most cases only be called by the layout manager.
//         void performDestroy(bool updateLocalNum = false);
//
//         // Apply the given sortlist to all the attributes.
//         void sort(SortList_t &);
//
//         //
//         // Global operations on all ghost attributes ... generally, these should
//         // only be used by the layout object
//         //
//
//         // Put the data for M particles starting from local index I in a Message.
//         // Return the number of particles put in the Message.  This is for building
//         // ghost particle interaction lists.
//         size_t ghostPutMessage(Message&, size_t, size_t);
//
//         // put the data for particles on a list into a Message, given list of indices
//         // Return the number of particles put in the Message.  This is for building
//         // ghost particle interaction lists.
//         size_t ghostPutMessage(Message&, const std::vector<size_t>&);
//
//         // Retrieve particles from the given message and sending node and store them.
//         // Return the number of particles retrieved.
//         size_t ghostGetMessage(Message&, int);
//
//         size_t ghostGetSingleMessage(Message&, int);
//
//         // delete M ghost particles, starting with the Ith particle.  This is
//         // always done immediately.
//         void ghostDestroy(size_t, size_t);
//
//         //
//         // I/O
//         //
//
//         // print out debugging information
//         void printDebug(Inform&);
//
//     protected:
//         // a virtual function which is called by this base class to get a
//         // specific instance of DataSourceObject based on the type of data
//         // and the connection method (the argument to the call).
//         virtual DataSourceObject *createDataSourceObject(const char *nm,
//                                                         DataConnect *dc, int tm) {
//             return make_DataSourceObject(nm, dc, tm, *this);
//         }
//
//         // list of destroy events for the next update.  The data
//         // is not actually destroyed until the update phase.
//         // Each destroy is stored as a pair of unsigned ints, the particle
//         // index I to start at and the number of particles M to destroy.
//         std::vector< std::pair<size_t,size_t> > DestroyList;
//
    private:
        // cannot use std::unique_ptr due to Kokkos
        std::shared_ptr<PLayout> layout_m;
//
//
//         // our current number of total and local atoms, and
//         // the number of particles we've deleted since the last update
//         // also, the number of ghost particles
        size_t totalNum_m;
        size_t localNum_m;
//         size_t DestroyNum;
//         size_t GhostNum;
//
//         // our list of attributes
        attribute_container_t attributes_m;

        // unique particle ID number generation value
        index_type nextID_m;
        index_type numNodes_m;

//
    };
}

#include "Particle/ParticleBase.hpp"

#endif
