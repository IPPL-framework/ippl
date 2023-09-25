//
// Class FieldBase
//   Base class for all user-defined field classes.
//
//   FieldBase is a container and manager for a set of fields.
//   The user must define a class derived from FieldBase which describes
//   what specific data attributes the field has (e.g., density or charge).
//   Each attribute is an instance of a FieldAttribute<T> class; FieldBase
//   keeps a list of pointers to these attributes, and performs field creation
//   and destruction.
//
//   FieldBase is templated on the FieldLayout mechanism for the fields.
//   This template parameter should be a class derived from FieldLayout.
//   FieldLayout-derived classes maintain the info on which fields are
//   located on which processor, and performs the specific communication
//   required between processors for the fields.  The FieldLayout is
//   templated on the type and dimension of the atom position attribute, and
//   FieldBase uses the same types for these items as the given
//   FieldLayout.
//
//   FieldBase and all derived classes have the following common
//   characteristics:
//       - The spatial positions of the N fields are stored in the
//         field_position_type variable R
//       - The global index of the N particles are stored in the
//         particle_index_type variable ID
//       - A pointer to an allocated layout class.  When you construct a
//         FieldBase, you must provide a layout instance, and FieldBase
//         will delete this instance when it (the FieldBase) is deleted.
//
//   To use this class, the user defines a derived class with the same
//   structure as in this example:
//
//     class UserField :
//              public FieldBase< ParticleSpatialLayout<double,3> > {
//     public:
//       // attributes for this class
//       FieldAttribute<double> rho;  // density
//
//       // constructor: add attributes to base class
//       UserParticles(ParticleSpatialLayout<double,2>* L) : ParticleBase(L) {
//         addAttribute(rho);
//       }
//     };
//
//   This example defines a user class with 3D position and two extra
//   attributes: a radius rad (double), and a velocity vel (a 3D Vector).
//
#ifndef IPPL_FIELD_BASE_H
#define IPPL_FIELD_BASE_H

#include <tuple>
#include <type_traits>
#include <vector>

#include "Types/IpplTypes.h"

#include "Utility/TypeUtils.h"

#include "FieldLayout/FieldLayout.h"

namespace ippl {

    /*!
     * @class FieldBase
     * @tparam FLayout the particle layout implementing an algorithm to
     * distribute the field among MPI ranks
     * @tparam IDProperties the view properties for field IDs
     */
    template <class FLayout, typename... IDProperties>
    class FieldBase {
        constexpr static bool EnableIDs = sizeof...(IDProperties) > 0;

    public:
        using vector_type            = typename FLayout::vector_type;
        using index_type             = typename FLayout::index_type;
        using field_position_type = typename FLayout::field_position_type;
        using field_index_type    = FieldAttrib<index_type, IDProperties...>;

        using Layout_t = FLayout;

        template <typename... Properties>
        using attribute_type = typename detail::FieldAttribBase<Properties...>;

        template <typename MemorySpace>
        using container_type = std::vector<attribute_type<MemorySpace>*>;

        using attribute_container_type =
            typename detail::ContainerForAllSpaces<container_type>::type;

        using bc_container_type = typename FLayout::bc_container_type;

        using hash_container_type = typename detail::ContainerForAllSpaces<detail::hash_type>::type;

        using size_type = detail::size_type;

    public:
        //! view of Field IDs
        field_index_type ID;

        /*!
         * If this constructor is used, the user must call 'initialize' with
         * a layout object in order to use this.
         */
        FieldBase();

        /*!
         * Ctor called when layout is provided with std::shared_ptr. It
         * calls the default ctor which then calls the private ctor. The
         * layout instance is moved to this class, hence, the argument
         * is null afterwards, i.e., layout == nullptr.
         * @param layout to be moved.
         */
        FieldBase(Layout_t& layout);

        /* cannot use '= default' since we get a
         * compiler warning otherwise:
         * warning: calling a __host__ function("std::vector< ::ippl::detail::ParticleAttribBase *,
         * ::std::allocator<
         * ::ippl::detail::ParticleAttribBase *> > ::~vector") from a __host__ __device__
         * function("ippl::ParticleBase<
         * ::ippl::ParticleLayout<double, (unsigned int)3u> > ::~ParticleBase") is not allowed
         */
        ~FieldBase() {}  // = default; //{ }

        /*!
         * Initialize the particle layout. Needs to be called
         * when the ParticleBase instance is constructed with the
         * default ctor.
         */
        void initialize(Layout_t& layout);

        /*!
         * @returns field layout
         */
        Layout_t& getLayout() { return *layout_m; }

        /*!
         * @returns field layout
         */
        const Layout_t& getLayout() const { return *layout_m; }

        /*!
         * Set all boundary conditions
         * @param bc the boundary conditions
         */
        //void setFieldBC(const bc_container_type& bcs) { layout_m->setFieldBC(bcs); }

        /*!
         * Set all boundary conditions to this BC
         * @param bc the boundary conditions
         */
        //void setFieldBC(BC bc) { layout_m->setFieldBC(bc); }

        /*!
         * Add particle attribute
         * @param pa attribute to be added to ParticleBase
         */
        template <typename MemorySpace>
        void addAttribute(detail::FieldAttribBase<MemorySpace>& pa);

        /*!
         * Get particle attribute
         * @param i attribute number in container
         * @returns a pointer to the attribute
         */
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        attribute_type<MemorySpace>* getAttribute(size_t i) {
            return attributes_m.template get<MemorySpace>()[i];
        }

        /*!
         * Calls a given function for all attributes in the bunch
         * @tparam MemorySpace the memory space of the attributes to visit (void to visit all of
         * them)
         * @tparam Functor the functor type
         * @param f a functor taking a single ParticleAttrib<MemorySpace>
         */
        template <typename MemorySpace = void, typename Functor>
        void forAllAttributes(Functor&& f) const {
            if constexpr (std::is_void_v<MemorySpace>) {
                attributes_m.forAll(f);
            } else {
                for (auto& attribute : attributes_m.template get<MemorySpace>()) {
                    f(attribute);
                }
            }
        }

        // Non-const variant of same function
        template <typename MemorySpace = void, typename Functor>
        void forAllAttributes(Functor&& f) {
            if constexpr (std::is_void_v<MemorySpace>) {
                attributes_m.forAll([&]<typename Attributes>(Attributes& atts) {
                    for (auto& attribute : atts) {
                        f(attribute);
                    }
                });
            } else {
                for (auto& attribute : attributes_m.template get<MemorySpace>()) {
                    f(attribute);
                }
            }
        }

        /*!
         * @returns the number of attributes
         */
        unsigned getAttributeNum() const {
            unsigned total = 0;
            detail::runForAllSpaces([&]<typename MemorySpace>() {
                total += attributes_m.template get<MemorySpace>().size();
            });
            return total;
        }

        /*!
         * Create nLocal processor local particles. This is a collective call,
         * i.e. all MPI ranks must call this.
         * @param nLocal number of local particles to be created
         */
        //void create(size_type nLocal);

        /*!
         * Create a new particle with a given ID. This is a collective call. If a process
         * passes a negative number, it does not create a particle.
         * @param id particle identity number
         */
        //void createWithID(index_type id);

        /*!
         * Create nTotal particles globally, equally distributed among all processors.
         * This is a collective call.
         * @param nTotal number of total particles to be created
         */
        //void globalCreate(size_type nTotal);

        /*!
         * Particle deletion Function. Partition the particles into a valid region
         * and an invalid region,
         * effectively deleting the invalid particles. This is a collective call.
         * @param invalid View marking which indices are invalid
         * @param destroyNum Total number of invalid particles
         */
        template <typename... Properties>
        void destroy(const Kokkos::View<bool*, Properties...>& invalid, const size_type destroyNum);

        // This is a collective call.
        void update() { layout_m->update(*this); }

        /*
         * The following functions should not be called in an application.
         */

        /* This function does not alter the totalNum_m member function. It should only be called
         * during the update function where we know the number of particles remains the same.
         */
        template <typename... Properties>
        void internalDestroy(const Kokkos::View<bool*, Properties...>& invalid,
                             const size_type destroyNum);

        /*!
         * Sends particles to another rank
         * @tparam HashType the hash view type
         * @param rank the destination rank
         * @param tag the MPI tag
         * @param sendNum the number of messages already sent (to distinguish the buffers)
         * @param requests destination vector in which to store the MPI requests for polling
         * purposes
         * @param hash a hash view indicating which particles need to be sent to which rank
         */
        template <typename HashType>
        void sendToRank(int rank, int tag, int sendNum, std::vector<MPI_Request>& requests,
                        const HashType& hash);

        /*!
         * Receives particles from another rank
         * @param rank the source rank
         * @param tag the MPI tag
         * @param recvNum the number of messages already received (to distinguish the buffers)
         * @param nRecvs the number of particles to receive
         */
        void recvFromRank(int rank, int tag, int recvNum, size_type nRecvs);

        /*!
         * Serialize to do MPI calls.
         * @param ar archive
         */
        template <typename Archive>
        void serialize(Archive& ar, size_type nsends);

        /*!
         * Deserialize to do MPI calls.
         * @param ar archive
         */
        template <typename Archive>
        void deserialize(Archive& ar, size_type nrecvs);

        /*!
         * Determine the total space necessary to store a certain number of particles
         * @tparam MemorySpace only consider attributes stored in this memory space
         * @param count particle number
         * @return Total size of a buffer packed with the given number of particles
         */
        template <typename MemorySpace>
        size_type packedSize(const size_type count) const;

    protected:
        /*!
         * Fill attributes of buffer.
         * @param buffer to send
         * @param hash function to access index.
         */
        void pack(const hash_container_type& hash);

        /*!
         * Fill my attributes.
         * @param buffer received
         */
        void unpack(size_type nrecvs);

    private:
        //! particle layout
        // cannot use std::unique_ptr due to Kokkos
        Layout_t* layout_m;

        //! all attributes
        attribute_container_type attributes_m;

        //! next unique particle ID
        index_type nextID_m;

        //! number of MPI ranks
        index_type numNodes_m;

        //! buffers for particle partitioning
        hash_container_type deleteIndex_m;
        hash_container_type keepIndex_m;
    };
}  // namespace ippl

#include "Field/FieldBase.hpp"

#endif
