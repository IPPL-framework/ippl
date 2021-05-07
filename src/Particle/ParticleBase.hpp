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

namespace ippl {

    template <class PLayout, class... Properties>
    ParticleBase<PLayout, Properties...>::ParticleBase()
    : layout_m(nullptr)
    , localNum_m(0)
    , attributes_m(0)
    , nextID_m(Ippl::Comm->myNode())
    , numNodes_m(Ippl::Comm->getNodes())
    {
        addAttribute(ID); // needs to be added first due to destroy function
        addAttribute(R);
    }

    template <class PLayout, class... Properties>
    ParticleBase<PLayout, Properties...>::ParticleBase(PLayout& layout)
    : ParticleBase()
    {
        initialize(layout);
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::addAttribute(detail::ParticleAttribBase<Properties...>& pa)
    {
        attributes_m.push_back(&pa);
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::initialize(PLayout& layout)
    {
//         PAssert(layout_m == nullptr);

        // save the layout, and perform setup tasks
        layout_m = &layout;
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::create(size_t nLocal)
    {
        PAssert(layout_m != nullptr);

        for (attribute_iterator it = attributes_m.begin();
             it != attributes_m.end(); ++it) {
            (*it)->create(nLocal);
        }

        // set the unique ID value for these new particles
        Kokkos::parallel_for("ParticleBase<PLayout, Properties...>::create(size_t)",
                             Kokkos::RangePolicy(localNum_m, nLocal),
                             KOKKOS_CLASS_LAMBDA(const std::int64_t i) {
                                 ID(i) = this->nextID_m + this->numNodes_m * i;
                             });
        nextID_m += numNodes_m * (nLocal - localNum_m);

        // remember that we're creating these new particles
        localNum_m += nLocal;
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::createWithID(index_type id)
    {
        PAssert(layout_m != nullptr);

        // temporary change
        index_type tmpNextID = nextID_m;
        nextID_m = id;
        numNodes_m = 0;

        create(1);

        nextID_m = tmpNextID;
        numNodes_m = Ippl::Comm->getNodes();
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::globalCreate(size_t nTotal)
    {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_t nLocal = nTotal / numNodes_m;

        const size_t rank = Ippl::Comm->myNode();

        size_t rest = nTotal - nLocal * rank;
        if (rank < rest)
            ++nLocal;

        create(nLocal);
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::destroy() {

        /* count the number of particles with ID == -1 and fill
         * a boolean view
         */
        size_t destroyNum = 0;
        Kokkos::View<bool*> invalidIndex("", localNum_m);
        Kokkos::parallel_reduce("Reduce in ParticleBase::destroy()",
                                localNum_m,
                                KOKKOS_CLASS_LAMBDA(const size_t i,
                                                    size_t& nInvalid)
                                {
                                    nInvalid += size_t(ID(i) < 0);
                                    invalidIndex(i) = (ID(i) < 0);
                                }, destroyNum);

        PAssert(destroyNum <= localNum_m);

        if (destroyNum == 0) {
            return;
        }

        /* Compute the prefix sum and store the new
         * particle indices in newIndex.
         */
        Kokkos::View<int*> newIndex("newIndex", localNum_m);
        Kokkos::parallel_scan("Scan in ParticleBase::destroy()",
                              localNum_m,
                              KOKKOS_LAMBDA(const size_t i, int& idx, const bool final)
                              {
                                  if (final) {
                                      newIndex(i) = idx;
                                  }

                                  if (!invalidIndex(i)) {
                                      idx += 1;
                                  }
                              });

        localNum_m -= destroyNum;

        // delete the invalide attribut indices
        for (attribute_iterator it = attributes_m.begin();
             it != attributes_m.end(); ++it)
        {
            (*it)->destroy(invalidIndex, newIndex, localNum_m);
        }
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::serialize(detail::Archive<Properties...>& ar) {
        using size_type = typename attribute_container_t::size_type;
        for (size_type i = 0; i < attributes_m.size(); ++i) {
            attributes_m[i]->serialize(ar);
        }
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::deserialize(detail::Archive<Properties...>& ar) {
        using size_type = typename attribute_container_t::size_type;
        for (size_type i = 0; i < attributes_m.size(); ++i) {
            attributes_m[i]->deserialize(ar);
        }
    }

 
    template <class PLayout, class... Properties>
//     template <class BufferType>
    void ParticleBase<PLayout, Properties...>::update()
    {
        PAssert(layout_m != nullptr);
//         layout_m->update<BufferType>(*this);
    }


    template <class PLayout, class... Properties>
    template <class Buffer>
    void ParticleBase<PLayout, Properties...>::pack(Buffer& buffer,
                                                    const hash_type& hash)
    {
        using size_type = typename attribute_container_t::size_type;
        for (size_type j = 0; j < attributes_m.size(); ++j) {
            attributes_m[j]->pack(buffer.getAttribute(j), hash);
        }
    }


    template <class PLayout, class... Properties>
    template <class Buffer>
    void ParticleBase<PLayout, Properties...>::unpack(Buffer& buffer)
    {
        using size_type = typename attribute_container_t::size_type;
        for (size_type j = 0; j < attributes_m.size(); ++j) {
            attributes_m[j]->unpack(buffer.getAttribute(j));
        }
    }
}
