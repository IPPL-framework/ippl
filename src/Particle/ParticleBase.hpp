//
// Class ParticleBase
//   Base class for all user-defined particle classes.
//
//   ParticleBase is a container and manager for a set of particles.
//   The user must define a class derived from ParticleBase which describes
//   what specific data attributes the particle has (e.g., mass or charge).
//   Each attribute is an instance of a ParticleAttribute<T> class; ParticleBase
//   keeps a list of pointers to these attributes, and performs particle creation
//   and destruction.
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
        pa.setParticleCount(localNum_m);
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::initialize(PLayout& layout)
    {
//         PAssert(layout_m == nullptr);

        // save the layout, and perform setup tasks
        layout_m = &layout;
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::create(size_type nLocal)
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
    void ParticleBase<PLayout, Properties...>::globalCreate(size_type nTotal)
    {
        PAssert(layout_m != nullptr);

        // Compute the number of particles local to each processor
        size_type nLocal = nTotal / numNodes_m;

        const size_t rank = Ippl::Comm->myNode();

        size_type rest = nTotal - nLocal * rank;
        if (rank < rest)
            ++nLocal;

        create(nLocal);
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::destroy(const Kokkos::View<bool*>& invalid, const size_type destroyNum) {
        PAssert(destroyNum <= localNum_m);

        // If there aren't any particles to delete, do nothing
        if (destroyNum == 0) return;

        // If we're deleting all the particles, there's no point in doing
        // anything because the valid region will be empty; we only need to
        // update the particle count
        if (destroyNum == localNum_m) {
            localNum_m = 0;
            return;
        }

        // Resize buffers, if necessary
        if (deleteIndex_m.size() < destroyNum) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            Kokkos::realloc(deleteIndex_m, destroyNum * overalloc);
            Kokkos::realloc(keepIndex_m, destroyNum * overalloc);
        }

        // Reset index buffer
        Kokkos::deep_copy(deleteIndex_m, -1);

        auto locDeleteIndex = deleteIndex_m;
        auto locKeepIndex = keepIndex_m;

        // Find the indices of the invalid particles in the valid region
        Kokkos::parallel_scan("Scan in ParticleBase::destroy()",
                              localNum_m - destroyNum,
                              KOKKOS_LAMBDA(const size_t i, int& idx, const bool final)
                              {
                                  if (final && invalid(i)) locDeleteIndex(idx) = i;
                                  if (invalid(i)) idx += 1;
                              });
        Kokkos::fence();

        // Determine the total number of invalid particles in the valid region
        size_type maxDeleteIndex = 0;
        Kokkos::parallel_reduce("Reduce in ParticleBase::destroy()", destroyNum,
                               KOKKOS_LAMBDA(const size_t i, size_t& maxIdx)
                               {
                                   if (locDeleteIndex(i) >= 0 && i > maxIdx) maxIdx = i;
                               }, Kokkos::Max<size_type>(maxDeleteIndex));

        // Find the indices of the valid particles in the invalid region
        Kokkos::parallel_scan("Second scan in ParticleBase::destroy()",
                              Kokkos::RangePolicy(localNum_m - destroyNum, localNum_m),
                              KOKKOS_LAMBDA(const size_t i, int& idx, const bool final)
                              {
                                  if (final && !invalid(i)) locKeepIndex(idx) = i;
                                  if (!invalid(i)) idx += 1;
                              });

        Kokkos::fence();

        localNum_m -= destroyNum;

        // Partition the attributes into valid and invalid regions
        for (attribute_iterator it = attributes_m.begin();
             it != attributes_m.end(); ++it)
        {
            (*it)->destroy(deleteIndex_m, keepIndex_m, maxDeleteIndex + 1);
        }
    }

    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::serialize(detail::Archive<Properties...>& ar, size_type nsends) {
        using size_type = typename attribute_container_t::size_type;
        for (size_type i = 0; i < attributes_m.size(); ++i) {
            attributes_m[i]->serialize(ar, nsends);
        }
    }


    template <class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::deserialize(detail::Archive<Properties...>& ar, size_type nrecvs) {
        using size_type = typename attribute_container_t::size_type;
        for (size_type i = 0; i < attributes_m.size(); ++i) {
            attributes_m[i]->deserialize(ar, nrecvs);
        }
    }

    template <class PLayout, class... Properties>
    detail::size_type ParticleBase<PLayout, Properties...>::packedSize(const size_type count) const {
        size_type total = 0;
        // Vector size type
        using vsize_t = typename attribute_container_t::size_type;
        for (vsize_t i = 0; i < attributes_m.size(); ++i) {
            total += attributes_m[i]->packedSize(count);
        }
        return total;
    }

    template <class PLayout, class... Properties>
    template <class Buffer>
    void ParticleBase<PLayout, Properties...>::pack(Buffer& buffer,
                                                    const hash_type& hash)
    {
        // Vector size type
        using vsize_t = typename attribute_container_t::size_type;
        for (vsize_t j = 0; j < attributes_m.size(); ++j) {
            attributes_m[j]->pack(buffer.getAttribute(j), hash);
        }
    }


    template <class PLayout, class... Properties>
    template <class Buffer>
    void ParticleBase<PLayout, Properties...>::unpack(Buffer& buffer, size_type nrecvs)
    {
        // Vector size type
        using vsize_t = typename attribute_container_t::size_type;
        for (vsize_t j = 0; j < attributes_m.size(); ++j) {
            attributes_m[j]->unpack(buffer.getAttribute(j), nrecvs);
        }
        localNum_m += nrecvs;
    }
}
