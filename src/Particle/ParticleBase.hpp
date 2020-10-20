/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/


namespace ippl {

    template<class PLayout, class... Properties>
    ParticleBase<PLayout, Properties...>::ParticleBase()
    : ParticleBase(nullptr)
    { }

    template<class PLayout, class... Properties>
    ParticleBase<PLayout, Properties...>::ParticleBase(std::shared_ptr<PLayout>& layout)
    : ParticleBase()
    {
        initialize(layout);
    }

    template<class PLayout, class... Properties>
    ParticleBase<PLayout, Properties...>::ParticleBase(std::shared_ptr<PLayout>&& layout)
    : layout_m(std::move(layout))
    , localNum_m(0)
    , destroyNum_m(0)
    , attributes_m(0)
    , nextID_m(Ippl::Comm->myNode())
    , numNodes_m(Ippl::Comm->getNodes())
    {
        addAttribute(ID); // needs to be added first due to destroy function
        addAttribute(R);
    }


    template<class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::addAttribute(ParticleAttribBase<Properties...>& pa)
    {
        attributes_m.push_back(&pa);
    }

    template<class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::initialize(std::shared_ptr<PLayout>& layout)
    {
        PAssert(layout != nullptr);

        // save the layout, and perform setup tasks
        layout_m = std::move(layout);
    }


    template<class PLayout, class... Properties>
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

    template<class PLayout, class... Properties>
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

    template<class PLayout, class... Properties>
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


    template<class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::destroy() {

        /* count the number of particles with ID == -1 and fill
         * a boolean view
         */
        Kokkos::View<bool*> invalidIndex("", localNum_m);
        Kokkos::parallel_reduce("Reduce in ParticleBase::destroy()",
                                localNum_m,
                                KOKKOS_CLASS_LAMBDA(const size_t i,
                                                    size_t& nInvalid)
                                {
                                    nInvalid += size_t(ID(i) < 0);
                                    invalidIndex(i) = (ID(i) < 0);
                                }, destroyNum_m);

        PAssert(destroyNum_m <= localNum_m);

        if (destroyNum_m == 0) {
            return;
        }

        /* Compute the prefix sum and store the new
         * particle indices in newIndex.
         */
        Kokkos::View<int*> newIndex("newIndex", localNum_m);
        Kokkos::parallel_scan("Scan in ParticleBase::destroy()",
                              localNum_m,
                              KOKKOS_LAMBDA(const int i, int& idx, const bool final)
                              {
                                  if (final) {
                                      newIndex(i) = idx;
                                  }

                                  if (!invalidIndex(i)) {
                                      idx += 1;
                                  }
                              });

        localNum_m -= destroyNum_m;

        // delete the invalide attribut indices
        for (attribute_iterator it = attributes_m.begin();
             it != attributes_m.end(); ++it)
        {
            (*it)->destroy(invalidIndex, newIndex, localNum_m);
        }
    }

//
//     /////////////////////////////////////////////////////////////////////
//     // delete M particles, starting with the Ith particle.  If the last argument
//     // is true, the destroy will be done immediately, otherwise the request
//     // will be cached.
//     template<class PLayout, class... Properties>
//     void ParticleBase<PLayout, Properties...>::destroy(size_t M, size_t I, bool doNow) {
//
//     // make sure we've been initialized
//     PAssert(Layout != 0);
//
//     if (M > 0) {
//         if (doNow) {
//         // find out if we are using optimized destroy method
//         bool optDestroy = getUpdateFlag(PLayout::OPTDESTROY);
//         // loop over attributes and carry out the destroy request
//         attrib_container_t::iterator abeg, aend = AttribList.end();
//         for (abeg = AttribList.begin(); abeg != aend; ++abeg)
//             (*abeg)->destroy(M,I,optDestroy);
//         LocalNum -= M;
//         }
//         else {
//         // add this group of particle indices to our list of items to destroy
//         std::pair<size_t,size_t> destroyEvent(I,M);
//         DestroyList.push_back(destroyEvent);
//         DestroyNum += M;
//         }
//
//         // remember we have this many more items to destroy (or have destroyed)
//         ADDIPPLSTAT(incParticlesDestroyed,M);
//     }
//     }
//
//
    template<class PLayout, class... Properties>
    void ParticleBase<PLayout, Properties...>::update()
    {
        PAssert(layout_m != nullptr);
        layout_m->update(*this);
    }
}
