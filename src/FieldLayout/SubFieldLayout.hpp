//
// Class SubFieldLayout
// SubFieldLayout provides a layout for a sub-region of a larger field.
// It ensures that the sub-region is partitioned in the same way as the original FieldLayout,
// maintaining consistent parallel decomposition and neighbor relationships within the sub-region.
//
#include "Ippl.h"

#include <cstdlib>
#include <limits>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include "Utility/PAssert.h"

#include "FieldLayout/SubFieldLayout.h"

namespace ippl {

    /**
     * @copydoc SubFieldLayout::SubFieldLayout(const mpi::Communicator&)
     * 
     * Creates a SubFieldLayout without specifying domains. The layout must be initialized
     * later using the initialize() methods. This constructor is useful when you need to
     * defer the layout configuration until more information is available.
     */
    template <unsigned Dim>
    SubFieldLayout<Dim>::SubFieldLayout(const mpi::Communicator& communicator)
        : FieldLayout<Dim>(communicator) {}

    /**
     * @copydoc SubFieldLayout::SubFieldLayout(mpi::Communicator, const NDIndex<Dim>&, const NDIndex<Dim>&, std::array<bool, Dim>, bool)
     * 
     * Implementation details:
     * Initializes both the full domain decomposition and the sub-domain layout. The sub-domain
     * must be contained within the full domain. All MPI ranks must have non-empty local domains
     * after intersection with the sub-domain, otherwise an exception will be thrown.
     * 
     * This constructor sets up the parallel decomposition based on the full domain, then
     * restricts the full domain to the specified sub-region while maintaining the same
     * partitioning structure.
     */
    template <unsigned Dim>
    SubFieldLayout<Dim>::SubFieldLayout(mpi::Communicator communicator, const NDIndex<Dim>& domain,
                                  const NDIndex<Dim>& subDomain, std::array<bool, Dim> isParallel, bool isAllPeriodic)
        : FieldLayout<Dim>(communicator) {
        initialize(domain, subDomain, isParallel, isAllPeriodic);
    }

    /**
     * @copydoc SubFieldLayout::SubFieldLayout(mpi::Communicator, const NDIndex<Dim>&, std::array<bool, Dim>, bool)
     * 
     * Implementation details:
     * Creates a SubFieldLayout where the sub-domain is the same as the full domain, making it
     * functionally equivalent to a regular FieldLayout.
     */
    template <unsigned Dim>
    SubFieldLayout<Dim>::SubFieldLayout(mpi::Communicator communicator, const NDIndex<Dim>& domain, 
                                  std::array<bool, Dim> isParallel, bool isAllPeriodic)
        : FieldLayout<Dim>(communicator) {
        initialize(domain, isParallel, isAllPeriodic);
    }

    /**
     * @copydoc SubFieldLayout::initialize(const NDIndex<Dim>&, const NDIndex<Dim>&, std::array<bool, Dim>, bool)
     * 
     * Implementation details:
     * This method first partitions the full domain for parallel processing,
     * then restricts each rank's local domain to the specified sub-domain.
     * 
     * The sub-domain must be contained within the full domain, and all MPI ranks must have 
     * non-empty local domains after intersection with the sub-domain, otherwise an exception
     * will be thrown.
     */
    template <unsigned Dim>
    void SubFieldLayout<Dim>::initialize(const NDIndex<Dim>& domain, const NDIndex<Dim>& subDomain, std::array<bool, Dim> isParallel,
                                      bool isAllPeriodic) {

        // Ensure the sub-domain is contained within the main domain
        PAssert(domain.contains(subDomain));

        // Call the base class initialize method to set up the main domain and parallel decomposition
        FieldLayout<Dim>::initialize(domain, isParallel, isAllPeriodic);

        unsigned int nRanks = this->comm.size();

        originDomain_m = domain;

        this->gDomain_m = subDomain;
        
        // Check if all ranks have a valid local domain that intersects with the sub-domain
        if (this->hLocalDomains_m(this->comm.rank()).intersect(subDomain).empty()) {
            throw std::runtime_error("SubFieldLayout:initialize: given subdomain is not valid, rank"
                + std::to_string(this->comm.rank()) + " has an empty local domain, choose a sub-domain that has content on all ranks");
        }

        // If the local domain is not contained in the sub-domain, change it to the intersection of the local domain and the sub-domain
        // This ensures that the sub-field layout is consistent with the original layout
        for (unsigned int rank = 0; rank < nRanks; ++rank) {
            if (!this->gDomain_m.contains(this->hLocalDomains_m(rank))) {
                this->hLocalDomains_m(rank) = this->hLocalDomains_m(rank).intersect(this->gDomain_m);
            }
        }

        this->findNeighbors();

        Kokkos::deep_copy(this->dLocalDomains_m, this->hLocalDomains_m);

        this->calcWidths();
    }

    /**
     * @copydoc SubFieldLayout::initialize(const NDIndex<Dim>&, std::array<bool, Dim>, bool)
     * 
     * Implementation details:
     * This method initializes the layout to use the entire domain as both the full domain
     * and the sub-domain, making it equivalent to a regular FieldLayout.
     */
    template <unsigned Dim>
    void SubFieldLayout<Dim>::initialize(const NDIndex<Dim>& domain, std::array<bool, Dim> isParallel, 
                                         bool isAllPeriodic) {
        // Call the base class initialize method to set up the main domain and parallel decomposition
        FieldLayout<Dim>::initialize(domain, isParallel, isAllPeriodic);

        originDomain_m = domain;
    }

}  // namespace ippl