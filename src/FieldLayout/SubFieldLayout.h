//
// Class SubFieldLayout
// SubFieldLayout provides a layout for a sub-region of a larger field.
// It ensures that the sub-region is partitioned in the same way as the original FieldLayout,
// maintaining consistent parallel decomposition and neighbor relationships within the sub-region.
//
#ifndef IPPL_SUB_FIELD_LAYOUT_H
#define IPPL_SUB_FIELD_LAYOUT_H

#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "Types/ViewTypes.h"

#include "Communicate/Communicator.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {

    /**
     * @class SubFieldLayout
     * @brief SubFieldLayout provides a layout for a sub-region of a larger field
     * 
     * SubFieldLayout extends FieldLayout to handle sub-regions of a larger computational domain.
     * It ensures that the sub-region is partitioned in the same way as the original FieldLayout,
     * maintaining consistent parallel decomposition and neighbor relationships within the sub-region.
     * 
     * @par Important Constraint:
     * SubFieldLayout only allows for sub-layouts that do NOT leave local domains empty.
     * All MPI ranks must have at least some portion of the sub-domain assigned to them.
     * If a sub-domain would result in empty local domains for some ranks, an exception
     * will be thrown during initialization.
     * 
     * @tparam Dim Number of spatial dimensions
     */
    template <unsigned Dim>
    class SubFieldLayout : public FieldLayout<Dim> {
    public:
        using NDIndex_t        = NDIndex<Dim>;
        using view_type        = typename detail::ViewType<NDIndex_t, 1>::view_type;
        using host_mirror_type = typename view_type::host_mirror_type;

        /**
         * @brief Default constructor, which should only be used if you are going to
         * call 'initialize' soon after (before using in any context)
         * 
         * @param communicator MPI communicator to use (defaults to MPI_COMM_WORLD)
         */
        SubFieldLayout(const mpi::Communicator& = MPI_COMM_WORLD);

        /**
         * @brief Constructor that creates a SubFieldLayout for a sub-region of a larger domain
         * 
         * @param communicator MPI communicator to use
         * @param domain The full domain that defines the partitioning
         * @param subDomain The sub-region within the full domain, which is partitioned in the same way as the fullomain
         * @param decomp Array specifying which dimensions should be parallel
         * @param isAllPeriodic Whether all dimensions have periodic boundary conditions
         */
        SubFieldLayout(mpi::Communicator, const NDIndex<Dim>& domain, const NDIndex<Dim>& subDomain, std::array<bool, Dim> decomp,
                    bool isAllPeriodic = false);

        /**
         * @brief Constructor for full-domain layout.
         * 
         * @param communicator MPI communicator to use
         * @param domain The full domain that defines the partitioning and is used as the sub-domain simultaneously
         * @param decomp Array specifying which dimensions should be parallel
         * @param isAllPeriodic Whether all dimensions have periodic boundary conditions
         */
        SubFieldLayout(mpi::Communicator, const NDIndex<Dim>& domain, std::array<bool, Dim> decomp,
                    bool isAllPeriodic = false);

        /**
         * @brief Destructor: Everything deletes itself automatically
         */
        virtual ~SubFieldLayout() = default;
        
        /**
         * @brief Initializes a SubFieldLayout with the sub-domain partitioned in the same way
         * as the original FieldLayout partitiones the full domain
         * 
         * @param domain The full domain to be partitioned
         * @param subDomain The sub-region within the full domain
         * @param decomp Array specifying which dimensions should be parallel
         * @param isAllPeriodic Whether all dimensions have periodic boundary conditions
         */
        void initialize(const NDIndex<Dim>& domain, const NDIndex<Dim>& subDomain, std::array<bool, Dim> decomp,
            bool isAllPeriodic = false);
            
        /**
         * @brief Initializes a SubFieldLayout using the domain as both the full domain and sub-domain
         * 
         * @param domain The domain to be partitioned
         * @param decomp Array specifying which dimensions should be parallel
         * @param isAllPeriodic Whether all dimensions have periodic boundary conditions
         */
        void initialize(const NDIndex<Dim>& domain, std::array<bool, Dim> decomp,
            bool isAllPeriodic = false);

        /**
         * @brief Return the original domain before sub-region extraction
         * 
         * @return Reference to the original full domain
         */
        const NDIndex<Dim>& getOriginDomain() const { return originDomain_m; }
                
        /**
         * @brief Compare SubFieldLayouts to see if they represent the same domain
         * 
         * @tparam Dim2 Dimension of the other SubFieldLayout
         * @param x The other SubFieldLayout to compare with
         * @return true if both the current domain, origin domain and local domains match
         */
        template <unsigned Dim2>
        bool operator==(const SubFieldLayout<Dim2>& x) const {
            // Ensure the dimensions match
            if (Dim2 != Dim) {
                return false;
            }

            // Check if the original and global domains match
            if (originDomain_m != x.getOriginDomain() || this->gDomain_m != x.getDomain()) {
                return false;
            }

            // Ensure the local domains match
            for (unsigned int rank = 0; rank < this->comm.size(); ++rank) {
                if (this->hLocalDomains_m(rank) != x.getLocalNDIndex(rank)) {
                    return false;
                }
            }

            // If all checks passed, the SubFieldLayouts matche
            return true;
        }

        /**
         * @brief Compare SubFieldLayout to a FieldLayout to see if they represent the same domain
         * 
         * @tparam Dim2 Dimension of the FieldLayout
         * @param x The FieldLayout to compare with
         * @return true if the SubFieldLayout's domain equals its original domain and matches the FieldLayout's domain and local domains
         */
        template <unsigned Dim2>
        bool operator==(const FieldLayout<Dim2>& x) const {
            // Ensure the dimensions match
            if (Dim2 != Dim) {
                return false;
            }

            // Check if the global domain matches the original domain and the FieldLayout's domain
            if (this->gDomain_m != originDomain_m || this->gDomain_m != x.getDomain()) {
                return false;
            }

            // Ensure the local domains match
            for (unsigned int rank = 0; rank < this->comm.size(); ++rank) {
                if (this->hLocalDomains_m(rank) != x.getLocalNDIndex(rank)) {
                    return false;
                }
            }

            // If all checks passed, the SubFieldLayout matches the FieldLayout
            return true;
        }

    private:
        /**
         * @brief Original global domain in which the sub-field is defined
         * 
         * This stores the full domain before any sub-region extraction,
         * allowing comparison with regular FieldLayouts.
         */
        NDIndex_t originDomain_m;
    };
}  // namespace ippl

#include "FieldLayout/SubFieldLayout.hpp"

#endif
