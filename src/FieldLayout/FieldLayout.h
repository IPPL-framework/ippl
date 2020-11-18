//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks. It performs the initial
//   partitioning. The user may request that a particular dimension not be
//   partitioned by flagging that axis as 'SERIAL' (instead of 'PARALLEL').
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_FIELD_LAYOUT_H
#define IPPL_FIELD_LAYOUT_H

#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"

#include <iostream>

namespace ippl {

    template <unsigned Dim> class FieldLayout;

    template <unsigned Dim>
    std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

    // enumeration used to select serial or parallel axes
    enum e_dim_tag { SERIAL=0, PARALLEL=1 } ;


    template<unsigned Dim>
    class FieldLayout
    {

    public:
        using NDIndex_t = NDIndex<Dim>;
        using view_type = typename detail::ViewType<NDIndex_t, 1>::view_type;
        using host_mirror_type = typename view_type::host_mirror_type;


        /*!
         * Default constructor, which should only be used if you are going to
         * call 'initialize' soon after (before using in any context)
         */
        FieldLayout();

        FieldLayout(const NDIndex<Dim>& domain, e_dim_tag *p=0) {
            initialize(domain, p);
        }

        // Destructor: Everything deletes itself automatically ... the base
        // class destructors inform all the FieldLayoutUser's we're going away.
        virtual ~FieldLayout();

        // Initialization functions, only to be called by the user of FieldLayout
        // objects when the FieldLayout was created using the default constructor;
        // otherwise these are only called internally by the various non-default
        // FieldLayout constructors:

        void initialize(const NDIndex<Dim>& domain, e_dim_tag *p=0);


        // Return the domain.
        const NDIndex<Dim>& getDomain() const { return gDomain_m; }

        // Compare FieldLayouts to see if they represent the same domain; if
        // dimensionalities are different, the NDIndex operator==() will return
        // false:
        template <unsigned Dim2>
        bool operator==(const FieldLayout<Dim2>& x) const {
            return gDomain_m == x.getDomain();
        }

        // for the requested dimension, report if the distribution is
        // SERIAL or PARALLEL
        e_dim_tag getDistribution(unsigned int d) const {
            e_dim_tag retval = PARALLEL;
            if (minWidth_m[d] == (unsigned int) gDomain_m[d].length())
                retval = SERIAL;
            return retval;
        }

        // for the requested dimension, report if the distribution was requested to
        // be SERIAL or PARALLEL
        e_dim_tag getRequestedDistribution(unsigned int d) const {
            return requestedLayout_m[d];
        }

        const NDIndex_t& getLocalNDIndex(int rank = Ippl::Comm->rank()) const;

    void write(std::ostream& = std::cout) const;

    private:
        //! Global domain
        NDIndex_t gDomain_m;

        //! Local domains (device view)
        view_type dLocalDomains_m;

        //! Local domains (host mirror view)
        host_mirror_type hLocalDomains_m;

        e_dim_tag requestedLayout_m[Dim];

        unsigned int minWidth_m[Dim];

        void calcWidths();
    };


    template<unsigned Dim>
    inline
    std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
        f.write(out);
        return out;
    }
}


#include "FieldLayout/FieldLayout.hpp"

#endif