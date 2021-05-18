//
// Class ORB for Domain Decomposition
//
// Simple domain decomposition using an Orthogonal Recursive Bisection,
// domain is divided recursively so as to even weights on each side of the cut,
// works with 2^n processors only. 
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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

#ifndef IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
#define IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Index/NDIndex.h"
#include "Index/Index.h"
#include "FieldLayout/FieldLayout.h"
#include "Region/NDRegion.h"
#include <mpi.h>
#include <fstream>

namespace ippl {

    /*
      @class OrthogonalRecursiveBisection
      @tparam T
      @tparam Dim dimension
      @tparam M mesh
    */
    template<class T, unsigned Dim, class M>
    class OrthogonalRecursiveBisection {
    public:
        using view_type = typename detail::ViewType<T, Dim>::view_type;

        // Weight for reduction
        Field<T,Dim> bf_m;

    public:

        /*!
          @param FieldLayout<Dim>& fl
          @param UniformCartesian<T,Dim>& mesh

          Initialize member field with mesh and field layout
        */    
        void initialize(FieldLayout<Dim>& fl, UniformCartesian<T,Dim>& mesh);


        /*!
          @param const ParticleAttrib<Vector<T,Dim>>& R particle positions
          @param FieldLayout<Dim>& fl
 
          - Performs scatter operation of particle positions in field (weights)
          - Repartition FieldLayout's global domain
        */
        bool binaryRepartition(const ParticleAttrib<Vector<T,Dim>>& R, FieldLayout<Dim>& fl, int step); 


        /*!
          @param NDIndex<Dim>& dom domain to reduce
  
          Find cutting axis as the longest axis of the field layout.
        */
         int findCutAxis(NDIndex<Dim>& dom); 


        /*!
          @param std::vector<T>& res result of reduction
          @param NDIndex<Dim>& dom domain to reduce
          @param int cutAxis

          Performs reduction on local field in all dimension except that determined by cutAxis,
          store result in res.  
        */
        void performReduction(std::vector<T>& res, unsigned int cutAxis, NDIndex<Dim>& dom); 
 

        /*!
          @param std::vector<T>& w
 
          Find median of array w, 
          does not return indices that would lead to domains of size 1 
        */
        int findMedian(std::vector<T>& w);


        /*!
          @param std::vector<NDIndex<Dim>>& domains
          @param std::vector<int>& procs
          @param int it iterator
          @param int cutAxis
          @param int median

          Split the domain given by the iterator along the cut axis at the median,
          the corresponding index will be cut between median and median+1
        */
        void cutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it, int cutAxis, int median);
 
        
        /*!
          @param const ParticleAttrib<Vector<T,Dim>>& r particle positions

          Scattering of particle positions in field using a CIC method
        */
        void scatterR(const ParticleAttrib<Vector<T,Dim>>& r);

    }; // class

} // namespace


#include "Decomposition/OrthogonalRecursiveBisection.hpp"

#endif // IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

