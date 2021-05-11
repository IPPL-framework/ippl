#ifndef IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
#define IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

/*
 * Class ORB for Domain Decomposition
 *
  @file OrthogonalRecursiveBisection.h
*/

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
  
          1. Performs scatter operation of particle positions in field
          2. Updates field layout by calling repartition on field
        */
        bool binaryRepartition(const ParticleAttrib<Vector<T,Dim>>& R, FieldLayout<Dim>& fl, int step); 


        /*!
          @param FieldLayout<Dim>& fl

          Performs recursive binary repartition on field layout using a field of weights.
          - Start with whole domain
          - Find cut axis as the longest axis
          - Perform reduction on all dimensions perp. to the cut axis
          - Find median of reduced weights
          - Divide field at median
        */
        bool calcBinaryRepartition(FieldLayout<Dim>& fl, int step);


        /*!
          @param NDIndex<Dim>& domain
  
          Find cutting axis as the longest axis of the field layout.
        */
         int findCutAxis(NDIndex<Dim>& domain); 


        /*!
          @param std::vector<T>& res result of reduction
          @param NDIndex<Dim>& dom domain to reduce
          @param int cutAxis

          Performs reduction on field BF in all dimension except that determined by cutAxis,
          store result in res.  
        */
        void performReduction(std::vector<T>& res, unsigned int cutAxis, NDIndex<Dim>& dom); 
 

        /*!
          @param std::vector<T>& w
 
          Find median of array w
        */
        int findMedian(std::vector<T>& w);


        /*!
          @param std::vector<NDIndex<Dim>>& domains
          @param std::vector<int>& procs
          @param int it iterator
          @param int cutAxis
          @param int median

          Cut field layout along the cut axis at the median
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

