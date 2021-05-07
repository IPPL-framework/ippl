#ifndef IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H
#define IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

/*
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
        using view_type_t = typename detail::ViewType<NDIndex<Dim>, 1>::view_type;
        using view_type = typename detail::ViewType<T, Dim>::view_type;
        using host_mirror_type = typename view_type_t::host_mirror_type;

        /*!
          @param const ParticleAttrib<Vector<T,Dim>>& R particle positions
          @param Field<T,Dim,M>& BF particle density
          @param FieldLayout<Dim>& FL
  
          1. Performs scatter operation of particle positions in field
          2. Updates field layout by calling repartition on field
        */
        bool BinaryRepartition(const ParticleAttrib<Vector<T,Dim>>& R, Field<T,Dim,M>& BF, FieldLayout<Dim>& FL, int step); 


        /*!
          @param FieldLayout<Dim>& FL
          @param Field<T, Dim>& BF

          Performs recursive binary repartition on field layout using a field of weights.
          - Start with whole domain
          - Find cut axis as the longest axis
          - Perform reduction on all dimensions perp. to the cut axis
          - Find median of reduced weights
          - Divide field at median
        */
        bool CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF, int step);


        /*!
          @param NDIndex<Dim>& domain
  
          Find cutting axis as the longest axis of the field layout.
        */
         int FindCutAxis(NDIndex<Dim>& domain); 


        /*!
          @param Field<T, Dim>& BF field of weights
          @param std::vector<T>& res result of reduction
          @param NDIndex<Dim>& dom domain to reduce
          @param int cutAxis

          Performs reduction on field BF in all dimension except that determined by cutAxis,
          store result in res.  
        */
        void PerformReduction(Field<T,Dim>& BF, std::vector<T>& res, unsigned int cutAxis, NDIndex<Dim>& dom); 
 

        /*!
          @param std::vector<T>& V
 
          Find median of array V
        */
        int FindMedian(std::vector<T>& V);


        /*!
          @param std::vector<NDIndex<Dim>>& domains
          @param std::vector<int>& procs
          @param int it iterator
          @param int cutAxis
          @param int median

          Cut field layout along the cut axis at the median
        */
        void CutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it, int cutAxis, int median);
 
        
        /*!
          @param Field<T,Dim,M>& f
          @param const ParticleAttrib<Vector<T,Dim>>& pr particle positions

          Scattering of particle positions in field using a CIC method
        */
        void scatterR(Field<T,Dim,M>& f, const ParticleAttrib<Vector<T,Dim>>& pr);

    }; // class

} // namespace


#include "Decomposition/OrthogonalRecursiveBisection.hpp"

#endif // IPPL_ORTHOGONAL_RECURSIVE_BISECTION_H

