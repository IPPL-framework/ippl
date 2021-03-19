#ifndef NPARTICLE_BALANCER_H
#define NPARTICLE_BALANCER_H

/*
  @file OrthogonalRecursiveBisection.h
*/

namespace ippl {

/*
  @class OrthogonalRecursiveBisection
*/
// template<class T, unsigned Dim, class M = UniformCartesian<T,Dim> >
template<class T, unsigned Dim, class M>
class OrthogonalRecursiveBisection {
public:
/*!
  @tparam T 
  @tparam Dim dimension
  @tparam Mesh
  @param ParticleBase<ParticleSpatialLayout<T,Dim,Mesh>>& P
  
  Performs scatter operation of particles into nodes.
  - Define a field layout FL (copy of one in P)
  - Create a field BF with using the mesh of P and the field layout
  - Scatter particles into field FL (MPI)
  - Repartition the field FL
  - Update P using the field FL
*/
static bool BinaryRepartition(ParticleBase<ParticleSpatialLayout<T,Dim,M> >& P); 


/*!
  @tparam T
  @tparam Dim dimension
  @param FieldLayout<Dim>& FL
  @param Field<T, Dim>& BF

  Performs recursive binary repartition on field layout using a field of weights.
  - Start with whole domain
  - Find cut axis as the longest axis
  - Perform reduction on all dimensions perp. to the cut axis
  - Find median of reduced weights
  - Divide field at median
*/
void CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF);


/*!
  @tparam Dim dimension
  @param FieldLayout<Dim>& FL
  
  Comment: Another domain as parameter needed?

  Find cutting axis as the longest axis of the field layout.
*/
int FindCutAxis(FieldLayout<Dim>& FL); 


/*!
  @tparam T
  @tparam Dim dimension
  @param Field<T, Dim>& BF
  @param std::vector<T>& res
  @param int cutAxis

  Performs reduction on field BF in all dimension except that determined by cutAxis,
  store result in res.  
*/
void PerformReduction(Field<T, Dim> BF, std::vector<T>& res, int cutAxis); 


/*!
  @tparam T
  @param std::vector<T>& V
 
  Find median of array V
*/
int FindMedian(std::vector<T>& V);


/*!
  @tparam Dim dimension
  @param FieldLayout<Dim>& FL
  @param int cutAxis
  @param int median

  Comment: perhaps also pass the procs...

  Cut field layout along the cut axis at the median
*/
void CutDomain(FieldLayout<Dim>& FL, int cutAxis, int median);

static void nTest() {
   std::cout << "Hello" << std::endl;
}

}; // class
} // namespace


















// #include "Decomposition/Decomposition.hpp"

#endif // NPARTICLE_BALANCER_H

