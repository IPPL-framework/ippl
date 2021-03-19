#ifndef NPARTICLE_BALANCER_H
#define NPARTICLE_BALANCER_H


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
template<class T, unsigned Dim, class Mesh>
bool
NBinaryRepartition(ippl::ParticleBase<ippl::ParticleSpatialLayout<T,Dim,Mesh> >& P); 


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
template<typename T, unsigned Dim>
void
NCalcBinaryRepartition(ippl::FieldLayout<Dim>& FL, ippl::Field<T, Dim>& BF);


/*!
  @tparam Dim dimension
  @param FieldLayout<Dim>& FL
  
  Comment: Another domain as parameter needed?

  Find cutting axis as the longest axis of the field layout.
*/
template<unsigned Dim>
int
NFindCutAxis(ippl::FieldLayout<Dim>& FL); 


/*!
  @tparam T
  @tparam Dim dimension
  @param Field<T, Dim>& BF
  @param std::vector<T>& res
  @param int cutAxis

  Performs reduction on field BF in all dimension except that determined by cutAxis,
  store result in res.  
*/
template<typename T, unsigned Dim>
void
NPerformReduction(ippl::Field<T, Dim> BF, std::vector<T>& res, int cutAxis); 


/*!
  @tparam T
  @param std::vector<T>& V
 
  Find median of array V
*/
template<typename T>
int
NFindMedian(std::vector<T>& V);


/*!
  @tparam Dim dimension
  @param FieldLayout<Dim>& FL
  @param int cutAxis
  @param int median

  Comment: perhaps also pass the procs...

  Cut field layout along the cut axis at the median
*/
template<unsigned Dim>
void
NCutDomain(ippl::FieldLayout<Dim>& FL, int cutAxis, int median);






















#include "Decomposition/Decomposition.hpp"

#endif // NPARTICLE_BALANCER_H

