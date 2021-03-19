// include files
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Region/RegionLayout.h"
#include "Index/NDIndex.h"
#include "FieldLayout/FieldLayout.h"

template < class T, unsigned Dim, class Mesh>
bool
NBinaryRepartition(ippl::ParticleBase<ippl::ParticleSpatialLayout<T,Dim,Mesh> >& P) {
   bool success = true;
  
   // Define field layout on which we will make the cuts
   // Layout_t& getLayout() { return *layout_m; }
   // const RegionLayout_t& getRegionLayout() const { return rlayout_m; }
   const ippl::detail::RegionLayout<T, Dim, Mesh>& RL = P.getLayout().getRegionLayout();
   /* 
   if (!RL.initialized())
      return false;
   
   ippl::FieldLayout<Dim> FL = P.getFieldLayout();

   // std::cout << FL << std::endl;
   */

   int localNumber = P.getLocalNum();

   return success;
}

template<typename T, unsigned Dim>
void
NCalcBinaryRepartition(ippl::FieldLayout<Dim>& FL, ippl::Field<T, Dim>& BF) {

}


template<unsigned Dim>
int
NFindCutAxis(ippl::FieldLayout<Dim>& FL) {
   return 0;
} 


template<typename T, unsigned Dim>
void
NPerformReduction(ippl::Field<T, Dim> BF, std::vector<T>& res, int cutAxis) {

}


template<typename T>
int
NFindMedian(std::vector<T>& V) {
   return 0;
}



template<unsigned Dim>
void
NCutDomain(ippl::FieldLayout<Dim>& FL, int cutAxis, int median) {

}
