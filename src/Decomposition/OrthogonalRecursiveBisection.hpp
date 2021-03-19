// include files
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Region/RegionLayout.h"
#include "Index/NDIndex.h"
#include "FieldLayout/FieldLayout.h"
#include "Region/NDRegion.h"

namespace ippl {

template < class T, unsigned Dim, class Mesh>
class OrthogonalRecursiveBisection {
bool
NBinaryRepartition(ParticleBase<ParticleSpatialLayout<T,Dim,M> >& P) {
   // Define field layout on which we will make the cuts
   const ippl::detail::RegionLayout<T, Dim, M>& RL = P.getLayout().getRegionLayout();
   
   // const NDIndex<Dim>& getDomain() const { return gDomain_m; }
   // const NDRegion_t& getDomain() const { return region_m; }
   
   // The field layout in P is transformed in a NDRegion in the region layout
   const ippl::NDRegion<T,Dim>& region = RL.getDomain();   

   // std::cout << FL << std::endl;
    

   /* ------------------------------ */
   FieldLayout<Dim>& FL = P.getLayout().getFieldLayout(); // ***Gets a warning about host__function called from host__device__function
   Mesh<T,Dim> mesh = P.getLayout().getMesh();
   // Field<T,Dim> BF(mesh, FL); // ***This doesn't work, problem with mesh, and UniformCartesian 
   
   // Scattering of particles in Field
   // ***The scatter methods are all commented in Interpolator.h... needs new writing?

   // Domain Decomposition
   // NCalcBinaryRepartition(FL, BF); 
   

   // Update particles
   // ...


   return true;
}

void
NCalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF) {
   // Start with whole domain
   // ippl::NDIndex<Dim> fdomain = FL.getDomain(); 
   
   // Find cut axis
   // int cutAxis = NFindCutAxis(FL);

   // Peform reduction with field of weights
   // std::vector<double> reduced;
   // NPerformReduction(BF, reduced, cutAxis); 
   
   // Find median of reduced weights
   // int median = NFindMedian(reduced);

   // Cut domain
}


int
NFindCutAxis(FieldLayout<Dim>& FL) {
   return 0;
} 


void
NPerformReduction(Field<T, Dim> BF, std::vector<T>& res, int cutAxis) {

}


int
NFindMedian(std::vector<T>& V) {
   return 0;
}



void
NCutDomain(FieldLayout<Dim>& FL, int cutAxis, int median) {

}

}; // class
}  // namespace
