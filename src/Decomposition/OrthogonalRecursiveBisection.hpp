// include files
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Region/RegionLayout.h"
#include "Index/NDIndex.h"
#include "Index/Index.h"
#include "FieldLayout/FieldLayout.h"
#include "Region/NDRegion.h"
#include <mpi.h>

namespace ippl {


    template < class T, unsigned Dim, class M>
    bool 
    OrthogonalRecursiveBisection<T,Dim,M>::BinaryRepartition(ParticleBase<ParticleSpatialLayout<T,Dim,M> >& P) {
       // Declaring field layout and field
       FieldLayout<Dim>& FL = P.getLayout().getFieldLayout();
       ParticleSpatialLayout<T,Dim,M> part = P.getLayout();
       UniformCartesian<T,Dim> mesh = P.getLayout().getMesh();
       Field<T,Dim,M> BF(mesh, FL);
   
       // Scattering of particle positions in field
       testScatter(BF, P.R);

       // Domain Decomposition
       if (Ippl::Comm->rank() == 0)
          CalcBinaryRepartition(FL, BF); 

       // Update particles
       std::cout << "ORB finished" << std::endl;

       return true;
    }

    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF) {
       int nprocs = Ippl::Comm->size();

       // Start with whole domain and total number of nodes
       view_type_t localDeviceDom = FL.getDeviceLocalDomains();
       host_mirror_type localHostDom = FL.getHostLocalDomains();
       std::vector<NDIndex<Dim>> domains = {FL.getLocalNDIndex()}; // Local domain
       std::vector<int> procs = {nprocs};

       /**PRINT**/
       std::cout << "FieldLayout.getLocalNDIndex(): " << domains[0] << std::endl;
       std::cout << "FieldLayout.getDeviceLocalDomains() (ghost): " << localDeviceDom.extent(0) << std::endl;
       std::cout << "FieldLayout.getHostLocalDomains() (ghost): " << localHostDom.extent(0) << std::endl;
       std::cout << "Number of nodes: " << nprocs << std::endl;
        
       // Start recursive repartition loop 
       int it = 0;
       // nprocs = 2; // shouldn't do this
       int maxprocs = nprocs; 
       while (maxprocs > 1) {
          // Find cut axis
          int cutAxis = FindCutAxis(domains[it]);
          
          /**PRINT**/
          std::cout << "cut axis: " << cutAxis << std::endl;
        
          // Peform reduction with field of weights
          std::vector<T> reduced;
          PerformReduction(BF, reduced, cutAxis); 

          /**PRINT**/
          std::cout << "reduced size: " << reduced.size() << std::endl;
          for (unsigned int i = 0; i < reduced.size(); i++)
             std::cout << "reduced[" << i << "]: " << reduced[i] << std::endl;
         
          // Find median of reduced weights
          int median = FindMedian(reduced);
       
          /**PRINT**/
          std::cout << "median: " << median << std::endl;
       
          // Cut domains and procs
          CutDomain(domains, procs, it, cutAxis, median);

          // Update max procs
          maxprocs = 0;
          for (unsigned int i = 0; i < procs.size(); i++) {
             if (procs[i] > maxprocs) {
                maxprocs = procs[i];
                it = i;
             } 
          }
       }
       
    }

    
    template < class T, unsigned Dim, class M>
    int
    OrthogonalRecursiveBisection<T,Dim,M>::FindCutAxis(NDIndex<Dim>& domain) {
       int cutAxis = 0;  
       unsigned int maxLength = 0;
       
       // Iterate along all the dimensions
       for (unsigned int d = 0; d < Dim; d++) {
          // Find longest domain size
          if (domain[d].length() > maxLength) {
             maxLength = domain[d].length();
             cutAxis = d;
          }
       }

       return cutAxis;
    } 

    
    // Comment: currently works for 3d only
    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T,Dim>& BF, std::vector<T>& res, unsigned int cutAxis) {
       // Return if Dim == 1
       if (Dim <= 1)
          return;

       using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
 
       // Get Field's weights
       const view_type& data = BF.getView();
       
       /***PRINT***/
       std::cout << "Field's domain: " << BF.getDomain() << std::endl;
       // std::cout << "BF.size(" << cutAxis << "): " << BF.size(cutAxis) << std::endl;
       std::cout << "BF.getView().extent(" << cutAxis << ") (size + 2 ghost): " << data.extent(cutAxis) << std::endl;
       
       // Iterate along cutAxis
       for (int i = 0; i < BF.size(cutAxis); i++) { 
          // Slicing view perpendincular to cutAxis
          auto data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
          switch (cutAxis) {
            case 1:
             data_i = Kokkos::subview(data, Kokkos::ALL, i, Kokkos::ALL);
             break;
            case 2:
             data_i = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, i);
             break;
          }
          // Perform reduction over weights
          T tempRes = T(0);
 
          /*
          Kokkos::parallel_for("Test for", mdrange_t({0,0},{data_i.extent(0), data_i.extent(1)}), 
                                                      KOKKOS_CLASS_LAMBDA (const int k, const int j) {
              data_i(k,j) = 1.0;       
          });*/
           
          Kokkos::parallel_reduce("Weight reduction", mdrange_t({0,0},{data_i.extent(0), data_i.extent(1)}),
                                                      KOKKOS_LAMBDA (const int k, const int j, T& weight) {
             weight += data_i(k,j);
          }, tempRes);

          Kokkos::fence();

          res.push_back(tempRes); 
       } 
    }


    template < class T, unsigned Dim, class M>
    int
    OrthogonalRecursiveBisection<T,Dim,M>::FindMedian(std::vector<T>& V) {
       // Get total sum of array
       T tot = T(0);
       for (unsigned int i = 0; i < V.size(); i++)
          tot += V[i];
       
       // Find position of median as half of total in array
       T half = tot / T(2);
       T sum = T(0);
       for (unsigned int i = 0; i < V.size(); i++) {
          sum += V[i];
          if (sum >= half) 
             return i;
       } 
       return 0;
    }


    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it, int cutAxis, int median) {
       // Cut field layout's domain in half at median along cutAxis
       NDIndex<Dim> leftDom, rightDom;
       domains[it].split(leftDom, rightDom, cutAxis, median);
       domains[it] = leftDom;
       domains.insert(domains.begin() + it + 1, 1, rightDom); // Not entirely sure about the +1
        
       // Cut procs in half
       int temp = procs[it];
       procs[it] = procs[it] / 2;
       procs.insert(procs.begin() + it + 1, 1, temp - procs[it]);       

       /***PRINT***/
       for (unsigned int i = 0; i < domains.size(); i++) {
          std::cout << domains[i] << std::endl;
          std::cout << procs[i] << std::endl;
       }
    }

}  // namespace
