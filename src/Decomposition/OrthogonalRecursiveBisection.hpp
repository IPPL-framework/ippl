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
       FieldLayout<Dim>& FL = P.getLayout().getFieldLayout();
       UniformCartesian<T,Dim> mesh = P.getLayout().getMesh();
       Field<T,Dim> BF(mesh, FL);
   
       // Scattering of particles in Field
       // ***The scatter methods are all commented in Interpolator.h... needs new writing?

       // Domain Decomposition
       int rank;
       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       if (rank == 0)
          CalcBinaryRepartition(FL, BF); 
   

       // Update particles
       // ...


       return true;
    }

    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF) {
       int nprocs;
       MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

       // Start with whole domain and total number of nodes
       std::vector<NDIndex<Dim>> domains = {FL.getDomain()};
       std::vector<int> procs = {nprocs};

       /**PRINT**/
       std::cout << "Field Layout's domain: " << domains[0] << std::endl;
       
       // Start recursive repartition loop 
       int it = 0;
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

    
    // PROBLEM: Dimension handling
    // PROBLEM: Subview handling
    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T, Dim> BF, std::vector<T>& res, unsigned int cutAxis) {
       // Return if Dim == 1
       if (Dim <= 1)
          return;
       
       const view_type& data = BF.getView();
       
       /***PRINT***/
       std::cout << "BF's domain: " << BF.getDomain() << std::endl;
       std::cout << "BF.size(" << cutAxis << "): " << BF.size(cutAxis) << std::endl;
       std::cout << "BF.getView().extent(" << cutAxis << "): " << data.extent(cutAxis) << std::endl;
       
       // std::cout << "Data(0,0,0): " << data(0,0,0) << std::endl;      
  
       // Iterate along cutAxis
       for (int i = 0; i < BF.size(cutAxis); i++) { 
          T weight = T(0);
          auto data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
          // Dirty
          switch (cutAxis) {
            case 0:
             // data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
             break;
            case 1:
             data_i = Kokkos::subview(data, Kokkos::ALL, i, Kokkos::ALL);
             break;
            case 2:
             data_i = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, i);
             break;
          }
          // Perform reduction over weights
          // parallel_reduce ? 
          for (unsigned int k = 0; k < data_i.extent(0); k++) {
             for (unsigned int j = 0; j < data_i.extent(1); j++) 
                // weight += data_i(k,j);
                weight += i + k + j;
          } 
          res.push_back(weight);
          /*
          parallel_reduce("weight reduction", data_i.extent(0), [=](int k, int j, T& weight) {
             weight += data_i(k, j);
          }, res);
          */         
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
