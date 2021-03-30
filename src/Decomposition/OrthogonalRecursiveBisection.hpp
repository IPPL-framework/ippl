// include files
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleAttrib.h"
#include "Region/RegionLayout.h"
#include "Index/NDIndex.h"
#include "Index/Index.h"
#include "FieldLayout/FieldLayout.h"
#include "Region/NDRegion.h"
#include <mpi.h>
#include <fstream>

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
       // if (Ippl::Comm->rank() == 0)
          CalcBinaryRepartition(FL, BF); 

       // Update particles
       // P.getLayout().setFieldLayout(FL);
       std::cout << "ORB finished" << std::endl;

       return true;
    }

    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF) {
       int nprocs = Ippl::Comm->size();

       // Start with whole domain and total number of nodes
       // view_type_t localDeviceDom = FL.getDeviceLocalDomains();
       // host_mirror_type localHostDom = FL.getHostLocalDomains();
       std::vector<NDIndex<Dim>> domains = {FL.getLocalNDIndex()}; // Local domain
       std::vector<int> procs = {nprocs};

       /**PRINT**/
       std::cout << "FieldLayout.getLocalNDIndex(): " << domains[0] << std::endl;
       // std::cout << "FieldLayout.getDeviceLocalDomains() (ghost): " << localDeviceDom.extent(0) << std::endl;
       // std::cout << "FieldLayout.getHostLocalDomains() (ghost): " << localHostDom.extent(0) << std::endl;
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
          std::vector<T> reduced(domains[it][cutAxis].length()); // Reserving space for reduced array for MPI
          PerformReduction(BF, reduced, domains[it], cutAxis); 
        
          //          
 
          // Find median of reduced weights
          int median = FindMedian(reduced);
          
          /**PRINT**/
          std::ofstream myfile;
          myfile.open ("run.txt");
          myfile << median << "\n";
          // std::cout << "reduced size: " << reduced.size() << std::endl;
          for (unsigned int i = 0; i < reduced.size(); i++){
             // std::cout << "reduced[" << i << "]: " << reduced[i] << std::endl;
             myfile << reduced[i] << "\n";
          }
          myfile.close();
         
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
       // Update local FieldLayout with new domains 
       // FL.setLocalDomain(domains); 
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
    // Comment: BF contains weights irrespectively of ranks...
    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T,Dim>& BF, std::vector<T>& res, NDIndex<Dim>& dom, unsigned int cutAxis) {
       if (Dim != 3)
          return;

       using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
 
       // Get Field's weights locally (per rank)
       const view_type& data = BF.getView();

       // Get Field's local domain's size
       // NDIndex<Dim> localDom = BF.getOwned();
       int sizeAxis = dom[cutAxis].length();
       int firstPerp1, firstPerp2, lastPerp1, lastPerp2;
       
       /***PRINT***/
       std::cout << "Field's local domain (not updated): " << BF.getOwned() << std::endl;
       std::cout << "Size: " << sizeAxis << std::endl;
       std::cout << "Domain to reduce: " << dom << std::endl;
       // std::cout << "dom[" << cutAxis << "].last(): " << dom[cutAxis].last() << std::endl;
       // std::cout << "dom[" << cutAxis << "].max(): " << dom[cutAxis].max() << std::endl;
       // std::cout << "dom[" << cutAxis << "].first(): " << dom[cutAxis].first() << std::endl;
       // std::cout << "Field's domain: " << BF.getDomain() << std::endl;
       // std::cout << "BF.size(" << cutAxis << "): " << BF.size(cutAxis) << std::endl;
       // std::cout << "BF.getView().extent(" << cutAxis << ") (size + 2 ghost): " << data.extent(cutAxis) << std::endl;
       
       // nghost -> extent() - nghost
       // Iterate along cutAxis
       std::vector<T> rankReduce;
       for (int i = 0; i < sizeAxis; i++) { 
          // Slicing view perpendincular to cutAxis
          auto data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
          firstPerp1 = dom[1].first(), lastPerp1 = dom[1].last(); 
          firstPerp2 = dom[2].first(), lastPerp2 = dom[2].last(); 
          switch (cutAxis) {
            case 1:
             data_i = Kokkos::subview(data, Kokkos::ALL, i, Kokkos::ALL);
             firstPerp1 = dom[0].first(), lastPerp1 = dom[0].last(); 
             firstPerp2 = dom[2].first(), lastPerp2 = dom[2].last(); 
             break;
            case 2:
             data_i = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, i);
             firstPerp1 = dom[0].first(), lastPerp1 = dom[0].last(); 
             firstPerp2 = dom[1].first(), lastPerp2 = dom[1].last(); 
             break;
          }
          // Perform reduction over weights
          T tempRes = T(0);
          
          // Kokkos::parallel_reduce("Weight reduction", mdrange_t({0,0},{data_i.extent(0), data_i.extent(1)}),
          Kokkos::parallel_reduce("Weight reduction", mdrange_t({firstPerp1, firstPerp2},{lastPerp1+2, lastPerp2+2}),
                                                      KOKKOS_CLASS_LAMBDA (const int k, const int j, T& weight) {
             weight += data_i(k,j);
          }, tempRes);

          Kokkos::fence();

          rankReduce.push_back(tempRes); 
       }
      
       // Reduce among ranks (this needs investigation concerning communication optimization)
       MPI_Allreduce(rankReduce.data(), res.data(), rankReduce.size(), MPI_DOUBLE, MPI_SUM, Ippl::getComm());
      
       // Testing Allreduce -> works 
       // for (unsigned int i = 0; i < rankReduce.size(); i++)
       //    std::cout << "rankReduce[" << i << "](" << Ippl::Comm->rank() << "): " << rankReduce[i] << std::endl;
       // if (Ippl::Comm->rank() == 0) { 
       //    for (unsigned int i = 0; i < res.size(); i++)
       //       std::cout << "res[" << i << "]: " << res[i] << std::endl;
       // }
       // MPI_Reduce(&sum_coord, &global_sum_coord, 1, MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());
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
          std::cout << "New domain: " << domains[i] << " (proc:" << procs[i] << ")" << std::endl;
       }
    }

}  // namespace
