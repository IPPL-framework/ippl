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
    OrthogonalRecursiveBisection<T,Dim,M>::BinaryRepartition(ParticleBase<ParticleSpatialLayout<T,Dim,M> >& P, FieldLayout<Dim>& FL, UniformCartesian<T,Dim>& mesh) {
       // Declaring field layout and field
       // FieldLayout<Dim>& FL = P.getLayout().getFieldLayout();
       // UniformCartesian<T,Dim> mesh = P.getLayout().getMesh();
       Field<T,Dim,M> BF(mesh, FL);
   
       // Scattering of particle positions in field
       testScatter(BF, P.R);

       // Domain Decomposition
       CalcBinaryRepartition(FL, BF); 

       // Update particles
       // P.getLayout().setFieldLayout(FL);
       // P.setFieldLayout<FieldLayout<Dim>>(FL); // ideally instead of line above
       // P.update();
 
       return true;
    }

    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF) {
       int nprocs = Ippl::Comm->size();

       // Start with whole domain and total number of nodes
       std::vector<NDIndex<Dim>> domains = {FL.getDomain()};
       // std::vector<NDIndex<Dim>> finalDomains[4]; 
       std::vector<int> procs = {nprocs};

       int split_rank;
       MPI_Comm_rank(comm_m, &split_rank);
       /**PRINT**/
       if (split_rank == 0)
          std::cout << "Domain to reduce with " << nprocs << " nodes: " << domains[0] << std::endl;
        
       // Start recursive repartition loop 
       int it = 0;
       int maxprocs = nprocs; 
       while (maxprocs > 1) {
          // Find cut axis
          int cutAxis = FindCutAxis(domains[it]);
          
          // Peform reduction with field of weights and communicate to the other ranks
          std::vector<T> reduced(domains[it][cutAxis].length()); // Reserving space for reduced array for MPI
          std::vector<T> rankReduce(domains[it][cutAxis].length()); // Reserving space for reduced array for MPI
          PerformReduction(BF, rankReduce, cutAxis, domains[it]); 
          MPI_Allreduce(rankReduce.data(), reduced.data(), rankReduce.size(), MPI_DOUBLE, MPI_SUM, comm_m);
        
          // Find median of reduced weights
          int median = FindMedian(reduced);
          
          /**PRINT**/
          if (split_rank == 0) {
          std::ofstream myfile;
          myfile.open ("run.txt");
          // myfile.open ("run" + std::to_string(split_rank) + ".txt");
          myfile << median << "\n";
          for (unsigned int i = 0; i < reduced.size(); i++){
             std::cout << "reduced[" << i << "]: " << reduced[i] << std::endl;
             myfile << reduced[i] << "\n";
          }
          myfile.close();
          }
         
          // Cut domains and procs
          CutDomain(domains, procs, it, cutAxis, median);

          /***PRINT***/
          if (split_rank == 0) {
          std::cout << "New domains:" << std::endl;
          for (unsigned int i = 0; i < domains.size(); i++) {
             std::cout << domains[i] << " (proc:" << procs[i] << ")" << std::endl;
          }}

          // Update max procs
          maxprocs = 0;
          for (unsigned int i = 0; i < procs.size(); i++) {
             if (procs[i] > maxprocs) {
                maxprocs = procs[i];
                it = i;
             } 
          }
       }
       
       // Update FieldLayout with new domains 
       // FL.updateLayout(domains);

       // Free communicators
       // if (comm_m != MPI_COMM_WORLD) // this test might be useless
       //    MPI_Comm_free(&comm_m);
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
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T,Dim>& BF, std::vector<T>& rankWeights, unsigned int cutAxis, NDIndex<Dim>& dom) {
      using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
      
      // Get Field's local domain's size and weights
      NDIndex<Dim> localDom = BF.getOwned();
      int axisSize = localDom[cutAxis].length();
      int localFirst = localDom[cutAxis].first();
      int nghost = BF.getNghost();
      const view_type& data = BF.getView();

      if (Dim == 2) {
        // Determining perpendicular bounds
        int perpAxis = (cutAxis+1) % 2;
        int inferior, superior;
        inferior = std::max(localDom[perpAxis].first(), dom[perpAxis].first()); 
        superior = std::min(localDom[perpAxis].last(), dom[perpAxis].last());

        if (superior < inferior) // Processor not involved in reduction
           return;    
/*
        // Reduction
        auto data_i = Kokkos::subview(data, 0, Kokkos::ALL);
        for (int i = nghost; i < axisSize+nghost; i++) {   // (Kokkos) Bounds take into account ghost layers
           // Slicing view perpendicular to cutAxis
           if (cutAxis == 0)
             data_i = Kokkos::subview(data, i, Kokkos::make_pair(inferior, superior));
           else
             data_i = Kokkos::subview(data, Kokkos::make_pair(inferior, superior), i); 

           // Perform reduction over weights
           T tempRes = T(0);
          
           // unsure about ghosts....
           Kokkos::parallel_reduce("Weight reduction", data_i.extent(0), KOKKOS_CLASS_LAMBDA (const int j, T& weight) {
              weight += data_i(j);
           }, tempRes);
           
           Kokkos::fence();

           rankWeights[localFirst + i - 1] = tempRes;
        } */         
      } else if (Dim == 3) {
 
       /***PRINT***/
       std::cout << "Local domain (owned): " << localDom << std::endl;
       std::cout << "Local axis size: " << axisSize << std::endl;
       std::cout << "Local first index: " << localFirst << std::endl;
       std::cout << "Data's extents (0,1,2): (" << data.extent(0) << "," << data.extent(1) << "," << data.extent(2) << ")" << std::endl;

       // std::vector<T> rankReduce(BF.getDomain()[cutAxis].length()); // [0,...,0]  
       
       /*// face has two directions: 1 and 2 
       int perpAxis1 = (cutAxis+1) % 3;
       int perpAxis2 = (cutAxis+2) % 3;
       // The 3 comes from make_pair(0,N) <-> 0,..,N-1 and there are 2 ghost layers
       int inferior1 = std::max(localDom[perpAxis1].first(), dom[perpAxis1].first());
       int inferior2 = std::max(localDom[perpAxis2].first(), dom[perpAxis2].first());
       int superior1 = std::min(localDom[perpAxis1].last(), dom[perpAxis1].last());
       int superior2 = std::min(localDom[perpAxis2].last(), dom[perpAxis2].last());
       }
       
       if (superior1 < inferior1 || superior2 < inferior2) // Processor is not involved in reduction
         return;

       */

       // auto data_i = Kokkos::subview(data, i, Kokkos::make_pair(inferior1, superior1), Kokkos::make_pair(inferior2, superior2));
       
       // Iterate along cutAxis
       auto data_i = Kokkos::subview(data, 0, Kokkos::ALL, Kokkos::ALL);
       for (int i = nghost; i < axisSize+nghost; i++) {   // (Kokkos) Bounds take into account ghost layers
       // for (int i = 0; i < axisSize; i++) {
          // Slicing view perpendicular to cutAxis
          switch (cutAxis) {
            default:
            case 0:
             data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
             break;
            case 1:
             data_i = Kokkos::subview(data, Kokkos::ALL, i, Kokkos::ALL);
             break;
            case 2:
             data_i = Kokkos::subview(data, Kokkos::ALL, Kokkos::ALL, i);
             break;
          }

          // Perform reduction over weights
          T tempRes = T(0);
          
          Kokkos::parallel_reduce("Weight reduction", mdrange_t({nghost, nghost},{data_i.extent(0)-nghost, data_i.extent(1)-nghost}),
                                                      KOKKOS_CLASS_LAMBDA (const int k, const int j, T& weight) {
             weight += data_i(k,j);
          }, tempRes);
          
          Kokkos::fence();

          rankWeights[localFirst + i - 1] = tempRes;
       }

       } // if (Dim == 3)

       std::cout << "rank " << Ippl::Comm->rank() << ": [";
       for (unsigned int i = 0; i < rankWeights.size(); i++) {
          std::cout << rankWeights[i];
          std::cout << (i+1 < rankWeights.size() ? ", " : "]");
       }
       std::cout << std::endl;
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
       domains.insert(domains.begin() + it + 1, 1, rightDom);
        
       // Cut procs in half
       int temp = procs[it];
       procs[it] = procs[it] / 2;
       procs.insert(procs.begin() + it + 1, 1, temp - procs[it]);       
      
       // Split communicator 
       /*
       int split_rank, split_size;
       MPI_Comm comm;
       MPI_Comm_rank(comm_m, &split_rank);
       MPI_Comm_size(comm_m, &split_size);
       (split_rank < split_size / 2) ? color_m = 0 : color_m = 1;
       MPI_Comm_split(comm_m, color_m, split_rank, &comm); // Split communicator in two
       if (comm != MPI_COMM_NULL)
          comm_m = comm;
       */       
    }

}  // namespace
