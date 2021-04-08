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
    OrthogonalRecursiveBisection<T,Dim,M>::BinaryRepartition(ParticleBase<ParticleSpatialLayout<T,Dim,M> >& P, FieldLayout<Dim>& FL, UniformCartesian<T,Dim> mesh) {
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
          
          // Peform reduction with field of weights
          std::vector<T> reduced(domains[it][cutAxis].length()); // Reserving space for reduced array for MPI
          PerformReduction(BF, reduced, cutAxis); 
        
          // Find median of reduced weights
          int median = FindMedian(reduced);
          
          /**PRINT**/
          if (split_rank == 0) {
          std::ofstream myfile;
          myfile.open ("run.txt");
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
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T,Dim>& BF, std::vector<T>& res, unsigned int cutAxis) {
       if (Dim != 3)
          return;

       using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
 
       // Get Field's weights locally (per rank)
       const view_type& data = BF.getView();

       // Get Field's local domain's size
       NDIndex<Dim> localDom = BF.getOwned();
       int axisSize = localDom[cutAxis].length();
       int localFirst = localDom[cutAxis].first();
       int nghost = BF.getNghost();
 
       /***PRINT***/
       std::cout << "Local domain (owned): " << localDom << std::endl;
       std::cout << "Local axis size: " << axisSize << std::endl;
       std::cout << "Local first index: " << localFirst << std::endl;
       std::cout << "Data's extents (0,1,2): (" << data.extent(0) << "," << data.extent(1) << "," << data.extent(2) << ")" << std::endl;

       std::vector<T> rankReduce(BF.getDomain()[cutAxis].length()); // [0,...,0]  
       
       /*
       int inferior1, superior1, inferior2, superior2; // face has two directions: 1 and 2 
       switch (cutAxis) {    // The 3 comes from make_pair(0,N) <-> 0,..,N-1 and there are 2 ghost layers
         default:
         case 0:
            inferior1 = dom[1].first();
            superior1 = dom[1].last()+3;
            inferior2 = dom[2].first();
            superior2 = dom[2].last()+3;
          break; 
         case 1:
            inferior1 = dom[0].first();
            superior1 = dom[0].last()+3;
            inferior2 = dom[1].first();
            superior2 = dom[1].last()+3;
          break; 
         case 2:
            inferior1 = dom[0].first();
            superior1 = dom[0].last()+3;
            inferior2 = dom[2].first();
            superior2 = dom[2].last()+3;
          break; 
       }
       */


       // Iterate along cutAxis
       for (int i = 1; i <= axisSize; i++) {   // (Kokkos) Bounds take into account ghost layers
       // for (int i = 0; i < axisSize; i++) {
          // Slicing view perpendicular to cutAxis
          // auto data_i = Kokkos::subview(data, i, Kokkos::make_pair(inferior1, superior1), Kokkos::make_pair(inferior2, superior2));
          auto data_i = Kokkos::subview(data, i, Kokkos::ALL, Kokkos::ALL);
          switch (cutAxis) {
            case 1:
             // data_i = Kokkos::subview(data, Kokkos::make_pair(inferior1, superior1), i, Kokkos::make_pair(inferior2, superior2));
             data_i = Kokkos::subview(data, Kokkos::ALL, i, Kokkos::ALL);
             break;
            case 2:
             // data_i = Kokkos::subview(data, Kokkos::make_pair(inferior1, superior1), Kokkos::make_pair(inferior2, superior2), i);
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

          rankReduce[localFirst + i - 1] = tempRes;
       }

       std::cout << "rank " << Ippl::Comm->rank() << ": [";
       for (unsigned int i = 0; i < rankReduce.size(); i++) {
          std::cout << rankReduce[i];
          std::cout << (i+1 < rankReduce.size() ? ", " : "]");
       }
       std::cout << std::endl;
  
       // Reduce among ranks (this needs investigation)
       MPI_Allreduce(rankReduce.data(), res.data(), rankReduce.size(), MPI_DOUBLE, MPI_SUM, comm_m);
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
