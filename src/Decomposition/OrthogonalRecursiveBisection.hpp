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
    OrthogonalRecursiveBisection<T,Dim,M>::BinaryRepartition(const ParticleAttrib<Vector<T,Dim>>& R, Field<T,Dim,M>& BF, FieldLayout<Dim>& FL, int step) {
       // Scattering of particle positions in field
       BF = 0.0;
       scatterR(BF, R);
       R.getView();

       // Domain Decomposition
       if (CalcBinaryRepartition(FL, BF, step))
          return true;
       else 
          return false;
    }

    template < class T, unsigned Dim, class M>
    bool 
    OrthogonalRecursiveBisection<T,Dim,M>::CalcBinaryRepartition(FieldLayout<Dim>& FL, Field<T, Dim>& BF, int step) {
       int nprocs = Ippl::Comm->size();
       
       // std::cout << "(after) BF.sum(): " << BF.sum() << " particles." << std::endl;
 
       // Start with whole domain and total number of nodes
       std::vector<NDIndex<Dim>> domains = {FL.getDomain()};
       std::vector<int> procs = {nprocs};

       // Arrays for reduction 
       std::vector<T> reduced, reducedRank;
 
       // Start recursive repartition loop 
       int it = 0;
       int maxprocs = nprocs; 
       // int loopstep = 1; // just for debugging
       while (maxprocs > 1) {
          // Find cut axis
          int cutAxis = FindCutAxis(domains[it]);
         
          // Reserve space
          reduced.resize(domains[it][cutAxis].length());
          reducedRank.resize(domains[it][cutAxis].length());

          // Peform reduction with field of weights and communicate to the other ranks
          PerformReduction(BF, reducedRank, cutAxis, domains[it]); 
          MPI_Allreduce(reducedRank.data(), reduced.data(), reducedRank.size(), MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        
          // Find median of reduced weights
          int median = FindMedian(reduced);
          
          /**PRINT**/
          /*
          if (Ippl::Comm->rank() == 0) {
             double total = 0.0;
             // std::ofstream myfile;
             // myfile.open ("run" + std::to_string(loopstep) + ".txt");
             // myfile << domains[it] << "\n";
             // myfile << median << "\n";
             for (unsigned int i = 0; i < reduced.size(); i++){
                total += reduced[i];
                std::cout << "reduced[" << i << "]: " << reduced[i] << std::endl;
                // myfile << reduced[i] << "\n";
             }
             std::cout << "Total number of particles: " << total << std::endl;
             // myfile.close();
             // std::cout << "STEP: " << loopstep << std::endl;
             // loopstep++;
          }
          */
          // Cut domains and procs
          CutDomain(domains, procs, it, cutAxis, median);

          /***PRINT***/
          /*
          if (Ippl::Comm->rank() == 0) {
          std::cout << "New domains:" << std::endl;
          for (unsigned int i = 0; i < domains.size(); i++) {
             std::cout << domains[i] << std::endl; // << " (proc:" << procs[i] << ")" << std::endl;
          }}
          */

          // Update max procs
          maxprocs = 0;
          for (unsigned int i = 0; i < procs.size(); i++) {
             if (procs[i] > maxprocs) {
                maxprocs = procs[i];
                it = i;
             } 
          }
               
          // Clear arrays' allocated space
          reduced.clear();
          reducedRank.clear();
       }

       /***PRINT***/
       if (Ippl::Comm->rank() == 0) {
       std::ofstream myfile;
       myfile.open ("domains" + std::to_string(step) + ".txt");
       // std::cout << "New domains: " << std::endl;
       for (unsigned int i = 0; i < domains.size(); i++) {
          // std::cout << domains[i] << std::endl;
          myfile << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                 << domains[i][0].first() << " " << domains[i][1].last() << " " << domains[i][2].first() << " "
                 << domains[i][0].last() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                 << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].last()
                 << "\n";
       }}
       
       // Update FieldLayout with new domains 
       if (domains.empty())
          return false;
       else {
          FL.updateLayout(domains);
          return true;
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

    
    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::PerformReduction(Field<T,Dim>& BF, std::vector<T>& rankWeights, unsigned int cutAxis, NDIndex<Dim>& dom) {
       // test if domains touch
       NDIndex<Dim> lDom = BF.getOwned();
       if (lDom[cutAxis].first() > dom[cutAxis].last() || lDom[cutAxis].last() < dom[cutAxis].first())
          return;
      
       // Get Field's local domain's size and weights
       int nghost = BF.getNghost();
       const view_type& data = BF.getView();
       int cutAxisFirst = std::max(lDom[cutAxis].first(), dom[cutAxis].first()) - lDom[cutAxis].first() + nghost;
       int cutAxisLast = std::min(lDom[cutAxis].last(), dom[cutAxis].last()) - lDom[cutAxis].first() + nghost;
       // Where to write in the reduced array
       unsigned int arrayStart = 0;
       if (dom[cutAxis].first() < lDom[cutAxis].first()) 
          arrayStart = lDom[cutAxis].first() - dom[cutAxis].first();
      
       // face has two directions: 1 and 2 
       int perpAxis1 = (cutAxis+1) % 3;
       int perpAxis2 = (cutAxis+2) % 3;
       // inf and sup bounds must be within the domain to reduce and translated for Kokkos' extens which start always at 0
       int inf1 = std::max(lDom[perpAxis1].first(), dom[perpAxis1].first()) - lDom[perpAxis1].first();
       int inf2 = std::max(lDom[perpAxis2].first(), dom[perpAxis2].first()) - lDom[perpAxis2].first();
       int sup1 = std::min(lDom[perpAxis1].last(), dom[perpAxis1].last()) - lDom[perpAxis1].first();
       int sup2 = std::min(lDom[perpAxis2].last(), dom[perpAxis2].last()) - lDom[perpAxis2].first();
       
       if (sup1 < inf1 || sup2 < inf2)  // Processor is not involved in reduction
          return;

       /***PRINT***/
       /*
       std::cout << "Domain to reduce: " << dom << " along " << cutAxis << std::endl;
       std::cout << "Local domain (owned): " << lDom << std::endl;
       std::cout << "Reduction sizes: (" << cutAxisFirst << ", " << cutAxisLast << ")" << std::endl;
       std::cout << "Array start: " << arrayStart << std::endl;
       */
       /*
       std::cout << "(inf1, sup1): (" << inf1 << ", " << sup1 << ")" << std::endl;
       std::cout << "(inf2, sup2): (" << inf2 << ", " << sup2 << ")" << std::endl;
       std::cout << "Data's extents (0,1,2): (" << data.extent(0) << "," << data.extent(1) << "," << data.extent(2) << ")" << std::endl;
       */
 
       // The +3 comes from make_pair(0,N) <-> 0,..,N-1 and there are 2 ghost layers
       sup1 += 3; sup2 += 3;
 
       // Iterate along cutAxis
       using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
       for (int i = cutAxisFirst; i <= cutAxisLast; i++) {  
          // Reducing over perpendicular plane defined by cutAxis
          T tempRes = T(0);
          switch (cutAxis) {
            default:
            case 0:
             Kokkos::parallel_reduce("Weight reduction", mdrange_t({inf1+nghost, inf2+nghost},{sup1-nghost, sup2-nghost}),
                                                         KOKKOS_LAMBDA (const int j, const int k, T& weight) {
                weight += data(i,j,k);
             }, tempRes);
             break;
            case 1:
             Kokkos::parallel_reduce("Weight reduction", mdrange_t({inf2+nghost, inf1+nghost},{sup2-nghost, sup1-nghost}),
                                                         KOKKOS_LAMBDA (const int j, const int k, T& weight) {
                weight += data(j,i,k);
             }, tempRes); 
             break;
            case 2:
             Kokkos::parallel_reduce("Weight reduction", mdrange_t({inf1+nghost, inf2+nghost},{sup1-nghost, sup2-nghost}),
                                                         KOKKOS_LAMBDA (const int j, const int k, T& weight) {
                weight += data(j,k,i);
             }, tempRes);
             break;
          }
          
          Kokkos::fence();
          
          rankWeights[arrayStart] = tempRes; arrayStart++;
       }

       /***PRINT***/
       /*
       T sum = T(0);
       std::cout << "rank " << Ippl::Comm->rank() << ": [";
       for (unsigned int i = 0; i < rankWeights.size(); i++) {
          std::cout << rankWeights[i];
          std::cout << (i+1 < rankWeights.size() ? ", " : "]");
          sum += rankWeights[i];
       }
       std::cout << " -> total: " << sum << std::endl;
       */  
    }
  

    // Potential problem: if there are zeros between weights, then the cut will be made most left, could be better?
    template < class T, unsigned Dim, class M>
    int
    OrthogonalRecursiveBisection<T,Dim,M>::FindMedian(std::vector<T>& V) {
       // Get total sum of array
       T tot = T(0);
       for (unsigned int i = 0; i < V.size(); i++)
          tot += V[i];
       
       // Find position of median as half of total in array
       T half = tot / T(2);
       T curr = T(0);
       for (unsigned int i = 0; i < V.size(); i++) {
          curr += V[i];
          if (curr >= half) { 
             T previous = curr - V[i];
             if ((curr + previous) < tot && V[i] != 0.0) // (curr - half < half - previous)
                return i;
             else
                return (i >= 1) ? (i-1) : 0;
          }
       } 
       return 0;
    }


    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::CutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it, int cutAxis, int median) {
       // Cut field layout's domain in half at median along cutAxis
       /*if (Ippl::Comm->rank() == 0) {
          std::cout << "Cutting " << domains[it] << " along " << cutAxis << " at first+median (" << domains[it][cutAxis].first() << "+" << median << ")" << std::endl;
       }*/
       NDIndex<Dim> leftDom, rightDom;
       domains[it].split(leftDom, rightDom, cutAxis, median + domains[it][cutAxis].first()); 
       domains[it] = leftDom;
       domains.insert(domains.begin() + it + 1, 1, rightDom);
        
       // Cut procs in half
       int temp = procs[it];
       procs[it] = procs[it] / 2;
       procs.insert(procs.begin() + it + 1, 1, temp - procs[it]);       
    }


    template < class T, unsigned Dim, class M>
    void 
    OrthogonalRecursiveBisection<T,Dim,M>::scatterR(Field<T, Dim, M>& f, const ParticleAttrib<Vector<T, Dim>>& pr) {

        typename Field<T, Dim, M>::view_type view = f.getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout(); 
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = f.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::scatterR",
            pr.getView().extent(0),
            KOKKOS_LAMBDA(const size_t idx)
            {
                // find nearest grid point
                vector_type l = (pr(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index = l;
                Vector<double, Dim> whi = l - index;
                Vector<double, Dim> wlo = 1.0 - whi;

                const size_t i = index[0] - lDom[0].first() + nghost;
                const size_t j = index[1] - lDom[1].first() + nghost;
                const size_t k = index[2] - lDom[2].first() + nghost;

                // scatter
                // const value_type& val = (idx);
                Kokkos::atomic_add(&view(i-1, j-1, k-1), wlo[0] * wlo[1] * wlo[2]);
                Kokkos::atomic_add(&view(i-1, j-1, k  ), wlo[0] * wlo[1] * whi[2]);
                Kokkos::atomic_add(&view(i-1, j,   k-1), wlo[0] * whi[1] * wlo[2]);
                Kokkos::atomic_add(&view(i-1, j,   k  ), wlo[0] * whi[1] * whi[2]);
                Kokkos::atomic_add(&view(i,   j-1, k-1), whi[0] * wlo[1] * wlo[2]);
                Kokkos::atomic_add(&view(i,   j-1, k  ), whi[0] * wlo[1] * whi[2]);
                Kokkos::atomic_add(&view(i,   j,   k-1), whi[0] * whi[1] * wlo[2]);
                Kokkos::atomic_add(&view(i,   j,   k  ), whi[0] * whi[1] * whi[2]);
            }
        );
            
        f.accumulateHalo();
    }

}  // namespace
