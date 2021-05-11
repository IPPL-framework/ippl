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

    template <class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::initialize(FieldLayout<Dim>& fl, UniformCartesian<T,Dim>& mesh) {
       bf_m.initialize(mesh, fl);
    }

    template <class T, unsigned Dim, class M>
    bool 
    OrthogonalRecursiveBisection<T,Dim,M>::binaryRepartition(const ParticleAttrib<Vector<T,Dim>>& R, FieldLayout<Dim>& fl, int step) {
       // Scattering of particle positions in field
       scatterR(R);

       // Domain Decomposition
       if (calcBinaryRepartition(fl, step))
          return true;
       else 
          return false;
    }

    template <class T, unsigned Dim, class M>
    bool 
    OrthogonalRecursiveBisection<T,Dim,M>::calcBinaryRepartition(FieldLayout<Dim>& fl, int step) {
       int nprocs = Ippl::Comm->size();
       
       std::cout << "bf_m.sum(): " << bf_m.sum() << " particles." << std::endl;
 
       // Start with whole domain and total number of nodes
       std::vector<NDIndex<Dim>> domains = {fl.getDomain()};
       std::vector<int> procs = {nprocs};

       // Arrays for reduction 
       std::vector<T> reduced, reducedRank;
 
       // Start recursive repartition loop 
       int it = 0;
       int maxprocs = nprocs; 
       while (maxprocs > 1) {
          // Find cut axis
          int cutAxis = findCutAxis(domains[it]);
         
          // Reserve space
          reduced.resize(domains[it][cutAxis].length());
          reducedRank.resize(domains[it][cutAxis].length());

          // Peform reduction with field of weights and communicate to the other ranks
          performReduction(reducedRank, cutAxis, domains[it]); 

          // Communicate to all the reduced weights
          MPI_Allreduce(reducedRank.data(), reduced.data(), reducedRank.size(), MPI_DOUBLE, MPI_SUM, Ippl::getComm());
        
          // Find median of reduced weights
          int median = findMedian(reduced);
          
          // Cut domains and procs
          cutDomain(domains, procs, it, cutAxis, median);

          // Update max procs
          maxprocs = 0;
          for (unsigned int i = 0; i < procs.size(); i++) {
             if (procs[i] > maxprocs) {
                maxprocs = procs[i];
                it = i;
             } 
          }
               
          // Clear all arrays
          reduced.clear();
          reducedRank.clear();
       }

       /***PRINT***/
       if (Ippl::Comm->rank() == 0) {
       std::ofstream myfile;
       std::ofstream finalDoms;
       myfile.open("domains" + std::to_string(step) + ".txt");
       finalDoms.open("newDomains.txt");
       finalDoms << "New domains: " << std::endl;
       for (unsigned int i = 0; i < domains.size(); i++) {
          finalDoms << domains[i] << std::endl;
          myfile << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                 << domains[i][0].first() << " " << domains[i][1].last() << " " << domains[i][2].first() << " "
                 << domains[i][0].last() << " " << domains[i][1].first() << " " << domains[i][2].first() << " "
                 << domains[i][0].first() << " " << domains[i][1].first() << " " << domains[i][2].last()
                 << "\n";
       }
       myfile.close();
       finalDoms.close(); 
       }

       // Update FieldLayout with new domains 
       if (domains.empty())
          return false;
       else {
          // Update FieldLayout with new indices
          fl.updateLayout(domains);
          
          // Update local field with new layout
          bf_m.updateLayout(fl);

          return true;
       }
    }

    
    template < class T, unsigned Dim, class M>
    int
    OrthogonalRecursiveBisection<T,Dim,M>::findCutAxis(NDIndex<Dim>& domain) {
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
    OrthogonalRecursiveBisection<T,Dim,M>::performReduction(std::vector<T>& rankWeights, unsigned int cutAxis, NDIndex<Dim>& dom) {
       // test if domains touch
       NDIndex<Dim> lDom = bf_m.getOwned();
       if (lDom[cutAxis].first() > dom[cutAxis].last() || lDom[cutAxis].last() < dom[cutAxis].first())
          return;
      
       // Get Field's local domain's size and weights
       int nghost = bf_m.getNghost();
       const view_type& data = bf_m.getView();
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
    }
  

    // Potential problem: if there are zeros between weights, then the cut will be made most left, could be better?
    // Important problem: w.size() < 4 should never happen...
    template < class T, unsigned Dim, class M>
    int
    OrthogonalRecursiveBisection<T,Dim,M>::findMedian(std::vector<T>& w) {
       // static_assert(w.size() >= 4, "Bisection cannot be performed!");
       
       // Special case when array must be cut in half in order to not have planes
       if (w.size() == 4)
          return 1;

       // Get total sum of array
       T tot = T(0);
       for (unsigned int i = 0; i < w.size(); i++)
          tot += w[i];
       
       // Find position of median as half of total in array
       T half = tot / T(2);
       T curr = T(0);
       for (unsigned int i = 0; i < w.size()-1; i++) {
          curr += w[i];
          if (curr >= half) {
             // If all particles are in the first plane, cut at 1 so to have size 2
             if (i == 0)
                return 1; 
             T previous = curr - w[i];
             // curr - half < half - previous
             if ((curr + previous) <= tot && curr != half) {    // if true then take current i, otherwise i-1
                if (i == w.size() - 2)
                   return (i-1);
                else
                   return i;
             } else {
                return (i > 1) ? (i-1) : 1;
             }
          }
       }
       // If all particles are in the last plane, cut two indices before the end so to have size 2
       return w.size()-3;
    }


    template < class T, unsigned Dim, class M>
    void
    OrthogonalRecursiveBisection<T,Dim,M>::cutDomain(std::vector<NDIndex<Dim>>& domains, std::vector<int>& procs, int it, int cutAxis, int median) {
       // Cut domains[it] in half at median along cutAxis
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
    OrthogonalRecursiveBisection<T,Dim,M>::scatterR(const ParticleAttrib<Vector<T, Dim>>& r) {

        bf_m = 0.0;

        typename Field<T, Dim, M>::view_type view = bf_m.getView();

        const M& mesh = bf_m.get_mesh();

        using vector_type = typename M::vector_type;

        const vector_type& dx = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx = 1.0 / dx;

        const FieldLayout<Dim>& layout = bf_m.getLayout(); 
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = bf_m.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::scatterR",
            r.getView().extent(0),
            KOKKOS_LAMBDA(const size_t idx)
            {
                // find nearest grid point
                vector_type l = (r(idx) - origin) * invdx + 0.5;
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
            
        bf_m.accumulateHalo();
    }

}  // namespace
