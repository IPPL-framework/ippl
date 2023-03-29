#include "Utility/IpplTimings.h"
namespace ippl {

    template <class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    void
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::initialize(FieldLayout<Dim>& fl, 
                                                     Mesh& mesh,
                                                      const Field<Tf,Dim, Mesh, Centering>& rho) {
       bf_m.initialize(mesh, fl);
       bf_m = rho;

    }

    template <class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    bool 
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::binaryRepartition(const ParticleAttrib<Vector<Tp,Dim>>& R, 
                                                             FieldLayout<Dim>& fl,
                                                             const bool& isFirstRepartition) {
       // Timings
       static IpplTimings::TimerRef tbasicOp = IpplTimings::getTimer("basicOperations");           
       static IpplTimings::TimerRef tperpReduction = IpplTimings::getTimer("perpReduction");           
       static IpplTimings::TimerRef tallReduce = IpplTimings::getTimer("allReduce");           
       static IpplTimings::TimerRef tscatter = IpplTimings::getTimer("scatterR");           

       //MPI datatype
       MPI_Datatype mpi_data= MPI_DATATYPE_NULL;
       if constexpr ( std::is_same_v<Tp, float> ) mpi_data = MPI_FLOAT;
       else if constexpr ( std::is_same_v<Tp, double> ) mpi_data = MPI_DOUBLE;
       
       // Scattering of particle positions in field
       // In case of first repartition we know the density from the
       // analytical expression and we use that for load balancing
       // and create particles. Note the particles are created only
       // after the first repartition and hence we cannot call scatter
       // before it.
       IpplTimings::startTimer(tscatter);
       if(!isFirstRepartition) {
          scatterR(R);
       }

       IpplTimings::stopTimer(tscatter);

       IpplTimings::startTimer(tbasicOp);

       // Get number of ranks
       int nprocs = Ippl::Comm->size();
       
       // Start with whole domain and total number of nodes
       std::vector<NDIndex<Dim>> domains = {fl.getDomain()};
       std::vector<int> procs = {nprocs};

       // Arrays for reduction 
       std::vector<Tp> reduced, reducedRank;
 
       // Start recursive repartition loop 
       unsigned int it = 0;
       int maxprocs = nprocs; 
       IpplTimings::stopTimer(tbasicOp);

       while (maxprocs > 1) {
          // Find cut axis
          IpplTimings::startTimer(tbasicOp);                                                    
          int cutAxis = findCutAxis(domains[it]);
          IpplTimings::stopTimer(tbasicOp);                                                    
          
          // Reserve space
          IpplTimings::startTimer(tperpReduction);                                                    
          reduced.resize(domains[it][cutAxis].length());
          reducedRank.resize(domains[it][cutAxis].length());

          std::fill(reducedRank.begin(), reducedRank.end(), 0.0);
          std::fill(reduced.begin(), reduced.end(), 0.0);

          // Peform reduction with field of weights and communicate to the other ranks
          perpendicularReduction(reducedRank, cutAxis, domains[it]); 
          IpplTimings::stopTimer(tperpReduction);                                                    

          // Communicate to all the reduced weights
          IpplTimings::startTimer(tallReduce);                                                    
          MPI_Allreduce(reducedRank.data(), reduced.data(), reducedRank.size(), 
                                            mpi_data/*DOUBLE*/, MPI_SUM, Ippl::getComm());
          IpplTimings::stopTimer(tallReduce);                                                    
        
          // Find median of reduced weights
          IpplTimings::startTimer(tbasicOp);
          // Initialize median to some value (1 is lower bound value)
          int median = 1;
          median = findMedian(reduced);
          IpplTimings::stopTimer(tbasicOp);

          // Cut domains and procs
          IpplTimings::startTimer(tbasicOp);
          cutDomain(domains, procs, it, cutAxis, median);

          // Update max procs
          maxprocs = 0;
          for (unsigned int i = 0; i < procs.size(); i++) {
             if (procs[i] > maxprocs) {
                maxprocs = procs[i];
                it = i;
             } 
          }
          IpplTimings::stopTimer(tbasicOp);                                                    
               
          // Clear all arrays
          IpplTimings::startTimer(tperpReduction);                                                    
          reduced.clear();
          reducedRank.clear();
          IpplTimings::stopTimer(tperpReduction);
       }

       // Check that no plane was obtained in the repartition
       IpplTimings::startTimer(tbasicOp);                                                    
       for (unsigned int i = 0; i < domains.size(); i++) {
          if (domains[i][0].length() == 1 || 
              domains[i][1].length() == 1 ||
              domains[i][2].length() == 1)
             return false;
       }

       // Update FieldLayout with new indices
       fl.updateLayout(domains);
         
       // Update local field with new layout
       bf_m.updateLayout(fl);
       IpplTimings::stopTimer(tbasicOp);                                                    

       return true;
    }

    
    template < class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    int
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::findCutAxis(NDIndex<Dim>& dom) {
       int cutAxis = 0;  
       unsigned int maxLength = 0;
       
       // Iterate along all the dimensions
       for (unsigned int d = 0; d < Dim; d++) {
          // Find longest domain size
          if (dom[d].length() > maxLength) {
             maxLength = dom[d].length();
             cutAxis = d;
          }
       }

       return cutAxis;
    } 

    
    template < class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    void
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::perpendicularReduction(
                                                         std::vector<Tp>& rankWeights, 
                                                         unsigned int cutAxis, 
                                                         NDIndex<Dim>& dom) {
       // Check if domains overlap, if not no need for reduction
       NDIndex<Dim> lDom = bf_m.getOwned();
       if (lDom[cutAxis].first() > dom[cutAxis].last() || 
           lDom[cutAxis].last() < dom[cutAxis].first())
          return;
       // Get field's local weights
       int nghost = bf_m.getNghost();
       const field_view_type data = bf_m.getView();
       // Determine the iteration bounds of the reduction
       int cutAxisFirst = std::max(lDom[cutAxis].first(), dom[cutAxis].first())
                                               - lDom[cutAxis].first() + nghost;
       int cutAxisLast = std::min(lDom[cutAxis].last(), dom[cutAxis].last())
                                            - lDom[cutAxis].first() + nghost;
       // Set iterator for where to write in the reduced array
       unsigned int arrayStart = 0;
       if (dom[cutAxis].first() < lDom[cutAxis].first()) 
          arrayStart = lDom[cutAxis].first() - dom[cutAxis].first();
       // Face of domain has two directions: 1 and 2 
       int perpAxis1 = (cutAxis+1) % Dim;
       int perpAxis2 = (cutAxis+2) % Dim;
       // inf and sup bounds must be within the domain to reduce, if not no need to reduce
       int inf1 = std::max(lDom[perpAxis1].first(), dom[perpAxis1].first())
                                         - lDom[perpAxis1].first() + nghost;
       int inf2 = std::max(lDom[perpAxis2].first(), dom[perpAxis2].first())
                                         - lDom[perpAxis2].first() + nghost;
       int sup1 = std::min(lDom[perpAxis1].last(), dom[perpAxis1].last()) 
                                       - lDom[perpAxis1].first() + nghost;
       int sup2 = std::min(lDom[perpAxis2].last(), dom[perpAxis2].last()) 
                                       - lDom[perpAxis2].first() + nghost;
       if (sup1 < inf1 || sup2 < inf2)  
          return;
       // The +1 is for Kokkos loop
       sup1++; sup2++;

       // Iterate along cutAxis
       using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;       
       for (int i = cutAxisFirst; i <= cutAxisLast; i++) {  
          // Reducing over perpendicular plane defined by cutAxis
          Tp tempRes = Tp(0);
          switch (cutAxis) {
            default:
            case 0:
             Kokkos::parallel_reduce("ORB weight reduction (0)", 
                                     mdrange_t({inf1, inf2},{sup1, sup2}),
                                     KOKKOS_LAMBDA(const int j, const int k, Tp& weight) {
                weight += data(i,j,k);
             }, tempRes);
             break;
            case 1:
             Kokkos::parallel_reduce("ORB weight reduction (1)", 
                                     mdrange_t({inf2, inf1},{sup2, sup1}),
                                     KOKKOS_LAMBDA(const int j, const int k, Tp& weight) {
                weight += data(j,i,k);
             }, tempRes); 
             break;
            case 2:
             Kokkos::parallel_reduce("ORB weight reduction (2)", 
                                     mdrange_t({inf1, inf2},{sup1, sup2}),
                                     KOKKOS_LAMBDA(const int j, const int k, Tp& weight) {
                weight += data(j,k,i);
             }, tempRes);
             break;
          }
          
          Kokkos::fence();
          
          rankWeights[arrayStart] = tempRes; 
          arrayStart++;
       }
    }
  

    template < class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    int
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::findMedian(std::vector<Tp>& w) {
       // Special case when array must be cut in half in order to not have planes
       if (w.size() == 4)
          return 1;

       // Get total sum of array
       Tp tot = std::accumulate(w.begin(), w.end(), Tp(0));
       
       // Find position of median as half of total in array
       Tp half = 0.5 * tot;
       Tp curr = Tp(0);
       // Do not need to iterate to full extent since it must not give planes
       for (unsigned int i = 0; i < w.size()-1; i++) {
          curr += w[i];
          if (curr >= half) {
             // If all particles are in the first plane, cut at 1 so to have size 2
             if (i == 0)
                return 1; 
             Tp previous = curr - w[i];
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


    template < class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    void
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::cutDomain(std::vector<NDIndex<Dim>>& domains, 
                                           std::vector<int>& procs, int it, int cutAxis, int median) {
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

    template < class Tf, unsigned Dim, class Mesh, class Centering, class Tp>
    void 
    OrthogonalRecursiveBisection<Tf,Dim,Mesh,Centering,Tp>::scatterR(const ParticleAttrib<Vector<Tp, Dim>>& r) {
        using vector_type = typename Mesh::vector_type;

        // Reset local field
        bf_m = 0.0;
        // Get local data
        typename Field<Tf, Dim, Mesh, Centering>::view_type view = bf_m.getView();
        const Mesh& mesh = bf_m.get_mesh();
        const FieldLayout<Dim>& layout = bf_m.getLayout(); 
        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();
        const int nghost = bf_m.getNghost();
 
        // Get spacings
        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        Kokkos::parallel_for(
            "ParticleAttrib::scatterR", r.getParticleCount(), KOKKOS_LAMBDA(const size_t idx) {
                // Find nearest grid point
                vector_type l = (r(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index = l;
                Vector<Tp, Dim> whi = l - index;
                Vector<Tp, Dim> wlo = 1.0 - whi;

                const size_t i = index[0] - lDom[0].first() + nghost;
                const size_t j = index[1] - lDom[1].first() + nghost;
                const size_t k = index[2] - lDom[2].first() + nghost;

                // Scatter
                Kokkos::atomic_add(&view(i - 1, j - 1, k - 1), wlo[0] * wlo[1] * wlo[2]);
                Kokkos::atomic_add(&view(i - 1, j - 1, k), wlo[0] * wlo[1] * whi[2]);
                Kokkos::atomic_add(&view(i - 1, j, k - 1), wlo[0] * whi[1] * wlo[2]);
                Kokkos::atomic_add(&view(i - 1, j, k), wlo[0] * whi[1] * whi[2]);
                Kokkos::atomic_add(&view(i, j - 1, k - 1), whi[0] * wlo[1] * wlo[2]);
                Kokkos::atomic_add(&view(i, j - 1, k), whi[0] * wlo[1] * whi[2]);
                Kokkos::atomic_add(&view(i, j, k - 1), whi[0] * whi[1] * wlo[2]);
                Kokkos::atomic_add(&view(i, j, k), whi[0] * whi[1] * whi[2]);
            });

        bf_m.accumulateHalo();
    }

}  // namespace ippl
