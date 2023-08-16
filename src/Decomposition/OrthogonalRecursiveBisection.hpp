#include "Utility/IpplTimings.h"

namespace ippl {

    template <class Field, class Tp>
    void OrthogonalRecursiveBisection<Field, Tp>::initialize(FieldLayout<Dim>& fl, mesh_type& mesh,
                                                             const Field& rho) {
        bf_m.initialize(mesh, fl);
        bf_m = rho;
    }

    template <class Field, class Tp>
    template <typename Attrib>
    bool OrthogonalRecursiveBisection<Field, Tp>::binaryRepartition(
        const Attrib& R, FieldLayout<Dim>& fl, const bool& isFirstRepartition) {
        // Timings
        static IpplTimings::TimerRef tbasicOp       = IpplTimings::getTimer("basicOperations");
        static IpplTimings::TimerRef tperpReduction = IpplTimings::getTimer("perpReduction");
        static IpplTimings::TimerRef tallReduce     = IpplTimings::getTimer("allReduce");
        static IpplTimings::TimerRef tscatter       = IpplTimings::getTimer("scatterR");

        // MPI datatype
        MPI_Datatype mpi_data = MPI_DATATYPE_NULL;
        if constexpr (std::is_same_v<Tf, float>) {
            mpi_data = MPI_FLOAT;
        } else if constexpr (std::is_same_v<Tf, double>) {
            mpi_data = MPI_DOUBLE;
        }

        // Scattering of particle positions in field
        // In case of first repartition we know the density from the
        // analytical expression and we use that for load balancing
        // and create particles. Note the particles are created only
        // after the first repartition and hence we cannot call scatter
        // before it.
        IpplTimings::startTimer(tscatter);
        if (!isFirstRepartition) {
            scatterR(R);
        }

        IpplTimings::stopTimer(tscatter);

        IpplTimings::startTimer(tbasicOp);

        // Get number of ranks
        int nprocs = Comm->size();

        // Start with whole domain and total number of nodes
        std::vector<NDIndex<Dim>> domains = {fl.getDomain()};
        std::vector<int> procs            = {nprocs};

        // Arrays for reduction
        std::vector<Tf> reduced, reducedRank;

        // Start recursive repartition loop
        unsigned int it = 0;
        int maxprocs    = nprocs;
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
            MPI_Allreduce(reducedRank.data(), reduced.data(), reducedRank.size(), mpi_data, MPI_SUM,
                          Comm->getCommunicator());
            IpplTimings::stopTimer(tallReduce);

            // Find median of reduced weights
            IpplTimings::startTimer(tbasicOp);
            // Initialize median to some value (1 is lower bound value)
            int median = 1;
            median     = findMedian(reduced);
            IpplTimings::stopTimer(tbasicOp);

            // Cut domains and procs
            IpplTimings::startTimer(tbasicOp);
            cutDomain(domains, procs, it, cutAxis, median);

            // Update max procs
            maxprocs = 0;
            for (unsigned int i = 0; i < procs.size(); i++) {
                if (procs[i] > maxprocs) {
                    maxprocs = procs[i];
                    it       = i;
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
        for (const auto& domain : domains) {
            for (const auto& axis : domain) {
                if (axis.length() == 1) {
                    return false;
                }
            }
        }

        // Update FieldLayout with new indices
        fl.updateLayout(domains);

        // Update local field with new layout
        bf_m.updateLayout(fl);
        IpplTimings::stopTimer(tbasicOp);

        return true;
    }

    template <class Field, class Tp>
    int OrthogonalRecursiveBisection<Field, Tp>::findCutAxis(NDIndex<Dim>& dom) {
        // Find longest domain size
        return std::distance(dom.begin(), std::max_element(dom.begin(), dom.end(),
                                                           [&](const Index& a, const Index& b) {
                                                               return a.length() < b.length();
                                                           }));
    }

    template <class Field, class Tp>
    void OrthogonalRecursiveBisection<Field, Tp>::perpendicularReduction(
        std::vector<Tf>& rankWeights, unsigned int cutAxis, NDIndex<Dim>& dom) {
        // Check if domains overlap, if not no need for reduction
        NDIndex<Dim> lDom = bf_m.getOwned();
        if (lDom[cutAxis].first() > dom[cutAxis].last()
            || lDom[cutAxis].last() < dom[cutAxis].first()) {
            return;
        }

        // Get field's local weights
        int nghost      = bf_m.getNghost();
        const auto data = bf_m.getView();

        // Determine the iteration bounds of the reduction
        int cutAxisFirst =
            std::max(lDom[cutAxis].first(), dom[cutAxis].first()) - lDom[cutAxis].first() + nghost;
        int cutAxisLast =
            std::min(lDom[cutAxis].last(), dom[cutAxis].last()) - lDom[cutAxis].first() + nghost;

        // Set iterator for where to write in the reduced array
        unsigned int arrayStart = 0;
        if (dom[cutAxis].first() < lDom[cutAxis].first()) {
            arrayStart = lDom[cutAxis].first() - dom[cutAxis].first();
        }

        // Find all the perpendicular axes
        using exec_space = typename Field::execution_space;
        using index_type = typename RangePolicy<Dim, exec_space>::index_type;
        Kokkos::Array<index_type, Dim> begin, end;
        for (unsigned d = 0; d < Dim; d++) {
            if (d == cutAxis) {
                continue;
            }

            int inf = std::max(lDom[d].first(), dom[d].first()) - lDom[d].first() + nghost;
            int sup = std::min(lDom[d].last(), dom[d].last()) - lDom[d].first() + nghost;
            // inf and sup bounds must be within the domain to reduce, if not no need to reduce
            if (sup < inf) {
                return;
            }

            begin[d] = inf;
            // The +1 is for Kokkos loop
            end[d] = sup + 1;
        }

        // Iterate along cutAxis
        for (int i = cutAxisFirst; i <= cutAxisLast; i++) {
            begin[cutAxis] = i;
            end[cutAxis]   = i + 1;

            // Reducing over perpendicular plane defined by cutAxis
            Tf tempRes = Tf(0);

            using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
            ippl::parallel_reduce(
                "ORB weight reduction", createRangePolicy<Dim, exec_space>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& args, Tf& weight) {
                    weight += apply(data, args);
                },
                Kokkos::Sum<Tf>(tempRes));

            Kokkos::fence();

            rankWeights[arrayStart++] = tempRes;
        }
    }

    template <class Field, class Tp>
    int OrthogonalRecursiveBisection<Field, Tp>::findMedian(std::vector<Tf>& w) {
        // Special case when array must be cut in half in order to not have planes
        if (w.size() == 4) {
            return 1;
        }
        // Get total sum of array
        Tf tot = std::accumulate(w.begin(), w.end(), Tf(0));

        // Find position of median as half of total in array
        Tf half = 0.5 * tot;
        Tf curr = Tf(0);
        // Do not need to iterate to full extent since it must not give planes
        for (unsigned int i = 0; i < w.size() - 1; i++) {
            curr += w[i];
            if (curr >= half) {
                // If all particles are in the first plane, cut at 1 so to have size 2
                if (i == 0) {
                    return 1;
                }
                Tf previous = curr - w[i];
                // curr - half < half - previous
                if ((curr + previous) <= tot
                    && curr != half) {  // if true then take current i, otherwise i-1
                    if (i == w.size() - 2) {
                        return (i - 1);
                    } else {
                        return i;
                    }
                } else {
                    return (i > 1) ? (i - 1) : 1;
                }
            }
        }
        // If all particles are in the last plane, cut two indices before the end so to have size 2
        return w.size() - 3;
    }

    template <class Field, class Tp>
    void OrthogonalRecursiveBisection<Field, Tp>::cutDomain(std::vector<NDIndex<Dim>>& domains,
                                                            std::vector<int>& procs, int it,
                                                            int cutAxis, int median) {
        // Cut domains[it] in half at median along cutAxis
        NDIndex<Dim> leftDom, rightDom;
        domains[it].split(leftDom, rightDom, cutAxis, median + domains[it][cutAxis].first());
        domains[it] = leftDom;
        domains.insert(domains.begin() + it + 1, rightDom);

        // Cut procs in half
        int temp  = procs[it];
        procs[it] = procs[it] / 2;
        procs.insert(procs.begin() + it + 1, temp - procs[it]);
    }

    template <class Field, class Tp>
    template <typename Attrib>
    void OrthogonalRecursiveBisection<Field, Tp>::scatterR(const Attrib& r) {
        using vector_type = typename mesh_type::vector_type;
        static_assert(
            Kokkos::SpaceAccessibility<typename Attrib::memory_space,
                                       typename Field::memory_space>::accessible,
            "Particle attribute memory space must be accessible from ORB field memory space");

        // Reset local field
        bf_m = 0.0;
        // Get local data
        auto view                      = bf_m.getView();
        const mesh_type& mesh          = bf_m.get_mesh();
        const FieldLayout<Dim>& layout = bf_m.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = bf_m.getNghost();

        // Get spacings
        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        using policy_type = Kokkos::RangePolicy<size_t, typename Field::execution_space>;

        Kokkos::parallel_for(
            "ParticleAttrib::scatterR", policy_type(0, r.getParticleCount()),
            KOKKOS_LAMBDA(const size_t idx) {
                // Find nearest grid point
                Vector<Tp, Dim> l      = (r(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index = l;
                Vector<Tf, Dim> whi    = l - index;
                Vector<Tf, Dim> wlo    = 1.0 - whi;

                Vector<size_t, Dim> args = index - lDom.first() + nghost;

                // Scatter
                scatterToField(std::make_index_sequence<1 << Dim>{}, view, wlo, whi, args);
            });

        bf_m.accumulateHalo();
    }

}  // namespace ippl
