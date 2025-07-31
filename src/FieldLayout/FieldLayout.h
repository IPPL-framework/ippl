//
// Class FieldLayout
//   FieldLayout describes how a given index space (represented by an NDIndex
//   object) is distributed among MPI ranks. It performs the initial
//   partitioning. The user may request that a particular dimension not be
//   partitioned by flagging that axis as 'SERIAL' (instead of 'PARALLEL').
//
#ifndef IPPL_FIELD_LAYOUT_H
#define IPPL_FIELD_LAYOUT_H

#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "Types/ViewTypes.h"

#include "Communicate/Communicator.h"
#include "Index/NDIndex.h"
#include "Partition/Partitioner.h"

namespace ippl {

    template <unsigned Dim>
    class FieldLayout;

    template <unsigned Dim>
    std::ostream& operator<<(std::ostream&, const FieldLayout<Dim>&);

    // enumeration used to describe a hypercube's relation to
    // a particular axis in a given bounded domain
    enum e_cube_tag {
        UPPER       = 0,
        LOWER       = 1,
        IS_PARALLEL = 2
    };

    namespace detail {
        /*!
         * Counts the hypercubes in a given dimension
         * @param dim the dimension
         * @return 3^n
         */
        KOKKOS_INLINE_FUNCTION constexpr unsigned int countHypercubes(unsigned int dim) {
            unsigned int ret = 1;
            for (unsigned int d = 0; d < dim; d++) {
                ret *= 3;
            }
            return ret;
        }

        /*!
         * Compile-time evaluation of x!
         * @param x input value
         * @return x factorial
         */
        constexpr unsigned int factorial(unsigned x) {
            return x == 0 ? 1 : x * factorial(x - 1);
        }

        /*!
         * Compile-time evaluation of binomial coefficients, aka
         * elements of Pascal's triangle, aka choices from combinatorics, etc
         * @param a number of options
         * @param b number of choices
         * @return a choose b
         */
        constexpr unsigned int binomialCoefficient(unsigned a, unsigned b) {
            return factorial(a) / (factorial(b) * factorial(a - b));
        }

        /*!
         * Compile-time evaluation of the number of hypercubes of dimension m in a hypercube
         * of dimension Dim
         * @tparam Dim parent hypercube dimension
         * @param m sub-hypercube dimension
         * @return The number of m-cubes in an n-cube
         */
        template <unsigned Dim>
        constexpr unsigned int countCubes(unsigned m) {
            return (1 << (Dim - m)) * binomialCoefficient(Dim, m);
        }

        /*!
         * Determines whether a facet is on the upper boundary
         * of its domain. For lower dimension hypercubes, determines
         * whether the component is on the upper boundary of
         * the domain along at least one axis
         * @param face the hypercube's index
         * @return Whether it touches any upper boundary
         */
        bool isUpper(unsigned int face);

        /*!
         * Determine the axis perpendicular to a given facet
         * (throws an exception if the index does not correspond
         * to a facet)
         * @param face the facet's index
         * @return The index of the axis perpendicular to that facet
         */
        unsigned int getFaceDim(unsigned int face);

        /*!
         * Converts between ternary encoding and face set indexing
         * @tparam Dim the number of dimensions
         * @param index the ternary-encoded index of a facet in [0, 3^Dim)
         * @return The index of that facet in a set of faces in [0, 2*Dim)
         */
        template <unsigned Dim>
        unsigned int indexToFace(unsigned int index) {
            // facets are group low/high by axis
            unsigned int axis = index / 2;
            // the digit to subtract is determined by whether the index describes an upper
            // face (even index) or lower face (odd index) and that digit's position is
            // determined by the axis of the face
            unsigned int toRemove = (2 - index % 2) * countHypercubes(axis);
            // start with all 2s (in base 3) and change the correct digit to get the encoded face
            return countHypercubes(Dim) - 1 - toRemove;
        }

        /*!
         * Computes the ternary-encoded index of a hypercube
         * @tparam Dim the number of dimensions in the full hypercube
         * @tparam CubeTags... variadic argument list, must be all e_cube_tag
         * @param tag(s...) the tags describing the hypercube of interest
         * @return The index of the desired hypercube
         */
        template <
            unsigned Dim, typename... CubeTags,
            typename = std::enable_if_t<sizeof...(CubeTags) == Dim - 1>,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<e_cube_tag, CubeTags>...>>>
        unsigned int getCube(e_cube_tag tag, CubeTags... tags) {
            if constexpr (Dim == 1) {
                return tag;
            } else {
                return tag + 3 * getCube<Dim - 1>(tags...);
            }
        }

        /*!
         * Utility function for getFace
         */
        template <size_t... Idx>
        unsigned int getFace_impl(const std::array<e_cube_tag, sizeof...(Idx)>& args,
                                  const std::index_sequence<Idx...>&) {
            return getCube<sizeof...(Idx)>(args[Idx]...);
        }

        /*!
         * Convenience alias for getCube for getting facets
         * @tparam Dim the number of dimensions in the parent hypercube
         * @param axis the axis perpendicular to the facet
         * @param side whether the facet is an upper or lower facet
         * @return The index of the facet
         */
        template <unsigned Dim>
        unsigned int getFace(unsigned int axis, e_cube_tag side) {
            std::array<e_cube_tag, Dim> args;
            args.fill(IS_PARALLEL);
            args[axis] = side;
            return getFace_impl(args, std::make_index_sequence<Dim>{});
        }
    }  // namespace detail

    template <unsigned Dim>
    class FieldLayout {
    public:
        using NDIndex_t        = NDIndex<Dim>;
        using view_type        = typename detail::ViewType<NDIndex_t, 1>::view_type;
        using host_mirror_type = typename view_type::host_mirror_type;

        struct bound_type {
            // lower bounds (ordering: x, y, z, ...)
            std::array<long, Dim> lo;
            // upper bounds (ordering: x, y, z, ...)
            std::array<long, Dim> hi;

            /*!
             * Compute the size of the region described by the bounds
             * @return Product of the axial dimensions of the region
             */
            long size() const {
                long total = 1;
                for (unsigned d = 0; d < Dim; d++) {
                    total *= hi[d] - lo[d];
                }
                return total;
            }
        };

        using rank_list   = std::vector<int>;
        using bounds_list = std::vector<bound_type>;

        using neighbor_list       = std::array<rank_list, detail::countHypercubes(Dim) - 1>;
        using neighbor_range_list = std::array<bounds_list, detail::countHypercubes(Dim) - 1>;

        /*!
         * Default constructor, which should only be used if you are going to
         * call 'initialize' soon after (before using in any context)
         */
        FieldLayout(const mpi::Communicator& = MPI_COMM_WORLD);

        FieldLayout(mpi::Communicator, const NDIndex<Dim>& domain, std::array<bool, Dim> decomp,
                    bool isAllPeriodic = false);

        // Destructor: Everything deletes itself automatically ... the base
        // class destructors inform all the FieldLayoutUser's we're going away.
        virtual ~FieldLayout() = default;

        // Initialization functions, only to be called by the user of FieldLayout
        // objects when the FieldLayout was created using the default constructor;
        // otherwise these are only called internally by the various non-default
        // FieldLayout constructors:

        void initialize(const NDIndex<Dim>& domain, std::array<bool, Dim> decomp,
                        bool isAllPeriodic = false);

        // Return the domain.
        const NDIndex<Dim>& getDomain() const { return gDomain_m; }

        // Compare FieldLayouts to see if they represent the same domain; if
        // dimensionalities are different, the NDIndex operator==() will return
        // false:
        template <unsigned Dim2>
        bool operator==(const FieldLayout<Dim2>& x) const {
            return gDomain_m == x.getDomain();
        }

        bool operator==(const FieldLayout<Dim>& x) const {
            for (unsigned int i = 0; i < Dim; ++i) {
                if (hLocalDomains_m(comm.rank())[i] != x.getLocalNDIndex()[i]) {
                    return false;
                }
            }
            return true;
        }

        // for the requested dimension, report if the distribution is
        // SERIAL or PARALLEL
        bool getDistribution(unsigned int d) const {
            return minWidth_m[d] == (unsigned int)gDomain_m[d].length();
        }

        // for the requested dimension, report if the distribution was requested to
        // be SERIAL or PARALLEL
        std::array<bool, Dim> isParallel() const { return isParallelDim_m; }

        // Get the local domain for the current rank.
        const NDIndex_t& getLocalNDIndex() const;

        // Get the local domain for a specific rank.
        const NDIndex_t& getLocalNDIndex(int rank) const;

        const host_mirror_type getHostLocalDomains() const;

        const view_type getDeviceLocalDomains() const;

        /*!
         * Get a list of all the neighbors, arranged by ternary encoding
         * of the hypercubes
         * @return List of list of neighbor ranks touching each boundary component
         */
        const neighbor_list& getNeighbors() const;

        /*!
         * Get the domain ranges corresponding to regions that should be sent
         * to neighbor ranks
         * @return Ranges to send
         */
        const neighbor_range_list& getNeighborsSendRange() const;

        /*!
         * Get the domain ranges corresponding to regions that should be received
         * from neighbor ranks
         * @return Ranges to receive
         */
        const neighbor_range_list& getNeighborsRecvRange() const;

        /*!
         * Given the index of a hypercube, find the index of the opposite hypercube,
         * i.e. the component with the same codimension belonging to a neighboring domain
         * that touches the hypercube with the given index, as determined by the
         * ternary encoding for hypercubes.
         *
         * For neighbor communication, the opposite component is the one that receives
         * sent data or sends us data to receive for a given component.
         *
         * The matching index is given by swapping alls 1s for 0s and vice versa in
         * the ternary encoding, while keeping the 2s unchanged. This can be understood
         * from the fact that if the local component is on the upper boundary of the local
         * domain, the neighbor component must be on the lower boundary of its local domain,
         * and vice versa. The 2s are unchanged because both the local component and the
         * neighbor component must be parallel to the same axes, otherwise their intersection
         * would have lower or higher dimension than the components themselves.
         * @param index index of the known component
         * @return Index of the matching component
         */
        static int getMatchingIndex(int index);

        /*!
         * Recursively finds neighbor ranks for layouts with all periodic boundary
         * conditions
         * @param nghost number of ghost cells
         * @param localDomain the rank's local domain
         * @param grown the local domain, grown by the number of ghost cells
         * @param neighborDomain a candidate neighbor rank's domain
         * @param rank the candidate neighbor's rank
         * @param offsets a dictionary containing offsets along different dimensions
         * @param d0 the dimension from which to start checking (default 0)
         * @param codim the codimension of overlapping regions to check (default 0)
         */
        void findPeriodicNeighbors(const int nghost, const NDIndex<Dim>& localDomain,
                                   NDIndex<Dim>& grown, NDIndex<Dim>& neighborDomain,
                                   const int rank, std::map<unsigned int, int>& offsets,
                                   unsigned d0 = 0, unsigned codim = 0);

        /*!
         * Finds all neighboring ranks based on the field layout
         * @param nghost number of ghost cells (default 1)
         */
        void findNeighbors(int nghost = 1);

        /*!
         * Adds a neighbor to the neighbor list
         * @param gnd the local domain, including ghost cells
         * @param nd the local domain
         * @param ndNeighbor the neighbor rank's domain
         * @param intersect the intersection of the domains
         * @param nghost number of ghost cells
         * @param rank the neighbor's rank
         */
        void addNeighbors(const NDIndex_t& gnd, const NDIndex_t& nd, const NDIndex_t& ndNeighbor,
                          const NDIndex_t& intersect, int nghost, int rank);

        void write(std::ostream& = std::cout) const;

        void updateLayout(const std::vector<NDIndex_t>& domains);

        bool isAllPeriodic_m;

        mpi::Communicator comm;

    private:
        /*!
         * Obtain the bounds to send / receive. The second domain, i.e.,
         * nd2, is grown by nghost cells in each dimension in order to
         * figure out the intersecting cells.
         * @param nd1 either remote or owned domain
         * @param nd2 either remote or owned domain
         * @param offset to map global to local grid point
         * @param nghost number of ghost cells per dimension
         */
        bound_type getBounds(const NDIndex_t& nd1, const NDIndex_t& nd2, const NDIndex_t& offset,
                             int nghost);

        int getPeriodicOffset(const NDIndex_t& nd, const unsigned int d, const int k);

    private:
        //! Global domain
        NDIndex_t gDomain_m;

        //! Local domains (device view)
        view_type dLocalDomains_m;

        //! Local domains (host mirror view)
        host_mirror_type hLocalDomains_m;

        std::array<bool, Dim> isParallelDim_m;

        unsigned int minWidth_m[Dim];

        neighbor_list neighbors_m;
        neighbor_range_list neighborsSendRange_m, neighborsRecvRange_m;

        void calcWidths();
    };

    template <unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const FieldLayout<Dim>& f) {
        f.write(out);
        return out;
    }
}  // namespace ippl

#include "FieldLayout/FieldLayout.hpp"

#endif
