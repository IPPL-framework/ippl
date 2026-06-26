//
// MULTIGRID
//

#ifndef IPPL_MULTIGRID
#define IPPL_MULTIGRID

#include "Kokkos_Core.hpp"
#include "Ippl.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include "Field/BcTypes.h"

#include "Types/Vector.h"

#include "Utility/IpplTimings.h"

#include "FieldLayout/SubFieldLayout.h"
#include "Index/Index.h"
#include "Index/NDIndex.h"
#include "Kokkos_NumericTraits.hpp"
#include "LinearSolvers/Preconditioner.h"

namespace ippl {

    namespace multigrid {
        /**
         * @brief Represents a single level in the multigrid hierarchy.
         *
         * Each level contains its own mesh, layout, and fields for the solution and the source.
         *
         * @tparam Field The type of field used on this level.
         */
        template <typename Field>
        struct Level {
            constexpr static unsigned Dim = Field::dim;
            using mesh_type               = typename Field::Mesh_t;
            using layout_type             = typename Field::Layout_t;

            std::shared_ptr<mesh_type> mesh_ptr;
            std::shared_ptr<layout_type> layout_ptr;

            ippl::Vector<int, Dim> nx;
            ippl::Vector<double, Dim> hx;
            ippl::Vector<double, Dim> origin;

            Field u, f;

            /**
             * @brief Construct a new Level object.
             *
             * @tparam BCType The type of boundary conditions.
             * @param m Shared pointer to the mesh.
             * @param l Shared pointer to the layout.
             * @param bcs Reference to the boundary conditions to apply.
             */
            template <typename BCType>
            Level(std::shared_ptr<mesh_type> m, std::shared_ptr<layout_type> l, BCType& bcs)
                : mesh_ptr(m)
                , layout_ptr(l)
                , u(*m, *l)
                , f(*m, *l) {
                // Apply boundary conditions to all fields
                u.setFieldBC(bcs);
                f.setFieldBC(bcs);

                // Extract grid info from the provided mesh and layout
                origin      = m->getOrigin();
                auto domain = l->getDomain();
                for (unsigned d = 0; d < Dim; ++d) {
                    nx[d] = domain[d].length();
                    hx[d] = m->getMeshSpacing(d);
                }
            }
        };

        /**
         * @brief Computes the diagonal element of the Laplacian operator for a given level.
         *
         * @tparam LevelType The type of the Level object.
         * @param lev The level for which to compute the diagonal.
         * @return The computed diagonal value.
         */
        template <typename LevelType>
        double compute_diag(const LevelType& lev) {
            double diag = 0.0;

            // We iterate up to LevelType::Dim.
            // The compiler knows Dim at compile time, so it can heavily
            // optimize or completely "unroll" this loop.
            for (unsigned d = 0; d < LevelType::Dim; ++d) {
                double ihxd2 = 1.0 / (lev.hx[d] * lev.hx[d]);
                diag += 2.0 * ihxd2;
            }

            return diag;
        }
    }  // namespace multigrid

    /**
     * @brief Multigrid preconditioner for linear solvers.
     *
     * This preconditioner uses a V-cycle multigrid approach to accelerate the convergence
     * of iterative solvers like PCG.
     *
     * @tparam Field The type of field used for the solution and source.
     * @tparam OperatorF The type of the operator (e.g., a Laplacian).
     */
    template <typename Field, typename OperatorF>
    struct multigrid_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

    public:
        /**
         * @brief Construct a new multigrid_preconditioner object.
         *
         * @param op The operator to be preconditioned.
         * @param pre_smooth_iters Number of pre-smoothing iterations (default: 2).
         * @param post_smooth_iters Number of post-smoothing iterations (default: 2).
         * @param omega_jacobi Jacobi relaxation parameter (default: 0.8).
         * @param min_cells_per_rank_per_dim Minimum number of cells per rank per dimension on the
         * coarsest level (default: 4).
         * @param communication Whether to perform halo communication (default: true).
         */
        multigrid_preconditioner(OperatorF&& op, unsigned pre_smooth_iters = 2,
                                 unsigned post_smooth_iters = 2, double omega_jacobi = 0.8,
                                 unsigned min_cells_per_rank_per_dim = 4, bool communication = true)
            : preconditioner<Field>("Multigrid")
            , op_(std::forward<OperatorF>(op))
            , nu1_(pre_smooth_iters)
            , nu2_(post_smooth_iters)
            , omega_(omega_jacobi)
            , min_cells_per_rank_per_dim_(min_cells_per_rank_per_dim)
            , communication_(communication) {}

        // --- DEBUGGING ---

        /**
         * @brief Enables or disables debug printing of the multigrid state.
         *
         * @param enabled True to enable, false to disable.
         */
        void setDebugPrint(bool enabled) { debug_print_ = enabled; }

        /**
         * @brief Prints the current state of all multigrid levels for debugging.
         *
         * @param label A label to identify the current state.
         */
        void printDebugState(const std::string& label) const { debug_print_all_levels(label); }

        // --- END OF DEBUGGING ---

        /**
         * @brief Applies the multigrid preconditioner to the input field.
         *
         * Writes the preconditioned field M^{-1} b into the caller-provided
         * result buffer (allocated once by the owning solver via init_fields),
         * avoiding a per-call Field allocation / return-by-value copy.
         *
         * @param b The input field (right-hand side).
         * @param result The output buffer that receives the preconditioned field.
         */
        void operator()(Field& b, Field& result) override {
            IpplTimings::TimerRef mg = IpplTimings::getTimer("MG-PRECOND");
            IpplTimings::startTimer(mg);

            L_[0].f = b.deepCopy();

            // Remove Volume average if periodic
            if (is_all_periodic_) {
                auto avg = L_[0].f.getVolumeAverage();
                L_[0].f  = L_[0].f - avg;
                if (communication_)
                    L_[0].f.fillHalo();
            }

            for (size_t level = 0; level < L_.size(); ++level)
                L_[level].u = 0.0;
            vcycle(0);

            // Remove Volume average if periodic
            if (is_all_periodic_) {
                auto avg = L_[0].u.getVolumeAverage();
                L_[0].u  = L_[0].u - avg;
                if (communication_)
                    L_[0].u.fillHalo();
            }

            // Write the result into the caller-provided buffer
            Kokkos::deep_copy(result.getView(), L_[0].u.getView());

            IpplTimings::stopTimer(mg);
        }

        /**
         * @brief Initializes the multigrid hierarchy of fields based on the input field.
         *
         * This function calculates the number of levels and creates the corresponding
         * meshes and layouts for each level.
         *
         * @param b The input field used to initialize the hierarchy.
         */
        void init_fields(Field& b) override {
            IpplTimings::TimerRef init_fields = IpplTimings::getTimer("init_fields");
            IpplTimings::startTimer(init_fields);

            auto& fine_mesh              = b.get_mesh();
            auto& fine_layout            = b.getLayout();
            auto fine_domain             = fine_layout.getDomain();
            std::array<bool, Dim> decomp = fine_layout.isParallel();

            auto bcs = b.getFieldBC();

            // 1. Calculate number of levels
            const auto& localFine = fine_layout.getLocalNDIndex();  // this rank's slab
            int min_local         = Kokkos::Experimental::finite_max<int>::value;

            for (unsigned d = 0; d < Dim; ++d) {
                // For serial dims, every rank's local extent == global extent, so this naturally
                // covers them too. We do not have to check if dim is parallel or not.
                min_local = Kokkos::min(min_local, static_cast<int>(localFine[d].length()));
            }

            // reduce over all ranks to find smallest dimensional extent
            int min_local_global = min_local;
            ippl::Comm->allreduce(min_local, min_local_global, 1, std::less<int>{});

            int min_cells_on_coarsest = min_cells_per_rank_per_dim_;
            int nlevels               = 1;
            int at_level              = min_local_global;
            while (at_level / 2 >= min_cells_on_coarsest) {
                at_level /= 2;
                ++nlevels;
            }

            // reserve full vector to avoid implicit copying when adding new elements
            L_.clear();
            L_.reserve(nlevels);

            // 2. Build the hierarchy (Meshes, Layouts, and Levels)
            for (int ell = 0; ell < nlevels; ++ell) {
                // Create SubFieldLayout using strided index over the original mesh
                ippl::NDIndex<Dim> sub_domain;
                ippl::Vector<double, Dim> level_hx;

                for (unsigned d = 0; d < Dim; ++d) {
                    const int stride = 1 << ell;  // 2^ell
                    // strided sub-index: first, last, stride
                    sub_domain[d] =
                        ippl::Index(fine_domain[d].first(), fine_domain[d].last(), stride);
                    level_hx[d] = fine_mesh.getMeshSpacing(d) * stride;
                }

                auto level_layout = std::make_shared<ippl::SubFieldLayout<Dim>>(
                    fine_layout.comm, fine_domain, sub_domain, decomp, fine_layout.isAllPeriodic_m);

                auto level_mesh =
                    std::make_shared<mesh_type>(sub_domain, level_hx, fine_mesh.getOrigin());

                // create BCs for each level
                ippl::BConds<Field, Dim> level_bcs;
                for (size_t i = 0; i < 2 * Dim; ++i) {
                    if (bcs[i]) {
                        int bc_type = bcs[i]->getBCType();

                        if (bc_type == ippl::PERIODIC_FACE) {
                            // PeriodicFace has internal MPI state (faceNeighbors_m, haloData_m).
                            // We MUST create a fresh instance for every level so it binds
                            // correctly.
                            level_bcs[i] = std::make_shared<ippl::PeriodicFace<Field>>(i);

                        } else if (ell > 0
                                   && (bc_type == ippl::CONSTANT_FACE
                                       || bc_type == ippl::ZERO_FACE)) {
                            // Error equation on coarse grids: inhomogeneous Dirichlet becomes Zero
                            level_bcs[i] = std::make_shared<ippl::ZeroFace<Field>>(i);

                        } else {
                            // For ell == 0 CONSTANT_FACE, or ZERO_FACE, or NO_FACE,
                            // they have no internal layout state. We can safely share the pointer
                            level_bcs[i] = bcs[i];
                        }
                    }
                }

                L_.emplace_back(level_mesh, level_layout, level_bcs);
            }

            IpplTimings::stopTimer(init_fields);
        }

        /**
         * @brief Restricts the residual from a fine level to the next coarser level.
         *
         * This function calculates the residual on the fine grid and then
         * averages it to obtain the source term on the coarser grid.
         *
         * @param level The index of the fine level.
         */
        void restrict_average(const size_t level) {
            IpplTimings::TimerRef restrict = IpplTimings::getTimer("restrict_fullweight");
            IpplTimings::startTimer(restrict);

            if (level >= L_.size() - 1) {
                std::cerr << "Trying to restrict at lowest level." << std::endl;
                return;
            }

            auto& lev_fine   = L_[level];
            auto& lev_coarse = L_[level + 1];

            // 1. Calculate and sync residual
            Field residual_fine = residual(lev_fine.u, lev_fine.f);
            residual_fine.fillHalo();
            lev_coarse.f = 0.0;

            // 2. Setup domains and views
            const auto lDomF = residual_fine.getLayout().getLocalNDIndex();
            const auto lDomC = lev_coarse.f.getLayout().getLocalNDIndex();

            const int nghF = residual_fine.getNghost();
            const int nghC = lev_coarse.f.getNghost();

            auto rf = residual_fine.getView();
            auto fc = lev_coarse.f.getView();

            // 3. Calculate number of children per coarser cell
            constexpr int num_children = 1 << Dim;                           // 2^Dim
            constexpr double denom     = static_cast<double>(num_children);  // 2^Dim

            using index_array_type = typename RangePolicy<Dim>::index_array_type;

            // 4. N-Dimensional Kokkos Loop
            ippl::parallel_for(
                "restrict_average", lev_coarse.f.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // Local coarse index -> local fine "lower corner" of the contained block
                    ippl::Vector<int, Dim> idxF_base;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const int localC = static_cast<int>(args[d]) - nghC;

                        // Coarse point in the original fine-grid index space.
                        const int globalC = lDomC[d].first() + localC * lDomC[d].stride();

                        // Same physical/index-space point, expressed as fine-level view index.
                        idxF_base[d] = (globalC - lDomF[d].first()) / lDomF[d].stride() + nghF;
                    }

                    // Sum the 2^Dim contained fine cells
                    double sum = 0.0;
                    for (int s = 0; s < num_children; ++s) {
                        Vector<int, Dim> idxF = idxF_base;
                        for (unsigned d = 0; d < Dim; ++d) {
                            idxF[d] += (s >> d) & 1;
                        }
                        sum += apply(rf, idxF);
                    }

                    apply(fc, args) = sum / denom;
                });

            ippl::fence();

            // Remove Volume average if periodic
            if (is_all_periodic_) {
                auto avg     = lev_coarse.f.getVolumeAverage();
                lev_coarse.f = lev_coarse.f - avg;
            }

            if (communication_)
                lev_coarse.f.fillHalo();

            IpplTimings::stopTimer(restrict);
        }

        /**
         * @brief Prolongs the correction from a coarse level and adds it to the fine level.
         *
         * This function interpolates the solution from the coarser grid back to the
         * finer grid and adds the correction to the existing solution.
         *
         * @param level The index of the fine level to be corrected.
         */
        void prolong_add(const size_t level) {
            IpplTimings::TimerRef prolong = IpplTimings::getTimer("prolong");
            IpplTimings::startTimer(prolong);

            if (level >= L_.size() - 1) {
                std::cerr << "Trying to prolong at invalid level" << std::endl;
                return;
            }

            auto& lev_fine   = L_[level];
            auto& lev_coarse = L_[level + 1];

            // 1. Sync coarse grid ghost cells (crucial because interpolation reads adjacent coarse
            // nodes)
            if (communication_)
                lev_coarse.u.fillHalo();

            // 2. Setup domains and views
            const auto lDomF = lev_fine.u.getLayout().getLocalNDIndex();
            const auto lDomC = lev_coarse.u.getLayout().getLocalNDIndex();

            const int nghF = lev_fine.u.getNghost();
            const int nghC = lev_coarse.u.getNghost();

            auto uf = lev_fine.u.getView();
            auto uc = lev_coarse.u.getView();

            const auto gDomF = lev_fine.u.getLayout().getDomain();

            // 3. Calculate number of corners
            constexpr int num_corners = 1 << Dim;  // 2^Dim contributing coarse cells

            using index_array_type = typename RangePolicy<Dim>::index_array_type;

            // 4. N-Dimensional Kokkos Loop over the Fine Grid
            ippl::parallel_for(
                "prolong_add", lev_fine.u.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // For each dim: which coarse cell contains us, and on which side?
                    //   sgn = -1 -> fine cell in lower half, neighbor is C-1
                    //   sgn = +1 -> fine cell in upper half, neighbor is C+1
                    ippl::Vector<int, Dim> idxC_base;
                    ippl::Vector<int, Dim> sgn;
                    for (unsigned d = 0; d < Dim; ++d) {
                        // Fine-level point in original fine-grid index space.
                        const int localF  = static_cast<int>(args[d]) - nghF;
                        const int globalF = lDomF[d].first() + localF * lDomF[d].stride();

                        // Logical index on this fine level:
                        // level 0: global 0,1,2,3 -> logical 0,1,2,3
                        // level 1: global 0,2,4,6 -> logical 0,1,2,3
                        // level 2: global 0,4,8   -> logical 0,1,2
                        const int logicalF = (globalF - gDomF[d].first()) / lDomF[d].stride();

                        const int logicalC = logicalF / 2;
                        sgn[d]             = (logicalF % 2 == 0) ? -1 : +1;

                        // Coarse-level global coordinate corresponding to logicalC.
                        const int globalC = gDomF[d].first() + logicalC * lDomC[d].stride();

                        // Map coarse global coordinate to coarse view index.
                        idxC_base[d] = (globalC - lDomC[d].first()) / lDomC[d].stride() + nghC;
                    }

                    // Tensor product of {3/4 (containing), 1/4 (neighbor)} per dim
                    double interp_val = 0.0;
                    for (int s = 0; s < num_corners; ++s) {
                        Vector<int, Dim> idxC = idxC_base;
                        double weight         = 1.0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            int b = (s >> d) & 1;  // 0 = self, 1 = neighbor
                            idxC[d] += b * sgn[d];
                            weight *= (b == 0) ? 0.75 : 0.25;
                        }
                        interp_val += weight * apply(uc, idxC);
                    }

                    // Multigrid: ADD the correction onto the existing fine solution
                    apply(uf, args) += interp_val;
                });
            ippl::fence();

            IpplTimings::stopTimer(prolong);
        }

    protected:
        std::vector<multigrid::Level<Field>> L_;
        OperatorF op_;
        unsigned nu1_, nu2_;
        double omega_;
        unsigned min_cells_per_rank_per_dim_;
        bool communication_;
        bool is_all_periodic_ = false;

        // --- DEBUGGING ---

        bool debug_print_ = false;

        /**
         * @brief Debug function to print the information and content of a field.
         *
         * @param field The field to print.
         * @param name The name of the field for the label.
         * @param level The level index of the field.
         */
        void debug_print_field(const Field& field, const std::string& name, size_t level) const {
            if (!debug_print_) {
                return;
            }

            Kokkos::fence();

            const int myRank = ippl::Comm->rank();
            const int nRanks = ippl::Comm->size();

            for (int rank = 0; rank < nRanks; ++rank) {
                if (rank == myRank) {
                    std::cout << "\n";
                    std::cout << "============================================================\n";
                    std::cout << "MG DEBUG | rank " << myRank << " | level " << level << " | "
                              << name << "\n";
                    std::cout << "owned domain: " << field.getOwned() << "\n";
                    std::cout << "allocated domain including ghosts: " << field.getAllocated()
                              << "\n";
                    std::cout << "nghost: " << field.getNghost() << "\n";
                    std::cout << "------------------------------------------------------------\n";

                    // field.write(std::cout);

                    std::cout << "============================================================\n";
                    std::cout << std::flush;
                }

                ippl::Comm->barrier();
            }
        }

        /**
         * @brief Debug function to print the state of a specific multigrid level.
         *
         * @param level The level index to print.
         * @param label A label for the state being printed.
         */
        void debug_print_level(size_t level, const std::string& label) const {
            if (!debug_print_) {
                return;
            }

            const auto& lev = L_[level];

            std::cout << "\nMG DEBUG LEVEL " << level << ": " << label << "\n";
            std::cout << "nx = " << lev.nx << "\n";
            std::cout << "hx = " << lev.hx << "\n";
            std::cout << "origin = " << lev.origin << "\n";

            debug_print_field(lev.u, "u", level);
            debug_print_field(lev.f, "f", level);
        }

        /**
         * @brief Debug function to print the state of all multigrid levels.
         *
         * @param label A label for the state being printed.
         */
        void debug_print_all_levels(const std::string& label) const {
            if (!debug_print_) {
                return;
            }

            for (size_t level = 0; level < L_.size(); ++level) {
                debug_print_level(level, label);
            }
        }
        // --- END OF DEBUGGING ---

        /**
         * @brief Computes the residual of the current solution.
         *
         * The residual is computed as r = f - A*u, where A is the operator.
         *
         * @param u The current solution field.
         * @param f The source field.
         * @return The computed residual field.
         */
        Field residual(Field& u, const Field& f) {
            IpplTimings::TimerRef resi = IpplTimings::getTimer("residual");
            IpplTimings::startTimer(resi);

            Field res = f.deepCopy();
            res       = f - op_(u);

            IpplTimings::stopTimer(resi);

            return res;
        };

        /**
         * @brief Performs a multigrid V-cycle.
         *
         * This function recursively implements the V-cycle: pre-smoothing, restriction,
         * coarse-grid solve (or recursion), prolongation, and post-smoothing.
         *
         * @param level The current level index in the V-cycle.
         */
        void vcycle(size_t level) {
            if (level == L_.size() - 1) {
                // Coarsest grid: just smooth a lot (or use a direct solver)
                smooth_jacobi(level, 50);
                return;
            }
            smooth_jacobi(level, nu1_);  // Pre-smoothing
            restrict_average(level);     // Pass level
            vcycle(level + 1);           // Recursively go down one level
            prolong_add(level);          // Pass level
            smooth_jacobi(level, nu2_);  // Post-smoothing
        }

        /**
         * @brief Performs Jacobi smoothing on a given level.
         *
         * @param level The level index to be smoothed.
         * @param iters The number of smoothing iterations to perform.
         */
        void smooth_jacobi(const size_t level, const unsigned iters) {
            IpplTimings::TimerRef jacobi = IpplTimings::getTimer("smooth_jacobi");
            IpplTimings::startTimer(jacobi);

            auto& lev     = L_[level];
            auto& u       = lev.u;
            const auto& f = lev.f;

            const auto diag = multigrid::compute_diag(lev);

            for (unsigned it = 0; it < iters; ++it) {
                if (communication_)
                    u.fillHalo();

                Field res = residual(u, f);
                u         = u + omega_ * (res / diag);
            }
            IpplTimings::stopTimer(jacobi);
        }
    };
}  // namespace ippl

#endif  // !IPPL_MULTIGRID
