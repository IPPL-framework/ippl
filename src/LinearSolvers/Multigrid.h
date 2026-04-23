//
// MULTIGRID
//

#ifndef IPPL_MULTIGRID
#define IPPL_MULTIGRID

#include "Kokkos_Core.hpp"
#include "IpplCore.h"

#include <cstddef>
#include <iostream>
#include <ostream>
#include <vector>

#include "Types/Vector.h"

#include "LinearSolvers/Preconditioner.h"
namespace ippl {
    namespace multigrid {
        template <typename Field>
        struct Level {
            constexpr static unsigned Dim = Field::dim;
            ippl::Vector<int, Dim> nx;
            ippl::Vector<double, Dim> hx;
            ippl::Vector<double, Dim> origin;

            Field u, f, r, t;
        };

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

        constexpr int power3(int d) {
            return (d == 0) ? 1 : 3 * power3(d - 1);
        }

        // Helper to unpack an array of indices into a Kokkos View
        template <typename ViewType, typename ArrayType, std::size_t... Is>
        KOKKOS_INLINE_FUNCTION double access_view_impl(const ViewType& view, const ArrayType& idx,
                                                       std::index_sequence<Is...>) {
            return view(idx[Is]...);
        }

        template <unsigned Dim, typename ViewType, typename ArrayType>
        KOKKOS_INLINE_FUNCTION double access_view(const ViewType& view, const ArrayType& idx) {
            return access_view_impl(view, idx, std::make_index_sequence<Dim>{});
        }
    }  // namespace multigrid

    template <typename Field, typename OperatorF>
    struct multigrid_preconditioner : public preconditioner<Field> {
        constexpr static unsigned Dim = Field::dim;
        using mesh_type               = typename Field::Mesh_t;
        using layout_type             = typename Field::Layout_t;

    public:
        multigrid_preconditioner(OperatorF&& op, std::vector<multigrid::Level<Field>>&& hierarchy,
                                 unsigned pre_smooth_iters = 2, unsigned post_smooth_iters = 2,
                                 double omega_jacobi = 0.8)
            : preconditioner<Field>("Multigrid")
            , initialized(true)
            , L(std::move(hierarchy))
            , nu1(pre_smooth_iters)
            , nu2(post_smooth_iters)
            , omega(omega_jacobi) {
            op_m = std::move(op);
        }
        Field operator()(const Field& b) override {
            L[0].f = b.deepCopy();
            for (size_t level = 0; level < L.size(); ++level)
                L[level].u = 0.0;
            vcycle(0);
            return L[0].u;
        }

    protected:
        bool initialized;
        std::vector<multigrid::Level<Field>> L;
        OperatorF op_m;
        unsigned nu1, nu2;
        double omega;

        Field residual(const Field& u, const Field& f) { return f - op_m(u); };

        void vcycle(size_t level) {
            auto& lev = L[level];

            if (level == L.size() - 1) {
                // Coarsest grid: just smooth a lot (or use a direct solver)
                smooth_jacobi(level, 50);
                return;
            }
            smooth_jacobi(level, nu1);   // Pre-smoothing
            restrict_fullweight(level);  // Pass level
            vcycle(level + 1);           // Recursively go down one level
            prolong_add(level);          // Pass level
            smooth_jacobi(level, nu2);   // Post-smoothing
        };

        void smooth_jacobi(const size_t level, const unsigned iters, double omega) {
            auto& lev = L[level];
            auto &u = lev.u, f = lev.f;

            const auto diag = multigrid::compute_diag(lev);

            for (unsigned it = 0; it < iters; ++it) {
                Field res = residual(u, f);
                u         = u + omega * (res / diag);
            }
        };

        // void restrict_fullweight(const size_t level) {
        //     if (level == L.size() - 1)
        //         std::cerr << "Trying to restrict at lowest level.";
        //
        //     auto& lev_fine   = L[level];
        //     auto& lev_coarse = L[level + 1];
        //     auto &u_fine = lev_fine.u, f_fine = lev_fine.f;
        //     auto &u_coarse = lev_coarse.u, f_coarse = lev_coarse.f;
        //
        //     Field residual_fine = residual(u_fine, f_fine);
        //
        //     residual_fine.fillHalo();
        //     f_coarse = 0.0;
        // };

        void restrict_fullweight(const size_t level) {
            if (level >= L.size() - 1) {
                std::cerr << "Trying to restrict at lowest level." << std::endl;
                return;
            }

            auto& lev_fine   = L[level];
            auto& lev_coarse = L[level + 1];

            // 1. Calculate and sync residual
            Field residual_fine = residual(lev_fine.u, lev_fine.f);
            residual_fine.fillHalo();  // IPPL halo exchange
            lev_coarse.f = 0.0;

            // 2. Setup domains and views
            const auto lDomF = residual_fine.getLayout().getLocalNDIndex();
            const auto lDomC = lev_coarse.f.getLayout().getLocalNDIndex();

            const int nghF = residual_fine.getNghost();
            const int nghC = lev_coarse.f.getNghost();

            auto rf = residual_fine.getView();
            auto fc = lev_coarse.f.getView();

            // N-dimensional grid sizes for boundaries
            Kokkos::Array<int, Dim> nxc;
            for (unsigned d = 0; d < Dim; ++d) {
                nxc[d] = lev_coarse.nx[d];
            }

            constexpr int stencil_size = multigrid::power3(Dim);
            constexpr double denom     = static_cast<double>(1 << (2 * Dim));  // 4^Dim

            // 3. N-Dimensional Kokkos Loop
            ippl::parallel_for(
                "restrict_fullweight", lev_coarse.f.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const auto... args) {
                    // Pack variadic arguments into a generic array
                    Kokkos::Array<int, Dim> idxC{static_cast<int>(args)...};
                    Kokkos::Array<int, Dim> global_idxC;

                    bool outside = false;

                    // Check domain bounds generically
                    for (unsigned d = 0; d < Dim; ++d) {
                        global_idxC[d] = idxC[d] + lDomC[d].first() - nghC;
                        if (global_idxC[d] < 0 || global_idxC[d] >= nxc[d]) {
                            outside = true;
                        }
                    }

                    if (outside)
                        return;

                    // Map coarse coordinates to fine coordinates
                    Kokkos::Array<int, Dim> idxF_center;
                    for (unsigned d = 0; d < Dim; ++d) {
                        int global_idxF = 2 * global_idxC[d];
                        idxF_center[d]  = global_idxF - lDomF[d].first() + nghF;
                    }

                    double sum = 0.0;

                    // Flat loop over the 3^Dim stencil
                    for (int s = 0; s < stencil_size; ++s) {
                        int temp   = s;
                        int zeroes = 0;
                        Kokkos::Array<int, Dim> idxF_current;

                        // Decode flat index 's' into N-dimensional offsets (-1, 0, 1)
                        for (unsigned d = 0; d < Dim; ++d) {
                            int offset = (temp % 3) - 1;
                            temp /= 3;

                            idxF_current[d] = idxF_center[d] + offset;
                            if (offset == 0)
                                zeroes++;
                        }

                        // Weight is 2^(number of zero-offsets)
                        const double w = static_cast<double>(1 << zeroes);

                        sum += w * multigrid::access_view<Dim>(rf, idxF_current);
                    }

                    // Write to coarse grid view using unpacking helper
                    multigrid::access_view_impl(fc, idxC, std::make_index_sequence<Dim>{}) =
                        sum / denom;
                });

            ippl::fence();

            // 4. Apply IPPL Boundary Conditions
            // Instead of hardcoding 0.0 on boundaries inside the Kokkos loop, we compute
            // the restriction everywhere on the physical domain, and then let IPPL's
            // generalized boundary condition mechanisms overwrite the edges.
            // Make sure your lev_coarse.f has its boundary conditions configured!

            // (Assuming you configured field BCs elsewhere in the code like:)
            // lev_coarse.f.setFieldBC(...);

            // Update the boundaries based on IPPL configurations:
            // lev_coarse.f.applyBoundaryConditions(); // Or however it is invoked in your IPPL
            // version
        }

        void prolong_add(const size_t level) {
            if (level >= L.size() - 1)
                std::cerr << "Trying to prolong at invalid level" << std::endl;
            return;

            auto& lev_fine   = L[level];
            auto& lev_coarse = L[level + 1];

            // 1. Sync coarse grid ghost cells (crucial because interpolation reads adjacent coarse
            // nodes)
            lev_coarse.u.fillHalo();

            // 2. Setup domains and views
            const auto lDomF = lev_fine.u.getLayout().getLocalNDIndex();
            const auto lDomC = lev_coarse.u.getLayout().getLocalNDIndex();

            const int nghF = lev_fine.u.getNghost();
            const int nghC = lev_coarse.u.getNghost();

            auto uf = lev_fine.u.getView();
            auto uc = lev_coarse.u.getView();

            Kokkos::Array<int, Dim> nxf;
            for (unsigned d = 0; d < Dim; ++d) {
                nxf[d] = lev_fine.nx[d];
            }

            constexpr int num_corners =
                1 << Dim;  // 2^Dim adjacent coarse nodes for N-linear interpolation

            // 3. N-Dimensional Kokkos Loop over the Fine Grid
            ippl::parallel_for(
                "prolong_add", lev_fine.u.getFieldRangePolicy(), KOKKOS_LAMBDA(const auto... args) {
                    Kokkos::Array<int, Dim> idxF{static_cast<int>(args)...};
                    Kokkos::Array<int, Dim> global_idxF;

                    bool outside = false;

                    // Compute global indices and check domain bounds
                    for (unsigned d = 0; d < Dim; ++d) {
                        global_idxF[d] = idxF[d] + lDomF[d].first() - nghF;
                        if (global_idxF[d] < 0 || global_idxF[d] >= nxf[d]) {
                            outside = true;
                        }
                    }

                    if (outside)
                        return;

                    // Find the base coarse coordinate (bottom-left-front equivalent) and the offset
                    // remainder
                    Kokkos::Array<int, Dim> idxC_base;
                    Kokkos::Array<int, Dim> rem;

                    for (unsigned d = 0; d < Dim; ++d) {
                        int global_idxC = global_idxF[d] / 2;
                        rem[d] = global_idxF[d] % 2;  // 0 if aligned with coarse node, 1 if halfway
                        idxC_base[d] = global_idxC - lDomC[d].first() + nghC;
                    }

                    double interp_val = 0.0;

                    // Loop over the 2^Dim corners of the containing coarse cell
                    for (int s = 0; s < num_corners; ++s) {
                        double weight = 1.0;
                        Kokkos::Array<int, Dim> idxC_current;

                        for (unsigned d = 0; d < Dim; ++d) {
                            int b = (s >> d) & 1;  // Extract the bit for dimension d (0 or 1)
                            idxC_current[d] = idxC_base[d] + b;

                            // N-linear interpolation weights:
                            // If exactly on a coarse point (rem == 0), weight is 1.0 for b=0, and
                            // 0.0 for b=1. If halfway (rem == 1), weight is 0.5 for both b=0 and
                            // b=1.
                            double w_d = (rem[d] == 0) ? (b == 0 ? 1.0 : 0.0) : 0.5;
                            weight *= w_d;
                        }

                        // Only read from memory if this corner actually contributes
                        if (weight > 0.0) {
                            interp_val += weight * multigrid::access_view<Dim>(uc, idxC_current);
                        }
                    }

                    // Multigrid prolongation ADDS the coarse correction to the existing fine
                    // solution
                    multigrid::access_view_impl(uf, idxF, std::make_index_sequence<Dim>{}) +=
                        interp_val;
                });

            ippl::fence();

            // Note: Boundary conditions on lev_fine.u are usually enforced during
            // the subsequent post-smoothing step (smooth_jacobi), so we don't
            // strictly need to call applyBoundaryConditions() right here.
        }
    };

}  // namespace ippl

#endif  // !IPPL_MULTIGRID
