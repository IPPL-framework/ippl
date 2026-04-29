#ifndef IPPL_FFT_TRANSFORM_PRUNEDRC_H
#define IPPL_FFT_TRANSFORM_PRUNEDRC_H

#include <array>
#include <mpi.h>

#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {
    //=========================================================================
    // Pruned Real-to-Complex Transform
    //
    // The key insight that makes this completely communication-free per
    // transform is the choice of heFFTe outbox:
    //
    //   R2C dim (d_r):
    //     Each rank's outbox lower bound in d_r = pruned field lower bound.
    //     The last rank (hi_pruned == K_r - 1) extends its outbox to cover
    //     [K_r, N_r/2], which are the "extra" full-complex modes discarded by
    //     the pruning.  This ensures every rank's outbox contains exactly the
    //     full-complex indices its pruned modes map to (fi = gi, direct).
    //
    //   Non-R2C dims:
    //     Every rank owns the FULL extent [0, N_d - 1].  The wrapping formula
    //     fj = gj < K/2 ? gj : N-K+gj  always maps into [0, N-1], which is
    //     always within this rank's outbox.
    //
    // Result: the local index in tempComplexFull_ is computable from the
    // pruned global index with no inter-rank communication.
    //=========================================================================

    template <typename RealField>
    class FFT<PrunedRCTransform, RealField> {
    public:
        static constexpr unsigned Dim = RealField::dim;

        using T         = typename RealField::value_type;
        using Complex_t = Kokkos::complex<T>;
        using MemSpace  = typename RealField::memory_space;
        using ExecSpace = typename RealField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

        using ComplexField =
            typename Field<Complex_t, Dim, typename RealField::Mesh_t,
                           typename RealField::Centering_t, ExecSpace>::uniform_type;

#ifdef IPPL_ENABLE_CUFFTMP
        using Backend_t = fft::CuFFTMpR2C<T, Dim, MemSpace>;
#else
        using Backend_t = fft::HeffteR2C<T, Dim, MemSpace>;
#endif
        using TempReal_t    = Kokkos::View<T***, Kokkos::LayoutLeft, MemSpace>;
        using TempComplex_t = Kokkos::View<Complex_t***, Kokkos::LayoutLeft, MemSpace>;

        FFT(const Layout_t& layoutReal,
            const Layout_t& layoutComplexFull,  // kept for API compatibility; outbox is recomputed
            const Layout_t& layoutComplexPruned, const PruningParams<Dim>& pruning,
            const ParameterList& params);

        void transform(TransformDirection direction, RealField& f, ComplexField& g);

    private:
        PruningParams<Dim> pruning_;
        std::unique_ptr<Backend_t> backend_;
        int r2c_dir_ = 0;

        // Local outbox origin and dimensions (= tempComplexFull_ dimensions).
        // For the R2C dim: origin = pruned field's lower bound on this rank.
        // For non-R2C dims: origin = 0, size = full global extent N_d.
        std::array<long long, 3> lowComplexFull_    = {};
        std::array<std::size_t, 3> fullComplexDims_ = {};

        // Global real-grid sizes for the wrapping formula:
        //   fj = gj < K/2 ? gj : N - K + gj
        std::array<long long, 3> globalRealDims_ = {};

        TempReal_t tempReal_;
        TempComplex_t tempComplexFull_;
    };

    //=========================================================================
    // Constructor
    //=========================================================================

    template <typename RealField>
    FFT<PrunedRCTransform, RealField>::FFT(const Layout_t& layoutReal,
                                           const Layout_t& /*layoutComplexFull*/,
                                           const Layout_t& layoutComplexPruned,
                                           const PruningParams<Dim>& pruning,
                                           const ParameterList& params)
        : pruning_(pruning) {
        static_assert(Dim == 3, "PrunedRCTransform currently only supports 3D");

        r2c_dir_ = params.get<int>("r2c_direction", 0);

        // Global real-grid sizes
        const auto& gDomReal = layoutReal.getDomain();
        for (int d = 0; d < 3; ++d)
            globalRealDims_[d] = gDomReal[d].length();

        // Inbox: this rank's local real slab
        std::array<long long, 3> lowReal, highReal;
        fft::domainToBounds<Dim>(layoutReal.getLocalNDIndex(), lowReal, highReal);

        // Outbox: custom — aligned to pruned in r2c_dir, full extent everywhere else
        //
        // In r2c_dir d_r:
        //   outbox lo = pruned lo   (so fi = gi_p is always within this rank's buffer)
        //   outbox hi = pruned hi   (except the last rank, which absorbs [K_r, N_r/2])
        //
        // In every other dim d:
        //   outbox = [0, N_d - 1]  (whole extent; wrapping formula always stays local)
        const auto& lDomPruned = layoutComplexPruned.getLocalNDIndex();

        std::array<long long, 3> lowOut, highOut;
        for (int d = 0; d < 3; ++d) {
            if (d == r2c_dir_) {
                const long long lo_p = lDomPruned[d].first();
                const long long hi_p = lDomPruned[d].last();
                const long long K_d  = static_cast<long long>(pruning_.n_modes[d]);
                const long long N_d  = globalRealDims_[d];
                const long long top  = N_d / 2;  // last valid index of R2C output (N/2+1 elements)

                lowOut[d] = lo_p;
                // If this rank owns the last pruned mode, extend outbox to cover
                // all remaining full-complex elements [K_d .. N_d/2]
                highOut[d] = (hi_p == K_d - 1) ? top : hi_p;
            } else {
                // Every rank owns the full extent of this dimension
                lowOut[d]  = 0;
                highOut[d] = globalRealDims_[d] - 1;
            }
        }

        heffte::box3d<long long> inbox{lowReal, highReal};
        heffte::box3d<long long> outbox{lowOut, highOut};

        backend_ =
            std::make_unique<Backend_t>(inbox, outbox, r2c_dir_, Comm->getCommunicator(), params);

        for (int d = 0; d < 3; ++d) {
            lowComplexFull_[d]  = lowOut[d];
            fullComplexDims_[d] = static_cast<std::size_t>(highOut[d] - lowOut[d] + 1);
        }
    }

    //=========================================================================
    // transform
    //=========================================================================

    template <typename RealField>
    void FFT<PrunedRCTransform, RealField>::transform(TransformDirection direction, RealField& f,
                                                      ComplexField& g) {
        auto fview    = f.getView();
        auto gview    = g.getView();
        const int ngf = f.getNghost();
        const int ngg = g.getNghost();

        // Ensure temp buffers
        if (tempReal_.size() != f.getOwned().size())
            tempReal_ = detail::shrinkView("pruned_r2c_real", fview, ngf);

        const std::size_t fullSize =
            fullComplexDims_[0] * fullComplexDims_[1] * fullComplexDims_[2];
        if (tempComplexFull_.size() != fullSize)
            tempComplexFull_ = TempComplex_t("pruned_r2c_complex_full", fullComplexDims_[0],
                                             fullComplexDims_[1], fullComplexDims_[2]);

        // Pruned domain info for the kernel
        const auto& lDomPruned = g.getLayout().getLocalNDIndex();
        const long long lp0    = lDomPruned[0].first();
        const long long lp1    = lDomPruned[1].first();
        const long long lp2    = lDomPruned[2].first();

        // Mode counts and real-grid sizes for the wrapping formula
        const long long K0 = pruning_.n_modes[0];
        const long long K1 = pruning_.n_modes[1];
        const long long K2 = pruning_.n_modes[2];
        const long long N0 = globalRealDims_[0];
        const long long N1 = globalRealDims_[1];
        const long long N2 = globalRealDims_[2];

        // outbox origins — only non-zero in r2c_dir
        const long long lcf0 = lowComplexFull_[0];
        const long long lcf1 = lowComplexFull_[1];
        const long long lcf2 = lowComplexFull_[2];
        const int r2c        = r2c_dir_;

        auto owned = g.getOwned();  // ghost-free extent of pruned field

        if (direction == FORWARD) {
            // 1. Strip ghosts → tempReal_
            auto tempreal = tempReal_;
            Kokkos::parallel_for(
                "r2c_copy_real_fwd",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {ngf, ngf, ngf}, {int(fview.extent(0)) - ngf, int(fview.extent(1)) - ngf,
                                      int(fview.extent(2)) - ngf}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    tempreal(i - ngf, j - ngf, k - ngf) = fview(i, j, k);
                });
            Kokkos::fence();

            // 2. Distributed R2C FFT → tempComplexFull_
            backend_->forward(tempReal_.data(), tempComplexFull_.data());

            // 3. Extract pruned modes from tempComplexFull_ → pruned output field
            //
            //    Every access is local by construction of the outbox:
            //      fi0_l = gi0 - lcf0  (R2C dim 0 assumed here; generalises below)
            //      fi1_l = wrap(gi1)   (lcf1=0 for non-R2C dims)
            //      fi2_l = wrap(gi2)
            auto& tcf = tempComplexFull_;
            Kokkos::parallel_for(
                "extract_pruned_r2c_fwd",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {int(owned[0].length()), int(owned[1].length()), int(owned[2].length())}),
                KOKKOS_LAMBDA(int i0, int i1, int i2) {
                    const long long gi0 = i0 + lp0;
                    const long long gi1 = i1 + lp1;
                    const long long gi2 = i2 + lp2;

                    const int fi0 =
                        (r2c == 0) ? int(gi0 - lcf0) : int((gi0 < K0 / 2) ? gi0 : (N0 - K0 + gi0));
                    const int fi1 =
                        (r2c == 1) ? int(gi1 - lcf1) : int((gi1 < K1 / 2) ? gi1 : (N1 - K1 + gi1));
                    const int fi2 =
                        (r2c == 2) ? int(gi2 - lcf2) : int((gi2 < K2 / 2) ? gi2 : (N2 - K2 + gi2));

                    gview(i0 + ngg, i1 + ngg, i2 + ngg) = tcf(fi0, fi1, fi2);
                });

        } else {  // BACKWARD

            // 1. Zero-fill tempComplexFull_, then scatter pruned modes into it
            Kokkos::deep_copy(tempComplexFull_, Complex_t(0, 0));

            auto& tcf = tempComplexFull_;
            Kokkos::parallel_for(
                "scatter_pruned_r2c_bwd",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {int(owned[0].length()), int(owned[1].length()), int(owned[2].length())}),
                KOKKOS_LAMBDA(int i0, int i1, int i2) {
                    const long long gi0 = i0 + lp0;
                    const long long gi1 = i1 + lp1;
                    const long long gi2 = i2 + lp2;

                    const int fi0 =
                        (r2c == 0) ? int(gi0 - lcf0) : int((gi0 < K0 / 2) ? gi0 : (N0 - K0 + gi0));
                    const int fi1 =
                        (r2c == 1) ? int(gi1 - lcf1) : int((gi1 < K1 / 2) ? gi1 : (N1 - K1 + gi1));
                    const int fi2 =
                        (r2c == 2) ? int(gi2 - lcf2) : int((gi2 < K2 / 2) ? gi2 : (N2 - K2 + gi2));

                    tcf(fi0, fi1, fi2) = gview(i0 + ngg, i1 + ngg, i2 + ngg);
                });
            Kokkos::fence();

            // 2. Distributed C2R backward
            backend_->backward(tempComplexFull_.data(), tempReal_.data());
            Kokkos::fence();

            // 3. Copy tempReal_ back (restore ghost padding)
            auto tempreal = tempReal_;
            Kokkos::parallel_for(
                "r2c_copy_real_bwd",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {ngf, ngf, ngf}, {int(fview.extent(0)) - ngf, int(fview.extent(1)) - ngf,
                                      int(fview.extent(2)) - ngf}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    fview(i, j, k) = tempreal(i - ngf, j - ngf, k - ngf);
                });
        }
    }

}  // namespace ippl

#endif