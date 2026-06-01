/*!
 * @file PrunedCC.h
 * @brief Pruned complex-to-complex FFT (PrunedCCTransform tag).
 *
 * Pruned FFTs keep only the lowest @c n_modes modes per axis. The
 * implementation runs @c 2^Dim sub-FFTs corresponding to all subsets of the
 * pruned axes, with optional concurrent execution on independent streams /
 * MPI communicator duplicates.
 */
#ifndef IPPL_FFT_TRANSFORM_PRUNEDCC_H
#define IPPL_FFT_TRANSFORM_PRUNEDCC_H

#include <array>
#include <mpi.h>
#include <type_traits>

#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

    namespace detail {
        /*!
         * @brief Run @p func(local) for @c local in [0, count) with optional
         *        OpenMP outer parallelism.
         *
         * On CPU execution spaces the loop is serial (Kokkos already
         * parallelizes the inner loops); on GPU execution spaces an OpenMP
         * outer loop is used so independent stream launches can overlap.
         *
         * @tparam IsCPU True when Kokkos is configured with an OpenMP-capable
         *               host execution space; false otherwise.
         */
        template <bool IsCPU, typename Func>
        inline void runConcurrentBatch(int count, Func&& func) {
            if constexpr (IsCPU) {
                // Serial outer loop for CPU (Kokkos will parallelize the inner loops)
                for (int local = 0; local < count; ++local) {
                    func(local);
                }
            } else {
                // OpenMP outer loop for GPU (to overlap asynchronous stream launches)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (int local = 0; local < count; ++local) {
                    func(local);
                }
            }
        }
    }  // namespace detail

    //=========================================================================
    // Pruned Complex-to-Complex Transform
    //=========================================================================

    /*!
     * @class FFT<PrunedCCTransform, ComplexField>
     * @brief Pruned C2C FFT keeping only the lowest n_modes per axis.
     *
     * Maintains @c 2^Dim heFFTe plans (one per subset of axes) and one
     * MPI communicator duplicate per concurrent sub-FFT. The number of
     * concurrent sub-FFTs is taken from the @c "num_concurrent_ffts"
     * parameter, clamped to [1, 2^Dim].
     *
     * @tparam ComplexField IPPL Field of Kokkos::complex elements.
     */
    template <typename ComplexField>
    class FFT<PrunedCCTransform, ComplexField> {
    public:
        static constexpr unsigned Dim   = ComplexField::dim;
        static constexpr int NumSubFFTs = 1 << Dim;

        using Complex_t = typename ComplexField::value_type;
        using T         = typename Complex_t::value_type;
        using MemSpace  = typename ComplexField::memory_space;
        using ExecSpace = typename ComplexField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

        using Backend_t  = fft::HeffteC2C<T, Dim, MemSpace>;
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::c2c;
        using GPUOps     = fft::Stream<MemSpace>;
        using Stream_t   = typename GPUOps::stream_type;
        using DeviceExec = typename GPUOps::exec_space;
        using TempView_t = typename Kokkos::View<typename ComplexField::view_type::data_type,
                                                 Kokkos::LayoutLeft, MemSpace>::uniform_type;

        /*!
         * @brief Build the pruned plan over the smaller of the two layouts.
         * @param layoutIn  Pre-pruning input layout.
         * @param layoutOut Post-pruning output layout.
         * @param pruning   Per-axis number of modes to retain.
         * @param params    Backend parameters; reads @c "num_concurrent_ffts".
         */
        FFT(const Layout_t& layoutIn, const Layout_t& layoutOut, const PruningParams<Dim>& pruning,
            const ParameterList& params)
            : pruning_(pruning)
            , numConcurrent_(std::clamp(params.get<int>("num_concurrent_ffts", 4), 1, NumSubFFTs)) {
            static_assert(Dim == 2 || Dim == 3, "Pruned FFT supports 2D and 3D");

            auto& prunedLayout =
                (layoutOut.getLocalNDIndex().size() < layoutIn.getLocalNDIndex().size()) ? layoutOut
                                                                                         : layoutIn;

            std::array<long long, 3> low, high;
            fft::domainToBounds<Dim>(prunedLayout.getLocalNDIndex(), low, high);
            heffte::box3d<long long> box{low, high};

            for (int s = 0; s < numConcurrent_; ++s) {
                MPI_Comm_dup(Comm->getCommunicator(), &comms_[s]);
                GPUOps::create(streams_[s]);
                backends_[s] = std::make_unique<Backend_t>(box, box, comms_[s], params);
            }
        }

        ~FFT() {
            for (int s = 0; s < numConcurrent_; ++s) {
                GPUOps::destroy(streams_[s]);
                MPI_Comm_free(&comms_[s]);
            }
        }

        /*!
         * @brief Pruned forward / backward C2C transform.
         * @param direction FORWARD or BACKWARD.
         * @param input     Pre-pruning input field.
         * @param output    Post-pruning output field.
         * @param dir       +1 / -1 swap-direction flag forwarded to the kernels.
         */
        void transform(TransformDirection direction, ComplexField& input, ComplexField& output,
                       int dir = 1) {
            if (direction == FORWARD) {
                forwardPruned(dir, input, output);
            } else {
                backwardPruned(dir, input, output);
            }
        }

        //! Forward pruned C2C kernel implementation (defined out-of-class below).
        void forwardPruned(int dir, ComplexField& input, ComplexField& output);
        //! Backward pruned C2C kernel implementation (defined out-of-class below).
        void backwardPruned(int dir, ComplexField& input, ComplexField& output);

    private:
        PruningParams<Dim> pruning_;
        int numConcurrent_;

        std::array<std::unique_ptr<Backend_t>, NumSubFFTs> backends_;
        std::array<TempView_t, NumSubFFTs> temps_;
        std::array<MPI_Comm, NumSubFFTs> comms_{};
        std::array<Stream_t, NumSubFFTs> streams_{};
    };

    //-------------------------------------------------------------------------
    // Forward Pruned C2C Implementation
    //-------------------------------------------------------------------------

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::forwardPruned(int dir, ComplexField& input,
                                                             ComplexField& output) {
        static IpplTimings::TimerRef twiddleTimer = IpplTimings::getTimer("TwiddleAdd");
        static IpplTimings::TimerRef subFFTTimer  = IpplTimings::getTimer("subFFTs");

        auto inView     = input.getView();
        auto outView    = output.getView();
        const int ngIn  = input.getNghost();
        const int ngOut = output.getNghost();

        const auto& lDomPruned = output.getLayout().getLocalNDIndex();
        const auto& gDomFull   = input.getLayout().getDomain();
        const auto& modes      = pruning_.n_modes;

        // Ensure temps
        for (int s = 0; s < numConcurrent_; ++s) {
            if (temps_[s].size() != output.getOwned().size()) {
                temps_[s] = detail::shrinkView("pruned_temp_" + std::to_string(s), outView, ngOut);
            }
        }

        Kokkos::deep_copy(outView, Complex_t(0, 0));

        double scale = 1.0;
        if (dir == 1) {
            for (unsigned d = 0; d < Dim; ++d) {
                scale *= double(modes[d]) / double(gDomFull[d].length());
            }
        }

        std::array<Vector<long, Dim>, NumSubFFTs> offsets;
        for (int k = 0; k < NumSubFFTs; ++k) {
            for (unsigned d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        auto owned           = output.getOwned();
        const int numBatches = (NumSubFFTs + numConcurrent_ - 1) / numConcurrent_;

        constexpr bool is_cpu = false
#ifdef KOKKOS_ENABLE_SERIAL
                                || std::is_same_v<ExecSpace, Kokkos::Serial>
#endif
#ifdef KOKKOS_ENABLE_OPENMP
                                || std::is_same_v<ExecSpace, Kokkos::OpenMP>
#endif
            ;

        const long ext0 = static_cast<long>(owned[0].length());
        const long ext1 = static_cast<long>(owned[1].length());
        [[maybe_unused]] const long ext2 =
            (Dim == 3) ? static_cast<long>(owned[Dim == 3 ? 2 : 0].length()) : 1L;

        for (int batch = 0; batch < numBatches; ++batch) {
            const int start = batch * numConcurrent_;
            const int end   = std::min(start + numConcurrent_, NumSubFFTs);
            const int count = end - start;

            IpplTimings::startTimer(subFFTTimer);

            // Using the wrapper to evaluate thread parallelism safely
            detail::runConcurrentBatch<is_cpu>(count, [&](int local) {
                const int k = start + local;
                auto offs   = offsets[k];
                auto& temp  = temps_[local];

                if constexpr (Dim == 3) {
                    auto copy_lambda = KOKKOS_LAMBDA(int i0, int i1, int i2) {
                        int si = i0 * 2 + int(offs[0]) + ngIn;
                        int sj = i1 * 2 + int(offs[1]) + ngIn;
                        int sk = i2 * 2 + int(offs[2]) + ngIn;
                        temp(i0, i1, i2) = inView(si, sj, sk);
                    };
                    if constexpr (is_cpu) {
                        Kokkos::parallel_for(
                            "strided_copy_forward",
                            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                                {0, 0, 0}, {ext0, ext1, ext2}),
                            copy_lambda);
                    } else {
                        auto exec = GPUOps::instance(streams_[local]);
                        Kokkos::parallel_for(
                            "strided_copy_forward",
                            Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<3>>(
                                exec, {0, 0, 0}, {ext0, ext1, ext2}),
                            copy_lambda);
                        GPUOps::sync(streams_[local]);
                    }
                } else {
                    auto copy_lambda = KOKKOS_LAMBDA(int i0, int i1) {
                        int si = i0 * 2 + int(offs[0]) + ngIn;
                        int sj = i1 * 2 + int(offs[1]) + ngIn;
                        temp(i0, i1) = inView(si, sj);
                    };
                    if constexpr (is_cpu) {
                        Kokkos::parallel_for(
                            "strided_copy_forward",
                            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                              {ext0, ext1}),
                            copy_lambda);
                    } else {
                        auto exec = GPUOps::instance(streams_[local]);
                        Kokkos::parallel_for(
                            "strided_copy_forward",
                            Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<2>>(exec, {0, 0},
                                                                               {ext0, ext1}),
                            copy_lambda);
                        GPUOps::sync(streams_[local]);
                    }
                }

                if (dir == 1) {
                    backends_[local]->forward(temp.data(), temp.data());
                } else {
                    backends_[local]->backward(temp.data(), temp.data());
                }
            });

            Kokkos::fence();
            IpplTimings::stopTimer(subFFTTimer);

            IpplTimings::startTimer(twiddleTimer);

            for (int local = 0; local < count; ++local) {
                const int k = start + local;
                auto offs   = offsets[k];
                auto& temp  = temps_[local];

                const long g0 = static_cast<long>(gDomFull[0].length());
                const long g1 = static_cast<long>(gDomFull[1].length());
                const long g2 =
                    (Dim == 3) ? static_cast<long>(gDomFull[Dim == 3 ? 2 : 0].length()) : 1L;
                const long m0 = static_cast<long>(modes[0]);
                const long m1 = static_cast<long>(modes[1]);
                const long m2 = (Dim == 3) ? static_cast<long>(modes[Dim == 3 ? 2 : 0]) : 1L;
                const int lf0 = localFirst[0];
                const int lf1 = localFirst[1];
                const int lf2 = (Dim == 3) ? localFirst[Dim == 3 ? 2 : 0] : 0;

                if constexpr (Dim == 3) {
                    Kokkos::parallel_for(
                        "twiddle_add_forward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                            {ngOut, ngOut, ngOut},
                            {int(outView.extent(0)) - ngOut, int(outView.extent(1)) - ngOut,
                             int(outView.extent(2)) - ngOut}),
                        KOKKOS_LAMBDA(int i, int j, int kk) {
                            int gi = i - ngOut + lf0;
                            int gj = j - ngOut + lf1;
                            int gk = kk - ngOut + lf2;

                            int64_t f0 = (gi < int64_t(m0) / 2) ? gi : int64_t(g0) - int64_t(m0) + gi;
                            int64_t f1 = (gj < int64_t(m1) / 2) ? gj : int64_t(g1) - int64_t(m1) + gj;
                            int64_t f2 = (gk < int64_t(m2) / 2) ? gk : int64_t(g2) - int64_t(m2) + gk;

                            Complex_t w(1.0, 0.0);
                            auto twiddle = [&](int64_t freq, int64_t N) {
                                double ang = -dir * 2.0 * M_PI * double(freq) / double(N);
                                return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                            };

                            if (offs[0]) w *= twiddle(f0, g0);
                            if (offs[1]) w *= twiddle(f1, g1);
                            if (offs[2]) w *= twiddle(f2, g2);

                            auto val = temp(i - ngOut, j - ngOut, kk - ngOut);
                            outView(i, j, kk) += w * val * scale;
                        });
                } else {
                    Kokkos::parallel_for(
                        "twiddle_add_forward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                            {ngOut, ngOut},
                            {int(outView.extent(0)) - ngOut, int(outView.extent(1)) - ngOut}),
                        KOKKOS_LAMBDA(int i, int j) {
                            int gi = i - ngOut + lf0;
                            int gj = j - ngOut + lf1;

                            int64_t f0 = (gi < int64_t(m0) / 2) ? gi : int64_t(g0) - int64_t(m0) + gi;
                            int64_t f1 = (gj < int64_t(m1) / 2) ? gj : int64_t(g1) - int64_t(m1) + gj;

                            Complex_t w(1.0, 0.0);
                            auto twiddle = [&](int64_t freq, int64_t N) {
                                double ang = -dir * 2.0 * M_PI * double(freq) / double(N);
                                return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                            };

                            if (offs[0]) w *= twiddle(f0, g0);
                            if (offs[1]) w *= twiddle(f1, g1);

                            auto val = temp(i - ngOut, j - ngOut);
                            outView(i, j) += w * val * scale;
                        });
                }
            }

            IpplTimings::stopTimer(twiddleTimer);
        }
    }

    //-------------------------------------------------------------------------
    // Backward Pruned C2C Implementation
    //-------------------------------------------------------------------------

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::backwardPruned(int dir, ComplexField& input,
                                                              ComplexField& output) {
        static IpplTimings::TimerRef subIFFTTimer      = IpplTimings::getTimer("subIFFTs");
        static IpplTimings::TimerRef stridedWriteTimer = IpplTimings::getTimer("StridedWrite");

        auto inView     = input.getView();   // Pruned frequency domain
        auto outView    = output.getView();  // Full spatial domain
        const int ngIn  = input.getNghost();
        const int ngOut = output.getNghost();

        const auto& lDomPruned = input.getLayout().getLocalNDIndex();
        const auto& gDomFull   = output.getLayout().getDomain();
        const auto& modes      = pruning_.n_modes;

        for (int s = 0; s < numConcurrent_; ++s) {
            if (temps_[s].size() != input.getOwned().size()) {
                temps_[s] =
                    detail::shrinkView("pruned_ifft_temp_" + std::to_string(s), inView, ngIn);
            }
        }

        Kokkos::deep_copy(outView, Complex_t(0, 0));

        std::array<Vector<long, Dim>, NumSubFFTs> offsets;
        for (int k = 0; k < NumSubFFTs; ++k) {
            for (unsigned d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        auto owned           = input.getOwned();
        const int numBatches = (NumSubFFTs + numConcurrent_ - 1) / numConcurrent_;

        constexpr bool is_cpu = false
#ifdef KOKKOS_ENABLE_SERIAL
                                || std::is_same_v<ExecSpace, Kokkos::Serial>
#endif
#ifdef KOKKOS_ENABLE_OPENMP
                                || std::is_same_v<ExecSpace, Kokkos::OpenMP>
#endif
            ;

        const long ext0 = static_cast<long>(owned[0].length());
        const long ext1 = static_cast<long>(owned[1].length());
        [[maybe_unused]] const long ext2 =
            (Dim == 3) ? static_cast<long>(owned[Dim == 3 ? 2 : 0].length()) : 1L;
        const long g0 = static_cast<long>(gDomFull[0].length());
        const long g1 = static_cast<long>(gDomFull[1].length());
        const long g2 = (Dim == 3) ? static_cast<long>(gDomFull[Dim == 3 ? 2 : 0].length()) : 1L;
        const long m0 = static_cast<long>(modes[0]);
        const long m1 = static_cast<long>(modes[1]);
        const long m2 = (Dim == 3) ? static_cast<long>(modes[Dim == 3 ? 2 : 0]) : 1L;
        const int lf0 = localFirst[0];
        const int lf1 = localFirst[1];
        const int lf2 = (Dim == 3) ? localFirst[Dim == 3 ? 2 : 0] : 0;

        for (int batch = 0; batch < numBatches; ++batch) {
            const int start = batch * numConcurrent_;
            const int end   = std::min(start + numConcurrent_, NumSubFFTs);
            const int count = end - start;

            IpplTimings::startTimer(subIFFTTimer);

            detail::runConcurrentBatch<is_cpu>(count, [&](int local) {
                const int k = start + local;
                auto offs   = offsets[k];
                auto& temp  = temps_[local];

                if constexpr (Dim == 3) {
                    auto multiply_lambda = KOKKOS_LAMBDA(int i0, int i1, int i2) {
                        int gi = i0 + lf0;
                        int gj = i1 + lf1;
                        int gk = i2 + lf2;

                        int64_t f0 = (gi < int64_t(m0) / 2) ? gi : int64_t(g0) - int64_t(m0) + gi;
                        int64_t f1 = (gj < int64_t(m1) / 2) ? gj : int64_t(g1) - int64_t(m1) + gj;
                        int64_t f2 = (gk < int64_t(m2) / 2) ? gk : int64_t(g2) - int64_t(m2) + gk;

                        Complex_t w(1.0, 0.0);
                        auto twiddle = [&](int64_t freq, int64_t N) {
                            double ang = dir * 2.0 * M_PI * double(freq) / double(N);
                            return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                        };

                        if (offs[0]) w *= twiddle(f0, g0);
                        if (offs[1]) w *= twiddle(f1, g1);
                        if (offs[2]) w *= twiddle(f2, g2);

                        auto input_val = inView(i0 + ngIn, i1 + ngIn, i2 + ngIn);
                        temp(i0, i1, i2) = w * input_val;
                    };
                    if constexpr (is_cpu) {
                        Kokkos::parallel_for(
                            "twiddle_multiply_backward",
                            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                                {0, 0, 0}, {ext0, ext1, ext2}),
                            multiply_lambda);
                    } else {
                        auto exec = GPUOps::instance(streams_[local]);
                        Kokkos::parallel_for(
                            "twiddle_multiply_backward",
                            Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<3>>(
                                exec, {0, 0, 0}, {ext0, ext1, ext2}),
                            multiply_lambda);
                        GPUOps::sync(streams_[local]);
                    }
                } else {
                    auto multiply_lambda = KOKKOS_LAMBDA(int i0, int i1) {
                        int gi = i0 + lf0;
                        int gj = i1 + lf1;

                        int64_t f0 = (gi < int64_t(m0) / 2) ? gi : int64_t(g0) - int64_t(m0) + gi;
                        int64_t f1 = (gj < int64_t(m1) / 2) ? gj : int64_t(g1) - int64_t(m1) + gj;

                        Complex_t w(1.0, 0.0);
                        auto twiddle = [&](int64_t freq, int64_t N) {
                            double ang = dir * 2.0 * M_PI * double(freq) / double(N);
                            return Complex_t(Kokkos::cos(ang), Kokkos::sin(ang));
                        };

                        if (offs[0]) w *= twiddle(f0, g0);
                        if (offs[1]) w *= twiddle(f1, g1);

                        auto input_val = inView(i0 + ngIn, i1 + ngIn);
                        temp(i0, i1) = w * input_val;
                    };
                    if constexpr (is_cpu) {
                        Kokkos::parallel_for(
                            "twiddle_multiply_backward",
                            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
                                                                              {ext0, ext1}),
                            multiply_lambda);
                    } else {
                        auto exec = GPUOps::instance(streams_[local]);
                        Kokkos::parallel_for(
                            "twiddle_multiply_backward",
                            Kokkos::MDRangePolicy<DeviceExec, Kokkos::Rank<2>>(exec, {0, 0},
                                                                               {ext0, ext1}),
                            multiply_lambda);
                        GPUOps::sync(streams_[local]);
                    }
                }

                if (dir == -1) {
                    backends_[local]->forward(temp.data(), temp.data());
                } else {
                    backends_[local]->backward(temp.data(), temp.data());
                }
            });

            Kokkos::fence();
            IpplTimings::stopTimer(subIFFTTimer);

            IpplTimings::startTimer(stridedWriteTimer);

            for (int local = 0; local < count; ++local) {
                const int k = start + local;
                auto offs   = offsets[k];
                auto& temp  = temps_[local];

                if constexpr (Dim == 3) {
                    Kokkos::parallel_for(
                        "strided_write_backward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                            {0, 0, 0}, {ext0, ext1, ext2}),
                        KOKKOS_LAMBDA(int i0, int i1, int i2) {
                            int oi              = i0 * 2 + int(offs[0]) + ngOut;
                            int oj              = i1 * 2 + int(offs[1]) + ngOut;
                            int ok              = i2 * 2 + int(offs[2]) + ngOut;
                            outView(oi, oj, ok) = temp(i0, i1, i2);
                        });
                } else {
                    Kokkos::parallel_for(
                        "strided_write_backward",
                        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {ext0, ext1}),
                        KOKKOS_LAMBDA(int i0, int i1) {
                            int oi          = i0 * 2 + int(offs[0]) + ngOut;
                            int oj          = i1 * 2 + int(offs[1]) + ngOut;
                            outView(oi, oj) = temp(i0, i1);
                        });
                }
            }

            IpplTimings::stopTimer(stridedWriteTimer);
        }
    }
}  // namespace ippl

#endif
