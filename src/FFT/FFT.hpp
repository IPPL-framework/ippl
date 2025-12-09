//
// Class FFT
//   The FFT class performs complex-to-complex,
//   real-to-complex on IPPL Fields.
//   FFT is templated on the type of transform to be performed,
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte. In making this interface,
//   we have referred Cabana library.
//   https://github.com/ECP-copa/Cabana.
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
/**
   Implementations for FFT constructor/destructor and transforms
*/

#include "Utility/IpplTimings.h"

#include "Field/BareField.h"

#include "FFT/NUFFT/NativeNUFFT.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {

    template <typename Field, template <typename...> class FFT, typename Backend, typename T>
    FFTBase<Field, FFT, Backend, T>::FFTBase(const Layout_t& layout, const ParameterList& params) {
        std::array<long long, 3> low;
        std::array<long long, 3> high;

        const NDIndex<Dim> lDom = layout.getLocalNDIndex();
        domainToBounds(lDom, low, high);

        heffte::box3d<long long> inbox  = {low, high};
        heffte::box3d<long long> outbox = {low, high};

        setup(inbox, outbox, params);
    }

    template <typename Field, template <typename...> class FFT, typename Backend, typename T>
    void FFTBase<Field, FFT, Backend, T>::domainToBounds(const NDIndex<Dim>& domain,
                                                         std::array<long long, 3>& low,
                                                         std::array<long long, 3>& high) {
        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for (size_t d = 0; d < Dim; ++d) {
            low[d]  = static_cast<long long>(domain[d].first());
            high[d] = static_cast<long long>(domain[d].length() + domain[d].first() - 1);
        }
    }

    /**
           setup performs the initialization necessary.
    */
    template <typename Field, template <typename...> class FFT, typename Backend, typename T>
    void FFTBase<Field, FFT, Backend, T>::setup(const heffte::box3d<long long>& inbox,
                                                const heffte::box3d<long long>& outbox,
                                                const ParameterList& params) {
        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

            switch (params.get<int>("comm")) {
                case a2a:
                    heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                    break;
                case a2av:
                    heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                    break;
                case p2p:
                    heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                    break;
                case p2p_pl:
                    heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                    break;
                default:
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        if constexpr (std::is_same_v<FFT<heffteBackend>, heffte::fft3d<heffteBackend>>) {
            heffte_m = std::make_shared<FFT<heffteBackend, long long>>(
                inbox, outbox, Comm->getCommunicator(), heffteOptions);
        } else {
            heffte_m = std::make_shared<FFT<heffteBackend, long long>>(
                inbox, outbox, params.get<int>("r2c_direction"), Comm->getCommunicator(),
                heffteOptions);
        }

        // heffte::gpu::device_set(Comm->rank() % heffte::gpu::device_count());
        if (workspace_m.size() < heffte_m->size_workspace()) {
            workspace_m = workspace_t(heffte_m->size_workspace());
        }
    }

    template <typename ComplexField>
    FFT<PrunedCCTransform, ComplexField>::~FFT() {
#ifdef Heffte_ENABLE_CUDA
        for (int k = 0; k < numConcurrentFFTs_; ++k) {
            cudaStreamDestroy(streams_m[k]);
        }
#elif defined(Heffte_ENABLE_ROCM)
        for (int k = 0; k < numConcurrentFFTs_; ++k) {
            hipStreamDestroy(streams_m[k]);
        }
#endif
        for (int k = 0; k < numConcurrentFFTs_; ++k) {
            MPI_Comm_free(&mpicomms_m[k]);
        }
    }

    template <typename ComplexField>
    void FFT<CCTransform, ComplexField>::warmup(ComplexField& f) {
        this->transform(FORWARD, f);
        this->transform(BACKWARD, f);
    }

    template <typename ComplexField>
    void FFT<CCTransform, ComplexField>::transform(TransformDirection direction, ComplexField& f) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) even though this
         *can be changed during heffte box creation
         */
        auto& tempField = this->tempField;
        if (tempField.size() != f.getOwned().size()) {
            tempField = detail::shrinkView("tempField", fview, nghost);
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempField, args - nghost).real(apply(fview, args).real());
                apply(tempField, args - nghost).imag(apply(fview, args).imag());
            });

        if (direction == FORWARD) {
            this->heffte_m->forward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                    heffte::scale::full);
        } else if (direction == BACKWARD) {
            this->heffte_m->backward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                     heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(fview, args).real() = apply(tempField, args - nghost).real();
                apply(fview, args).imag() = apply(tempField, args - nghost).imag();
            });
    }

    //========================================================================
    // FFT PrunedCCTransform Constructor and Methods
    //========================================================================

    template <typename ComplexField>
    typename FFT<PrunedCCTransform, ComplexField>::device_exec_space
    FFT<PrunedCCTransform, ComplexField>::get_exec_instance(int k) const {
#if defined(KOKKOS_ENABLE_CUDA)
        return device_exec_space(streams_m[k]);
#elif defined(KOKKOS_ENABLE_HIP)
        return device_exec_space(streams_m[k]);
#else
        return device_exec_space();
#endif
    }

    template <typename ComplexField>
    FFT<PrunedCCTransform, ComplexField>::FFT(const Layout_t& layoutInput,
                                              const Layout_t& layoutOutput,
                                              const PruningParams<Dim>& pruning,
                                              const ParameterList& params)
        : pruning_(pruning) {
        static_assert(Dim == 3, "Parallel pruned FFT currently only supports 3D");

        // Get number of concurrent FFTs (1-8, default to 2)
        int numConcurrent  = params.get<int>("num_concurrent_ffts", 2);
        numConcurrentFFTs_ = std::max(1, std::min(numConcurrent, numSubFFTs));

        // Setup heFFTe with the full (input) grid domain
        std::array<long long, 3> lowInput, highInput;
        const NDIndex<Dim>& lDomInput = layoutInput.getLocalNDIndex();
        this->domainToBounds(lDomInput, lowInput, highInput);

        heffte::box3d<long long> inbox  = {lowInput, highInput};
        heffte::box3d<long long> outbox = {lowInput, highInput};
        this->setup(inbox, outbox, params);

        auto output_size = layoutOutput.getLocalNDIndex().size();
        auto input_size  = layoutInput.getLocalNDIndex().size();

        // Setup pruned domain
        auto& pruned_layout            = output_size < input_size ? layoutOutput : layoutInput;
        const NDIndex<Dim>& lDomOutput = pruned_layout.getLocalNDIndex();
        std::array<long long, 3> lowInputPruned, highInputPruned;
        this->domainToBounds(lDomOutput, lowInputPruned, highInputPruned);

        heffte::box3d<long long> inboxPruned  = {lowInputPruned, highInputPruned};
        heffte::box3d<long long> outboxPruned = {lowInputPruned, highInputPruned};

        heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();

        if (!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

            switch (params.get<int>("comm")) {
                case a2a:
                    heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                    break;
                case a2av:
                    heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                    break;
                case p2p:
                    heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                    break;
                case p2p_pl:
                    heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                    break;
                default:
                    throw IpplException("FFT::setup", "Unrecognized heffte communication type");
            }
        }

        // Create streams and heFFTe instances only for the concurrent slots we need
        for (int slot = 0; slot < numConcurrentFFTs_; ++slot) {
            MPI_Comm_dup(MPI_COMM_WORLD, &mpicomms_m[slot]);

#ifdef Heffte_ENABLE_CUDA
            cudaStreamCreate(&streams_m[slot]);
#elif defined(Heffte_ENABLE_ROCM)
            hipStreamCreate(&streams_m[slot]);
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            pruned_heffte_m[slot] = std::make_shared<BaseFFTType<heffteBackend, long long>>(
                streams_m[slot], inboxPruned, outboxPruned, mpicomms_m[slot], heffteOptions);
#else
            pruned_heffte_m[slot] = std::make_shared<BaseFFTType<heffteBackend, long long>>(
                inboxPruned, outboxPruned, mpicomms_m[slot], heffteOptions);
#endif

            workspaces_m[slot] = workspace_t(pruned_heffte_m[slot]->size_workspace());
        }
    }

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::forward_stride2_pruned_3d(int dir,
                                                                         ComplexField& input,
                                                                         ComplexField& output) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");
        static IpplTimings::TimerRef TwiddleAdd = IpplTimings::getTimer("TwiddleAdd");
        static IpplTimings::TimerRef PrunedFFT  = IpplTimings::getTimer("prunedFFT");
        static IpplTimings::TimerRef SubFFTs    = IpplTimings::getTimer("subFFTs");

        IpplTimings::startTimer(PrunedFFT);

        auto& input_view     = input.getView();
        auto& output_view    = output.getView();
        const int nghost_in  = input.getNghost();
        const int nghost_out = output.getNghost();

        const auto& lDomPruned = output.getLayout().getLocalNDIndex();
        const auto& gDomFull   = input.getLayout().getDomain();

        const size_t K0 = pruning_.n_modes[0];
        const size_t K1 = pruning_.n_modes[1];
        const size_t K2 = pruning_.n_modes[2];

        assert(K0 * 2 == gDomFull[0].length() && K1 * 2 == gDomFull[1].length()
               && K2 * 2 == gDomFull[2].length());

        auto& pruned_modes = pruning_.n_modes;

        // Allocate temp buffers only for the concurrent slots we have
        for (int slot = 0; slot < numConcurrentFFTs_; ++slot) {
            if (tempFieldInputs[slot].size() != output.getOwned().size()) {
                tempFieldInputs[slot] = detail::shrinkView(
                    "tempFieldPrunedFFT_" + std::to_string(slot), output_view, nghost_out);
            }
        }

        Kokkos::deep_copy(output_view, Complex_t(0, 0));

        double scaling_factor = 1.0;
        for (int d = 0; d < Dim; ++d) {
            scaling_factor *= static_cast<double>(pruning_.n_modes[d])
                              / static_cast<double>(gDomFull[d].length());
        }
        if (dir == -1) {
            scaling_factor = 1.0;
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;

        // Compute all offsets upfront for all 8 sub-FFTs
        std::array<Vector<long, Dim>, numSubFFTs> offsets;
        for (int k = 0; k < numSubFFTs; ++k) {
            for (int d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        // Get output extents for MDRangePolicy
        auto owned = output.getOwned();
        Kokkos::Array<long, Dim> lo, hi;
        for (unsigned d = 0; d < Dim; ++d) {
            lo[d] = 0;
            hi[d] = owned[d].length();
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        // Process sub-FFTs in batches based on available concurrent slots
        const int numBatches = (numSubFFTs + numConcurrentFFTs_ - 1) / numConcurrentFFTs_;

        for (int batch = 0; batch < numBatches; ++batch) {
            const int batchStart = batch * numConcurrentFFTs_;
            const int batchEnd   = std::min(batchStart + numConcurrentFFTs_, numSubFFTs);
            const int batchSize  = batchEnd - batchStart;

            // Phase 1: Submit strided copies + FFTs for this batch in parallel
            IpplTimings::startTimer(SubFFTs);

            // Launch all sub-FFTs in this batch on separate streams
            Kokkos::parallel_for(
                "PrunedFFT parallel sub-FFTs batch",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, batchSize),
                [&](const int localIdx) {
                    const int k    = batchStart + localIdx;  // Which sub-FFT (0-7)
                    const int slot = localIdx;               // Which slot to use (0 to batchSize-1)

                    auto offs                = offsets[k];
                    auto& tempFieldInputView = tempFieldInputs[slot];
                    auto exec_instance       = get_exec_instance(slot);

                    // Strided copy on this slot's stream
                    using mdrange_policy_t =
                        Kokkos::MDRangePolicy<device_exec_space, Kokkos::Rank<Dim>>;
                    mdrange_policy_t policy(exec_instance, {lo[0], lo[1], lo[2]},
                                            {hi[0], hi[1], hi[2]});

                    Kokkos::parallel_for(
                        "strided input copy PrunedFFT", policy,
                        KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
                            index_array_type args = {i0, i1, i2};
                            auto strided_in       = args * 2 + offs + nghost_in;

                            apply(tempFieldInputView, args)
                                .real(apply(input_view, strided_in).real());
                            apply(tempFieldInputView, args)
                                .imag(apply(input_view, strided_in).imag());
                        });

                    // FFT on same stream - automatically waits for copy
                    if (dir == 1) {
                        pruned_heffte_m[slot]->forward(
                            tempFieldInputs[slot].data(), tempFieldInputs[slot].data(),
                            workspaces_m[slot].data(), heffte::scale::full);
                    } else {
                        pruned_heffte_m[slot]->backward(
                            tempFieldInputs[slot].data(), tempFieldInputs[slot].data(),
                            workspaces_m[slot].data(), heffte::scale::none);
                    }
                });

            // Wait for all streams in this batch to complete before accumulation
            Kokkos::fence();
            IpplTimings::stopTimer(SubFFTs);

            // Phase 2: Accumulate results with twiddle factors for this batch
            // Must be done before next batch overwrites temp buffers
            IpplTimings::startTimer(TwiddleAdd);

            for (int localIdx = 0; localIdx < batchSize; ++localIdx) {
                const int k    = batchStart + localIdx;
                const int slot = localIdx;

                auto offs                = offsets[k];
                auto& tempFieldInputView = tempFieldInputs[slot];

                ippl::parallel_for(
                    "PrunedFFT sub-FFT twiddle factor addition",
                    getRangePolicy(output_view, nghost_out),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        auto g = (args - nghost_out) + localFirst;
                        Vector<int64_t, Dim> freq{};
                        for (int d = 0; d < Dim; ++d) {
                            if (g[d] < static_cast<int64_t>(pruned_modes[d]) / 2) {
                                freq[d] = g[d];
                            } else {
                                freq[d] = static_cast<int64_t>(gDomFull[d].length())
                                          - static_cast<int64_t>(pruned_modes[d]) + g[d];
                            }
                        }

                        Complex_t w = 1.0;
                        for (int d = 0; d < Dim; ++d) {
                            double ang = -dir * 2.0 * M_PI * (static_cast<double>(freq[d]))
                                         / static_cast<double>(gDomFull[d].length());
                            w *= (offs[d] == 0) ? Complex_t(1.0, 0.0)
                                                : Complex_t(std::cos(ang), std::sin(ang));
                        }
                        auto fft_input = apply(tempFieldInputView, args - nghost_out);
                        apply(output_view, args).imag() += (w * fft_input * scaling_factor).imag();
                        apply(output_view, args).real() += (w * fft_input * scaling_factor).real();
                    });
            }

            IpplTimings::stopTimer(TwiddleAdd);
        }

        IpplTimings::stopTimer(PrunedFFT);
    }

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::backward_stride2_pruned_3d(
        int dir,
        ComplexField& input,  // pruned frequency domain (K per dim)
        ComplexField& output  // full spatial domain (N = 2K per dim)
    ) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

        static IpplTimings::TimerRef TwiddleMul   = IpplTimings::getTimer("TwiddleMul");
        static IpplTimings::TimerRef PrunedIFFT   = IpplTimings::getTimer("prunedIFFT");
        static IpplTimings::TimerRef SubIFFTs     = IpplTimings::getTimer("subIFFTs");
        static IpplTimings::TimerRef StridedWrite = IpplTimings::getTimer("StridedWrite");

        IpplTimings::startTimer(PrunedIFFT);

        auto& input_view  = input.getView();
        auto& output_view = output.getView();

        const int nghost_in  = input.getNghost();
        const int nghost_out = output.getNghost();

        const auto& lDomPruned = input.getLayout().getLocalNDIndex();
        const auto& gDomFull   = output.getLayout().getDomain();

        const size_t K0 = pruning_.n_modes[0];
        const size_t K1 = pruning_.n_modes[1];
        const size_t K2 = pruning_.n_modes[2];

        assert(K0 * 2 == gDomFull[0].length() && K1 * 2 == gDomFull[1].length()
               && K2 * 2 == gDomFull[2].length());

        auto& pruned_modes = pruning_.n_modes;

        // Allocate temp buffers only for the concurrent slots we have
        for (int slot = 0; slot < numConcurrentFFTs_; ++slot) {
            if (tempFieldInputs[slot].size() != input.getOwned().size()) {
                tempFieldInputs[slot] = detail::shrinkView(
                    "tempFieldPrunedIFFT_" + std::to_string(slot), input_view, nghost_in);
            }
        }

        // Zero the output
        Kokkos::deep_copy(output_view, Complex_t(0, 0));

        using index_array_type = typename RangePolicy<Dim>::index_array_type;

        // Compute all offsets upfront for all 8 sub-FFTs
        std::array<Vector<long, Dim>, numSubFFTs> offsets;
        for (int k = 0; k < numSubFFTs; ++k) {
            for (int d = 0; d < Dim; ++d) {
                offsets[k][d] = (k >> d) & 1;
            }
        }

        // Get input extents for MDRangePolicy
        auto owned = input.getOwned();
        Kokkos::Array<long, Dim> lo, hi;
        for (unsigned d = 0; d < Dim; ++d) {
            lo[d] = 0;
            hi[d] = owned[d].length();
        }

        Vector<int, Dim> localFirst;
        for (unsigned d = 0; d < Dim; ++d) {
            localFirst[d] = lDomPruned[d].first();
        }

        // Process sub-IFFTs in batches based on available concurrent slots
        const int numBatches = (numSubFFTs + numConcurrentFFTs_ - 1) / numConcurrentFFTs_;

        for (int batch = 0; batch < numBatches; ++batch) {
            const int batchStart = batch * numConcurrentFFTs_;
            const int batchEnd   = std::min(batchStart + numConcurrentFFTs_, numSubFFTs);
            const int batchSize  = batchEnd - batchStart;

            // Phase 1: Apply twiddle factors and run sub-IFFTs for this batch
            IpplTimings::startTimer(SubIFFTs);

            Kokkos::parallel_for(
                "PrunedIFFT parallel sub-IFFTs batch",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, batchSize),
                [&](const int localIdx) {
                    const int k    = batchStart + localIdx;
                    const int slot = localIdx;

                    auto offs                = offsets[k];
                    auto& tempFieldInputView = tempFieldInputs[slot];
                    auto exec_instance       = get_exec_instance(slot);

                    using mdrange_policy_t =
                        Kokkos::MDRangePolicy<device_exec_space, Kokkos::Rank<Dim>>;
                    mdrange_policy_t policy(exec_instance, {lo[0], lo[1], lo[2]},
                                            {hi[0], hi[1], hi[2]});

                    // Apply twiddle factors to input and copy to temp buffer
                    Kokkos::parallel_for(
                        "twiddle multiply PrunedIFFT", policy,
                        KOKKOS_LAMBDA(const int i0, const int i1, const int i2) {
                            index_array_type args = {i0, i1, i2};
                            auto g                = args + localFirst;

                            // Map pruned frequency index to full frequency index
                            Vector<int64_t, Dim> freq;
                            for (int d = 0; d < Dim; ++d) {
                                if (g[d] < static_cast<int64_t>(pruned_modes[d]) / 2) {
                                    freq[d] = g[d];
                                } else {
                                    freq[d] = static_cast<int64_t>(gDomFull[d].length())
                                              - static_cast<int64_t>(pruned_modes[d]) + g[d];
                                }
                            }

                            // Compute twiddle factor (positive sign for inverse)
                            Complex_t w = 1.0;
                            for (int d = 0; d < Dim; ++d) {
                                if (offs[d] != 0) {
                                    double ang = 2.0 * M_PI * static_cast<double>(freq[d])
                                                 / static_cast<double>(gDomFull[d].length());
                                    w *= Complex_t(std::cos(ang), std::sin(ang));
                                }
                            }

                            auto input_val = apply(input_view, args + nghost_in);
                            apply(tempFieldInputView, args).real((w * input_val).real());
                            apply(tempFieldInputView, args).imag((w * input_val).imag());
                        });

                    // Backward FFT on same stream
                    pruned_heffte_m[slot]->backward(tempFieldInputs[slot].data(),
                                                    tempFieldInputs[slot].data(),
                                                    workspaces_m[slot].data(), heffte::scale::none);
                });

            // Wait for all streams in this batch
            Kokkos::fence();
            IpplTimings::stopTimer(SubIFFTs);

            // Phase 2: Strided write to output for this batch
            // Must be done before next batch overwrites temp buffers
            IpplTimings::startTimer(StridedWrite);

            for (int localIdx = 0; localIdx < batchSize; ++localIdx) {
                const int k    = batchStart + localIdx;
                const int slot = localIdx;

                auto offs                = offsets[k];
                auto& tempFieldInputView = tempFieldInputs[slot];

                ippl::parallel_for(
                    "PrunedIFFT strided output write", getRangePolicy(tempFieldInputView, 0),
                    KOKKOS_LAMBDA(const index_array_type& args) {
                        auto strided_out = args * 2 + offs + nghost_out;

                        auto ifft_result = apply(tempFieldInputView, args);
                        apply(output_view, strided_out).real(ifft_result.real());
                        apply(output_view, strided_out).imag(ifft_result.imag());
                    });
            }

            IpplTimings::stopTimer(StridedWrite);
        }

        IpplTimings::stopTimer(PrunedIFFT);
    }

    template <typename ComplexField>
    void FFT<PrunedCCTransform, ComplexField>::transform(TransformDirection direction,
                                                         ComplexField& input, ComplexField& output,
                                                         int dir) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");
        if (direction == FORWARD && Dim == 3) {
            forward_stride2_pruned_3d(dir, input, output);
            return;
        } else if (direction == BACKWARD && Dim == 3) {
            backward_stride2_pruned_3d(dir, input, output);
            return;
        }

        throw std::runtime_error(
            "Currently, pruning is only implemented and tested in Dim 3. You can"
            "likely call the above with other dimensions, but it is not testsed.");
        //
        // auto& input_view     = input.getView();
        // auto& output_view    = output.getView();
        // const int nghost_in  = input.getNghost();
        // const int nghost_out = output.getNghost();
        //
        // // Allocate temp view for full grid (heFFTe works on full grid)
        // auto& tempField        = this->tempField;
        // using index_array_type = typename RangePolicy<Dim>::index_array_type;
        //
        // if (direction == FORWARD) {
        //     if (tempField.size() != input.getOwned().size()) {
        //         tempField = detail::shrinkView("tempField", input_view, nghost_in);
        //     }
        //
        //     // Forward: full grid -> full FFT -> extract first K/2 and last K/2 modes
        //     // Copy full input to temp
        //     ippl::parallel_for(
        //         "copy input pruned FFT", getRangePolicy(input_view, nghost_in),
        //         KOKKOS_LAMBDA(const index_array_type& args) {
        //             apply(tempField, args - nghost_in).real(apply(input_view, args).real());
        //             apply(tempField, args - nghost_in).imag(apply(input_view, args).imag());
        //         });
        //
        //     // Perform full FFT in-place
        //     this->heffte_m->forward(tempField.data(), tempField.data(), this->workspace_m.data(),
        //                             heffte::scale::full);
        //
        //     // Extract modes: first K/2 positive and last K/2 negative frequencies
        //     // FFT ordering: [0, 1, ..., N/2-1, -N/2, ..., -1]
        //     // Pruned: [0, ..., K/2-1] and [-K/2, ..., -1]
        //     const auto& lDomFull   = input.getLayout().getLocalNDIndex();
        //     const auto& lDomPruned = output.getLayout().getLocalNDIndex();
        //
        //     const size_t N0 = lDomFull[0].length();
        //     const size_t N1 = lDomFull[1].length();
        //     const size_t N2 = lDomFull[2].length();
        //
        //     const size_t K0 = pruning_.n_modes[0];
        //     const size_t K1 = pruning_.n_modes[1];
        //     const size_t K2 = pruning_.n_modes[2];
        //
        //     ippl::parallel_for(
        //         "extract pruned modes",
        //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        //             {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
        //             {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() +
        //             1}),
        //         KOKKOS_LAMBDA(const ippl::Vector<int64_t, 3>& idx) {
        //             int64_t gi = idx[0];  // Global index in pruned grid
        //             int64_t gj = idx[1];
        //             int64_t gk = idx[2];
        //
        //             // Map pruned index to full grid index
        //             // Positive freqs [0, K/2-1] map to [0, K/2-1]
        //             // Negative freqs [K/2, K-1] map to [N-K/2, N-1]
        //             auto map_index = [](int64_t pruned_idx, size_t K, size_t N) -> int64_t {
        //                 if (pruned_idx < K / 2) {
        //                     return pruned_idx;  // Positive frequencies
        //                 } else {
        //                     return N - K + pruned_idx;  // Negative frequencies
        //                 }
        //             };
        //
        //             int64_t fi = map_index(gi, K0, N0);
        //             int64_t fj = map_index(gj, K1, N1);
        //             int64_t fk = map_index(gk, K2, N2);
        //
        //             // Check if this mode exists in local full domain
        //             if (fi >= lDomFull[0].first() && fi <= lDomFull[0].last()
        //                 && fj >= lDomFull[1].first() && fj <= lDomFull[1].last()
        //                 && fk >= lDomFull[2].first() && fk <= lDomFull[2].last()) {
        //                 int li_full = fi - lDomFull[0].first();
        //                 int lj_full = fj - lDomFull[1].first();
        //                 int lk_full = fk - lDomFull[2].first();
        //
        //                 int li_pruned = gi - lDomPruned[0].first() + nghost_out;
        //                 int lj_pruned = gj - lDomPruned[1].first() + nghost_out;
        //                 int lk_pruned = gk - lDomPruned[2].first() + nghost_out;
        //
        //                 output_view(li_pruned, lj_pruned, lk_pruned) =
        //                     tempField(li_full, lj_full, lk_full);
        //             } else {
        //                 assert(false);
        //                 int li_pruned = gi - lDomPruned[0].first() + nghost_out;
        //                 int lj_pruned = gj - lDomPruned[1].first() + nghost_out;
        //                 int lk_pruned = gk - lDomPruned[2].first() + nghost_out;
        //
        //                 output_view(li_pruned, lj_pruned, lk_pruned) = Complex_t(0, 0);
        //             }
        //         });
        //
        // } else if (direction == BACKWARD) {
        //     if (tempField.size() != output.getOwned().size()) {
        //         tempField = detail::shrinkView("tempField", output_view, nghost_in);
        //     }
        //
        //     // Backward: pruned modes -> zero-pad to full -> full IFFT
        //     // Zero-initialize temp for full grid
        //     Kokkos::deep_copy(tempField, Complex_t(0, 0));
        //
        //     const auto& lDomFull   = output.getLayout().getLocalNDIndex();
        //     const auto& lDomPruned = input.getLayout().getLocalNDIndex();
        //
        //     const size_t N0 = lDomFull[0].length();
        //     const size_t N1 = lDomFull[1].length();
        //     const size_t N2 = lDomFull[2].length();
        //
        //     const size_t K0 = pruning_.n_modes[0];
        //     const size_t K1 = pruning_.n_modes[1];
        //     const size_t K2 = pruning_.n_modes[2];
        //
        //     // Place pruned modes at correct positions in full grid
        //     ippl::parallel_for(
        //         "zeropad pruned modes",
        //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        //             {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
        //             {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() +
        //             1}),
        //         KOKKOS_LAMBDA(const ippl::Vector<int64_t, 3>& idx) {
        //             int64_t gi = idx[0];  // Global index in pruned grid
        //             int64_t gj = idx[1];
        //             int64_t gk = idx[2];
        //
        //             // Map pruned index to full grid index
        //             auto map_index = [](int64_t pruned_idx, size_t K, size_t N) -> int64_t {
        //                 if (pruned_idx < K / 2) {
        //                     return pruned_idx;  // Positive frequencies
        //                 } else {
        //                     return N - K + pruned_idx;  // Negative frequencies
        //                 }
        //             };
        //
        //             int64_t fi = map_index(gi, K0, N0);
        //             int64_t fj = map_index(gj, K1, N1);
        //             int64_t fk = map_index(gk, K2, N2);
        //
        //             // Check if this mode should be placed in local full domain
        //             if (fi >= lDomFull[0].first() && fi <= lDomFull[0].last()
        //                 && fj >= lDomFull[1].first() && fj <= lDomFull[1].last()
        //                 && fk >= lDomFull[2].first() && fk <= lDomFull[2].last()) {
        //                 int li_full = fi - lDomFull[0].first();
        //                 int lj_full = fj - lDomFull[1].first();
        //                 int lk_full = fk - lDomFull[2].first();
        //
        //                 int li_pruned = gi - lDomPruned[0].first() + nghost_in;
        //                 int lj_pruned = gj - lDomPruned[1].first() + nghost_in;
        //                 int lk_pruned = gk - lDomPruned[2].first() + nghost_in;
        //
        //                 // auto &val_tmpfield = tempField(li_full, lj_full, lk_full);
        //                 auto val_input = input_view(li_pruned, lj_pruned, lk_pruned);
        //                 tempField(li_full, lj_full, lk_full) = val_input;
        //             }
        //         });
        //
        //     // Perform full IFFT in-place
        //     this->heffte_m->backward(tempField.data(), tempField.data(),
        //     this->workspace_m.data(),
        //                              heffte::scale::none);
        //
        //     // Copy result to output
        //     ippl::parallel_for(
        //         "copy output pruned FFT backward", getRangePolicy(output_view, nghost_out),
        //         KOKKOS_LAMBDA(const index_array_type& args) {
        //             apply(output_view, args).real() = apply(tempField, args - nghost_out).real();
        //             apply(output_view, args).imag() = apply(tempField, args - nghost_out).imag();
        //         });
        // } else {
        //     throw std::logic_error("Only FORWARD and BACKWARD are allowed as directions");
        // }
    }

    //========================================================================
    // FFT RCTransform Constructors
    //========================================================================

    /**
     *Create a new FFT object of type RCTransform, with given input and output
     *layouts and heffte parameters.
     */

    template <typename RealField>
    FFT<RCTransform, RealField>::FFT(const Layout_t& layoutInput, const Layout_t& layoutOutput,
                                     const ParameterList& params) {
        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> lowInput;
        std::array<long long, 3> highInput;
        std::array<long long, 3> lowOutput;
        std::array<long long, 3> highOutput;

        const NDIndex<Dim>& lDomInput  = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();

        this->domainToBounds(lDomInput, lowInput, highInput);
        this->domainToBounds(lDomOutput, lowOutput, highOutput);

        heffte::box3d<long long> inbox  = {lowInput, highInput};
        heffte::box3d<long long> outbox = {lowOutput, highOutput};

        this->setup(inbox, outbox, params);
    }

    template <typename RealField>
    void FFT<RCTransform, RealField>::warmup(RealField& f, ComplexField& g) {
        this->transform(FORWARD, f, g);
        this->transform(BACKWARD, f, g);
    }

    template <typename RealField>
    void FFT<RCTransform, RealField>::transform(TransformDirection direction, RealField& f,
                                                ComplexField& g) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

        auto fview        = f.getView();
        auto gview        = g.getView();
        const int nghostf = f.getNghost();
        const int nghostg = g.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto& tempFieldf = this->tempField;
        auto& tempFieldg = this->tempFieldComplex;
        if (tempFieldf.size() != f.getOwned().size()) {
            tempFieldf = detail::shrinkView("tempFieldf", fview, nghostf);
        }
        if (tempFieldg.size() != g.getOwned().size()) {
            tempFieldg = detail::shrinkView("tempFieldg", gview, nghostg);
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos f field in FFT", getRangePolicy(fview, nghostf),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempFieldf, args - nghostf) = apply(fview, args);
            });
        ippl::parallel_for(
            "copy from Kokkos g field in FFT", getRangePolicy(gview, nghostg),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempFieldg, args - nghostg).real(apply(gview, args).real());
                apply(tempFieldg, args - nghostg).imag(apply(gview, args).imag());
            });

        if (direction == FORWARD) {
            this->heffte_m->forward(tempFieldf.data(), tempFieldg.data(), this->workspace_m.data(),
                                    heffte::scale::full);
        } else if (direction == BACKWARD) {
            this->heffte_m->backward(tempFieldg.data(), tempFieldf.data(), this->workspace_m.data(),
                                     heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos f field FFT", getRangePolicy(fview, nghostf),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(fview, args) = apply(tempFieldf, args - nghostf);
            });

        ippl::parallel_for(
            "copy to Kokkos g field FFT", getRangePolicy(gview, nghostg),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(gview, args).real() = apply(tempFieldg, args - nghostg).real();
                apply(gview, args).imag() = apply(tempFieldg, args - nghostg).imag();
            });
    }
    //========================================================================
    // FFT PrunedRCTransform Constructor and Methods
    //========================================================================

    template <typename RealField>
    FFT<PrunedRCTransform, RealField>::FFT(const Layout_t& layoutReal,
                                           const Layout_t& layoutComplexFull,
                                           const Layout_t& layoutComplexPruned,
                                           const PruningParams<Dim>& pruning,
                                           const ParameterList& params)
        : pruning_(pruning) {
        // Setup heFFTe with real input and full complex output
        std::array<long long, 3> lowReal, highReal, lowComplexFull, highComplexFull;

        const NDIndex<Dim>& lDomReal        = layoutReal.getLocalNDIndex();
        const NDIndex<Dim>& lDomComplexFull = layoutComplexFull.getLocalNDIndex();

        this->domainToBounds(lDomReal, lowReal, highReal);
        this->domainToBounds(lDomComplexFull, lowComplexFull, highComplexFull);

        heffte::box3d<long long> inbox  = {lowReal, highReal};
        heffte::box3d<long long> outbox = {lowComplexFull, highComplexFull};

        this->setup(inbox, outbox, params);
    }

    template <typename RealField>
    void FFT<PrunedRCTransform, RealField>::transform(TransformDirection direction, RealField& f,
                                                      ComplexField& g) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

        auto fview        = f.getView();
        auto gview        = g.getView();
        const int nghostf = f.getNghost();
        const int nghostg = g.getNghost();

        using index_array_type = typename RangePolicy<Dim>::index_array_type;

        if (direction == FORWARD) {
            // Forward: real (full) -> full complex FFT -> extract pruned modes

            // Prepare temp views (no ghost layers)
            auto& tempFieldf = this->tempField;
            auto& tempFieldg = this->tempFieldComplex;
            if (tempFieldf.size() != f.getOwned().size()) {
                tempFieldf = detail::shrinkView("tempFieldf", fview, nghostf);
            }

            // We need a temp field for FULL complex output
            const auto& lDomReal = f.getLayout().getLocalNDIndex();
            // Full complex size: (N0/2+1) x N1 x N2 for r2c in dimension 0
            size_t fullComplexSize =
                (lDomReal[0].length() / 2 + 1) * lDomReal[1].length() * lDomReal[2].length();
            if (tempFieldg.size() != fullComplexSize) {
                tempFieldg = Kokkos::View<Complex_t***, Kokkos::LayoutLeft,
                                          typename RealField::memory_space>(
                    "tempFieldg", lDomReal[0].length() / 2 + 1, lDomReal[1].length(),
                    lDomReal[2].length());
            }

            // Copy real field to temp (no ghosts)
            ippl::parallel_for(
                "copy real field pruned R2C", getRangePolicy(fview, nghostf),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    apply(tempFieldf, args - nghostf) = apply(fview, args);
                });

            // R2C FFT: real -> full complex
            this->heffte_m->forward(tempFieldf.data(), tempFieldg.data(), this->workspace_m.data(),
                                    heffte::scale::full);

            // Extract pruned modes from full complex to output
            // R2C layout: dimension 0 has [0, N/2] (only positive freqs)
            //             dimensions 1,2 have [0, ..., N/2-1, -N/2, ..., -1]
            const auto& lDomPruned = g.getLayout().getLocalNDIndex();

            const size_t N0 = lDomReal[0].length();
            const size_t N1 = lDomReal[1].length();
            const size_t N2 = lDomReal[2].length();

            const size_t K0 = pruning_.n_modes[0];  // For R2C dim, just first K0 modes
            const size_t K1 = pruning_.n_modes[1];
            const size_t K2 = pruning_.n_modes[2];

            ippl::parallel_for(
                "extract pruned modes R2C",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
                    {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() + 1}),
                KOKKOS_LAMBDA(const ippl::Vector<int64_t, 3>& idx) {
                    int64_t gi = idx[0];  // Global index in pruned grid
                    int64_t gj = idx[1];
                    int64_t gk = idx[2];

                    // Map pruned index to full grid index
                    // Dimension 0 (R2C): only positive freqs, direct mapping
                    // Dimensions 1,2: positive [0, K/2) -> [0, K/2), negative [K/2, K) -> [N-K/2,
                    // N)
                    int64_t fi = gi;  // R2C dimension - direct mapping
                    int64_t fj = (gj < static_cast<int64_t>(K1 / 2)) ? gj : (N1 - K1 + gj);
                    int64_t fk = (gk < static_cast<int64_t>(K2 / 2)) ? gk : (N2 - K2 + gk);

                    // tempFieldg has no ghosts, indexed from 0
                    int li_full = fi;
                    int lj_full = fj;
                    int lk_full = fk;

                    // gview has ghosts
                    int li_pruned = gi - lDomPruned[0].first() + nghostg;
                    int lj_pruned = gj - lDomPruned[1].first() + nghostg;
                    int lk_pruned = gk - lDomPruned[2].first() + nghostg;

                    gview(li_pruned, lj_pruned, lk_pruned) = tempFieldg(li_full, lj_full, lk_full);
                });

        } else if (direction == BACKWARD) {
            // Backward: pruned complex -> zero-pad to full complex -> C2R IFFT

            // Prepare temp views
            auto& tempFieldf = this->tempField;
            auto& tempFieldg = this->tempFieldComplex;
            if (tempFieldf.size() != f.getOwned().size()) {
                tempFieldf = detail::shrinkView("tempFieldf", fview, nghostf);
            }

            const auto& lDomReal = f.getLayout().getLocalNDIndex();
            size_t fullComplexSize =
                (lDomReal[0].length() / 2 + 1) * lDomReal[1].length() * lDomReal[2].length();
            if (tempFieldg.size() != fullComplexSize) {
                tempFieldg = Kokkos::View<Complex_t***, Kokkos::LayoutLeft,
                                          typename RealField::memory_space>(
                    "tempFieldg", lDomReal[0].length() / 2 + 1, lDomReal[1].length(),
                    lDomReal[2].length());
            }

            // Zero-initialize full complex temp
            Kokkos::deep_copy(tempFieldg, Complex_t(0, 0));

            // Copy pruned modes to correct positions in full complex
            const auto& lDomPruned = g.getLayout().getLocalNDIndex();

            const size_t N0 = lDomReal[0].length();
            const size_t N1 = lDomReal[1].length();
            const size_t N2 = lDomReal[2].length();

            const size_t K0 = pruning_.n_modes[0];
            const size_t K1 = pruning_.n_modes[1];
            const size_t K2 = pruning_.n_modes[2];

            ippl::parallel_for(
                "zeropad pruned modes R2C",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
                    {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() + 1}),
                KOKKOS_LAMBDA(const ippl::Vector<int64_t, 3>& idx) {
                    int64_t gi = idx[0];
                    int64_t gj = idx[1];
                    int64_t gk = idx[2];

                    // Map pruned index to full grid index
                    int64_t fi = gi;  // R2C dimension - direct mapping
                    int64_t fj = (gj < static_cast<int64_t>(K1 / 2)) ? gj : (N1 - K1 + gj);
                    int64_t fk = (gk < static_cast<int64_t>(K2 / 2)) ? gk : (N2 - K2 + gk);

                    // tempFieldg has no ghosts
                    int li_full = fi;
                    int lj_full = fj;
                    int lk_full = fk;

                    // gview has ghosts
                    int li_pruned = gi - lDomPruned[0].first() + nghostg;
                    int lj_pruned = gj - lDomPruned[1].first() + nghostg;
                    int lk_pruned = gk - lDomPruned[2].first() + nghostg;

                    tempFieldg(li_full, lj_full, lk_full) = gview(li_pruned, lj_pruned, lk_pruned);
                });

            // C2R IFFT: full complex -> real
            this->heffte_m->backward(tempFieldg.data(), tempFieldf.data(), this->workspace_m.data(),
                                     heffte::scale::none);

            // Copy real result to output (with ghosts)
            ippl::parallel_for(
                "copy real field from temp", getRangePolicy(fview, nghostf),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    apply(fview, args) = apply(tempFieldf, args - nghostf);
                });

        } else {
            throw std::logic_error("Only FORWARD and BACKWARD are allowed as directions");
        }
    }

    template <typename Field>
    void FFT<SineTransform, Field>::warmup(Field& f) {
        this->transform(FORWARD, f);
        this->transform(BACKWARD, f);
    }

    template <typename Field>
    void FFT<SineTransform, Field>::transform(TransformDirection direction, Field& f) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");
#ifdef Heffte_ENABLE_FFTW
        if (direction == FORWARD) {
            f = f / 8.0;
        }
#endif

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto& tempField = this->tempField;
        if (tempField.size() != f.getOwned().size()) {
            tempField = detail::shrinkView("tempField", fview, nghost);
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempField, args - nghost) = apply(fview, args);
            });

        if (direction == FORWARD) {
            this->heffte_m->forward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                    heffte::scale::full);
        } else if (direction == BACKWARD) {
            this->heffte_m->backward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                     heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(fview, args) = apply(tempField, args - nghost);
            });
#ifdef Heffte_ENABLE_FFTW
        if (direction == BACKWARD) {
            f = f * 8.0;
        }
#endif
    }

    template <typename Field>
    void FFT<CosTransform, Field>::warmup(Field& f) {
        this->transform(FORWARD, f);
        this->transform(BACKWARD, f);
    }

    template <typename Field>
    void FFT<CosTransform, Field>::transform(TransformDirection direction, Field& f) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");
#ifdef Heffte_ENABLE_FFTW
        if (direction == FORWARD) {
            f = f / 8.0;
        }
#endif

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto& tempField = this->tempField;
        if (tempField.size() != f.getOwned().size()) {
            tempField = detail::shrinkView("tempField", fview, nghost);
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempField, args - nghost) = apply(fview, args);
            });

        if (direction == FORWARD) {
            this->heffte_m->forward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                    heffte::scale::full);
        } else if (direction == BACKWARD) {
            this->heffte_m->backward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                     heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(fview, args) = apply(tempField, args - nghost);
            });
#ifdef Heffte_ENABLE_FFTW
        if (direction == BACKWARD) {
            f = f * 8.0;
        }
#endif
    }

    template <typename Field>
    void FFT<Cos1Transform, Field>::warmup(Field& f) {
        this->transform(FORWARD, f);
        this->transform(BACKWARD, f);
    }

    template <typename Field>
    void FFT<Cos1Transform, Field>::transform(TransformDirection direction, Field& f) {
        static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

/**
 * This rescaling is needed to match the normalization constant
 * between fftw and the other gpu interfaces. fftw rescales with an extra factor of 8.
 */
#ifdef Heffte_ENABLE_FFTW
        if (direction == FORWARD) {
            f = f / 8.0;
        }
#endif

        auto fview       = f.getView();
        const int nghost = f.getNghost();

        /**
         *This copy to a temporary Kokkos view is needed because of following
         *reasons:
         *1) heffte wants the input and output fields without ghost layers
         *2) heffte accepts data in layout left (by default) eventhough this
         *can be changed during heffte box creation
         */
        auto& tempField = this->tempField;
        if (tempField.size() != f.getOwned().size()) {
            tempField = detail::shrinkView("tempField", fview, nghost);
        }

        using index_array_type = typename RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "copy from Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(tempField, args - nghost) = apply(fview, args);
            });

        if (direction == FORWARD) {
            this->heffte_m->forward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                    heffte::scale::full);
        } else if (direction == BACKWARD) {
            this->heffte_m->backward(tempField.data(), tempField.data(), this->workspace_m.data(),
                                     heffte::scale::none);
        } else {
            throw std::logic_error("Only 1:forward and -1:backward are allowed as directions");
        }

        ippl::parallel_for(
            "copy to Kokkos FFT", getRangePolicy(fview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                apply(fview, args) = apply(tempField, args - nghost);
            });

/**
 * This rescaling is needed to match the normalization constant
 * between fftw and the other gpu interfaces. fftw rescales with an extra factor of 8.
 */
#ifdef Heffte_ENABLE_FFTW
        if (direction == BACKWARD) {
            f = f * 8.0;
        }
#endif
    }

    template <typename RealField>
    FFT<NUFFTransform, RealField>::FFT(const Layout_t& layout, const detail::size_type& localNp,
                                       int type, const ParameterList& params) {
        /**
         * FINUFFT requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to fill in other
         * dimensions to be 1. Note this is different from Heffte
         * where we fill 0.
         */

        std::array<int64_t, 3> nmodes;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        nmodes.fill(1);

        for (size_t d = 0; d < Dim; ++d) {
            nmodes[d] = layout.getDomain()[d].length();
        }
        use_kokkos_nufft = params.get<bool>("use_kokkos_nufft", false);
        use_finufft      = params.get<bool>("use_finufft", false);

        type_m = type;

        use_upsampled_inputs_m = params.get<bool>("use_upsampled_inputs", false);

        if (tempField_m.size() < lDom.size()) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }
        for (size_t d = 0; d < Dim; ++d) {
            if (tempR_m[d].size() < localNp) {
                Kokkos::realloc(tempR_m[d], localNp);
            }
        }
        if (tempQ_m.size() < localNp) {
            Kokkos::realloc(tempQ_m, localNp);
        }
        setup(layout, nmodes, params);
    }

    /**
        setup performs the initialization necessary.
    */
    template <typename RealField>
    void FFT<NUFFTransform, RealField>::setup(const Layout_t& layout,
                                              std::array<int64_t, 3>& nmodes,
                                              const ParameterList& params) {
        tol_m = params.get<T>("tolerance", 1e-6);

        if (use_kokkos_nufft) {
#ifdef KOKKOS_NUFFT_AVAILABLE
            this->n_modes = nmodes;

            // Setup kokkos_nufft
            nufft::array<typename RealField::memory_space::size_type, Dim> n_modes;
            for (int d = 0; d < Dim; ++d) {
                n_modes[d] = nmodes[d];
            }

            typename kokkos_nufft_t::Config cfg{tol_m};
            kokkos_nufft_plan = std::make_unique<kokkos_nufft_t>(n_modes, cfg);
#else
            throw IpplException("FFT<NUFFTransform>::setup",
                                "kokkos_nufft requested but not available");
#endif
        } else if (use_finufft) {
#ifdef ENABLE_FINUFFT

#if ENABLE_GPU_NUFFT
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);
#else
            finufft_opts opts;
            finufft_default_opts(&opts);
#endif

            tol_m = params.get<T>("tolerance");
#ifdef ENABLE_GPU_NUFFT
            opts.gpu_method      = params.get<int>("gpu_method", opts.gpu_method);
            opts.gpu_sort        = params.get<int>("gpu_sort", opts.gpu_sort);
            opts.gpu_kerevalmeth = params.get<int>("gpu_kerevalmeth", opts.gpu_kerevalmeth);
            opts.gpu_binsizex    = params.get<int>("gpu_binsizex", opts.gpu_binsizex);
            opts.gpu_binsizey    = params.get<int>("gpu_binsizey", opts.gpu_binsizey);
            opts.gpu_binsizez    = params.get<int>("gpu_binsizez", opts.gpu_binsizez);
            opts.gpu_maxsubprobsize =
                params.get<int>("gpu_maxsubprobsize", opts.gpu_maxsubprobsize);
#else
            opts.spread_sort        = params.get<int>("spread_sort");
            opts.spread_kerevalmeth = params.get<int>("spread_kerevalmeth");
            opts.nthreads           = params.get<int>("nthreads");
#endif

#ifdef ENABLE_GPU_NUFFT
            opts.gpu_maxbatchsize = 0;  // default option. ignored for ntransf = 1 which
                                        //  is our case
            // For Perlmutter since the mask to hide the other GPUs in the node is
            // somehow not working there
            // opts.gpu_device_id = (int)(Ippl::Comm->rank() % 4);
#endif

            int iflag;

            if (type_m == 1) {
                iflag = -1;
            } else if (type_m == 2) {
                iflag = 1;
            } else {
                throw std::logic_error("Only type 1 and type 2 NUFFT are allowed now");
            }

            // dim in finufft is int
            int dim = static_cast<int>(Dim);
            ier_m   = nufft_m.makeplan(type_m, dim, this->n_modes.data(), iflag, 1, tol_m, &plan_m,
                                       &opts);
#else
            throw IpplException(
                "FFT<NUFFTransform>::setup",
                "FINUFFT requested but not available (IPPL_ENABLE_FINUFFT not set)");
#endif
        } else {
            // Use native NUFFT implementation (default path)
            using NativeNUFFT_t = NUFFT::NativeNUFFT<Dim, T, typename RealField::execution_space>;

            Vector<size_t, Dim> n_modes_vec;
            for (unsigned d = 0; d < Dim; ++d) {
                n_modes_vec[d] = nmodes[d];
            }

            typename NativeNUFFT_t::Config cfg;
            cfg.tol   = tol_m;
            cfg.sigma = params.get<T>("sigma", T(2.0));
            cfg.scatter_config =
                Interpolation::ScatterConfig::get_default<typename RealField::execution_space>();
            cfg.gather_config =
                Interpolation::GatherConfig::get_default<typename RealField::execution_space>();

            // Configure spread method
            std::string spread_method = params.get<std::string>("spread_method", "none");
            if (spread_method == "atomic") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::Atomic;
            }
            if (spread_method == "output_focused") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::OutputFocused;
            }
            if (spread_method == "tiled") {
                cfg.scatter_config.method = Interpolation::ScatterMethod::Tiled;
            }
            if (params.contains("tile_size_3d")) {
                cfg.scatter_config.tile_size_3d = params.get<int>("tile_size_3d");
            }
            if (params.contains("z_tiles")) {
                cfg.scatter_config.z_tiles = params.get<int>("z_tiles");
            }
            if (params.contains("team_size")) {
                cfg.scatter_config.team_size = params.get<int>("team_size");
            }

            std::string gather_method = params.get<std::string>("gather_method", "none");
            if (gather_method == "tiled") {
                cfg.gather_config.method = Interpolation::GatherMethod::Tiled;
            }
            if (gather_method == "native") {
                cfg.gather_config.method = Interpolation::GatherMethod::Native;
            }
            if (gather_method == "atomic") {
                cfg.gather_config.method = Interpolation::GatherMethod::Atomic;
            }
            if (gather_method == "atomic_sort") {
                cfg.gather_config.method = Interpolation::GatherMethod::AtomicSort;
            }

            auto* nufft_ptr = new NativeNUFFT_t(n_modes_vec, use_upsampled_inputs_m, cfg);
            nufft_ptr->initialize(layout, MPI_COMM_WORLD);
            native_nufft_ = static_cast<void*>(nufft_ptr);
        }
    }

#ifdef KOKKOS_NUFFT_AVAILABLE
    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transform_kokkos_nufft(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        ComplexField& f) {
        auto fview       = f.getView();
        auto Rview       = R.getView();
        auto Qview       = Q.getView();
        const int nghost = f.getNghost();
        auto localNp     = R.getParticleCount();

        // Get layout information
        const Layout_t& layout               = f.getLayout();
        const UniformCartesian<T, Dim>& mesh = f.get_mesh();
        const Vector<T, Dim>& dx             = mesh.getMeshSpacing();
        const auto& domain                   = layout.getDomain();

        Vector<T, Dim> Len;
        for (unsigned d = 0; d < Dim; ++d) {
            Len[d] = dx[d] * domain[d].length();
        }
        const double pi = std::acos(-1.0);

        // Create views compatible with kokkos_nufft
        using ExecSpace = typename RealField::execution_space;
        using MemSpace  = typename ExecSpace::memory_space;

        Kokkos::View<T* [Dim], MemSpace> x_nufft("x", localNp);
        Kokkos::View<Kokkos::complex<T>*, MemSpace> c_nufft("c", localNp);

        // Convert particle positions and charges
        Kokkos::parallel_for(
            "prepare_kokkos_nufft", localNp, KOKKOS_LAMBDA(const size_t i) {
                for (size_t d = 0; d < Dim; ++d) {
                    x_nufft(i, d) = Rview(i)[d] * (2.0 * pi / Len[d]);
                }
                c_nufft(i) = Kokkos::complex<T>(Qview(i), 0.0);
            });

        // Create output view
        auto f_nufft = [&]() {
            const auto& lDom = layout.getLocalNDIndex();
            if constexpr (Dim == 1) {
                return Kokkos::View<Kokkos::complex<T>*, MemSpace>("f", lDom[0].length());
            } else if constexpr (Dim == 2) {
                return Kokkos::View<Kokkos::complex<T>**, MemSpace>("f", lDom[0].length(),
                                                                    lDom[1].length());
            } else {
                return Kokkos::View<Kokkos::complex<T>***, MemSpace>(
                    "f", lDom[0].length(), lDom[1].length(), lDom[2].length());
            }
        }();

        // Execute transform
        if (type_m == 1) {
            kokkos_nufft_plan->type1(x_nufft, c_nufft, f_nufft);
        } else if (type_m == 2) {
            // For Type 2: FFT-shift input field (no conjugation, we'll conjugate output instead)
            const auto& lDom = layout.getLocalNDIndex();
            const size_t nx  = lDom[0].length();
            const size_t ny  = lDom[1].length();
            const size_t nz  = lDom[2].length();

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "prepare_type2_input_fftshift",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    const size_t ii = i - nghost;
                    const size_t jj = j - nghost;
                    const size_t kk = k - nghost;

                    // FFT-shift: map from corner to centered convention
                    const size_t ii_shift = (ii + nx / 2) % nx;
                    const size_t jj_shift = (jj + ny / 2) % ny;
                    const size_t kk_shift = (kk + nz / 2) % nz;

                    // Dummy captures for nvcc
                    (void)fview;
                    (void)f_nufft;

                    // Just FFT-shift, no conjugation
                    if constexpr (Dim == 3) {
                        f_nufft(ii_shift, jj_shift, kk_shift) = fview(i, j, k);
                    } else if constexpr (Dim == 2) {
                        f_nufft(ii_shift, jj_shift) = fview(i, j, k);
                    } else {
                        f_nufft(ii_shift) = fview(i, j, k);
                    }
                });

            kokkos_nufft_plan->type2(f_nufft, x_nufft, c_nufft);
        }
        Kokkos::fence();

        // Copy results back with FFT-shift and conjugate to match FINUFFT convention
        if (type_m == 1) {
            const auto& lDom   = layout.getLocalNDIndex();
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "copy_to_field_with_fftshift_conj",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // kokkos_nufft returns FFT-shifted and conjugated output
                    const size_t nx = lDom[0].length();
                    const size_t ny = lDom[1].length();
                    const size_t nz = lDom[2].length();

                    const size_t ii = i - nghost;
                    const size_t jj = j - nghost;
                    const size_t kk = k - nghost;

                    // FFT-shift: map from centered to corner convention
                    const size_t ii_shift = (ii + nx / 2) % nx;
                    const size_t jj_shift = (jj + ny / 2) % ny;
                    const size_t kk_shift = (kk + nz / 2) % nz;

                    (void)fview;
                    (void)f_nufft;
                    if constexpr (Dim == 3) {
                        fview(i, j, k) = Kokkos::conj(f_nufft(ii_shift, jj_shift, kk_shift));
                    } else if constexpr (Dim == 2) {
                        fview(i, j, k) = Kokkos::conj(f_nufft(ii_shift, jj_shift));
                    } else {
                        fview(i, j, k) = Kokkos::conj(f_nufft(ii_shift));
                    }
                });
        } else if (type_m == 2) {
            // Type 2: Conjugate output to fix sign (kokkos uses -1, FINUFFT uses +1)
            Kokkos::parallel_for(
                "copy_to_particles_conj", localNp, KOKKOS_LAMBDA(const size_t i) {
                    // Conjugate and take real part
                    auto val      = c_nufft(i);
                    auto conj_val = Kokkos::conj(val);
                    Qview(i)      = conj_val.real();
                });
        }
    }
#endif

    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transform(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        typename FFT<NUFFTransform, RealField>::ComplexField& f) {
        auto localNp = R.getParticleCount();

        const Layout_t& layout               = f.getLayout();
        const UniformCartesian<T, Dim>& mesh = f.get_mesh();
        const Vector<T, Dim>& dx             = mesh.getMeshSpacing();
        const Vector<T, Dim>& origin         = mesh.getOrigin();
        const auto& domain                   = layout.getDomain();
        Vector<T, Dim> Len;
        Vector<int, Dim> N;
        const int nghost = f.getNghost();

        for (unsigned d = 0; d < Dim; ++d) {
            N[d]   = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        const double pi = std::acos(-1.0);
        if (use_kokkos_nufft) {
#ifdef KOKKOS_NUFFT_AVAILABLE
            transform_kokkos_nufft(R, Q, f);
#endif
        } else if (use_finufft) {
#ifdef ENABLE_FINUFFT
            auto fview = f.getView();
            auto Rview = R.getView();
            auto Qview = Q.getView();

            auto tempField                                = tempField_m;
            auto tempQ                                    = tempQ_m;
            Kokkos::View<T*, Kokkos::LayoutLeft> tempR[3] = {};

            for (size_t d = 0; d < Dim; ++d) {
                tempR[d] = tempR_m[d];
            }
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

            Kokkos::parallel_for(
                "copy from field data NUFFT",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
#ifdef ENABLE_GPU_NUFFT
                    tempField(i - nghost, j - nghost, k - nghost).x = fview(i, j, k).real();
                    tempField(i - nghost, j - nghost, k - nghost).y = fview(i, j, k).imag();
#else
                    tempField(i - nghost, j - nghost, k - nghost).real(fview(i, j, k).real());
                    tempField(i - nghost, j - nghost, k - nghost).imag(fview(i, j, k).imag());
#endif
                });

            Kokkos::parallel_for(
                "copy from particle data NUFFT", localNp, KOKKOS_LAMBDA(const size_t i) {
                    for (size_t d = 0; d < Dim; ++d) {
                        // tempR[d](i) = (Rview(i)[d] - (twopiFactor * 2.0 * pi)) * (2.0 * pi /
                        // Len[d]);
                        tempR[d](i) = Rview(i)[d] * (2.0 * pi / Len[d]);
                        // tempR[d](i) = Rview(i)[d];
                    }
#ifdef ENABLE_GPU_NUFFT
                    tempQ(i).x = Qview(i);
                    tempQ(i).y = 0.0;
#else
                                     tempQ(i).real(Qview(i));
                                     tempQ(i).imag(0.0);
#endif
                });

            ier_m = nufft_m.setpts(plan_m, localNp, tempR[0].data(), tempR[1].data(),
                                   tempR[2].data(), 0, NULL, NULL, NULL);

            ier_m = nufft_m.execute(plan_m, tempQ.data(), tempField.data());
            Kokkos::fence();

            if (type_m == 1) {
                Kokkos::parallel_for(
                    "copy to field data NUFFT",
                    mdrange_type({nghost, nghost, nghost},
                                 {fview.extent(0) - nghost, fview.extent(1) - nghost,
                                  fview.extent(2) - nghost}),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
#ifdef ENABLE_GPU_NUFFT
                        fview(i, j, k).real() = tempField(i - nghost, j - nghost, k - nghost).x;
                        fview(i, j, k).imag() = tempField(i - nghost, j - nghost, k - nghost).y;
#else
                        fview(i, j, k).real() =
                            tempField(i - nghost, j - nghost, k - nghost).real();
                        fview(i, j, k).imag() =
                            tempField(i - nghost, j - nghost, k - nghost).imag();
#endif
                    });
            } else if (type_m == 2) {
                Kokkos::parallel_for(
                    "copy to particle data NUFFT", localNp, KOKKOS_LAMBDA(const size_t i) {
#ifdef ENABLE_GPU_NUFFT
                        Qview(i) = tempQ(i).x;
#else
                                         Qview(i) = tempQ(i).real();
#endif
                    });
            }
#else
            throw IpplException("FFT<NUFFTransform>::transform",
                                "FINUFFT requested but not available");
#endif
        } else {
            auto localNp                         = R.getParticleCount();
            const Layout_t& layout               = f.getLayout();
            const UniformCartesian<T, Dim>& mesh = f.get_mesh();
            const Vector<T, Dim>& dx             = mesh.getMeshSpacing();
            const Vector<T, Dim>& origin         = mesh.getOrigin();
            const auto& domain                   = layout.getDomain();
            Vector<T, Dim> Len;
            Vector<int, Dim> N;
            const int nghost = f.getNghost();
            for (unsigned d = 0; d < Dim; ++d) {
                N[d]   = domain[d].length();
                Len[d] = dx[d] * N[d];
            }
            const double pi = std::acos(-1.0);

            // Use native NUFFT implementation (default path)
            using NativeNUFFT_t = NUFFT::NativeNUFFT<Dim, T, typename RealField::execution_space>;
            auto* nufft         = static_cast<NativeNUFFT_t*>(native_nufft_);
            auto Rview          = R.getView();
            auto fview          = f.getView();
            Kokkos::parallel_for(
                "Scale particles to 2pi", localNp, KOKKOS_LAMBDA(const size_t i) {
                    for (size_t d = 0; d < Dim; ++d) {
                        Rview(i)[d] *= (2.0 * pi / Len[d]);
                    }
                });

            if (type_m == 1) {
                nufft->type1(R, Q, f, use_upsampled_inputs_m);
            } else if (type_m == 2) {
                nufft->type2(
                    f, R, Q,
                    use_upsampled_inputs_m);  // Note: argument order is different for type2
            }
            Kokkos::parallel_for(
                "Roll back the scaling", localNp, KOKKOS_LAMBDA(const size_t i) {
                    for (size_t d = 0; d < Dim; ++d) {
                        Rview(i)[d] *= (Len[d] / (2.0 * pi));
                    }
                });
        }
    }

    template <typename RealField>
    FFT<NUFFTransform, RealField>::~FFT() {
#ifdef ENABLE_FINUFFT
        if (use_finufft) {
            ier_m = nufft_m.destroy(plan_m);
        }
#endif
        // Clean up native NUFFT (when not using kokkos_nufft or finufft)
        if (!use_kokkos_nufft && !use_finufft && native_nufft_) {
            using NativeNUFFT_t = NUFFT::NativeNUFFT<RealField::dim, typename RealField::value_type,
                                                     typename RealField::execution_space>;
            delete static_cast<NativeNUFFT_t*>(native_nufft_);
            native_nufft_ = nullptr;
        }
    }

}  // namespace ippl

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
