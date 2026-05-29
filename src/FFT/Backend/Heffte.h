#ifndef IPPL_FFT_BACKEND_HEFFTE_H
#define IPPL_FFT_BACKEND_HEFFTE_H

#include <heffte_fft3d.h>
#include <heffte_fft3d_r2c.h>
#include <memory>

#include "Utility/ParameterList.h"

#include "Field/BareField.h"

#include "FFT/Traits.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {
    namespace fft {

        /*!
         * @brief Multiply a complex array by a real scalar in-place on @p MemSpace.
         *
         * Used as a post-FFT normalization helper for the heFFTe backends.
         *
         * @tparam T        Real (precision) type of the complex elements.
         * @tparam MemSpace Kokkos memory space the buffer lives in.
         * @param data  Raw pointer to the device-/host-resident buffer.
         * @param scale Scalar multiplier applied to every element.
         * @param size  Number of complex elements in @p data.
         */
        template <typename T, typename MemSpace>
        inline void applyScale(Kokkos::complex<T>* data, T scale, size_t size) {
            Kokkos::View<Kokkos::complex<T>*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "Heffte_scale_complex",
                Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        /*!
         * @brief Multiply a real array by a scalar in-place on @p MemSpace.
         *
         * Real-buffer counterpart to applyScale used by trigonometric transforms.
         *
         * @tparam T        Element value type.
         * @tparam MemSpace Kokkos memory space the buffer lives in.
         * @param data  Raw pointer to the buffer.
         * @param scale Scalar multiplier applied to every element.
         * @param size  Number of elements in @p data.
         */
        template <typename T, typename MemSpace>
        inline void applyScaleReal(T* data, T scale, size_t size) {
            Kokkos::View<T*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "Heffte_scale_real",
                Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        /*!
         * @brief Compute the total global FFT grid size by reducing the upper
         *        bounds of @p inbox and @p outbox across @p comm.
         *
         * Each rank only knows its own local box; the global size requires an
         * MPI_MAX reduction over the upper-corner indices.
         *
         * @param inbox  Local input box (low/high inclusive corner indices).
         * @param outbox Local output box (low/high inclusive corner indices).
         * @param comm   MPI communicator over which the FFT is distributed.
         * @return Total number of points in the global FFT grid.
         */
        inline size_t computeGlobalSize(const heffte::box3d<long long>& inbox,
                                        const heffte::box3d<long long>& outbox, MPI_Comm comm) {
            long long local_max[3], global_max[3];
            local_max[0] = std::max(inbox.high[0], outbox.high[0]) + 1;
            local_max[1] = std::max(inbox.high[1], outbox.high[1]) + 1;
            local_max[2] = std::max(inbox.high[2], outbox.high[2]) + 1;

            MPI_Allreduce(local_max, global_max, 3, MPI_LONG_LONG, MPI_MAX, comm);

            return static_cast<size_t>(global_max[0]) * static_cast<size_t>(global_max[1])
                   * static_cast<size_t>(global_max[2]);
        }

        /*!
         * @brief Translate IPPL @p params into a heFFTe plan_options struct.
         *
         * If `use_heffte_defaults` is set, the heFFTe library defaults are
         * returned unchanged (with GPU-aware MPI enabled). Otherwise the
         * pencil/reorder flags, GPU-aware flag (only for GPU backends), and
         * communication algorithm (FFTComm enum: a2a, a2av, p2p, p2p_pl)
         * are pulled from @p params.
         *
         * @tparam HeffteBackendT Concrete heFFTe backend (e.g. heffte::backend::cufft).
         * @param  params         IPPL parameter list with FFT tuning knobs.
         * @return Configured heFFTe plan_options.
         * @throws IpplException on an unknown communication enum value.
         */
        template <typename HeffteBackendT>
        heffte::plan_options makeHeffteOptions(const ParameterList& params) {
            auto opts = heffte::default_options<HeffteBackendT>();

            if (!params.get<bool>("use_heffte_defaults")) {
                opts.use_pencils = params.get<bool>("use_pencils");
                opts.use_reorder = params.get<bool>("use_reorder");

                if constexpr (is_available_v<HeffteGPU>) {
                    opts.use_gpu_aware = params.get<bool>("use_gpu_aware");
                }

                switch (params.get<int>("comm")) {
                    case a2a:
                        opts.algorithm = heffte::reshape_algorithm::alltoall;
                        break;
                    case a2av:
                        opts.algorithm = heffte::reshape_algorithm::alltoallv;
                        break;
                    case p2p:
                        opts.algorithm = heffte::reshape_algorithm::p2p;
                        break;
                    case p2p_pl:
                        opts.algorithm = heffte::reshape_algorithm::p2p_plined;
                        break;
                    default:
                        throw IpplException("FFT", "Unknown communication type");
                }
            } else {
                opts.use_gpu_aware = true;
                opts.algorithm     = heffte::reshape_algorithm::p2p_plined;
            }
            return opts;
        }

        //=============================================================================
        // heFFTe C2C
        //=============================================================================

        /*!
         * @class HeffteC2C
         * @brief Thin wrapper around heffte::fft3d for complex-to-complex transforms.
         *
         * Owns the heFFTe plan and a workspace large enough to handle the
         * configured maximum batch size. forward() applies full normalization
         * (heffte::scale::full); backward() applies none, so a forward followed
         * by a backward returns the input.
         *
         * @tparam T        Real precision type (float / double).
         * @tparam Dim      Spatial dimension (only 2 and 3 are supported).
         * @tparam MemSpace Kokkos memory space holding the input/output buffers.
         */
        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteC2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

            /*!
             * @brief Construct a heFFTe C2C plan over the given box decomposition.
             *
             * @param inbox        Local input box (inclusive corner indices).
             * @param outbox       Local output box (inclusive corner indices).
             * @param comm         MPI communicator participating in the transform.
             * @param params       FFT parameter list (see makeHeffteOptions).
             * @param maxBatchSize Maximum batch size for batched transforms.
             */
            HeffteC2C(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox,
                      MPI_Comm comm, const ParameterList& params, int maxBatchSize = 1)
                : maxBatchSize_(maxBatchSize) {
                static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

                auto opts = makeHeffteOptions<backend_t>(params);
                heffte_   = std::make_shared<heffte_t>(inbox, outbox, comm, opts);

                // Allocate workspace for maximum batch size
                workspace_ = workspace_t(heffte_->size_workspace() * maxBatchSize);

                localSize_  = heffte_->size_outbox();
                globalSize_ = computeGlobalSize(inbox, outbox, comm);
            }

            //! Single forward C2C transform with full normalization.
            void forward(complex_t* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::full);
            }

            //! Single backward C2C transform (no normalization).
            void backward(complex_t* in, complex_t* out) {
                heffte_->backward(in, out, workspace_.data(), heffte::scale::none);
            }

            //! Batched forward C2C transform; @p batchSize must be <= max_batch_size().
            void forward(int batchSize, complex_t* in, complex_t* out) {
                assert(batchSize <= maxBatchSize_ && "Batch size exceeds allocated workspace");
                heffte_->forward(batchSize, in, out, workspace_.data(), heffte::scale::full);
            }

            //! Batched backward C2C transform; @p batchSize must be <= max_batch_size().
            void backward(int batchSize, complex_t* in, complex_t* out) {
                assert(batchSize <= maxBatchSize_ && "Batch size exceeds allocated workspace");
                heffte_->backward(batchSize, in, out, workspace_.data(), heffte::scale::none);
            }

            //! @return Per-plan heFFTe workspace size (single batch slot).
            size_t workspace_size() const { return heffte_->size_workspace(); }
            //! @return Number of local complex elements after the transform.
            size_t local_size() const { return localSize_; }
            //! @return Total number of points in the global FFT grid.
            size_t global_size() const { return globalSize_; }
            //! @return Local input-box size as reported by heFFTe.
            size_t size_inbox() const { return heffte_->size_inbox(); }
            //! @return Local output-box size as reported by heFFTe.
            size_t size_outbox() const { return heffte_->size_outbox(); }
            //! @return Maximum batch size the workspace was allocated for.
            int max_batch_size() const { return maxBatchSize_; }

        private:
            std::shared_ptr<heffte_t> heffte_;
            workspace_t workspace_;
            size_t localSize_;
            size_t globalSize_;
            int maxBatchSize_;
        };

        //=============================================================================
        // heFFTe R2C
        //=============================================================================

        /*!
         * @class HeffteR2C
         * @brief Wrapper around heffte::fft3d_r2c for real-to-complex transforms.
         *
         * forward() consumes a real buffer and writes the half-complex spectrum;
         * backward() does the inverse. Normalization matches HeffteC2C: forward
         * is fully normalized, backward is unscaled.
         *
         * @tparam T        Real precision type.
         * @tparam Dim      Spatial dimension.
         * @tparam MemSpace Kokkos memory space holding the buffers.
         */
        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteR2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d_r2c<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

            /*!
             * @brief Construct an R2C plan over the given decomposition.
             *
             * @param inbox         Local real-input box.
             * @param outbox        Local complex-output box (Hermitian-symmetric).
             * @param r2c_direction Axis along which the half-complex output lives.
             * @param comm          MPI communicator.
             * @param params        FFT parameter list (see makeHeffteOptions).
             */
            HeffteR2C(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox,
                      int r2c_direction, MPI_Comm comm, const ParameterList& params) {
                auto opts  = makeHeffteOptions<backend_t>(params);
                heffte_    = std::make_shared<heffte_t>(inbox, outbox, r2c_direction, comm, opts);
                workspace_ = workspace_t(heffte_->size_workspace());

                local_complex_size_ = heffte_->size_outbox();

                // For R2C, normalize by the global REAL size
                // inbox is the real box
                long long local_max[3], global_max[3];
                local_max[0] = inbox.high[0] + 1;
                local_max[1] = inbox.high[1] + 1;
                local_max[2] = inbox.high[2] + 1;

                MPI_Allreduce(local_max, global_max, 3, MPI_LONG_LONG, MPI_MAX, comm);

                global_real_size_ = static_cast<size_t>(global_max[0])
                                    * static_cast<size_t>(global_max[1])
                                    * static_cast<size_t>(global_max[2]);
            }

            //! Forward R2C transform with full normalization (real -> half-complex).
            void forward(T* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::full);
            }

            //! Backward C2R transform without normalization (half-complex -> real).
            void backward(complex_t* in, T* out) {
                heffte_->backward(in, out, workspace_.data(), heffte::scale::none);
            }

        private:
            std::shared_ptr<heffte_t> heffte_;
            workspace_t workspace_;
            size_t local_complex_size_;
            size_t global_real_size_;
        };

        //=============================================================================
        // heFFTe Trigonometric (Sine, Cos, Cos1)
        //=============================================================================

        /*!
         * @class HeffteTrig
         * @brief Wrapper around heffte::fft3d for sine / cosine transforms.
         *
         * Specializations are generated by IPPL_FFT_DEFINE_HEFFTE_TRIG for each
         * transform tag (SineTransform, CosTransform, Cos1Transform). The tag
         * selects the corresponding heFFTe backend.
         *
         * @tparam T        Real precision type.
         * @tparam Dim      Spatial dimension.
         * @tparam MemSpace Kokkos memory space holding the buffers.
         * @tparam Tag      One of SineTransform, CosTransform, Cos1Transform.
         */
        template <typename T, unsigned Dim, typename MemSpace, typename Tag>
        class HeffteTrig;

#define IPPL_FFT_DEFINE_HEFFTE_TRIG(TagType, member)                                              \
    template <typename T, unsigned Dim, typename MemSpace>                                        \
    class HeffteTrig<T, Dim, MemSpace, TagType> {                                                 \
    public:                                                                                       \
        using backend_t   = typename HeffteBackend<MemSpace>::member;                             \
        using heffte_t    = heffte::fft3d<backend_t, long long>;                                  \
        using workspace_t = typename heffte_t::template buffer_container<T>;                      \
                                                                                                  \
        HeffteTrig(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox, \
                   MPI_Comm comm, const ParameterList& params) {                                  \
            auto opts    = makeHeffteOptions<backend_t>(params);                                  \
            heffte_      = std::make_shared<heffte_t>(inbox, outbox, comm, opts);                 \
            workspace_   = workspace_t(heffte_->size_workspace());                                \
            local_size_  = heffte_->size_outbox();                                                \
            global_size_ = computeGlobalSize(inbox, outbox, comm);                                \
        }                                                                                         \
                                                                                                  \
        void forward(T* in, T* out) {                                                             \
            heffte_->forward(in, out, workspace_.data(), heffte::scale::full);                    \
        }                                                                                         \
                                                                                                  \
        void backward(T* in, T* out) {                                                            \
            heffte_->backward(in, out, workspace_.data(), heffte::scale::none);                   \
        }                                                                                         \
                                                                                                  \
    private:                                                                                      \
        std::shared_ptr<heffte_t> heffte_;                                                        \
        workspace_t workspace_;                                                                   \
        size_t local_size_;                                                                       \
        size_t global_size_;                                                                      \
    };

        IPPL_FFT_DEFINE_HEFFTE_TRIG(SineTransform, sin)
        IPPL_FFT_DEFINE_HEFFTE_TRIG(CosTransform, cos)
        IPPL_FFT_DEFINE_HEFFTE_TRIG(Cos1Transform, cos1)

#undef IPPL_FFT_DEFINE_HEFFTE_TRIG

    }  // namespace fft
}  // namespace ippl

#endif
