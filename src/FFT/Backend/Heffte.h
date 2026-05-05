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

        template <typename T, typename MemSpace>
        inline void applyScale(Kokkos::complex<T>* data, T scale, size_t size) {
            Kokkos::View<Kokkos::complex<T>*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "Heffte_scale_complex",
                Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        template <typename T, typename MemSpace>
        inline void applyScaleReal(T* data, T scale, size_t size) {
            Kokkos::View<T*, MemSpace> view(data, size);
            Kokkos::parallel_for(
                "Heffte_scale_real",
                Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size),
                KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
            Kokkos::fence();
        }

        // Helper to compute global FFT size from box coordinates
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
        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteC2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

            // Standard constructor
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

            // Single transform
            void forward(complex_t* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::full);
            }

            void backward(complex_t* in, complex_t* out) {
                heffte_->backward(in, out, workspace_.data(), heffte::scale::none);
            }

            // Batched transform
            void forward(int batchSize, complex_t* in, complex_t* out) {
                assert(batchSize <= maxBatchSize_ && "Batch size exceeds allocated workspace");
                heffte_->forward(batchSize, in, out, workspace_.data(), heffte::scale::full);
            }

            void backward(int batchSize, complex_t* in, complex_t* out) {
                assert(batchSize <= maxBatchSize_ && "Batch size exceeds allocated workspace");
                heffte_->backward(batchSize, in, out, workspace_.data(), heffte::scale::none);
            }

            // Accessors
            size_t workspace_size() const { return heffte_->size_workspace(); }
            size_t local_size() const { return localSize_; }
            size_t global_size() const { return globalSize_; }
            size_t size_inbox() const { return heffte_->size_inbox(); }
            size_t size_outbox() const { return heffte_->size_outbox(); }
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

        template <typename T, unsigned Dim, typename MemSpace>
        class HeffteR2C {
        public:
            using complex_t   = Kokkos::complex<T>;
            using backend_t   = typename HeffteBackend<MemSpace>::c2c;
            using heffte_t    = heffte::fft3d_r2c<backend_t, long long>;
            using workspace_t = typename heffte_t::template buffer_container<complex_t>;

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

            void forward(T* in, complex_t* out) {
                heffte_->forward(in, out, workspace_.data(), heffte::scale::full);
            }

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