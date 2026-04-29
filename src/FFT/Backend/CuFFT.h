#ifndef IPPL_FFT_BACKEND_CUFFT_H
#define IPPL_FFT_BACKEND_CUFFT_H

#ifdef KOKKOS_ENABLE_CUDA

#include <array>
#include <cufft.h>
#include <cuda_runtime.h>
#include <heffte.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

namespace ippl {
namespace fft {

    namespace detail {
        inline void checkCufftResult(cufftResult result, const char* context) {
            if (result != CUFFT_SUCCESS) {
                std::string msg =
                    std::string(context) + " (error code: " + std::to_string(result) + ")";
                throw IpplException("cuFFT", msg.c_str());
            }
        }

        inline void checkCudaError(cudaError_t err, const char* context) {
            if (err != cudaSuccess) {
                std::string msg = std::string(context) + ": " + cudaGetErrorString(err);
                throw IpplException("cuFFT", msg.c_str());
            }
        }
    }  // namespace detail

    namespace detail {
        template <typename T>
        __global__ void cufftScaleKernel(T* data, size_t n, double scale) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx].x *= scale;
                data[idx].y *= scale;
            }
        }
    }  // namespace detail

    //=========================================================================
    // CuFFTC2C - Single-node cuFFT with Batched Support
    //=========================================================================

    template <typename T, unsigned Dim, typename MemSpace>
    class CuFFTC2C {
    public:
        static_assert(std::is_same_v<MemSpace, Kokkos::CudaSpace>,
                      "CuFFTC2C requires Kokkos::CudaSpace");
        static_assert(Dim == 3, "CuFFTC2C only supports 3D");
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "CuFFTC2C only supports float and double precision");

        using complex_t      = Kokkos::complex<T>;
        using cuda_complex_t = std::conditional_t<std::is_same_v<T, float>,
                                                   cufftComplex,
                                                   cufftDoubleComplex>;

        static constexpr cufftType fft_type = std::is_same_v<T, float> ? CUFFT_C2C : CUFFT_Z2Z;

        // Constructor matching HeFFTe/cuFFTMp interface
        CuFFTC2C(const heffte::box3d<long long>& inbox,
                 const heffte::box3d<long long>& outbox,
                 MPI_Comm comm,
                 const ParameterList& /*params*/,
                 int maxBatchSize = 1)
            : maxBatchSize_(maxBatchSize)
            , comm_(comm)
        {
            using detail::checkCudaError;
            using detail::checkCufftResult;

            // Extract local dimensions from inbox
            for (int d = 0; d < 3; ++d) {
                lowerIn_[d]    = inbox.low[d];
                upperIn_[d]    = inbox.high[d] + 1;
                lowerOut_[d]   = outbox.low[d];
                upperOut_[d]   = outbox.high[d] + 1;
                localSize_[d]  = upperIn_[d] - lowerIn_[d];
            }

            // Compute global size via MPI reduction
            std::array<long long, 3> localMax;
            for (int d = 0; d < 3; ++d) {
                localMax[d] = std::max(upperIn_[d], upperOut_[d]);
            }
            MPI_Allreduce(localMax.data(), globalSize_.data(), 3, MPI_LONG_LONG, MPI_MAX, comm);

            localElements_  = localSize_[0] * localSize_[1] * localSize_[2];
            globalElements_ = globalSize_[0] * globalSize_[1] * globalSize_[2];

            // Create CUDA stream
            checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

            // cuFFT expects row-major (C-order) dimensions. The Kokkos views are
            // LayoutLeft, so dimension 0 is fastest-varying - pass extents reversed.
            int n[3] = {
                static_cast<int>(localSize_[2]),
                static_cast<int>(localSize_[1]),
                static_cast<int>(localSize_[0])
            };

            int inembed[3] = {n[0], n[1], n[2]};
            int onembed[3] = {n[0], n[1], n[2]};
            int istride    = 1;
            int ostride    = 1;
            int idist      = static_cast<int>(localElements_);
            int odist      = static_cast<int>(localElements_);

            // Create batched plan
            checkCufftResult(
                cufftPlanMany(&planBatched_, 3, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              fft_type, maxBatchSize),
                "Failed to create batched cuFFT plan");

            checkCufftResult(cufftSetStream(planBatched_, stream_),
                             "Failed to set stream on batched plan");

            // Create single-transform plan if needed
            if (maxBatchSize > 1) {
                checkCufftResult(
                    cufftPlanMany(&planSingle_, 3, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  fft_type, 1),
                    "Failed to create single cuFFT plan");

                checkCufftResult(cufftSetStream(planSingle_, stream_),
                                 "Failed to set stream on single plan");
            } else {
                planSingle_ = planBatched_;
            }
        }

        ~CuFFTC2C() {
            if (planBatched_) cufftDestroy(planBatched_);
            if (maxBatchSize_ > 1 && planSingle_) cufftDestroy(planSingle_);
            if (stream_) cudaStreamDestroy(stream_);
        }

        // Non-copyable
        CuFFTC2C(const CuFFTC2C&)            = delete;
        CuFFTC2C& operator=(const CuFFTC2C&) = delete;

        // Movable
        CuFFTC2C(CuFFTC2C&& other) noexcept
            : planBatched_(other.planBatched_)
            , planSingle_(other.planSingle_)
            , stream_(other.stream_)
            , comm_(other.comm_)
            , maxBatchSize_(other.maxBatchSize_)
            , localElements_(other.localElements_)
            , globalElements_(other.globalElements_)
            , localSize_(other.localSize_)
            , globalSize_(other.globalSize_)
            , lowerIn_(other.lowerIn_)
            , upperIn_(other.upperIn_)
            , lowerOut_(other.lowerOut_)
            , upperOut_(other.upperOut_)
        {
            other.planBatched_ = 0;
            other.planSingle_  = 0;
            other.stream_      = nullptr;
        }

        CuFFTC2C& operator=(CuFFTC2C&& other) noexcept {
            if (this != &other) {
                if (planBatched_) cufftDestroy(planBatched_);
                if (maxBatchSize_ > 1 && planSingle_) cufftDestroy(planSingle_);
                if (stream_) cudaStreamDestroy(stream_);

                planBatched_     = other.planBatched_;
                planSingle_      = other.planSingle_;
                stream_          = other.stream_;
                comm_            = other.comm_;
                maxBatchSize_    = other.maxBatchSize_;
                localElements_   = other.localElements_;
                globalElements_  = other.globalElements_;
                localSize_       = other.localSize_;
                globalSize_      = other.globalSize_;
                lowerIn_         = other.lowerIn_;
                upperIn_         = other.upperIn_;
                lowerOut_        = other.lowerOut_;
                upperOut_        = other.upperOut_;

                other.planBatched_ = 0;
                other.planSingle_  = 0;
                other.stream_      = nullptr;
            }
            return *this;
        }

        // Single transform
        void forward(complex_t* in, complex_t* out) {
            execute(planSingle_, in, out, CUFFT_FORWARD);
            applyScaling(out, localElements_, T(1) / static_cast<T>(globalElements_));
            detail::checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
        }

        void backward(complex_t* in, complex_t* out) {
            execute(planSingle_, in, out, CUFFT_INVERSE);
            detail::checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
        }

        // Batched transform
        void forward(int batchSize, complex_t* in, complex_t* out) {
            if (batchSize > maxBatchSize_) {
                throw IpplException("CuFFTC2C", "Batch size exceeds plan capacity");
            }

            if (batchSize == maxBatchSize_) {
                execute(planBatched_, in, out, CUFFT_FORWARD);
            } else {
                // Execute individual transforms for partial batch
                for (int b = 0; b < batchSize; ++b) {
                    execute(planSingle_,
                            in + b * localElements_,
                            out + b * localElements_,
                            CUFFT_FORWARD);
                }
            }
            applyScaling(out, localElements_ * batchSize, T(1) / static_cast<T>(globalElements_));
            detail::checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
        }

        void backward(int batchSize, complex_t* in, complex_t* out) {
            if (batchSize > maxBatchSize_) {
                throw IpplException("CuFFTC2C", "Batch size exceeds plan capacity");
            }

            if (batchSize == maxBatchSize_) {
                execute(planBatched_, in, out, CUFFT_INVERSE);
            } else {
                for (int b = 0; b < batchSize; ++b) {
                    execute(planSingle_,
                            in + b * localElements_,
                            out + b * localElements_,
                            CUFFT_INVERSE);
                }
            }
            detail::checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
        }

        // Set external stream
        void setStream(cudaStream_t stream) {
            stream_ = stream;
            detail::checkCufftResult(cufftSetStream(planBatched_, stream_),
                                     "Failed to set stream on batched plan");
            if (maxBatchSize_ > 1) {
                detail::checkCufftResult(cufftSetStream(planSingle_, stream_),
                                         "Failed to set stream on single plan");
            }
        }

        // Accessors
        size_t workspace_size() const { return 0; }  // cuFFT manages internally
        size_t local_size() const { return localElements_; }
        size_t global_size() const { return globalElements_; }
        size_t size_inbox() const { return localElements_; }
        size_t size_outbox() const { return localElements_; }
        int max_batch_size() const { return maxBatchSize_; }
        cudaStream_t stream() const { return stream_; }
        const std::array<long long, 3>& local_dims() const { return localSize_; }
        const std::array<long long, 3>& global_dims() const { return globalSize_; }

    private:
        void execute(cufftHandle plan, complex_t* in, complex_t* out, int direction) {
            auto* inPtr  = reinterpret_cast<cuda_complex_t*>(in);
            auto* outPtr = reinterpret_cast<cuda_complex_t*>(out);

            if constexpr (std::is_same_v<T, float>) {
                detail::checkCufftResult(cufftExecC2C(plan, inPtr, outPtr, direction),
                                         "cuFFT C2C execution failed");
            } else {
                detail::checkCufftResult(cufftExecZ2Z(plan, inPtr, outPtr, direction),
                                         "cuFFT Z2Z execution failed");
            }
        }

        void applyScaling(complex_t* data, size_t count, T scale) {
            auto* ptr                  = reinterpret_cast<cuda_complex_t*>(data);
            constexpr size_t blockSize = 256;
            size_t numBlocks           = (count + blockSize - 1) / blockSize;
            detail::cufftScaleKernel<<<numBlocks, blockSize, 0, stream_>>>(
                ptr, count, static_cast<double>(scale));
        }

        cufftHandle planBatched_ = 0;
        cufftHandle planSingle_  = 0;
        cudaStream_t stream_     = nullptr;
        MPI_Comm comm_;

        int maxBatchSize_;
        size_t localElements_;
        size_t globalElements_;

        std::array<long long, 3> localSize_;
        std::array<long long, 3> globalSize_;
        std::array<long long, 3> lowerIn_, upperIn_;
        std::array<long long, 3> lowerOut_, upperOut_;
    };

}  // namespace fft
}  // namespace ippl

#endif  // KOKKOS_ENABLE_CUDA

#endif  // IPPL_FFT_BACKEND_CUFFT_H