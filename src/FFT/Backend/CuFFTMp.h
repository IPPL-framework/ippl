/*!
 * @file CuFFTMp.h
 * @brief Multi-node cuFFTMp wrappers for IPPL FFT transforms.
 *
 * Provides C2C, R2C and pruned variants on top of cuFFTMp. Includes small
 * CUDA helpers for layout transposes (LayoutLeft <-> LayoutRight) since
 * cuFFTMp expects row-major data while IPPL uses Kokkos LayoutLeft views.
 */
#ifndef IPPL_FFT_BACKEND_CUFFTMP_H
#define IPPL_FFT_BACKEND_CUFFTMP_H

#include <array>
#include <cufftMp.h>
#include <mpi.h>
#include <type_traits>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

#include "FFT/Traits.h"

namespace ippl {
    namespace fft {

        namespace detail {
            /*!
             * @brief Throw IpplException if @p result is not CUFFT_SUCCESS.
             * @param result  cuFFT API return code.
             * @param context Human-readable label for the failing call.
             */
            inline void checkCufftResult(cufftResult result, const char* context) {
                if (result != CUFFT_SUCCESS) {
                    std::string msg =
                        std::string(context) + " (error code: " + std::to_string(result) + ")";
                    throw IpplException("cuFFTMp", msg.c_str());
                }
            }

            /*!
             * @brief Throw IpplException if @p err is not cudaSuccess.
             * @param err     CUDA runtime error code.
             * @param context Human-readable label for the failing call.
             */
            inline void checkCudaError(cudaError_t err, const char* context) {
                if (err != cudaSuccess) {
                    std::string msg = std::string(context) + ": " + cudaGetErrorString(err);
                    throw IpplException("cuFFTMp", msg.c_str());
                }
            }

            /*!
             * @brief CUDA kernel that transposes a 3D buffer from LayoutLeft
             *        to LayoutRight indexing.
             *
             * LayoutLeft:  src[i + j*n0 + k*n0*n1]  (i is fastest-varying).
             * LayoutRight: dst[k + j*n2 + i*n1*n2]  (k is fastest-varying).
             *
             * @tparam T Element type (cufftComplex / cufftDoubleComplex / real).
             */
            template <typename T>
            __global__ void transposeL2R(T* __restrict__ dst, const T* __restrict__ src, int n0,
                                         int n1, int n2) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                int j = blockIdx.y * blockDim.y + threadIdx.y;
                int k = blockIdx.z * blockDim.z + threadIdx.z;

                if (i < n0 && j < n1 && k < n2) {
                    size_t src_idx = i + j * n0 + k * n0 * n1;  // LayoutLeft
                    size_t dst_idx = k + j * n2 + i * n1 * n2;  // LayoutRight
                    dst[dst_idx]   = src[src_idx];
                }
            }

            /*!
             * @brief CUDA kernel inverse of transposeL2R (LayoutRight -> LayoutLeft).
             */
            template <typename T>
            __global__ void transposeR2L(T* __restrict__ dst, const T* __restrict__ src, int n0,
                                         int n1, int n2) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                int j = blockIdx.y * blockDim.y + threadIdx.y;
                int k = blockIdx.z * blockDim.z + threadIdx.z;

                if (i < n0 && j < n1 && k < n2) {
                    size_t src_idx = k + j * n2 + i * n1 * n2;  // LayoutRight
                    size_t dst_idx = i + j * n0 + k * n0 * n1;  // LayoutLeft
                    dst[dst_idx]   = src[src_idx];
                }
            }
        }  // namespace detail

        namespace detail {
            /*!
             * @brief CUDA kernel that scales each complex element by @p scale in-place.
             *
             * @tparam T cuFFT complex type.
             */
            template <typename T>
            __global__ void cufftMpScaleKernel(T* data, size_t n, double scale) {
                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx].x *= scale;
                    data[idx].y *= scale;
                }
            }
        }  // namespace detail

        //=============================================================================
        // cuFFTMp C2C Backend
        //=============================================================================

        /*!
         * @class CuFFTMpC2C
         * @brief Distributed-memory complex-to-complex FFT via cuFFTMp.
         *
         * Configures a cuFFTMp 3D plan over the supplied MPI communicator,
         * allocates an internal CUDA stream, and exposes the IPPL-uniform
         * forward()/backward() interface. Only 3D float / double precision
         * is supported (compile-time enforced via static_assert).
         *
         * @tparam T        Real precision (float / double).
         * @tparam Dim      Spatial dimension (only 3D).
         * @tparam MemSpace Kokkos memory space holding the buffers.
         */
        template <typename T, unsigned Dim, typename MemSpace>
        class CuFFTMpC2C {
        public:
            using complex_t = Kokkos::complex<T>;
            using cuda_complex_t =
                std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;

            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "cuFFTMp only supports float and double precision");
            static_assert(Dim == 3, "cuFFTMp backend currently only supports 3D transforms");
            static_assert(is_available_v<CuFFTMp>, "cuFFTMp not available");

            /*!
             * @brief Create the cuFFTMp plan and CUDA stream.
             *
             * @param inbox  Local input box (inclusive corner indices).
             * @param outbox Local output box (inclusive corner indices).
             * @param comm   MPI communicator that participates in the transform.
             */
            CuFFTMpC2C(const heffte::box3d<long long>& inbox,
                       const heffte::box3d<long long>& outbox, MPI_Comm comm,
                       const ParameterList& /*params*/)
                : comm_(comm) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");
                checkCufftResult(cufftCreate(&handle_), "Failed to create cuFFT handle");
                checkCufftResult(cufftSetStream(handle_, stream_), "Failed to set stream");

                cufftType type = std::is_same_v<T, float> ? CUFFT_C2C : CUFFT_Z2Z;

                for (int d = 0; d < 3; ++d) {
                    lower_in_[d]  = inbox.low[d];
                    upper_in_[d]  = inbox.high[d] + 1;
                    lower_out_[d] = outbox.low[d];
                    upper_out_[d] = outbox.high[d] + 1;
                }

                for (int d = 0; d < 3; ++d) {
                    local_size_[d] = upper_in_[d] - lower_in_[d];
                }

                // Row-major strides (required by cuFFTMp - must be decreasing)
                std::array<long long, 3> strides;
                strides[0] = local_size_[1] * local_size_[2];
                strides[1] = local_size_[2];
                strides[2] = 1;

                std::array<long long, 3> local_max;
                for (int d = 0; d < 3; ++d) {
                    local_max[d] = std::max(upper_in_[d], upper_out_[d]);
                }
                MPI_Allreduce(local_max.data(), global_size_.data(), 3, MPI_LONG_LONG, MPI_MAX,
                              comm);

                int n[3] = {static_cast<int>(global_size_[0]), static_cast<int>(global_size_[1]),
                            static_cast<int>(global_size_[2])};

                total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
                local_elements_ = local_size_[0] * local_size_[1] * local_size_[2];

                checkCufftResult(cufftMpMakePlanDecomposition(
                                     handle_, 3, n, lower_in_.data(), upper_in_.data(),
                                     strides.data(), lower_out_.data(), upper_out_.data(),
                                     strides.data(), type, &comm_, CUFFT_COMM_MPI, &worksize_),
                                 "Failed to create cuFFTMp decomposition plan");

                checkCufftResult(cufftXtMalloc(handle_, &desc_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
                                 "Failed to allocate descriptor");
            }

            ~CuFFTMpC2C() {
                if (desc_)
                    cufftXtFree(desc_);
                if (handle_)
                    cufftDestroy(handle_);
                if (stream_)
                    cudaStreamDestroy(stream_);
            }

            CuFFTMpC2C(const CuFFTMpC2C&)            = delete;
            CuFFTMpC2C& operator=(const CuFFTMpC2C&) = delete;

            CuFFTMpC2C(CuFFTMpC2C&& other) noexcept
                : handle_(other.handle_)
                , comm_(other.comm_)
                , stream_(other.stream_)
                , desc_(other.desc_)
                , worksize_(other.worksize_)
                , total_elements_(other.total_elements_)
                , local_elements_(other.local_elements_)
                , global_size_(other.global_size_)
                , local_size_(other.local_size_)
                , lower_in_(other.lower_in_)
                , upper_in_(other.upper_in_)
                , lower_out_(other.lower_out_)
                , upper_out_(other.upper_out_) {
                other.handle_ = 0;
                other.stream_ = nullptr;
                other.desc_   = nullptr;
            }

            CuFFTMpC2C& operator=(CuFFTMpC2C&& other) noexcept {
                if (this != &other) {
                    if (desc_)
                        cufftXtFree(desc_);
                    if (handle_)
                        cufftDestroy(handle_);
                    if (stream_)
                        cudaStreamDestroy(stream_);

                    handle_         = other.handle_;
                    comm_           = other.comm_;
                    stream_         = other.stream_;
                    desc_           = other.desc_;
                    worksize_       = other.worksize_;
                    total_elements_ = other.total_elements_;
                    local_elements_ = other.local_elements_;
                    global_size_    = other.global_size_;
                    local_size_     = other.local_size_;
                    lower_in_       = other.lower_in_;
                    upper_in_       = other.upper_in_;
                    lower_out_      = other.lower_out_;
                    upper_out_      = other.upper_out_;

                    other.handle_ = 0;
                    other.stream_ = nullptr;
                    other.desc_   = nullptr;
                }
                return *this;
            }

            //! Forward C2C transform (LayoutLeft -> LayoutRight transpose,
            //! cuFFTMp forward, transpose back, normalize by 1/N).
            void forward(complex_t* in, complex_t* out) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                cuda_complex_t* desc_data =
                    static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);

                // Transpose input: LayoutLeft -> LayoutRight (into descriptor buffer)
                launchTransposeL2R(desc_data, reinterpret_cast<cuda_complex_t*>(in));

                // Execute forward FFT
                checkCufftResult(cufftXtExecDescriptor(handle_, desc_, desc_, CUFFT_FORWARD),
                                 "Forward FFT execution failed");

                // Transpose output: LayoutRight -> LayoutLeft
                launchTransposeR2L(reinterpret_cast<cuda_complex_t*>(out), desc_data);

                // Apply scaling (1/N)
                T scale = T(1) / static_cast<T>(total_elements_);
                applyScaling(reinterpret_cast<cuda_complex_t*>(out), local_elements_, scale);

                checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
            }

            //! Backward C2C transform (unscaled), with the same layout transpose dance.
            void backward(complex_t* in, complex_t* out) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                cuda_complex_t* desc_data =
                    static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);

                // Transpose input: LayoutLeft -> LayoutRight (into descriptor buffer)
                launchTransposeL2R(desc_data, reinterpret_cast<cuda_complex_t*>(in));

                // Execute backward FFT
                checkCufftResult(cufftXtExecDescriptor(handle_, desc_, desc_, CUFFT_INVERSE),
                                 "Backward FFT execution failed");

                // Transpose output: LayoutRight -> LayoutLeft
                launchTransposeR2L(reinterpret_cast<cuda_complex_t*>(out), desc_data);

                checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
            }

            //! @return Per-rank cuFFTMp workspace size in bytes.
            std::size_t workspace_size() const { return worksize_; }

        private:
            void launchTransposeL2R(cuda_complex_t* dst, const cuda_complex_t* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_size_[0] + block.x - 1) / block.x,
                          (local_size_[1] + block.y - 1) / block.y,
                          (local_size_[2] + block.z - 1) / block.z);
                detail::transposeL2R<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_size_[0]), static_cast<int>(local_size_[1]),
                    static_cast<int>(local_size_[2]));
            }

            void launchTransposeR2L(cuda_complex_t* dst, const cuda_complex_t* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_size_[0] + block.x - 1) / block.x,
                          (local_size_[1] + block.y - 1) / block.y,
                          (local_size_[2] + block.z - 1) / block.z);
                detail::transposeR2L<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_size_[0]), static_cast<int>(local_size_[1]),
                    static_cast<int>(local_size_[2]));
            }

            void applyScaling(cuda_complex_t* data, size_t count, T scale) {
                constexpr size_t blockSize = 256;
                size_t numBlocks           = (count + blockSize - 1) / blockSize;
                detail::cufftMpScaleKernel<<<numBlocks, blockSize, 0, stream_>>>(
                    data, count, static_cast<double>(scale));
            }

            cufftHandle handle_ = 0;
            MPI_Comm comm_;
            cudaStream_t stream_ = nullptr;
            cudaLibXtDesc* desc_ = nullptr;

            size_t worksize_       = 0;
            size_t total_elements_ = 0;
            size_t local_elements_ = 0;
            std::array<long long, 3> global_size_;
            std::array<long long, 3> local_size_;
            std::array<long long, 3> lower_in_, upper_in_;
            std::array<long long, 3> lower_out_, upper_out_;
        };

        //=============================================================================
        // cuFFTMp R2C Backend
        //=============================================================================

        /*!
         * @class CuFFTMpR2C
         * @brief Distributed real-to-complex FFT via cuFFTMp.
         *
         * Holds two cuFFTMp plans (R2C and C2R) plus a CUDA stream. Mirrors
         * the IPPL R2C backend interface. Forward = real -> half-complex
         * (normalized by 1/N); backward = half-complex -> real (unscaled).
         *
         * @tparam T        Real precision (float / double).
         * @tparam Dim      Spatial dimension (only 3D).
         * @tparam MemSpace Kokkos memory space.
         */
        template <typename T, unsigned Dim, typename MemSpace>
        class CuFFTMpR2C {
        public:
            using complex_t = Kokkos::complex<T>;
            using cuda_complex_t =
                std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;

            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "cuFFTMp only supports float and double precision");
            static_assert(Dim == 3, "cuFFTMp backend currently only supports 3D transforms");
            static_assert(is_available_v<CuFFTMp>, "cuFFTMp not available");

            /*!
             * @brief Build the R2C and C2R cuFFTMp plans.
             * @param inbox  Local real-input box.
             * @param outbox Local complex-output box (Hermitian-symmetric).
             * @param comm   MPI communicator participating in the transform.
             */
            CuFFTMpR2C(const heffte::box3d<long long>& inbox,
                       const heffte::box3d<long long>& outbox, int /*r2c_direction*/,
                       MPI_Comm comm, const ParameterList& /*params*/)
                : comm_(comm) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                checkCudaError(cudaStreamCreate(&stream_), "Failed to create CUDA stream");
                checkCufftResult(cufftCreate(&handle_r2c_), "Failed to create R2C handle");
                checkCufftResult(cufftCreate(&handle_c2r_), "Failed to create C2R handle");
                checkCufftResult(cufftSetStream(handle_r2c_, stream_), "Failed to set stream");
                checkCufftResult(cufftSetStream(handle_c2r_, stream_), "Failed to set stream");

                std::array<long long, 3> lower_real, upper_real;
                std::array<long long, 3> lower_complex, upper_complex;

                for (int d = 0; d < 3; ++d) {
                    lower_real[d]    = inbox.low[d];
                    upper_real[d]    = inbox.high[d] + 1;
                    lower_complex[d] = outbox.low[d];
                    upper_complex[d] = outbox.high[d] + 1;
                }

                for (int d = 0; d < 3; ++d) {
                    local_real_size_[d]    = upper_real[d] - lower_real[d];
                    local_complex_size_[d] = upper_complex[d] - lower_complex[d];
                }

                // Row-major strides for real data
                std::array<long long, 3> strides_real;
                strides_real[0] = local_real_size_[1] * local_real_size_[2];
                strides_real[1] = local_real_size_[2];
                strides_real[2] = 1;

                // Row-major strides for complex data
                std::array<long long, 3> strides_complex;
                strides_complex[0] = local_complex_size_[1] * local_complex_size_[2];
                strides_complex[1] = local_complex_size_[2];
                strides_complex[2] = 1;

                std::array<long long, 3> local_max;
                for (int d = 0; d < 3; ++d) {
                    local_max[d] = std::max(upper_real[d], upper_complex[d]);
                }
                MPI_Allreduce(local_max.data(), global_size_.data(), 3, MPI_LONG_LONG, MPI_MAX,
                              comm);

                int n[3] = {static_cast<int>(global_size_[0]), static_cast<int>(global_size_[1]),
                            static_cast<int>(global_size_[2])};

                total_elements_ = static_cast<size_t>(n[0]) * n[1] * n[2];
                local_real_elements_ =
                    local_real_size_[0] * local_real_size_[1] * local_real_size_[2];
                local_complex_elements_ =
                    local_complex_size_[0] * local_complex_size_[1] * local_complex_size_[2];

                size_t worksize_r2c = 0;
                size_t worksize_c2r = 0;
                checkCufftResult(
                    cufftMpMakePlanDecomposition(
                        handle_r2c_, 3, n, lower_real.data(), upper_real.data(),
                        strides_real.data(), lower_complex.data(), upper_complex.data(),
                        strides_complex.data(), CUFFT_R2C, &comm_, CUFFT_COMM_MPI, &worksize_r2c),
                    "Failed to create R2C plan");

                checkCufftResult(
                    cufftMpMakePlanDecomposition(
                        handle_c2r_, 3, n, lower_real.data(), upper_real.data(),
                        strides_real.data(), lower_complex.data(), upper_complex.data(),
                        strides_complex.data(), CUFFT_C2R, &comm_, CUFFT_COMM_MPI, &worksize_c2r),
                    "Failed to create C2R plan");

                worksize_ = std::max(worksize_r2c, worksize_c2r);

                checkCufftResult(
                    cufftXtMalloc(handle_r2c_, &desc_, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT),
                    "Failed to allocate R2C descriptor");
            }

            ~CuFFTMpR2C() {
                if (desc_)
                    cufftXtFree(desc_);
                if (handle_r2c_)
                    cufftDestroy(handle_r2c_);
                if (handle_c2r_)
                    cufftDestroy(handle_c2r_);
                if (stream_)
                    cudaStreamDestroy(stream_);
            }

            CuFFTMpR2C(const CuFFTMpR2C&)            = delete;
            CuFFTMpR2C& operator=(const CuFFTMpR2C&) = delete;

            CuFFTMpR2C(CuFFTMpR2C&& other) noexcept
                : handle_r2c_(other.handle_r2c_)
                , handle_c2r_(other.handle_c2r_)
                , comm_(other.comm_)
                , stream_(other.stream_)
                , desc_(other.desc_)
                , worksize_(other.worksize_)
                , total_elements_(other.total_elements_)
                , local_real_elements_(other.local_real_elements_)
                , local_complex_elements_(other.local_complex_elements_)
                , global_size_(other.global_size_)
                , local_real_size_(other.local_real_size_)
                , local_complex_size_(other.local_complex_size_) {
                other.handle_r2c_ = 0;
                other.handle_c2r_ = 0;
                other.stream_     = nullptr;
                other.desc_       = nullptr;
            }

            CuFFTMpR2C& operator=(CuFFTMpR2C&& other) noexcept {
                if (this != &other) {
                    if (desc_)
                        cufftXtFree(desc_);
                    if (handle_r2c_)
                        cufftDestroy(handle_r2c_);
                    if (handle_c2r_)
                        cufftDestroy(handle_c2r_);
                    if (stream_)
                        cudaStreamDestroy(stream_);

                    handle_r2c_             = other.handle_r2c_;
                    handle_c2r_             = other.handle_c2r_;
                    comm_                   = other.comm_;
                    stream_                 = other.stream_;
                    desc_                   = other.desc_;
                    worksize_               = other.worksize_;
                    total_elements_         = other.total_elements_;
                    local_real_elements_    = other.local_real_elements_;
                    local_complex_elements_ = other.local_complex_elements_;
                    global_size_            = other.global_size_;
                    local_real_size_        = other.local_real_size_;
                    local_complex_size_     = other.local_complex_size_;

                    other.handle_r2c_ = 0;
                    other.handle_c2r_ = 0;
                    other.stream_     = nullptr;
                    other.desc_       = nullptr;
                }
                return *this;
            }

            //! Forward R2C transform (real -> half-complex), normalized by 1/N.
            void forward(T* in, complex_t* out) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                T* desc_data = static_cast<T*>(desc_->descriptor->data[0]);

                // Transpose real input: LayoutLeft -> LayoutRight
                launchTransposeRealL2R(desc_data, in);

                checkCufftResult(cufftXtExecDescriptor(handle_r2c_, desc_, desc_, CUFFT_FORWARD),
                                 "R2C execution failed");

                // Transpose complex output: LayoutRight -> LayoutLeft
                cuda_complex_t* complex_desc =
                    static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);
                launchTransposeComplexR2L(reinterpret_cast<cuda_complex_t*>(out), complex_desc);

                T scale = T(1) / static_cast<T>(total_elements_);
                applyScaling(reinterpret_cast<cuda_complex_t*>(out), local_complex_elements_,
                             scale);

                checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
            }

            //! Backward C2R transform (half-complex -> real), unscaled.
            void backward(complex_t* in, T* out) {
                using detail::checkCudaError;
                using detail::checkCufftResult;

                cuda_complex_t* complex_desc =
                    static_cast<cuda_complex_t*>(desc_->descriptor->data[0]);

                // Transpose complex input: LayoutLeft -> LayoutRight
                launchTransposeComplexL2R(complex_desc, reinterpret_cast<cuda_complex_t*>(in));

                checkCufftResult(cufftXtExecDescriptor(handle_c2r_, desc_, desc_, CUFFT_INVERSE),
                                 "C2R execution failed");

                // Transpose real output: LayoutRight -> LayoutLeft
                T* real_desc = static_cast<T*>(desc_->descriptor->data[0]);
                launchTransposeRealR2L(out, real_desc);

                checkCudaError(cudaStreamSynchronize(stream_), "Stream sync failed");
            }

            //! @return Per-rank cuFFTMp workspace size in bytes (max of R2C/C2R).
            std::size_t workspace_size() const { return worksize_; }

        private:
            void launchTransposeRealL2R(T* dst, const T* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_real_size_[0] + block.x - 1) / block.x,
                          (local_real_size_[1] + block.y - 1) / block.y,
                          (local_real_size_[2] + block.z - 1) / block.z);
                detail::transposeL2R<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_real_size_[0]),
                    static_cast<int>(local_real_size_[1]), static_cast<int>(local_real_size_[2]));
            }

            void launchTransposeRealR2L(T* dst, const T* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_real_size_[0] + block.x - 1) / block.x,
                          (local_real_size_[1] + block.y - 1) / block.y,
                          (local_real_size_[2] + block.z - 1) / block.z);
                detail::transposeR2L<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_real_size_[0]),
                    static_cast<int>(local_real_size_[1]), static_cast<int>(local_real_size_[2]));
            }

            void launchTransposeComplexL2R(cuda_complex_t* dst, const cuda_complex_t* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_complex_size_[0] + block.x - 1) / block.x,
                          (local_complex_size_[1] + block.y - 1) / block.y,
                          (local_complex_size_[2] + block.z - 1) / block.z);
                detail::transposeL2R<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_complex_size_[0]),
                    static_cast<int>(local_complex_size_[1]),
                    static_cast<int>(local_complex_size_[2]));
            }

            void launchTransposeComplexR2L(cuda_complex_t* dst, const cuda_complex_t* src) {
                dim3 block(8, 8, 8);
                dim3 grid((local_complex_size_[0] + block.x - 1) / block.x,
                          (local_complex_size_[1] + block.y - 1) / block.y,
                          (local_complex_size_[2] + block.z - 1) / block.z);
                detail::transposeR2L<<<grid, block, 0, stream_>>>(
                    dst, src, static_cast<int>(local_complex_size_[0]),
                    static_cast<int>(local_complex_size_[1]),
                    static_cast<int>(local_complex_size_[2]));
            }

            void applyScaling(cuda_complex_t* data, size_t count, T scale) {
                constexpr size_t blockSize = 256;
                size_t numBlocks           = (count + blockSize - 1) / blockSize;
                detail::cufftMpScaleKernel<<<numBlocks, blockSize, 0, stream_>>>(
                    data, count, static_cast<double>(scale));
            }

            cufftHandle handle_r2c_ = 0;
            cufftHandle handle_c2r_ = 0;
            MPI_Comm comm_;
            cudaStream_t stream_ = nullptr;
            cudaLibXtDesc* desc_ = nullptr;

            size_t worksize_               = 0;
            size_t total_elements_         = 0;
            size_t local_real_elements_    = 0;
            size_t local_complex_elements_ = 0;
            std::array<long long, 3> global_size_;
            std::array<long long, 3> local_real_size_;
            std::array<long long, 3> local_complex_size_;
        };

    }  // namespace fft
}  // namespace ippl

#endif
