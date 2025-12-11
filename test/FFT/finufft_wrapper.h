#ifndef FINUFFT_WRAPPER_H
#define FINUFFT_WRAPPER_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef ENABLE_GPU_NUFFT
#include <cufinufft.h>
#else
#include <finufft.h>
#endif

namespace finufft_wrapper {

    /**
     * @brief Configuration for FINUFFT transforms
     */
    template <typename T>
    struct Config {
        T tolerance = T(1e-6);
        int type    = 1;  // 1 or 2

        // GPU-specific options
        int gpu_method         = 1;   // 1=nonuniform pts driven, 2=subproblem
        int gpu_sort           = 1;   // 0=no sort, 1=sort
        int gpu_kerevalmeth    = 1;   // 0=direct, 1=Horner
        int gpu_binsizex       = -1;  // auto
        int gpu_binsizey       = -1;
        int gpu_binsizez       = -1;
        int gpu_maxsubprobsize = 1024;

        // CPU-specific options
        int nthreads           = 0;  // 0=use all
        int spread_sort        = 2;  // 0=no, 1=yes, 2=heuristic
        int spread_kerevalmeth = 1;
    };

    /**
     * @brief Direct FINUFFT/cuFINUFFT wrapper for 3D transforms
     *
     * This bypasses any application-level wrappers and calls FINUFFT directly.
     */
    template <typename T>
    class DirectNUFFT3D {
    public:
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float and double are supported");

#ifdef ENABLE_GPU_NUFFT
        using plan_type =
            std::conditional_t<std::is_same_v<T, float>, cufinufftf_plan, cufinufft_plan>;
        using complex_type =
            std::conditional_t<std::is_same_v<T, float>, cuFloatComplex, cuDoubleComplex>;
#else
        using plan_type = std::conditional_t<std::is_same_v<T, float>, finufftf_plan, finufft_plan>;
        using complex_type =
            std::conditional_t<std::is_same_v<T, float>, fftwf_complex, fftw_complex>;
#endif

    private:
        plan_type plan_     = nullptr;
        int type_           = 0;
        int64_t n_modes_[3] = {0, 0, 0};
        bool plan_created_  = false;

    public:
        DirectNUFFT3D() = default;

        ~DirectNUFFT3D() { destroy(); }

        // Non-copyable
        DirectNUFFT3D(const DirectNUFFT3D&)            = delete;
        DirectNUFFT3D& operator=(const DirectNUFFT3D&) = delete;

        // Movable
        DirectNUFFT3D(DirectNUFFT3D&& other) noexcept {
            plan_ = other.plan_;
            type_ = other.type_;
            for (int d = 0; d < 3; ++d)
                n_modes_[d] = other.n_modes_[d];
            plan_created_       = other.plan_created_;
            other.plan_         = nullptr;
            other.plan_created_ = false;
        }

        DirectNUFFT3D& operator=(DirectNUFFT3D&& other) noexcept {
            if (this != &other) {
                destroy();
                plan_ = other.plan_;
                type_ = other.type_;
                for (int d = 0; d < 3; ++d)
                    n_modes_[d] = other.n_modes_[d];
                plan_created_       = other.plan_created_;
                other.plan_         = nullptr;
                other.plan_created_ = false;
            }
            return *this;
        }

        /**
         * @brief Create a FINUFFT plan
         *
         * @param n_modes Grid dimensions [nx, ny, nz]
         * @param config Configuration options
         */
        void make_plan(const int64_t n_modes[3], const Config<T>& config) {
            destroy();  // Clean up any existing plan

            type_ = config.type;
            for (int d = 0; d < 3; ++d) {
                n_modes_[d] = n_modes[d];
            }

            // iflag: +1 for type1 (NUFFT), -1 for type2 (adjoint)
            int iflag   = (type_ == 1) ? 1 : -1;
            int ntransf = 1;
            int ier     = 0;

#ifdef ENABLE_GPU_NUFFT
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);

            opts.gpu_method      = config.gpu_method;
            opts.gpu_sort        = config.gpu_sort;
            opts.gpu_kerevalmeth = config.gpu_kerevalmeth;
            if (config.gpu_binsizex > 0)
                opts.gpu_binsizex = config.gpu_binsizex;
            if (config.gpu_binsizey > 0)
                opts.gpu_binsizey = config.gpu_binsizey;
            if (config.gpu_binsizez > 0)
                opts.gpu_binsizez = config.gpu_binsizez;
            opts.gpu_maxsubprobsize = config.gpu_maxsubprobsize;
            opts.gpu_maxbatchsize   = 0;

            if constexpr (std::is_same_v<T, float>) {
                ier = cufinufftf_makeplan(type_, 3, const_cast<int64_t*>(n_modes_), iflag, ntransf,
                                          config.tolerance, &plan_, &opts);
            } else {
                ier = cufinufft_makeplan(type_, 3, const_cast<int64_t*>(n_modes_), iflag, ntransf,
                                         config.tolerance, &plan_, &opts);
            }
#else
            finufft_opts opts;
            finufft_default_opts(&opts);

            opts.nthreads           = config.nthreads;
            opts.spread_sort        = config.spread_sort;
            opts.spread_kerevalmeth = config.spread_kerevalmeth;

            if constexpr (std::is_same_v<T, float>) {
                ier = finufftf_makeplan(type_, 3, const_cast<int64_t*>(n_modes_), iflag, ntransf,
                                        config.tolerance, &plan_, &opts);
            } else {
                ier = finufft_makeplan(type_, 3, const_cast<int64_t*>(n_modes_), iflag, ntransf,
                                       config.tolerance, &plan_, &opts);
            }
#endif

            if (ier != 0) {
                throw std::runtime_error("FINUFFT makeplan failed with error code: "
                                         + std::to_string(ier));
            }

            plan_created_ = true;
        }

        /**
         * @brief Set nonuniform points
         *
         * @param n_pts Number of nonuniform points
         * @param x X coordinates (in [-pi, pi])
         * @param y Y coordinates (in [-pi, pi])
         * @param z Z coordinates (in [-pi, pi])
         */
        void set_points(int64_t n_pts, T* x, T* y, T* z) {
            if (!plan_created_) {
                throw std::runtime_error("Plan not created. Call make_plan first.");
            }

            int ier = 0;

#ifdef ENABLE_GPU_NUFFT
            if constexpr (std::is_same_v<T, float>) {
                ier = cufinufftf_setpts(plan_, n_pts, x, y, z, 0, nullptr, nullptr, nullptr);
            } else {
                ier = cufinufft_setpts(plan_, n_pts, x, y, z, 0, nullptr, nullptr, nullptr);
            }
#else
            if constexpr (std::is_same_v<T, float>) {
                ier = finufftf_setpts(plan_, n_pts, x, y, z, 0, nullptr, nullptr, nullptr);
            } else {
                ier = finufft_setpts(plan_, n_pts, x, y, z, 0, nullptr, nullptr, nullptr);
            }
#endif

            if (ier != 0) {
                throw std::runtime_error("FINUFFT setpts failed with error code: "
                                         + std::to_string(ier));
            }
        }

        /**
         * @brief Execute the transform
         *
         * For type 1: c (input strengths) -> f (output grid)
         * For type 2: f (input grid) -> c (output values at points)
         *
         * @param c Complex values at nonuniform points
         * @param f Complex values on uniform grid (size nx*ny*nz)
         */
        void execute(complex_type* c, complex_type* f) {
            if (!plan_created_) {
                throw std::runtime_error("Plan not created. Call make_plan first.");
            }

            int ier = 0;

#ifdef ENABLE_GPU_NUFFT
            if constexpr (std::is_same_v<T, float>) {
                ier = cufinufftf_execute(plan_, c, f);
            } else {
                ier = cufinufft_execute(plan_, c, f);
            }
#else
            if constexpr (std::is_same_v<T, float>) {
                ier = finufftf_execute(plan_, c, f);
            } else {
                ier = finufft_execute(plan_, c, f);
            }
#endif

            if (ier != 0) {
                throw std::runtime_error("FINUFFT execute failed with error code: "
                                         + std::to_string(ier));
            }
        }

        /**
         * @brief Destroy the plan and free resources
         */
        void destroy() {
            if (plan_created_ && plan_ != nullptr) {
#ifdef ENABLE_GPU_NUFFT
                if constexpr (std::is_same_v<T, float>) {
                    cufinufftf_destroy(plan_);
                } else {
                    cufinufft_destroy(plan_);
                }
#else
                if constexpr (std::is_same_v<T, float>) {
                    finufftf_destroy(plan_);
                } else {
                    finufft_destroy(plan_);
                }
#endif
                plan_         = nullptr;
                plan_created_ = false;
            }
        }

        bool is_valid() const { return plan_created_; }
        int type() const { return type_; }
        const int64_t* n_modes() const { return n_modes_; }
    };

    /**
     * @brief Kokkos-compatible wrapper that handles memory transfers
     *
     * Note: Each instance is tied to a specific transform type (1 or 2).
     * Create separate instances for different transform types.
     */
    template <typename T, typename ExecSpace = Kokkos::DefaultExecutionSpace>
    class KokkosNUFFT3D {
    public:
        using real_type    = T;
        using complex_type = Kokkos::complex<T>;
        using memory_space = typename ExecSpace::memory_space;

#ifdef ENABLE_GPU_NUFFT
        using finufft_complex =
            std::conditional_t<std::is_same_v<T, float>, cuFloatComplex, cuDoubleComplex>;
#else
        using finufft_complex =
            std::conditional_t<std::is_same_v<T, float>, fftwf_complex, fftw_complex>;
#endif

    private:
        DirectNUFFT3D<T> nufft_;
        Config<T> config_;

        // Coordinate buffers (LayoutLeft for FINUFFT compatibility)
        Kokkos::View<T*, Kokkos::LayoutLeft, memory_space> x_, y_, z_;

        // Complex buffers
        Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space> c_;
        Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space> f_;

        int64_t n_pts_      = 0;
        int64_t n_modes_[3] = {0, 0, 0};
        bool initialized_   = false;

    public:
        KokkosNUFFT3D() = default;

        /**
         * @brief Initialize the NUFFT
         *
         * @param n_modes Grid dimensions [nx, ny, nz]
         * @param max_pts Maximum number of nonuniform points
         * @param config Configuration options (including type = 1 or 2)
         */
        void initialize(const int64_t n_modes[3], int64_t max_pts, const Config<T>& config) {
            config_ = config;
            for (int d = 0; d < 3; ++d) {
                n_modes_[d] = n_modes[d];
            }

            // Allocate coordinate buffers
            x_ = Kokkos::View<T*, Kokkos::LayoutLeft, memory_space>("x", max_pts);
            y_ = Kokkos::View<T*, Kokkos::LayoutLeft, memory_space>("y", max_pts);
            z_ = Kokkos::View<T*, Kokkos::LayoutLeft, memory_space>("z", max_pts);

            // Allocate complex buffers
            c_ = Kokkos::View<complex_type*, Kokkos::LayoutLeft, memory_space>("c", max_pts);
            f_ = Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space>(
                "f", n_modes[0], n_modes[1], n_modes[2]);

            // Create FINUFFT plan
            nufft_.make_plan(n_modes, config);
            initialized_ = true;

            std::cout << "KokkosNUFFT3D initialized: type=" << config.type << ", modes=["
                      << n_modes[0] << "," << n_modes[1] << "," << n_modes[2] << "]"
                      << ", max_pts=" << max_pts << ", tol=" << config.tolerance << std::endl;
        }

        int type() const { return config_.type; }
        bool is_initialized() const { return initialized_; }

        /**
         * @brief Execute Type 1 transform (nonuniform -> uniform)
         *
         * @param positions Particle positions View<Vector<T,3>*> in [-pi, pi]
         * @param strengths Particle strengths View<T*>
         * @param output Output field View<complex<T>***>
         * @param n_pts Number of points to transform
         */
        template <typename PosView, typename StrengthView, typename OutputView>
        void type1(const PosView& positions, const StrengthView& strengths, OutputView& output,
                   int64_t n_pts) {
            if (!initialized_) {
                throw std::runtime_error("KokkosNUFFT3D not initialized");
            }
            if (config_.type != 1) {
                throw std::runtime_error("This instance was initialized for type "
                                         + std::to_string(config_.type) + ", not type 1");
            }

            n_pts_ = n_pts;

            // Copy positions to separate x, y, z arrays
            auto x = x_;
            auto y = y_;
            auto z = z_;
            auto c = c_;

            Kokkos::parallel_for(
                "copy_positions_strengths", Kokkos::RangePolicy<ExecSpace>(0, n_pts),
                KOKKOS_LAMBDA(const int64_t i) {
                    x(i) = positions(i)[0];
                    y(i) = positions(i)[1];
                    z(i) = positions(i)[2];
                    c(i) = complex_type(strengths(i), T(0));
                });
            Kokkos::fence();

            // Set points and execute
            nufft_.set_points(n_pts, x_.data(), y_.data(), z_.data());
            nufft_.execute(reinterpret_cast<finufft_complex*>(c_.data()),
                           reinterpret_cast<finufft_complex*>(f_.data()));
            Kokkos::fence();

            // Copy result to output (handle ghost cells if present)
            auto f           = f_;
            const int64_t nx = n_modes_[0];
            const int64_t ny = n_modes_[1];
            const int64_t nz = n_modes_[2];

            // Detect ghost cells from output extent
            const int nghost = (output.extent(0) > size_t(nx)) ? (output.extent(0) - nx) / 2 : 0;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>;
            Kokkos::parallel_for(
                "copy_to_output", mdrange_type({0, 0, 0}, {nx, ny, nz}),
                KOKKOS_LAMBDA(const int64_t i, const int64_t j, const int64_t k) {
                    output(i + nghost, j + nghost, k + nghost) = f(i, j, k);
                });
            Kokkos::fence();
        }

        /**
         * @brief Execute Type 2 transform (uniform -> nonuniform)
         *
         * @param input Input field View<complex<T>***>
         * @param positions Particle positions View<Vector<T,3>*> in [-pi, pi]
         * @param output Output values View<T*> (real part of result)
         * @param n_pts Number of points to transform
         */
        template <typename InputView, typename PosView, typename OutputView>
        void type2(const InputView& input, const PosView& positions, OutputView& output,
                   int64_t n_pts) {
            if (!initialized_) {
                throw std::runtime_error("KokkosNUFFT3D not initialized");
            }
            if (config_.type != 2) {
                throw std::runtime_error("This instance was initialized for type "
                                         + std::to_string(config_.type) + ", not type 2");
            }

            n_pts_ = n_pts;

            // Copy positions to separate x, y, z arrays
            auto x = x_;
            auto y = y_;
            auto z = z_;

            Kokkos::parallel_for(
                "copy_positions", Kokkos::RangePolicy<ExecSpace>(0, n_pts),
                KOKKOS_LAMBDA(const int64_t i) {
                    x(i) = positions(i)[0];
                    y(i) = positions(i)[1];
                    z(i) = positions(i)[2];
                });

            // Copy input field to internal buffer (handle ghost cells)
            auto f           = f_;
            const int64_t nx = n_modes_[0];
            const int64_t ny = n_modes_[1];
            const int64_t nz = n_modes_[2];

            const int nghost = (input.extent(0) > size_t(nx)) ? (input.extent(0) - nx) / 2 : 0;

            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>;
            Kokkos::parallel_for(
                "copy_from_input", mdrange_type({0, 0, 0}, {nx, ny, nz}),
                KOKKOS_LAMBDA(const int64_t i, const int64_t j, const int64_t k) {
                    f(i, j, k) = input(i + nghost, j + nghost, k + nghost);
                });
            Kokkos::fence();

            // Set points and execute
            nufft_.set_points(n_pts, x_.data(), y_.data(), z_.data());
            nufft_.execute(reinterpret_cast<finufft_complex*>(c_.data()),
                           reinterpret_cast<finufft_complex*>(f_.data()));
            Kokkos::fence();

            // Copy real part of result to output
            auto c = c_;
            Kokkos::parallel_for(
                "copy_result", Kokkos::RangePolicy<ExecSpace>(0, n_pts),
                KOKKOS_LAMBDA(const int64_t i) { output(i) = c(i).real(); });
            Kokkos::fence();
        }

        // Accessors for internal buffers (useful for debugging)
        auto& x() { return x_; }
        auto& y() { return y_; }
        auto& z() { return z_; }
        auto& c() { return c_; }
        auto& f() { return f_; }
    };

}  // namespace finufft_wrapper

#endif  // FINUFFT_WRAPPER_H