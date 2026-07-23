/*!
 * @file NUFFT.hpp
 * @brief Implementation of FFT<NUFFTransform, RealField> declared in NUFFT.h.
 *
 * Contains the Kokkos functors used to copy particles / fields between IPPL
 * and FINUFFT-friendly buffers (with optional fftshift) plus the dispatch
 * to the native and FINUFFT backends.
 */
#ifndef IPPL_FFT_TRANSFORM_NUFFT_HPP
#define IPPL_FFT_TRANSFORM_NUFFT_HPP

namespace ippl {

    namespace detail {

        //=====================================================================
        // Functors for NUFFT operations (must be outside class for NVCC)
        //=====================================================================

        /**
         * @brief Functor to scale particle positions by a factor
         */
        template <typename T, unsigned Dim, typename RView>
        struct ScalePositionsFunctor {
            RView Rview_m;
            Vector<T, Dim> scale_m;

            ScalePositionsFunctor(RView Rview, Vector<T, Dim> scale)
                : Rview_m(Rview)
                , scale_m(scale) {}

            KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
                for (unsigned d = 0; d < Dim; ++d) {
                    Rview_m(i)[d] *= scale_m[d];
                }
            }
        };

#ifdef ENABLE_FINUFFT

        /**
         * @brief Functor to copy field data to FINUFFT temp buffer with optional ifftshift
         */
        template <typename FieldView, typename TempFieldView>
        struct CopyFieldToTempFunctor {
            FieldView fview_m;
            TempFieldView tempField_m;
            int nghost_m;
            int nx_m, ny_m, nz_m;
            bool applyShift_m;

            CopyFieldToTempFunctor(FieldView fview, TempFieldView tempField, int nghost, int nx = 0,
                                   int ny = 0, int nz = 0, bool applyShift = false)
                : fview_m(fview)
                , tempField_m(tempField)
                , nghost_m(nghost)
                , nx_m(nx)
                , ny_m(ny)
                , nz_m(nz)
                , applyShift_m(applyShift) {}

            KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
                int li = i - nghost_m;
                int lj = j - nghost_m;
                int lk = k - nghost_m;

                int di = li, dj = lj, dk = lk;
                if (applyShift_m) {
                    di = (li + nx_m / 2) % nx_m;
                    dj = (lj + ny_m / 2) % ny_m;
                    dk = (lk + nz_m / 2) % nz_m;
                }

                auto& dst = tempField_m(di, dj, dk);
                auto src  = fview_m(i, j, k);
#ifdef ENABLE_GPU_NUFFT
                dst.x = src.real();
                dst.y = src.imag();
#else
                dst.real(src.real());
                dst.imag(src.imag());
#endif
            }
        };

        /**
         * @brief Functor to copy field data from FINUFFT temp buffer with fftshift
         */
        template <typename FieldView, typename TempFieldView>
        struct CopyFieldFromTempFunctor {
            FieldView fview_m;
            TempFieldView tempField_m;
            int nghost_m;
            int nx_m, ny_m, nz_m;

            CopyFieldFromTempFunctor(FieldView fview, TempFieldView tempField, int nghost, int nx,
                                     int ny, int nz)
                : fview_m(fview)
                , tempField_m(tempField)
                , nghost_m(nghost)
                , nx_m(nx)
                , ny_m(ny)
                , nz_m(nz) {}

            KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int k) const {
                int li = i - nghost_m;
                int lj = j - nghost_m;
                int lk = k - nghost_m;

                int si = (li + nx_m / 2) % nx_m;
                int sj = (lj + ny_m / 2) % ny_m;
                int sk = (lk + nz_m / 2) % nz_m;

                auto src = tempField_m(si, sj, sk);
#ifdef ENABLE_GPU_NUFFT
                fview_m(i, j, k).real() = src.x;
                fview_m(i, j, k).imag() = src.y;
#else
                fview_m(i, j, k).real() = src.real();
                fview_m(i, j, k).imag() = src.imag();
#endif
            }
        };

        /**
         * @brief Functor to copy particle data to FINUFFT temp buffers
         */
        template <typename T, unsigned Dim, typename RView, typename QView, typename TempRView,
                  typename TempQView>
        struct CopyParticlesToTempFunctor {
            RView Rview_m;
            QView Qview_m;
            TempRView tempRx_m, tempRy_m, tempRz_m;
            TempQView tempQ_m;
            Vector<T, Dim> scale_m;

            CopyParticlesToTempFunctor(RView Rview, QView Qview, TempRView tempRx, TempRView tempRy,
                                       TempRView tempRz, TempQView tempQ, Vector<T, Dim> scale)
                : Rview_m(Rview)
                , Qview_m(Qview)
                , tempRx_m(tempRx)
                , tempRy_m(tempRy)
                , tempRz_m(tempRz)
                , tempQ_m(tempQ)
                , scale_m(scale) {}

            KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
                tempRx_m(i) = Rview_m(i)[0] * scale_m[0];
                tempRy_m(i) = Rview_m(i)[1] * scale_m[1];
                tempRz_m(i) = Rview_m(i)[2] * scale_m[2];

#ifdef ENABLE_GPU_NUFFT
                tempQ_m(i).x = Qview_m(i);
                tempQ_m(i).y = T(0);
#else
                tempQ_m(i).real(Qview_m(i));
                tempQ_m(i).imag(T(0));
#endif
            }
        };

        /**
         * @brief Functor to copy particle data from FINUFFT temp buffer
         */
        template <typename QView, typename TempQView>
        struct CopyParticlesFromTempFunctor {
            QView Qview_m;
            TempQView tempQ_m;

            CopyParticlesFromTempFunctor(QView Qview, TempQView tempQ)
                : Qview_m(Qview)
                , tempQ_m(tempQ) {}

            KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
#ifdef ENABLE_GPU_NUFFT
                Qview_m(i) = tempQ_m(i).x;
#else
                Qview_m(i) = tempQ_m(i).real();
#endif
            }
        };

#endif  // ENABLE_FINUFFT

    }  // namespace detail

    //=========================================================================
    // Constructor / Destructor
    //=========================================================================

    template <typename RealField>
    FFT<NUFFTransform, RealField>::FFT(const Layout_t& layout, detail::size_type localNp, int type,
                                       const ParameterList& params)
        : type_m(type)
        , tol_m(params.get<T>("tolerance", T(1e-6)))
        , useFinufft_m(params.get<bool>("use_finufft", false))
        , useUpsampledInputs_m(params.get<bool>("use_upsampled_inputs", false))
        , useR2C_m(params.get<bool>("use_r2c", false))
        , r2cDir_m(params.get<int>("r2c_direction", 0))
        , lockMethod_m(params.get<bool>("lock_method", false)) {
        const auto& domain = layout.getDomain();
        for (unsigned d = 0; d < Dim; ++d) {
            nModes_m[d] = domain[d].length();
        }

#ifdef ENABLE_FINUFFT
        // allocateFinufftBuffers and the rest of the FINUFFT/cuFINUFFT
        // pipeline are hardcoded for 3D (lDom[2], Rank<3> MDRangePolicy, ...).
        // For Dim != 3 we never call into them -- the native path handles
        // 2D NUFFT -- so we also skip the buffer allocation here, which
        // would otherwise read past the local NDIndex for Dim < 3.
        if constexpr (Dim == 3) {
            allocateFinufftBuffers(layout, localNp);
        } else {
            (void)layout;
            (void)localNp;
        }
#else
        (void)localNp;
#endif

        initBackend(layout, params);
    }

    template <typename RealField>
    FFT<NUFFTransform, RealField>::~FFT() {
        cleanupBackend();
    }

    //=========================================================================
    // Backend Initialization
    //=========================================================================

    template <typename RealField>
    void FFT<NUFFTransform, RealField>::initBackend(const Layout_t& layout,
                                                    const ParameterList& params) {
        // FINUFFT path is 3D-only (see FFT::transform()); fall back to the
        // native backend for any other Dim so the FINUFFT-enabled 2D unit
        // tests don't tear down on the plan setup.
        if constexpr (Dim == 3 && fft::is_available_v<fft::Finufft>) {
            if (useFinufft_m) {
                initFinufft(params);
                return;
            }
        }

        initNative(layout, params);
    }

    template <typename RealField>
    void FFT<NUFFTransform, RealField>::initNative(const Layout_t& layout,
                                                   const ParameterList& params) {
        Vector<std::size_t, Dim> nModesVec;
        for (unsigned d = 0; d < Dim; ++d) {
            nModesVec[d] = nModes_m[d];
        }

        typename NativeNUFFT_t::Config cfg;
        cfg.tol   = tol_m;
        cfg.sigma = params.get<T>("sigma", T(2.0));

        cfg.scatter_config = Interpolation::ScatterConfig<Dim>::template get_default<ExecSpace>();
        cfg.gather_config  = Interpolation::GatherConfig<Dim>::template get_default<ExecSpace>();

        cfg.scatter_config.lock_method = lockMethod_m;

        std::string spreadMethod = params.get<std::string>("spread_method", "none");
        if (spreadMethod == "atomic") {
            cfg.scatter_config.method = Interpolation::ScatterMethod::Atomic;
        } else if (spreadMethod == "output_focused"
                   || spreadMethod == "output_focused_zbatched") {
            // The "_zbatched" alias is kept so old test parameter sets still
            // resolve; both map to OutputFocused (the z_batches knob lives on
            // ScatterConfig, not in the method enum).
            cfg.scatter_config.method = Interpolation::ScatterMethod::OutputFocused;
        } else if (spreadMethod == "tiled") {
            cfg.scatter_config.method = Interpolation::ScatterMethod::Tiled;
        }

        std::string gatherMethod = params.get<std::string>("gather_method", "none");
        if (gatherMethod == "atomic") {
            cfg.gather_config.method = Interpolation::GatherMethod::Atomic;
        } else if (gatherMethod == "atomic_sort") {
            cfg.gather_config.method = Interpolation::GatherMethod::AtomicSort;
        }

        if (params.contains("tile_size_3d")) {
            cfg.scatter_config.tile_size.fill(params.get<int>("tile_size_3d"));
        }
        if (params.contains("team_size")) {
            cfg.scatter_config.team_size = params.get<int>("team_size");
            cfg.gather_config.team_size  = params.get<int>("team_size");
        }

        nativeNufft_m = std::make_unique<NativeNUFFT_t>(nModesVec, useUpsampledInputs_m, cfg);
        nativeNufft_m->initialize(layout, Comm->getCommunicator());
    }

    template <typename RealField>
    void FFT<NUFFTransform, RealField>::cleanupBackend() {
#ifdef ENABLE_FINUFFT
        if (useFinufft_m && finufftPlan_m) {
            Traits_t::destroy(finufftPlan_m);
            finufftPlan_m = FinufftPlan_t{};
        }
#endif
    }

    template <typename RealField>
    void FFT<NUFFTransform, RealField>::allocateFinufftBuffers(const Layout_t& layout,
                                                               detail::size_type localNp) {
#ifdef ENABLE_FINUFFT
        const auto& lDom = layout.getLocalNDIndex();
        Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());

        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::realloc(tempR_m[d], localNp);
        }
        Kokkos::realloc(tempQ_m, localNp);
#else
        (void)layout;
        (void)localNp;

        throw std::runtime_error("FINUFFT is not activated. Rebuild with -DIPPL_ENABLE_FINUFFT=ON");
#endif
    }

    template <typename RealField>
    void FFT<NUFFTransform, RealField>::initFinufft(const ParameterList& params) {
#ifdef ENABLE_FINUFFT
#ifdef ENABLE_GPU_NUFFT
        if (Comm->size() != 1) {
            throw IpplException(
                "FFT<NUFFTransform>",
                "cuFINUFFT supports only single-rank execution; cuFINUFFT has no MPI "
                "decomposition. Use native IPPL NUFFT for distributed runs.");
        }

        if constexpr (!detail::is_cufinufft_memory_space_v<MemSpace>) {
            throw IpplException(
                "FFT<NUFFTransform>",
                "cuFINUFFT requires CUDA-backed IPPL/Kokkos memory. Host-only SERIAL fields "
                "are not supported because no host-to-CUDA staging backend is provided.");
        }
#endif

        FinufftOpts_t opts;
        Traits_t::defaultOpts(&opts);

#ifdef ENABLE_GPU_NUFFT
        opts.gpu_method         = params.get<int>("gpu_method", opts.gpu_method);
        opts.gpu_sort           = params.get<int>("gpu_sort", opts.gpu_sort);
        opts.gpu_kerevalmeth    = params.get<int>("gpu_kerevalmeth", opts.gpu_kerevalmeth);
        opts.gpu_binsizex       = params.get<int>("gpu_binsizex", opts.gpu_binsizex);
        opts.gpu_binsizey       = params.get<int>("gpu_binsizey", opts.gpu_binsizey);
        opts.gpu_binsizez       = params.get<int>("gpu_binsizez", opts.gpu_binsizez);
        opts.gpu_maxsubprobsize = params.get<int>("gpu_maxsubprobsize", opts.gpu_maxsubprobsize);
        opts.gpu_maxbatchsize   = 0;
#else
        opts.spread_sort        = params.get<int>("spread_sort", opts.spread_sort);
        opts.spread_kerevalmeth = params.get<int>("spread_kerevalmeth", opts.spread_kerevalmeth);
        opts.nthreads           = params.get<int>("nthreads", opts.nthreads);
#endif

        int iflag = (type_m == 1) ? -1 : 1;
        int dim   = static_cast<int>(Dim);

        int err = Traits_t::makeplan(type_m, dim, nModes_m.data(), iflag, 1, tol_m, &finufftPlan_m,
                                     &opts);

        if (err != 0) {
            throw IpplException("FFT<NUFFTransform>", "FINUFFT makeplan failed");
        }
#else
        (void)params;
        throw std::runtime_error("FINUFFT is not activated. Rebuild with -DIPPL_ENABLE_FINUFFT=ON");
#endif  // ENABLE_FINUFFT
    }

    //=========================================================================
    // Transform Dispatch
    //=========================================================================

    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transform(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        ComplexField& f) {
        // The FINUFFT/cuFINUFFT path is hardcoded for Dim == 3 (3D mode-grid
        // scratch + Rank<3> MDRangePolicy in transformFinufft). For Dim != 3
        // we skip the constexpr branch entirely so the 3D-only template body
        // is never instantiated -- otherwise the unit tests that exercise the
        // 2D type instantiations of FFT<NUFFTransform, ...> fail to compile
        // under a FINUFFT-enabled build.
        if constexpr (Dim == 3 && fft::is_available_v<fft::Finufft>) {
            if (useFinufft_m) {
                transformFinufft(R, Q, f);
                return;
            }
        }

        transformNative(R, Q, f);
    }

    //=========================================================================
    // Native NUFFT Transform
    //=========================================================================

    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transformNative(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        ComplexField& f) {
        const auto localNp = R.getParticleCount();
        const auto& layout = f.getLayout();
        const auto& mesh   = f.get_mesh();
        const auto& dx     = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();

        Vector<T, Dim> Len;
        for (unsigned d = 0; d < Dim; ++d) {
            int fullLength = domain[d].length();
            if (useR2C_m && static_cast<int>(d) == r2cDir_m) {
                fullLength = 2 * (fullLength - 1);
            }
            Len[d] = dx[d] * fullLength;
        }

        constexpr T twoPi = T(2.0 * M_PI);
        auto Rview        = R.getView();

        Vector<T, Dim> scaleToTwoPi, scaleBack;
        for (unsigned d = 0; d < Dim; ++d) {
            scaleToTwoPi[d] = twoPi / Len[d];
            scaleBack[d]    = Len[d] / twoPi;
        }

        using ScaleFunctor = detail::ScalePositionsFunctor<T, Dim, decltype(Rview)>;
        Kokkos::parallel_for("NUFFT_scale_to_2pi", Kokkos::RangePolicy<ExecSpace>(0, localNp),
                             ScaleFunctor(Rview, scaleToTwoPi));

        if (type_m == 1) {
            nativeNufft_m->type1(R, Q, f, useUpsampledInputs_m);
        } else if (type_m == 2) {
            nativeNufft_m->type2(f, R, Q, useUpsampledInputs_m);
        } else {
            throw IpplException("FFT<NUFFTransform>", "Only type 1 and type 2 NUFFT supported");
        }

        Kokkos::parallel_for("NUFFT_scale_back", Kokkos::RangePolicy<ExecSpace>(0, localNp),
                             ScaleFunctor(Rview, scaleBack));
    }

    //=========================================================================
    // FINUFFT Transform
    //=========================================================================
    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transformFinufft(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        ComplexField& f) {
#ifdef ENABLE_FINUFFT
        const auto localNp = R.getParticleCount();
        const auto& layout = f.getLayout();
        const auto& mesh   = f.get_mesh();
        const auto& dx     = mesh.getMeshSpacing();
        const auto& domain = layout.getDomain();
        const int nghost   = f.getNghost();

        Vector<T, Dim> Len;
        for (unsigned d = 0; d < Dim; ++d) {
            Len[d] = dx[d] * domain[d].length();
        }

        constexpr T twoPi = T(2.0 * M_PI);

        auto fview = f.getView();
        auto Rview = R.getView();
        auto Qview = Q.getView();

        const auto& lDom = layout.getLocalNDIndex();
        if (tempField_m.extent(0) != static_cast<std::size_t>(lDom[0].length())
            || tempField_m.extent(1) != static_cast<std::size_t>(lDom[1].length())
            || tempField_m.extent(2) != static_cast<std::size_t>(lDom[2].length())) {
            Kokkos::realloc(tempField_m, lDom[0].length(), lDom[1].length(), lDom[2].length());
        }

        if (tempQ_m.extent(0) < localNp) {
            Kokkos::realloc(tempQ_m, localNp);
        }

        for (unsigned d = 0; d < Dim; ++d) {
            if (tempR_m[d].extent(0) < localNp) {
                Kokkos::realloc(tempR_m[d], localNp);
            }
        }

        auto tempField = tempField_m;
        auto tempQ     = tempQ_m;
        auto tempRx    = tempR_m[0];
        auto tempRy    = tempR_m[1];
        auto tempRz    = tempR_m[2];

        Vector<T, Dim> scale;
        for (unsigned d = 0; d < Dim; ++d) {
            scale[d] = twoPi / Len[d];
        }

        using mdrange_type = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
        using CopyToTemp   = detail::CopyFieldToTempFunctor<decltype(fview), decltype(tempField)>;

        int nx         = lDom[0].length();
        int ny         = lDom[1].length();
        int nz         = lDom[2].length();
        bool needShift = (type_m == 2);

        Kokkos::parallel_for(
            "FINUFFT_copy_field_to_temp",
            mdrange_type({nghost, nghost, nghost}, {static_cast<int>(fview.extent(0)) - nghost,
                                                    static_cast<int>(fview.extent(1)) - nghost,
                                                    static_cast<int>(fview.extent(2)) - nghost}),
            CopyToTemp(fview, tempField, nghost, nx, ny, nz, needShift));

        using CopyParticles =
            detail::CopyParticlesToTempFunctor<T, Dim, decltype(Rview), decltype(Qview),
                                               decltype(tempRx), decltype(tempQ)>;

        Kokkos::parallel_for("FINUFFT_copy_particles_to_temp", localNp,
                             CopyParticles(Rview, Qview, tempRx, tempRy, tempRz, tempQ, scale));

        Kokkos::fence();

        int err = Traits_t::setpts(finufftPlan_m, static_cast<FinufftCount_t>(localNp),
                                   tempRx.data(), tempRy.data(), tempRz.data(), FinufftCount_t{0},
                                   nullptr, nullptr, nullptr);

        if (err != 0) {
            throw IpplException("FFT<NUFFTransform>", "FINUFFT setpts failed");
        }

        err = Traits_t::execute(finufftPlan_m, tempQ.data(), tempField.data());

        if (err != 0) {
            throw IpplException("FFT<NUFFTransform>", "FINUFFT execute failed");
        }

        Kokkos::fence();

        if (type_m == 1) {
            using CopyFromTemp =
                detail::CopyFieldFromTempFunctor<decltype(fview), decltype(tempField)>;

            Kokkos::parallel_for("FINUFFT_copy_field_from_temp",
                                 mdrange_type({nghost, nghost, nghost},
                                              {static_cast<int>(fview.extent(0)) - nghost,
                                               static_cast<int>(fview.extent(1)) - nghost,
                                               static_cast<int>(fview.extent(2)) - nghost}),
                                 CopyFromTemp(fview, tempField, nghost, nx, ny, nz));
        } else if (type_m == 2) {
            using CopyBack = detail::CopyParticlesFromTempFunctor<decltype(Qview), decltype(tempQ)>;

            Kokkos::parallel_for("FINUFFT_copy_particles_from_temp", localNp,
                                 CopyBack(Qview, tempQ));
        }
#else
        throw std::runtime_error("FINUFFT is not activated. Rebuild with -DIPPL_ENABLE_FINUFFT=ON");
        (void)R;
        (void)Q;
        (void)f;
#endif  // ENABLE_FINUFFT
    }
}  // namespace ippl

#endif  // IPPL_FFT_TRANSFORM_NUFFT_HPP
