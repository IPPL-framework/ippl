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
            nmodes[d] = lDom[d].length();
            ;
        }

        type_m = type;
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
        setup(nmodes, params);
    }

    /**
        setup performs the initialization necessary.
    */
    template <typename RealField>
    void FFT<NUFFTransform, RealField>::setup(std::array<int64_t, 3>& nmodes,
                                              const ParameterList& params) {
#ifdef ENABLE_GPU_NUFFT
        cufinufft_opts opts;
        cufinufft_default_opts(&opts);
#else
        finufft_opts opts;
        finufft_default_opts(&opts);
#endif
        tol_m = 1e-6;

        if (!params.get<bool>("use_finufft_defaults")) {
            tol_m = params.get<T>("tolerance");
#ifdef ENABLE_GPU_NUFFT
            opts.gpu_method         = params.get<int>("gpu_method");
            opts.gpu_sort           = params.get<int>("gpu_sort");
            opts.gpu_kerevalmeth    = params.get<int>("gpu_kerevalmeth");
            opts.gpu_binsizex       = params.get<int>("gpu_binsizex");
            opts.gpu_binsizey       = params.get<int>("gpu_binsizey");
            opts.gpu_binsizez       = params.get<int>("gpu_binsizez");
            opts.gpu_maxsubprobsize = params.get<int>("gpu_maxsubprobsize");
#else
            opts.spread_sort        = params.get<int>("spread_sort");
            opts.spread_kerevalmeth = params.get<int>("spread_kerevalmeth");
            opts.nthreads           = params.get<int>("nthreads");
#endif
        }

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
        ier_m   = nufft_m.makeplan(type_m, dim, nmodes.data(), iflag, 1, tol_m, &plan_m, &opts);
    }

    template <typename RealField>
    template <class... Properties>
    void FFT<NUFFTransform, RealField>::transform(
        const ParticleAttrib<Vector<T, Dim>, Properties...>& R, ParticleAttrib<T, Properties...>& Q,
        typename FFT<NUFFTransform, RealField>::ComplexField& f) {
        auto fview       = f.getView();
        auto Rview       = R.getView();
        auto Qview       = Q.getView();
        const int nghost = f.getNghost();

        auto localNp = R.getParticleCount();

        const Layout_t& layout               = f.getLayout();
        const UniformCartesian<T, Dim>& mesh = f.get_mesh();
        const Vector<T, Dim>& dx             = mesh.getMeshSpacing();
        const Vector<T, Dim>& origin         = mesh.getOrigin();
        const auto& domain                   = layout.getDomain();
        Vector<T, Dim> Len;
        Vector<int, Dim> N;

        for (unsigned d = 0; d < Dim; ++d) {
            N[d]   = domain[d].length();
            Len[d] = dx[d] * N[d];
        }

        const double pi = std::acos(-1.0);

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
                    // tempR[d](i) = (Rview(i)[d] - (twopiFactor * 2.0 * pi)) * (2.0 * pi / Len[d]);
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

        ier_m = nufft_m.setpts(plan_m, localNp, tempR[0].data(), tempR[1].data(), tempR[2].data(),
                               0, NULL, NULL, NULL);

        ier_m = nufft_m.execute(plan_m, tempQ.data(), tempField.data());
        Kokkos::fence();

        if (type_m == 1) {
            Kokkos::parallel_for(
                "copy to field data NUFFT",
                mdrange_type(
                    {nghost, nghost, nghost},
                    {fview.extent(0) - nghost, fview.extent(1) - nghost, fview.extent(2) - nghost}),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
#ifdef ENABLE_GPU_NUFFT
                    fview(i, j, k).real() = tempField(i - nghost, j - nghost, k - nghost).x;
                    fview(i, j, k).imag() = tempField(i - nghost, j - nghost, k - nghost).y;
#else
                    fview(i, j, k).real() = tempField(i - nghost, j - nghost, k - nghost).real();
                    fview(i, j, k).imag() = tempField(i - nghost, j - nghost, k - nghost).imag();
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
    }

    template <typename RealField>
    FFT<NUFFTransform, RealField>::~FFT() {
        ier_m = nufft_m.destroy(plan_m);
    }

}  // namespace ippl

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End: