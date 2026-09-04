/*!
 * @file Trig.h
 * @brief Trigonometric (sine / cosine / cosine-Type-I) FFT specializations.
 */
#ifndef IPPL_FFT_TRANSFORM_TRIG_H
#define IPPL_FFT_TRANSFORM_TRIG_H

#include "Utility/ParameterList.h"
#include "Utility/ViewUtils.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"


namespace ippl {

    namespace fft {

        /*!
         * @class TrigBase
         * @brief Shared implementation for sine / cosine transforms.
         *
         * Holds a HeffteTrig backend, an internal LayoutLeft scratch buffer,
         * and the FFTW-only renormalization (FFTW emits trig outputs scaled
         * by 8 in 3D; other backends do not).
         *
         * @tparam Field IPPL Field of real elements.
         * @tparam Tag   One of SineTransform, CosTransform, Cos1Transform.
         */
        template <typename Field, typename Tag>
        class TrigBase {
        public:
            static constexpr unsigned Dim = Field::dim;

            using T          = typename Field::value_type;
            using MemSpace   = typename Field::memory_space;
            using ExecSpace  = typename Field::execution_space;
            using Layout_t   = FieldLayout<Dim>;
            using Backend_t  = HeffteTrig<T, Dim, MemSpace, Tag>;
            using TempView_t = typename Kokkos::View<typename Field::view_type::data_type,
                                                     Kokkos::LayoutLeft, MemSpace>::uniform_type;

            //! Build the heFFTe trig plan for the local subdomain in @p layout.
            TrigBase(const Layout_t& layout, const ParameterList& params) {
                static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

                std::array<long long, 3> low, high;
                domainToBounds<Dim>(layout.getLocalNDIndex(), low, high);
                heffte::box3d<long long> box{low, high};

                backend_ = std::make_unique<Backend_t>(box, box, Comm->getCommunicator(), params);
            }

            //! Run a forward + backward pair to JIT-compile / warm caches.
            void warmup(Field& f) {
                transform(FORWARD, f);
                transform(BACKWARD, f);
            }

            /*!
             * @brief In-place trig transform of @p f.
             *
             * When FFTW is the backend the input is rescaled by
             * @c 1/fftw_trig_scale() on FORWARD and @c fftw_trig_scale() on
             * BACKWARD so the result is consistent with the other backends.
             */
            void transform(TransformDirection direction, Field& f) {
                // FFTW scaling
                if constexpr (is_available_v<FFTW>) {
                    if (direction == FORWARD)
                        f = f / fftw_trig_scale();
                }

                auto view    = f.getView();
                const int ng = f.getNghost();

                ensureTemp(f);
                fft::copyToTemp<ExecSpace, decltype(temp_), decltype(view)>(temp_, view, ng);

                if (direction == FORWARD) {
                    backend_->forward(temp_.data(), temp_.data());
                } else {
                    backend_->backward(temp_.data(), temp_.data());
                }

                fft::copyFromTemp<ExecSpace, decltype(view), decltype(temp_)>(view, temp_, ng);

                if constexpr (is_available_v<FFTW>) {
                    if (direction == BACKWARD)
                        f = f * fftw_trig_scale();
                }
            }

        private:
            std::unique_ptr<Backend_t> backend_;
            TempView_t temp_;

            void ensureTemp(const Field& f) {
                if (temp_.size() != f.getOwned().size()) {
                    temp_ = ippl::detail::shrinkView("fft_trig_temp", f.getView(), f.getNghost());
                }
            }
        };

    }  // namespace fft

    //! @name Trigonometric FFT specializations
    //! Thin tag-aliases over fft::TrigBase. The transform tag selects the
    //! corresponding heFFTe sine/cosine backend.
    //! @{
    template <typename Field>
    class FFT<SineTransform, Field> : public fft::TrigBase<Field, SineTransform> {
    public:
        using MemSpace      = typename Field::memory_space;
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::sin;
        using fft::TrigBase<Field, SineTransform>::TrigBase;
    };

    template <typename Field>
    class FFT<CosTransform, Field> : public fft::TrigBase<Field, CosTransform> {
    public:
        using MemSpace      = typename Field::memory_space;
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::cos;
        using fft::TrigBase<Field, CosTransform>::TrigBase;
    };

    template <typename Field>
    class FFT<Cos1Transform, Field> : public fft::TrigBase<Field, Cos1Transform> {
    public:
        using MemSpace      = typename Field::memory_space;
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::cos1;
        using fft::TrigBase<Field, Cos1Transform>::TrigBase;
    };
    //! @}

}  // namespace ippl

#endif
