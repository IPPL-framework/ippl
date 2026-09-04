/*!
 * @file RC.h
 * @brief Real-to-complex FFT specialization (RCTransform tag).
 */
#ifndef IPPL_FFT_TRANSFORM_RC_H
#define IPPL_FFT_TRANSFORM_RC_H

#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

    /*!
     * @class FFT<RCTransform, RealField>
     * @brief Real-to-complex / complex-to-real FFT over IPPL Fields.
     *
     * Forward transforms a real field into its half-complex spectrum, the
     * backward transforms back. Selects cuFFTMp on CUDA-MP builds, otherwise
     * the heFFTe R2C backend.
     *
     * @tparam RealField IPPL Field of real-valued elements.
     */
    template <typename RealField>
    class FFT<RCTransform, RealField> {
    public:
        static constexpr unsigned Dim = RealField::dim;

        using T         = typename RealField::value_type;
        using Complex_t = Kokkos::complex<T>;
        using MemSpace  = typename RealField::memory_space;
        using ExecSpace = typename RealField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

        using ComplexField = typename Field<Complex_t, Dim, typename RealField::Mesh_t,
                                            typename RealField::Centering_t,
                                            ExecSpace>::uniform_type;

#if defined(IPPL_ENABLE_CUFFTMP) && defined(KOKKOS_ENABLE_CUDA)
        using Backend_t = std::conditional_t<fft::use_cufftmp_v<MemSpace>,
                                             fft::CuFFTMpR2C<T, Dim, MemSpace>,
                                             fft::HeffteR2C<T, Dim, MemSpace>>;
#else
        using Backend_t = fft::HeffteR2C<T, Dim, MemSpace>;
#endif
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::c2c;
        using TempReal_t    = typename Kokkos::View<typename RealField::view_type::data_type,
                                                    Kokkos::LayoutLeft, MemSpace>::uniform_type;
        using TempComplex_t = typename Kokkos::View<typename ComplexField::view_type::data_type,
                                                    Kokkos::LayoutLeft, MemSpace>::uniform_type;

        /*!
         * @brief Build the R2C plan for the given layouts.
         * @param layoutIn  Real-input field layout.
         * @param layoutOut Complex-output field layout (Hermitian-symmetric).
         * @param params    Backend parameter list (R2C axis taken from key
         *                  `r2c_direction`, default 0).
         */
        FFT(const Layout_t& layoutIn, const Layout_t& layoutOut, const ParameterList& params) {
            static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

            std::array<long long, 3> lowIn, highIn, lowOut, highOut;
            fft::domainToBounds<Dim>(layoutIn.getLocalNDIndex(), lowIn, highIn);
            fft::domainToBounds<Dim>(layoutOut.getLocalNDIndex(), lowOut, highOut);

            int r2c_dir = params.get<int>("r2c_direction", 0);
            backend_    = std::make_unique<Backend_t>(heffte::box3d<long long>{lowIn, highIn},
                                                     heffte::box3d<long long>{lowOut, highOut},
                                                     r2c_dir, Comm->getCommunicator(), params);
        }

        //! Execute one forward + one backward to JIT-compile / warm caches.
        void warmup(RealField& f, ComplexField& g) {
            transform(FORWARD, f, g);
            transform(BACKWARD, f, g);
        }

        /*!
         * @brief Forward (real -> complex) or backward (complex -> real) transform.
         * @param direction FORWARD or BACKWARD.
         * @param f         Real field (input on FORWARD, output on BACKWARD).
         * @param g         Complex field (output on FORWARD, input on BACKWARD).
         */
        void transform(TransformDirection direction, RealField& f, ComplexField& g) {
            auto fview    = f.getView();
            auto gview    = g.getView();
            const int ngf = f.getNghost();
            const int ngg = g.getNghost();

            ensureTemps(f, g);
            fft::copyToTemp<ExecSpace, decltype(tempReal_), decltype(fview)>(tempReal_, fview, ngf);
            fft::copyToTemp<ExecSpace, decltype(tempComplex_), decltype(gview)>(tempComplex_, gview,
                                                                                ngg);

            if (direction == FORWARD) {
                backend_->forward(tempReal_.data(), tempComplex_.data());
            } else {
                backend_->backward(tempComplex_.data(), tempReal_.data());
            }

            fft::copyFromTemp<ExecSpace, decltype(fview), decltype(tempReal_)>(fview, tempReal_,
                                                                               ngf);
            fft::copyFromTemp<ExecSpace, decltype(gview), decltype(tempComplex_)>(
                gview, tempComplex_, ngg);
        }

    private:
        std::unique_ptr<Backend_t> backend_;
        TempReal_t tempReal_;
        TempComplex_t tempComplex_;

        void ensureTemps(const RealField& f, const ComplexField& g) {
            if (tempReal_.size() != f.getOwned().size()) {
                tempReal_ = detail::shrinkView("fft_rc_real", f.getView(), f.getNghost());
            }
            if (tempComplex_.size() != g.getOwned().size()) {
                tempComplex_ = detail::shrinkView("fft_rc_complex", g.getView(), g.getNghost());
            }
        }
    };

}  // namespace ippl

#endif
