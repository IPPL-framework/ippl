/*!
 * @file CC.h
 * @brief Complex-to-complex FFT specialization (CCTransform tag).
 */
#ifndef IPPL_FFT_TRANSFORM_CC_H
#define IPPL_FFT_TRANSFORM_CC_H

#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

    /*!
     * @class FFT<CCTransform, ComplexField>
     * @brief In-place complex-to-complex FFT over a complex IPPL Field.
     *
     * Selects cuFFTMp when @c IPPL_ENABLE_CUFFTMP is defined, otherwise the
     * heFFTe C2C backend. The transform is performed on a contiguous
     * LayoutLeft scratch view; ghost cells are stripped on the way in and
     * restored on the way out.
     *
     * @tparam ComplexField Field whose value_type is a Kokkos::complex.
     */
    template <typename ComplexField>
    class FFT<CCTransform, ComplexField> {
    public:
        static constexpr unsigned Dim = ComplexField::dim;

        using Complex_t  = typename ComplexField::value_type;
        using T          = typename Complex_t::value_type;
        using MemSpace   = typename ComplexField::memory_space;
        using ExecSpace  = typename ComplexField::execution_space;
        using Layout_t   = FieldLayout<Dim>;

#if defined(IPPL_ENABLE_CUFFTMP) && defined(KOKKOS_ENABLE_CUDA)
        using Backend_t = std::conditional_t<fft::use_cufftmp_v<MemSpace>,
                                             fft::CuFFTMpC2C<T, Dim, MemSpace>,
                                             fft::HeffteC2C<T, Dim, MemSpace>>;
#else
        using Backend_t  = fft::HeffteC2C<T, Dim, MemSpace>;
#endif
        using heffteBackend = typename fft::HeffteBackend<MemSpace>::c2c;
        using TempView_t = typename Kokkos::View<typename ComplexField::view_type::data_type,
                                                 Kokkos::LayoutLeft, MemSpace>::uniform_type;

        /*!
         * @brief Build the FFT plan for the local subdomain described by @p layout.
         * @param layout Field layout giving the local NDIndex and MPI partition.
         * @param params FFT-tuning parameters forwarded to the backend.
         */
        FFT(const Layout_t& layout, const ParameterList& params) {
            static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

            std::array<long long, 3> low, high;
            fft::domainToBounds<Dim>(layout.getLocalNDIndex(), low, high);
            heffte::box3d<long long> box{low, high};

            backend_ = std::make_unique<Backend_t>(box, box, Comm->getCommunicator(), params);
        }

        //! Run a forward + backward pair on @p f to JIT compile / cache backend kernels.
        void warmup(ComplexField& f) {
            transform(FORWARD, f);
            transform(BACKWARD, f);
        }

        /*!
         * @brief In-place FFT of @p f.
         * @param direction FORWARD or BACKWARD.
         * @param f         Field to transform; modified in place.
         */
        void transform(TransformDirection direction, ComplexField& f) {
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
        }

    private:
        std::unique_ptr<Backend_t> backend_;
        TempView_t temp_;

        void ensureTemp(const ComplexField& f) {
            if (temp_.size() != f.getOwned().size()) {
                temp_ = detail::shrinkView("fft_cc_temp", f.getView(), f.getNghost());
            }
        }
    };

}  // namespace ippl

#endif
