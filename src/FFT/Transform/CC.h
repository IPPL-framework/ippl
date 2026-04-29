#ifndef IPPL_FFT_TRANSFORM_CC_H
#define IPPL_FFT_TRANSFORM_CC_H

#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

    template <typename ComplexField>
    class FFT<CCTransform, ComplexField> {
    public:
        static constexpr unsigned Dim = ComplexField::dim;

        using Complex_t  = typename ComplexField::value_type;
        using T          = typename Complex_t::value_type;
        using MemSpace   = typename ComplexField::memory_space;
        using ExecSpace  = typename ComplexField::execution_space;
        using Layout_t   = FieldLayout<Dim>;

#ifdef IPPL_ENABLE_CUFFTMP
        using Backend_t = fft::CuFFTMpC2C<T, Dim, MemSpace>;
#else
        using Backend_t  = fft::HeffteC2C<T, Dim, MemSpace>;
#endif
        using TempView_t = Kokkos::View<Complex_t***, Kokkos::LayoutLeft, MemSpace>;

        FFT(const Layout_t& layout, const ParameterList& params) {
            static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

            std::array<long long, 3> low, high;
            fft::domainToBounds<Dim>(layout.getLocalNDIndex(), low, high);
            heffte::box3d<long long> box{low, high};

            backend_ = std::make_unique<Backend_t>(box, box, Comm->getCommunicator(), params);
        }

        void warmup(ComplexField& f) {
            transform(FORWARD, f);
            transform(BACKWARD, f);
        }

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