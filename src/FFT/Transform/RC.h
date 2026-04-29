#ifndef IPPL_FFT_TRANSFORM_RC_H
#define IPPL_FFT_TRANSFORM_RC_H

#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {

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

#ifdef IPPL_ENABLE_CUFFTMP
        using Backend_t = fft::CuFFTMpR2C<T, Dim, MemSpace>;
#else
        using Backend_t = fft::HeffteR2C<T, Dim, MemSpace>;
#endif
        using TempReal_t    = Kokkos::View<T***, Kokkos::LayoutLeft, MemSpace>;
        using TempComplex_t = Kokkos::View<Complex_t***, Kokkos::LayoutLeft, MemSpace>;

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

        void warmup(RealField& f, ComplexField& g) {
            transform(FORWARD, f, g);
            transform(BACKWARD, f, g);
        }

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
