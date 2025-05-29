//
// Class FFT
//   The FFT class performs complex-to-complex,
//   real-to-complex on IPPL Fields.
//   FFT is templated on the type of transform to be performed,
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte. In making this interface,
//   we have referred Cabana library
//   https://github.com/ECP-copa/Cabana.
//
//

#ifndef IPPL_FFT_FFT_H
#define IPPL_FFT_FFT_H

#include <Kokkos_Complex.hpp>
#include <array>
#include <heffte_fft3d.h>
#include <heffte_fft3d_r2c.h>
#include <memory>
#include <type_traits>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Index/NDIndex.h"

namespace heffte {
    template <>
    struct is_ccomplex<Kokkos::complex<float>> : std::true_type {};

    template <>
    struct is_zcomplex<Kokkos::complex<double>> : std::true_type {};
}  // namespace heffte

namespace ippl {

    /**
       Tag classes for Fourier transforms
    */
    class CCTransform {};
    class RCTransform {};
    class SineTransform {};
    class CosTransform {};
    /**
       Tag classes for Cosine of type 1 transforms
    */
    class Cos1Transform {};

    enum FFTComm {
        a2av   = 0,
        a2a    = 1,
        p2p    = 2,
        p2p_pl = 3
    };

    enum TransformDirection {
        FORWARD,
        BACKWARD
    };

    namespace detail {
        /*!
         * Wrapper type for heFFTe backends, templated
         * on the Kokkos memory space
         */
        template <typename>
        struct HeffteBackendType;

#if defined(Heffte_ENABLE_FFTW)
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::fftw;
            using backendSine = heffte::backend::fftw_sin;
            using backendCos  = heffte::backend::fftw_cos;
            using backendCos1 = heffte::backend::fftw_cos1;
        };
#elif defined(Heffte_ENABLE_MKL)
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::mkl;
            using backendSine = heffte::backend::mkl_sin;
            using backendCos  = heffte::backend::mkl_cos;
        };
#endif

#ifdef Heffte_ENABLE_CUDA
#ifdef KOKKOS_ENABLE_CUDA
        template <>
        struct HeffteBackendType<Kokkos::CudaSpace> {
            using backend     = heffte::backend::cufft;
            using backendSine = heffte::backend::cufft_sin;
            using backendCos  = heffte::backend::cufft_cos;
            using backendCos1 = heffte::backend::cufft_cos1;
        };
#else
#error cuFFT backend is enabled for heFFTe but CUDA is not enabled for Kokkos!
#endif
#endif

#ifdef KOKKOS_ENABLE_HIP
#ifdef Heffte_ENABLE_ROCM
        template <>
        struct HeffteBackendType<Kokkos::HIPSpace> {
            using backend     = heffte::backend::rocfft;
            using backendSine = heffte::backend::rocfft_sin;
            using backendCos  = heffte::backend::rocfft_cos;
            using backendCos1 = heffte::backend::rocfft_cos1;
        };
#else
        template <>
        struct HeffteBackendType<Kokkos::HIPSpace> {
            using backend     = heffte::backend::stock;
            using backendSine = heffte::backend::stock_sin;
            using backendCos  = heffte::backend::stock_cos;
            using backendCos1 = heffte::backend::stock_cos1;
        };
#endif
#endif

#if !defined(Heffte_ENABLE_MKL) && !defined(Heffte_ENABLE_FFTW)
        /**
         * Use heFFTe's inbuilt 1D fft computation on CPUs if no
         * vendor specific or optimized backend is found
         */
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::stock;
            using backendSine = heffte::backend::stock_sin;
            using backendCos  = heffte::backend::stock_cos;
            using backendCos1 = heffte::backend::stock_cos1;
        };
#endif

    }  // namespace detail

    template <typename Field, template <typename...> class FFT, typename Backend,
              typename BufferType = typename Field::value_type>
    class FFTBase {
        constexpr static unsigned Dim = Field::dim;

    public:
        using heffteBackend = Backend;
        using workspace_t   = typename FFT<heffteBackend>::template buffer_container<BufferType>;
        using Layout_t      = FieldLayout<Dim>;

        FFTBase(const Layout_t& layout, const ParameterList& params);
        ~FFTBase() = default;

    protected:
        FFTBase() = default;

        void domainToBounds(const NDIndex<Dim>& domain, std::array<long long, 3>& low,
                            std::array<long long, 3>& high);
        void setup(const heffte::box3d<long long>& inbox, const heffte::box3d<long long>& outbox,
                   const ParameterList& params);

        std::shared_ptr<FFT<heffteBackend, long long>> heffte_m;
        workspace_t workspace_m;

        template <typename FieldType>
        using temp_view_type =
            typename Kokkos::View<typename FieldType::view_type::data_type, Kokkos::LayoutLeft,
                                  typename FieldType::memory_space>::uniform_type;
        temp_view_type<Field> tempField;
    };

#define IN_PLACE_FFT_BASE_CLASS(Field, Backend) \
    FFTBase<Field, heffte::fft3d,               \
            typename detail::HeffteBackendType<typename Field::memory_space>::Backend>
#define EXT_FFT_BASE_CLASS(Field, Backend, Type)                                       \
    FFTBase<Field, heffte::fft3d_r2c,                                                  \
            typename detail::HeffteBackendType<typename Field::memory_space>::Backend, \
            typename Type>

    /**
       Non-specialized FFT class.  We specialize based on Transform tag class
    */
    template <class Transform, typename Field>
    class FFT {};

    /**
       complex-to-complex FFT class
    */
    template <typename ComplexField>
    class FFT<CCTransform, ComplexField> : public IN_PLACE_FFT_BASE_CLASS(ComplexField, backend) {
        constexpr static unsigned Dim = ComplexField::dim;
        using Base                    = IN_PLACE_FFT_BASE_CLASS(ComplexField, backend);

    public:
        using Complex_t = typename ComplexField::value_type;

        using Base::Base;
        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /*!
         * Warmup the FFT object by forward & backward FFT on an empty field
         * @param f Field whose transformation to compute (and overwrite)
         */
        void warmup(ComplexField& f);

        /*!
         * Perform in-place FFT
         * @param direction Forward or backward transformation
         * @param f Field whose transformation to compute (and overwrite)
         */
        void transform(TransformDirection direction, ComplexField& f);
    };

    /**
       real-to-complex FFT class
    */
    template <typename RealField>
    class FFT<RCTransform, RealField>
        : public EXT_FFT_BASE_CLASS(RealField, backend,
                                    Kokkos::complex<typename RealField::value_type>) {
        constexpr static unsigned Dim = RealField::dim;
        using Real_t                  = typename RealField::value_type;
        using Base                    = EXT_FFT_BASE_CLASS(RealField, backend,
                                                           Kokkos::complex<typename RealField::value_type>);

    public:
        using Complex_t    = Kokkos::complex<Real_t>;
        using ComplexField = typename Field<Complex_t, Dim, typename RealField::Mesh_t,
                                            typename RealField::Centering_t,
                                            typename RealField::execution_space>::uniform_type;

        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /** Create a new FFT object with the layout for the input and output Fields
         * and parameters for heffte.
         */
        FFT(const Layout_t& layoutInput, const Layout_t& layoutOutput, const ParameterList& params);

        /*!
         * Warmup the FFT object by forward & backward FFT on an empty field
         * @param f Field whose transformation to compute
         * @param g Field in which to store the transformation
         */
        void warmup(RealField& f, ComplexField& g);

        /*!
         * Perform FFT
         * @param direction Forward or backward transformation
         * @param f Field whose transformation to compute
         * @param g Field in which to store the transformation
         */
        void transform(TransformDirection direction, RealField& f, ComplexField& g);

    private:
        typename Base::template temp_view_type<ComplexField> tempFieldComplex;
    };

    /**
       Sine transform class
    */
    template <typename Field>
    class FFT<SineTransform, Field> : public IN_PLACE_FFT_BASE_CLASS(Field, backendSine) {
        constexpr static unsigned Dim = Field::dim;
        using Base                    = IN_PLACE_FFT_BASE_CLASS(Field, backendSine);

    public:
        using Base::Base;
        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /*!
         * Warmup the FFT object by forward & backward FFT on an empty field
         * @param f Field whose transformation to compute (and overwrite)
         */
        void warmup(Field& f);

        /*!
         * Perform in-place FFT
         * @param direction Forward or backward transformation
         * @param f Field whose transformation to compute (and overwrite)
         */
        void transform(TransformDirection direction, Field& f);
    };
    /**
       Cosine transform class
    */
    template <typename Field>
    class FFT<CosTransform, Field> : public IN_PLACE_FFT_BASE_CLASS(Field, backendCos) {
        constexpr static unsigned Dim = Field::dim;
        using Base                    = IN_PLACE_FFT_BASE_CLASS(Field, backendCos);

    public:
        using Base::Base;
        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /*!
         * Warmup the FFT object by forward & backward FFT on an empty field
         * @param f Field whose transformation to compute (and overwrite)
         */
        void warmup(Field& f);

        /*!
         * Perform in-place FFT
         * @param direction Forward or backward transformation
         * @param f Field whose transformation to compute (and overwrite)
         */
        void transform(TransformDirection direction, Field& f);
    };
    /**
       Cosine type 1 transform class
    */
    template <typename Field>
    class FFT<Cos1Transform, Field> : public IN_PLACE_FFT_BASE_CLASS(Field, backendCos1) {
        constexpr static unsigned Dim = Field::dim;
        using Base                    = IN_PLACE_FFT_BASE_CLASS(Field, backendCos1);

    public:
        using Base::Base;
        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /*!
         * Warmup the FFT object by forward & backward FFT on an empty field
         * @param f Field whose transformation to compute (and overwrite)
         */
        void warmup(Field& f);

        /*!
         * Perform in-place FFT
         * @param direction Forward or backward transformation
         * @param f Field whose transformation to compute (and overwrite)
         */
        void transform(TransformDirection direction, Field& f);
    };
}  // namespace ippl

#include "FFT/FFT.hpp"

#endif  // IPPL_FFT_FFT_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
