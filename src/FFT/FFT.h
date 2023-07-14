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
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
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

    enum FFTComm {
        a2av   = 0,
        a2a    = 1,
        p2p    = 2,
        p2p_pl = 3
    };

    namespace detail {

        /*!
         * Wrapper type for heFFTe backends, templated
         * on the Kokkos memory space
         */
        template <typename>
        struct HeffteBackendType;

#ifdef Heffte_ENABLE_FFTW
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::fftw;
            using backendSine = heffte::backend::fftw_sin;
            using backendCos  = heffte::backend::fftw_cos;
        };
#endif
#ifdef Heffte_ENABLE_MKL
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::mkl;
            using backendSine = heffte::backend::mkl_sin;
            using backendCos  = heffte::backend::mkl_cos;
        };
#endif
#if defined(Heffte_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA)
        template <>
        struct HeffteBackendType<Kokkos::CudaSpace> {
            using backend     = heffte::backend::cufft;
            using backendSine = heffte::backend::cufft_sin;
            using backendCos  = heffte::backend::cufft_cos;
        };
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(Heffte_ENABLE_MKL) && !defined(Heffte_ENABLE_FFTW)
        /**
         * Use heFFTe's inbuilt 1D fft computation on CPUs if no
         * vendor specific or optimized backend is found
         */
        template <>
        struct HeffteBackendType<Kokkos::HostSpace> {
            using backend     = heffte::backend::stock;
            using backendSine = heffte::backend::stock_sin;
            using backendCos  = heffte::backend::stock_cos;
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
        typedef FieldLayout<Dim> Layout_t;

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

        Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft> tempField;
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
        typedef typename ComplexField::value_type Complex_t;

        using Base::Base;
        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /** Do the inplace FFT: specify +1 or -1 to indicate forward or inverse
            transform. The output is over-written in the input.
        */
        void transform(int direction, ComplexField& f);
    };

    /**
       real-to-complex FFT class
    */
    template <typename RealField>
    class FFT<RCTransform, RealField>
        : public EXT_FFT_BASE_CLASS(RealField, backend,
                                    Kokkos::complex<typename RealField::value_type>) {
        constexpr static unsigned Dim = RealField::dim;
        typedef typename RealField::value_type Real_t;
        using Base = EXT_FFT_BASE_CLASS(RealField, backend,
                                        Kokkos::complex<typename RealField::value_type>);

    public:
        typedef Kokkos::complex<Real_t> Complex_t;
        using ComplexField =
            Field<Complex_t, Dim, typename RealField::Mesh_t, typename RealField::Centering_t,
                  typename RealField::execution_space>;

        using typename Base::heffteBackend, typename Base::workspace_t, typename Base::Layout_t;

        /** Create a new FFT object with the layout for the input and output Fields
         * and parameters for heffte.
         */
        FFT(const Layout_t& layoutInput, const Layout_t& layoutOutput, const ParameterList& params);

        /** Do the FFT: specify +1 or -1 to indicate forward or inverse
            transform.
        */
        void transform(int direction, RealField& f, ComplexField& g);

    protected:
        Kokkos::View<typename ComplexField::view_type::data_type, Kokkos::LayoutLeft>
            tempFieldComplex;
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

        /** Do the inplace FFT: specify +1 or -1 to indicate forward or inverse
            transform. The output is over-written in the input.
        */
        void transform(int direction, Field& f);
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

        /** Do the inplace FFT: specify +1 or -1 to indicate forward or inverse
            transform. The output is over-written in the input.
        */
        void transform(int direction, Field& f);
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
