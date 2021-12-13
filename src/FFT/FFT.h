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

#include <heffte_fft3d.h>
#include <heffte_fft3d_r2c.h>
#include <array>
#include <memory>
#include <type_traits>

#include "FieldLayout/FieldLayout.h"
#include "Field/Field.h"
#include "Utility/ParameterList.h"
#include "Utility/IpplException.h"

namespace ippl {

    /**
       Tag classes for CC type of Fourier transforms
    */
    class CCTransform {};
    /**
       Tag classes for RC type of Fourier transforms
    */
    class RCTransform {};

    enum FFTComm {
        a2av = 0,
        a2a = 1,
        p2p = 2,
        p2p_pl = 3
    };

    //class FFTParams {
    //    bool alltoall = true;
    //    bool pencils = true;
    //    bool reorder = true;
    //    int  rcdirection = 0;
    //    public:
    //        void setAllToAll( bool value ) { alltoall = value; }
    //        void setPencils( bool value ) { pencils = value; }
    //        void setReorder( bool value ) { reorder = value; }
    //        void setRCDirection( int value ) { rcdirection = value; }
    //        bool getAllToAll() const { return alltoall; }
    //        bool getPencils() const { return pencils; }
    //        bool getReorder() const { return reorder; }
    //        int  getRCDirection() const { return rcdirection; }
    //};

    namespace detail {

        template <class T>
        struct HeffteBackendType {};

#ifdef Heffte_ENABLE_FFTW
        template <>
        struct HeffteBackendType<float> {
            using backend = heffte::backend::fftw;
            using complexType = std::complex<float>;
        };
        template <>
        struct HeffteBackendType<double> {
            using backend = heffte::backend::fftw;
            using complexType = std::complex<double>;
        };
#endif
#ifdef Heffte_ENABLE_MKL
        template <>
        struct HeffteBackendType<float> {
            using backend = heffte::backend::mkl;
            using complexType = std::complex<float>;
        };
        template <>
        struct HeffteBackendType<double> {
            using backend = heffte::backend::mkl;
            using complexType = std::complex<double>;
        };
#endif
#ifdef Heffte_ENABLE_CUDA
#ifdef KOKKOS_ENABLE_CUDA
        template <>
        struct HeffteBackendType<double> {
            using backend = heffte::backend::cufft;
            using complexType = cufftDoubleComplex;
        };
        template <>
        struct HeffteBackendType<float> {
            using backend = heffte::backend::cufft;
            using complexType = cufftComplex;
        };
#endif
#endif
    }

    /**
       Non-specialized FFT class.  We specialize based on Transform tag class
    */
    template <class Transform, size_t Dim, class T>
    class FFT {};

    /**
       complex-to-complex FFT class
    */
    template <size_t Dim, class T>
    class FFT<CCTransform,Dim,T> {

    public:

        typedef FieldLayout<Dim> Layout_t;
        typedef Kokkos::complex<T> Complex_t;
        typedef Field<Complex_t,Dim> ComplexField_t;

        using heffteBackend = typename detail::HeffteBackendType<T>::backend;
        using heffteComplex_t = typename detail::HeffteBackendType<T>::complexType;
        using workspace_t = typename heffte::fft3d<heffteBackend>::template buffer_container<heffteComplex_t>;

        /** Create a new FFT object with the layout for the input Field and
         * parameters for heffte.
        */
        FFT(const Layout_t& layout, const ParameterList& params);

        // Destructor
        ~FFT() = default;

        /** Do the inplace FFT: specify +1 or -1 to indicate forward or inverse
            transform. The output is over-written in the input.
        */
        void transform(int direction, ComplexField_t& f);


    private:
        //using long long = detail::long long;

        /**
           setup performs the initialization necessary.
        */
        void setup(const std::array<long long, Dim>& low,
                   const std::array<long long, Dim>& high,
                   const ParameterList& params);

        std::shared_ptr<heffte::fft3d<heffteBackend, long long>> heffte_m;
        workspace_t workspace_m;

    };


    /**
       real-to-complex FFT class
    */
    template <size_t Dim, class T>
    class FFT<RCTransform,Dim,T> {

    public:

        typedef FieldLayout<Dim> Layout_t;
        typedef Field<T,Dim> RealField_t;

        using heffteBackend = typename detail::HeffteBackendType<T>::backend;
        using heffteComplex_t = typename detail::HeffteBackendType<T>::complexType;
        using workspace_t = typename heffte::fft3d_r2c<heffteBackend>::template buffer_container<heffteComplex_t>;

        typedef Kokkos::complex<T> Complex_t;
        //typedef heffteComplex_t Complex_t;
        typedef Field<Complex_t,Dim> ComplexField_t;

        /** Create a new FFT object with the layout for the input and output Fields
         * and parameters for heffte.
        */
        FFT(const Layout_t& layoutInput, const Layout_t& layoutOutput, const ParameterList& params);


        ~FFT() = default;

        /** Do the FFT: specify +1 or -1 to indicate forward or inverse
            transform.
        */
        void transform(int direction, RealField_t& f, ComplexField_t& g);


    private:
        //using long long = detail::long long;

        /**
           setup performs the initialization necessary after the transform
           directions have been specified.
        */
        void setup(const std::array<long long, Dim>& lowInput,
                   const std::array<long long, Dim>& highInput,
                   const std::array<long long, Dim>& lowOutput,
                   const std::array<long long, Dim>& highOutput,
                   const ParameterList& params);


        std::shared_ptr<heffte::fft3d_r2c<heffteBackend, long long>> heffte_m;
        workspace_t workspace_m;

    };

}
#include "FFT/FFT.hpp"
#endif // IPPL_FFT_FFT_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
