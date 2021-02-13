//
// IPPL FFT
//
// Copyright (c) 2008-2018
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved.
//
// OPAL is licensed under GNU GPL version 3.
//

/**
   The FFT class performs complex-to-complex, real-to-complex on IPPL Fields. 
   FFT is templated on the type of transform to be performed, 
   the dimensionality of the Field to transform, and the
   floating-point precision type of the Field (float or double).
*/

#ifndef IPPL_FFT_FFT_H
#define IPPL_FFT_FFT_H

//#include "FFT/Kokkos_FFTBase.h"
#include <heffte_fft3d.h>
#include <array>
#include <memory>
#include <type_traits>

// forward declarations
//template <unsigned Dim> class FieldLayout;
#include "FieldLayout/FieldLayout.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh, class Cell> class Field;
    
    /**
       Tag classes for CC type of Fourier transforms
    */
    class CCTransform {};
    /**
       Tag classes for RC type of Fourier transforms
    */
    class RCTransform {};
    
    /**
       Tag classes for FFT scalings
    */
    
    //class FFTScaleFull {};
    //
    //class FFTScaleNone {};
    //
    //class FFTScaleSymmetric {};

    class HeffteParams {
        bool alltoall = true;
        bool pencils = true;
        bool reorder = true;
        int  rcdirection = 0;
        public:
            void setAllToAll( bool value ) { alltoall = value; }
            void setPencils( bool value ) { pencils = value; }
            void setReorder( bool value ) { reorder = value; }
            void setRCDirection( int value ) { rcdirection = value; }
            bool getAllToAll() const { return alltoall; }
            bool getPencils() const { return pencils; }
            bool getReorder() const { return reorder; }
            int  getRCDirection() const { return rcdirection; }
    };

    namespace detail {
    //    template <class ScaleType>
    //    class HeffteScaling {};
    //    
    //    template <>
    //    class HeffteScaling<FFTScaleNone> {
    //        static const auto scaling_type = heffte::scale::none;
    //    };
    //    
    //    template <>
    //    class HeffteScaling<FFTScaleFull> {
    //        static const auto scaling_type = heffte::scale::full;
    //    };

    //    template <>
    //    class HeffteScaling<FFTScaleSymmetric> {
    //        static const auto scaling_type = heffte::scale::symmetric;
    //    };

        template <typename>
        struct isCudaComplexT : public std::false_type {};
#ifdef KOKKOS_ENABLE_CUDA
        template <>
        struct isCudaComplexT<cufftComplex> : public std::true_type {};
        
        template <>
        struct isCudaComplexT<cufftDoubleComplex> : public std::true_type {};
#endif
        template <class T>
        struct isCudaComplex : public isCudaComplexT<typename std::remove_cv<T>::type>::type
        {
        };
          
        template <typename>
        struct isStdComplexT : public std::false_type {};
        
        template <class T>
        struct isStdComplexT<std::complex<T>> : public std::true_type {};
        
        template <class T>
        struct isStdComplex : public isStdComplexT<typename std::remove_cv<T>::type>::type
        {
        };
        
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
        struct HeffteBackendType<float> {
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
    //class FFT : public FFTBase<Dim,T> {};
    class FFT {};
    
    /**
       complex-to-complex FFT class
    */
    template <size_t Dim, class T>
    //class FFT<CCTransform,Dim,T> : public FFTBase<Dim,T> {
    class FFT<CCTransform,Dim,T> {
    
    public:
    
        typedef FieldLayout<Dim> Layout_t;
        typedef Kokkos::complex<T> Complex_t;
        //typedef std::complex<T> Complex_t;
        typedef Field<Complex_t,Dim> ComplexField_t;
        //typedef typename FFTBase<Dim,T>::Domain_t Domain_t;

        using heffteBackend = typename detail::HeffteBackendType<T>::backend;
        using heffteComplex_t = typename detail::HeffteBackendType<T>::complexType;
        

        /** Create a new FFT object with the layout for the input Field and parameters
         *  for heffte.
        */
        FFT(const Layout_t& layout, const HeffteParams& params);
    
    
        // Destructor
        //~FFT(void);
        ~FFT() = default;
//#ifdef KOKKOS_ENABLE_CUDA
//        KOKKOS_INLINE_FUNCTION void 
//        copyFromKokkosComplex( Complex_t& fVal, heffteComplex_t& tempFieldVal )
//        {
//            tempFieldVal.x = fVal.real();
//            tempFieldVal.y = fVal.imag();
//            //return tempFieldVal;
//        }
//
//        KOKKOS_INLINE_FUNCTION void 
//        copyToKokkosComplex( heffteComplex_t& tempFieldVal, Complex_t& fVal )
//        {
//            fVal.real() = tempFieldVal.x;
//            fVal.imag() = tempFieldVal.y;
//            //return fVal;
//        }
//#endif
    
        /** Do the FFT: specify +1 or -1 to indicate forward or inverse
            transform, or specify the user-defined name string for the direction.
        */
        void transform(int direction, ComplexField_t& f); 
        /**
           invoke using string for direction name
        */
        //void transform(const char* directionName, ComplexField_t& f);
    
    
    private:
    
        /**
           setup performs all the initializations necessary after the transform
           directions have been specified.
        */
        void setup(const std::array<int, Dim>& low, 
                   const std::array<int, Dim>& high,
                   const HeffteParams& params);
    
        //template <class ComplexType>
        //KOKKOS_INLINE_FUNCTION ComplexType 
        //copyFromKokkosComplex( Complex_t fVal, ComplexType tempFieldVal,
        //                       typename std::enable_if<( detail::isCudaComplex<ComplexType>::value ),
        //                       int>::type* = 0 )
        //{
        //    tempFieldVal.x = fVal.real();
        //    tempFieldVal.y = fVal.imag();
        //    return tempFieldVal;
        //}

        //template <class ComplexType>
        //KOKKOS_INLINE_FUNCTION Complex_t 
        //copyToKokkosComplex( ComplexType tempFieldVal, Complex_t fVal,
        //                       typename std::enable_if<( detail::isCudaComplex<ComplexType>::value ),
        //                       int>::type* = 0 )
        //{
        //    fVal.real() = tempFieldVal.x;
        //    fVal.imag() = tempFieldVal.y;
        //    return fVal;
        //}
        
        //template <class ComplexType>
        //KOKKOS_INLINE_FUNCTION ComplexType 
        //copyFromKokkosComplex( Complex_t fVal, ComplexType tempFieldVal,
        //                       typename std::enable_if<( detail::isStdComplex<ComplexType>::value ),
        //                       int>::type* = 0 )
        //{
        //    tempFieldVal.real( fVal.real() );
        //    tempFieldVal.imag( fVal.imag() );
        //    return tempFieldVal;
        //}
        //template <class ComplexType>
        //KOKKOS_INLINE_FUNCTION Complex_t 
        //copyToKokkosComplex( ComplexType tempFieldVal, Complex_t fVal,
        //                       typename std::enable_if<( detail::isStdComplex<ComplexType>::value ),
        //                       int>::type* = 0 )
        //{
        //    fVal.real() = tempFieldVal.real();
        //    fVal.imag() = tempFieldVal.imag();
        //    return fVal;
        //}

        std::shared_ptr<heffte::fft3d<heffteBackend>> heffte_m;
        //Kokkos::View<heffteComplex_t*> tempField_m;
        //Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight> tempField_m;
        //Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight> tempField_m;
    
    };
    
    template <class T, class... Params>
    Kokkos::View<T***, Params..., Kokkos::MemoryUnmanaged>
    createView( const std::array<int, 3>& length, T* data );

    /**
       real-to-complex FFT class
    */
    template <size_t Dim, class T>
    class FFT<RCTransform,Dim,T> {
    
    public:
    
        typedef FieldLayout<Dim> Layout_t;
        typedef Kokkos::complex<T> Complex_t;
        typedef Field<T,Dim> RealField_t;
        typedef Field<Complex_t,Dim> ComplexField_t;

        using heffteBackend = typename detail::HeffteBackendType<T>::backend;
        using heffteComplex_t = typename detail::HeffteBackendType<T>::complexType;
        

        /** Create a new FFT object with the layout for the input and output Fields and parameters
         *  for heffte.
        */
        FFT(const Layout_t& layoutInput, const Layout_t& layoutOutput, const HeffteParams& params);
    
    
        ~FFT() = default;
    
        /** Do the FFT: specify +1 or -1 to indicate forward or inverse
            transform.
        */
        void transform(int direction, RealField_t& f, ComplexField_t& g); 
    
    
    private:
    
        /**
           setup performs the initialization necessary after the transform
           directions have been specified.
        */
        void setup(const std::array<int, Dim>& lowInput, 
                   const std::array<int, Dim>& highInput,
                   const std::array<int, Dim>& lowOutput, 
                   const std::array<int, Dim>& highOutput,
                   const HeffteParams& params);
    

        std::shared_ptr<heffte::fft3d_r2c<heffteBackend>> heffte_m;
    
    };

}
#include "FFT/Kokkos_FFT.hpp"
#endif // IPPL_FFT_FFT_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
