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

#include "FFT/Kokkos_FFTBase.h"

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
        public:
            void setAllToAll( bool value ) { alltoall = value; }
            void setPencils( bool value ) { pencils = value; }
            void setReorder( bool value ) { reorder = value; }
            bool getAllToAll() const { return alltoall; }
            bool getPencils() const { return pencils; }
            bool getReorder() const { return reorder; }
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

        template <class T>
        struct HeffteBackendType {};

#ifdef Heffte_ENABLE_FFTW
        template <class T>
        struct HeffeBackendType<T> {
            using backend = heffte::backend::fftw;
            using complexType = std::complex<T>;
        }
#endif
#ifdef Heffte_ENABLE_MKL
        template <class T>
        struct HeffeBackendType<T> {
            using backend = heffte::backend::mkl;
            using complexType = std::complex<T>;
        }
#endif
#ifdef Heffte_ENABLE_CUDA
#ifdef Kokkos_ENABLE_CUDA
        template <>
        struct HeffeBackendType<double> {
            using backend = heffte::backend::cufft;
            using complexType = cufftDoubleComplex;
        }
        template <>
        struct HeffeBackendType<float> {
            using backend = heffte::backend::cufft;
            using complexType = cufftComplex;
        }
#endif
#endif
    }

    /**
       Non-specialized FFT class.  We specialize based on Transform tag class
    */
    template <class Transform, size_t Dim, class T>
    class FFT : public FFTBase<Dim,T> {};
    
    /**
       complex-to-complex FFT class
    */
    template <size_t Dim, class T>
    class FFT<CCTransform,Dim,T> : public FFTBase<Dim,T> {
    
    public:
    
        typedef FieldLayout<Dim> Layout_t;
        typedef Kokkos::complex<T> Complex_t;
        typedef Field<Complex_t,Dim> ComplexField_t;
        //typedef typename FFTBase<Dim,T>::Domain_t Domain_t;

        using heffteBackend = detail::HeffteBackendType<T>::backend;
        using heffteComplex_t = detail::HeffteBackendType<T>::complexType;
        

        /** Create a new FFT object with the layout for the input Field and parameters
         *  for heffte.
        */
        FFT(const Layout_t& layout, const HeffteParams& params);
    
    
        // Destructor
        ~FFT(void);
    
        /** Do the FFT: specify +1 or -1 to indicate forward or inverse
            transform, or specify the user-defined name string for the direction.
        */
        void transform(int direction, ComplexField_t& f);
        /**
           invoke using string for direction name
        */
        void transform(const char* directionName, ComplexField_t& f);
    
    
    private:
    
        /**
           setup performs all the initializations necessary after the transform
           directions have been specified.
        */
        void setup(void);
    
        /**
           How the temporary field's are laid out; these are computed from the
           input Field's domain. This will be allocated as an array of FieldLayouts
           with nTransformDims elements. Each is SERIAL along the zeroth dimension
           and the axes are permuted so that the transform direction is first
        */
        //Layout_t** tempLayouts_m;
    
        /** The array of temporary fields, one for each transform direction
            These use the corresponding tempLayouts.
        */
        //ComplexField_t** tempFields_m;

         std::shared_ptr<heffte::fft3d<heffteBackend>> heffte_m;
         Kokkos::View<heffteComplex_t*> tempField_m;
    
    };
    
    
    /**
       invoke CC transform using direction name string
    */
    template <size_t Dim, class T>
    inline void
    FFT<CCTransform,Dim,T>::transform(
        const char* directionName,
        typename FFT<CCTransform,Dim,T>::ComplexField_t& f)
    {
        int dir = this->getDirection(directionName);
        transform(dir, f);
        return;
    }
    
    
    ///**
    //   1D complex-to-complex FFT class
    //*/
    //template <class T>
    //class FFT<CCTransform,1U,T> : public FFTBase<1U,T> {
    //
    //public:
    //
    //    // typedefs
    //    typedef FieldLayout<1U> Layout_t;
    //    typedef std::complex<T> Complex_t;
    //    typedef BareField<Complex_t,1U> ComplexField_t;
    //    typedef LField<Complex_t,1U> ComplexLField_t;
    //    typedef typename FFTBase<1U,T>::Domain_t Domain_t;
    //
    //    // Constructors:
    //
    //    /** Create a new FFT object with the given domain for the input Field.
    //        Specify which dimensions to transform along.
    //        Optional argument compressTemps indicates whether or not to compress
    //        temporary Fields in between uses.
    //    */
    //    FFT(const Domain_t& cdomain, const bool transformTheseDims[1U],
    //        const bool& compressTemps=false);
    //    /** Create a new FFT object with the given domain for the input Field.
    //        Transform along all dimensions.
    //        Optional argument compressTemps indicates whether or not to compress
    //        temporary Fields in between uses.
    //
    //    */
    //    FFT(const Domain_t& cdomain, const bool& compressTemps=false);
    //
    //    // Destructor
    //    ~FFT(void);
    //
    //    /** Do the FFT: specify +1 or -1 to indicate forward or inverse
    //        transform, or specify the user-defined name string for the direction.
    //        User provides separate input and output fields
    //        optional argument constInput indicates whether or not to treat the
    //        input Field argument f as const.  If not, we can use it as a temporary
    //        in order to avoid an additional data transpose.
    //    */
    //    void transform(int direction, ComplexField_t& f, ComplexField_t& g,
    //                   const bool& constInput=false);
    //    /**
    //       invoke using string for direction name
    //    */
    //    void transform(const char* directionName, ComplexField_t& f,
    //                   ComplexField_t& g, const bool& constInput=false);
    //
    //    /**
    //       overloaded versions which perform the FFT "in place"
    //    */
    //    void transform(int direction, ComplexField_t& f);
    //    void transform(const char* directionName, ComplexField_t& f);
    //
    //private:
    //
    //    /**
    //       setup performs all the initializations necessary after the transform
    //       directions have been specified.
    //    */
    //    void setup(void);
    //
    //    /**
    //       The temporary field layout
    //    */
    //    Layout_t* tempLayouts_m;
    //
    //    /**
    //       The temporary field
    //    */
    //    ComplexField_t* tempFields_m;
    //
    //};
    //
    //
    //// inline function definitions
    //
    ///**
    //   invoke two-field transform function using direction name string
    //*/
    //template <class T>
    //inline void
    //FFT<CCTransform,1U,T>::transform(
    //    const char* directionName,
    //    typename FFT<CCTransform,1U,T>::ComplexField_t& f,
    //    typename FFT<CCTransform,1U,T>::ComplexField_t& g,
    //    const bool& constInput)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f, g, constInput);
    //    return;
    //}
    //
    ///**
    //   invoke in-place transform function using direction name string
    //*/
    //template <class T>
    //inline void
    //FFT<CCTransform,1U,T>::transform(
    //    const char* directionName,
    //    typename FFT<CCTransform,1U,T>::ComplexField_t& f)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f);
    //    return;
    //}
    
    
    ///**
    //   real-to-complex FFT class
    //*/
    //template <size_t Dim, class T>
    //class FFT<RCTransform,Dim,T> : public FFTBase<Dim,T> {
    //
    //private:
    //
    //public:
    //
    //    // typedefs
    //    typedef FieldLayout<Dim> Layout_t;
    //    typedef BareField<T,Dim> RealField_t;
    //    typedef LField<T,Dim> RealLField_t;
    //    typedef std::complex<T> Complex_t;
    //    typedef BareField<Complex_t,Dim> ComplexField_t;
    //    typedef LField<Complex_t,Dim> ComplexLField_t;
    //    typedef typename FFTBase<Dim, T>::Domain_t Domain_t;
    //
    //    // Constructors:
    //
    //    /** Create a new FFT object with the given domains for input/output Fields
    //        Specify which dimensions to transform along.
    //        Optional argument compress indicates whether or not to compress
    //        temporary Fields in between uses.
    //    */
    //    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
    //        const bool transformTheseDims[Dim], const bool& compressTemps=false);
    //
    //    /**
    //       Same as above, but transform all dims:
    //    */
    //    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
    //        const bool& compressTemps=false, int serialAxes = 1);
    //
    //    // Destructor
    //    ~FFT(void);
    //
    //    /** real-to-complex FFT: specify +1 or -1 to indicate forward or inverse
    //        transform, or specify the user-defined name string for the direction.
    //        Supply a second BareField to store the output.
    //        optional argument constInput indicates whether or not to treat the
    //        input Field argument f as const.  If not, we can use it as a temporary
    //        in order to avoid an additional data transpose.
    //    */
    //    void transform(int direction, RealField_t& f, ComplexField_t& g,
    //                   const bool& constInput=false);
    //    void transform(const char* directionName, RealField_t& f,
    //                   ComplexField_t& g, const bool& constInput=false);
    //
    //    /** real-to-complex FFT on GPU: transfer the real field to GPU execute FFT
    //        return the pointer to memory on GPU where complex results are stored
    //    */
    //    /** complex-to-real FFT
    //        Same as above, but with input and output field types reversed.
    //    */
    //    void transform(int direction, ComplexField_t& f, RealField_t& g,
    //                   const bool& constInput=false);
    //    void transform(const char* directionName, ComplexField_t& f,
    //                   RealField_t& g, const bool& constInput=false);
    //
    //    /** complex-to-real FFT on GPU: pass pointer to GPU memory where complex field
    //        is stored, do the inverse FFT and transfer real field back to host memory
    //    */
    //
    //private:
    //
    //    /**
    //       setup performs all the initializations necessary after the transform
    //       directions have been specified.
    //    */
    //    void setup(void);
    //
    //    /** How the temporary fields are laid out; these are computed from the
    //        input Field's domain. This will be allocated as an array of FieldLayouts
    //        with nTransformDims elements. Each is SERIAL along the zeroth dimension
    //        and the axes are permuted so that the transform direction is first
    //    */
    //    Layout_t** tempLayouts_m;
    //
    //    /**
    //       extra layout for the one real Field needed
    //    */
    //    Layout_t* tempRLayout_m;
    //
    //    /** The array of temporary fields, one for each transform direction
    //        These use the corresponding tempLayouts.
    //    */
    //    ComplexField_t** tempFields_m;
    //
    //    /**
    //       We need one real internal Field in this case.
    //    */
    //    RealField_t* tempRField_m;
    //
    //    /**
    //       domain of the resulting complex fields
    //       const Domain_t& complexDomain_m;
    //    */
    //    Domain_t complexDomain_m;
    //
    //    /**
    //       number of axes to make serial
    //    */
    //    int serialAxes_m;
    //};
    //
    //// Inline function definitions
    //
    ///**
    //   invoke real-to-complex transform using string for transform direction
    //*/
    //template <size_t Dim, class T>
    //inline void
    //FFT<RCTransform,Dim,T>::transform(
    //    const char* directionName,
    //    typename FFT<RCTransform,Dim,T>::RealField_t& f,
    //    typename FFT<RCTransform,Dim,T>::ComplexField_t& g,
    //    const bool& constInput)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f, g, constInput);
    //    return;
    //}
    //
    ///**
    //   invoke complex-to-real transform using string for transform direction
    //*/
    //template <size_t Dim, class T>
    //inline void
    //FFT<RCTransform,Dim,T>::transform(
    //    const char* directionName,
    //    typename FFT<RCTransform,Dim,T>::ComplexField_t& f,
    //    typename FFT<RCTransform,Dim,T>::RealField_t& g,
    //    const bool& constInput)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f, g, constInput);
    //    return;
    //}
    //
    //
    ///**
    //   1D real-to-complex FFT class
    //*/
    //template <class T>
    //class FFT<RCTransform,1U,T> : public FFTBase<1U,T> {
    //
    //public:
    //
    //    // typedefs
    //    typedef FieldLayout<1U> Layout_t;
    //    typedef BareField<T,1U> RealField_t;
    //    typedef LField<T,1U> RealLField_t;
    //    typedef std::complex<T> Complex_t;
    //    typedef BareField<Complex_t,1U> ComplexField_t;
    //    typedef LField<Complex_t,1U> ComplexLField_t;
    //    typedef typename FFTBase<1U,T>::Domain_t Domain_t;
    //
    //    // Constructors:
    //
    //    /**
    //       Create a new FFT object with the given domains for input/output Fields
    //       Specify which dimensions to transform along.
    //       Optional argument compress indicates whether or not to compress
    //       temporary Fields in between uses.
    //    */
    //    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
    //        const bool transformTheseDims[1U], const bool& compressTemps=false);
    //    /**
    //       Same as above, but transform all dims:
    //    */
    //    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
    //        const bool& compressTemps=false);
    //
    //    /**
    //       Destructor
    //    */
    //    ~FFT(void);
    //
    //    /**
    //       real-to-complex FFT: specify +1 or -1 to indicate forward or inverse
    //       transform, or specify the user-defined name string for the direction.
    //       Supply a second BareField to store the output.
    //       optional argument constInput indicates whether or not to treat the
    //       input Field argument f as const.  If not, we can use it as a temporary
    //       in order to avoid an additional data transpose.
    //    */
    //    void transform(int direction, RealField_t& f, ComplexField_t& g,
    //                   const bool& constInput=false);
    //    void transform(const char* directionName, RealField_t& f,
    //                   ComplexField_t& g, const bool& constInput=false);
    //
    //    /**
    //       complex-to-real FFT
    //       Same as above, but with input and output field types reversed.
    //    */
    //    void transform(int direction, ComplexField_t& f, RealField_t& g,
    //                   const bool& constInput=false);
    //    void transform(const char* directionName, ComplexField_t& f,
    //                   RealField_t& g, const bool& constInput=false);
    //
    //private:
    //
    //    /**
    //       setup performs all the initializations necessary after the transform
    //       directions have been specified.
    //    */
    //    void setup(void);
    //
    //    /**
    //       The temporary field layout
    //    */
    //    Layout_t* tempLayouts_m;
    //
    //    /**
    //       The temporary field
    //    */
    //    ComplexField_t* tempFields_m;
    //
    //    /**
    //       Real field layout
    //    */
    //    Layout_t* tempRLayout_m;
    //
    //    /**
    //       We need one real internal Field in this case.
    //       domain of the resulting complex fields
    //    */
    //    //  const Domain_t& complexDomain_m;
    //    Domain_t complexDomain_m;
    //};
    //
    ///**
    //   invoke real-to-complex transform using string for transform direction
    //*/
    //template <class T>
    //inline void
    //FFT<RCTransform,1U,T>::transform(
    //    const char* directionName,
    //    typename FFT<RCTransform,1U,T>::RealField_t& f,
    //    typename FFT<RCTransform,1U,T>::ComplexField_t& g,
    //    const bool& constInput)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f, g, constInput);
    //    return;
    //}
    //
    ///**
    //   invoke complex-to-real transform using string for transform direction
    //*/
    //template <class T>
    //inline void
    //FFT<RCTransform,1U,T>::transform(
    //    const char* directionName,
    //    typename FFT<RCTransform,1U,T>::ComplexField_t& f,
    //    typename FFT<RCTransform,1U,T>::RealField_t& g,
    //    const bool& constInput)
    //{
    //    int dir = this->getDirection(directionName);
    //    transform(dir, f, g, constInput);
    //    return;
    //}
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
