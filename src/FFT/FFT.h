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
   The FFT class performs complex-to-complex, real-to-complex, or sine
   transforms on IPPL Fields.  FFT is templated on the type of transform
   to be performed, the dimensionality of the Field to transform, and the
   floating-point precision type of the Field (float or double).
*/

#ifndef IPPL_FFT_FFT_H
#define IPPL_FFT_FFT_H

#include "FFT/FFTBase.h"

// forward declarations
//template <unsigned Dim> class FieldLayout;
#include "FieldLayout/FieldLayout.h"
template <class T, unsigned Dim> class BareField;
template <class T, unsigned Dim> class LField;



/**
   Tag classes for CC type of Fourier transforms
*/
class CCTransform {};
/**
   Tag classes for RC type of Fourier transforms
*/
class RCTransform {};
/**
   Tag classes for sine types of Fourier transforms
*/
class SineTransform {};

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
    typedef std::complex<T> Complex_t;
    typedef BareField<Complex_t,Dim> ComplexField_t;
    typedef LField<Complex_t,Dim> ComplexLField_t;
    typedef typename FFTBase<Dim,T>::Domain_t Domain_t;

    /** Create a new FFT object with the given domain for the input Field.
        Specify which dimensions to transform along.
        Optional argument compressTemps indicates whether or not to compress
        temporary Fields in between uses.
    */
    FFT(const Domain_t& cdomain, const bool transformTheseDims[Dim],
        const bool& compressTemps=false);

    /**
       Create a new FFT object of type CCTransform, with a
       given domain. Default case of transforming along all dimensions.
       Note this was formerly in the .cpp file, but the IBM linker
       could not find it!
    */
FFT(const Domain_t& cdomain, const bool& compressTemps=false)
    : FFTBase<Dim,T>(FFT<CCTransform,Dim,T>::ccFFT, cdomain,compressTemps) {

        // construct array of axis lengths
        int lengths[Dim];
        size_t d;
        for (d=0; d<Dim; ++d)
            lengths[d] = cdomain[d].length();

        // construct array of transform types for FFT Engine, compute normalization
        int transformTypes[Dim];
        T& normFact = this->getNormFact();
        normFact = 1.0;
        for (d=0; d<Dim; ++d) {
            transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all transforms are complex-to-complex
            normFact /= lengths[d];
        }

        // set up FFT Engine
        this->getEngine().setup(Dim, transformTypes, lengths);
        // set up the temporary fields
        setup();
    }


    // Destructor
    ~FFT(void);

    /** Do the FFT: specify +1 or -1 to indicate forward or inverse
        transform, or specify the user-defined name string for the direction.
        User provides separate input and output fields
        optional argument constInput indicates whether or not to treat the
        input Field argument f as const.  If not, we can use it as a temporary
        in order to avoid an additional data transpose.
    */
    void transform(int direction, ComplexField_t& f, ComplexField_t& g,
                   const bool& constInput=false);
    /**
       invoke using string for direction name
    */
    void transform(const char* directionName, ComplexField_t& f,
                   ComplexField_t& g, const bool& constInput=false);

    /** overloaded versions which perform the FFT "in place"
     */
    void transform(int direction, ComplexField_t& f);

    void transform(const char* directionName, ComplexField_t& f) {
        // invoke in-place transform function using direction name string
        int direction = this->getDirection(directionName);

        // Check domain of incoming Field
        const Layout_t& in_layout = f.getLayout();
        const Domain_t& in_dom = in_layout.getDomain();
        PAssert_EQ(this->checkDomain(this->getDomain(),in_dom), true);

        // Common loop iterate and other vars:
        size_t d;
        int idim;            // idim loops over the number of transform dims.
        int begdim, enddim;  // beginning and end of transform dim loop
        size_t nTransformDims = this->numTransformDims();
        // Field* for temp Field management:
        ComplexField_t* temp = &f;
        // Local work array passed to FFT:
        Complex_t* localdata;

        // Loop over the dimensions be transformed:
        begdim = (direction == +1) ? 0 : (nTransformDims-1);
        enddim = (direction == +1) ? nTransformDims : -1;
        for (idim = begdim; idim != enddim; idim += direction) {
      
            // Now do the serial transforms along this dimension:

            bool skipTranspose = false;
            // if this is the first transform dimension, we might be able
            // to skip the transpose into the first temporary Field
            if (idim == begdim) {
                // get domain for comparison
                const Domain_t& first_dom = tempLayouts_m[idim]->getDomain();
                // check that zeroth axis is the same and is serial
                // and that there are no guard cells
                skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                                  (in_dom[0].length() == first_dom[0].length()) &&
                                  (in_layout.getDistribution(0) == SERIAL) &&
                                  (f.getGC() == FFT<CCTransform,Dim,T>::nullGC) );
            }

            // if this is the last transform dimension, we might be able
            // to skip the last temporary and transpose right into f
            if (idim == enddim-direction) {
                // get domain for comparison
                const Domain_t& last_dom = tempLayouts_m[idim]->getDomain();
                // check that zeroth axis is the same and is serial
                // and that there are no guard cells
                skipTranspose = ( (in_dom[0].sameBase(last_dom[0])) &&
                                  (in_dom[0].length() == last_dom[0].length()) &&
                                  (in_layout.getDistribution(0) == SERIAL) &&
                                  (f.getGC() == FFT<CCTransform,Dim,T>::nullGC) );
            }

            if (!skipTranspose) {
                // transpose and permute to Field with transform dim first
                (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                    (*temp)[temp->getLayout().getDomain()];

                // Compress out previous iterate's storage:
                if (this->compressTemps() && temp != &f) *temp = 0;
                temp = tempFields_m[idim];  // Field* management aid
            }
            else if (idim == enddim-direction && temp != &f) {
                // last transform and we can skip the last temporary field
                // so do the transpose here using f instead

                // transpose and permute to Field with transform dim first
                f[in_dom] = (*temp)[temp->getLayout().getDomain()];

                // Compress out previous iterate's storage:
                if (this->compressTemps()) *temp = 0;
                temp = &f;  // Field* management aid
            }
      

      
            // Loop over all the Vnodes, working on the LField in each.
            typename ComplexField_t::const_iterator_if l_i, l_end = temp->end_if();
            for (l_i = temp->begin_if(); l_i != l_end; ++l_i) {

                // Get the LField
                ComplexLField_t* ldf = (*l_i).second.get();
                // make sure we are uncompressed
                ldf->Uncompress();
                // get the raw data pointer
                localdata = ldf->getP();

                // Do 1D complex-to-complex FFT's on all the strips in the LField:
                int nstrips = 1, length = ldf->size(0);
                for (d=1; d<Dim; ++d) nstrips *= ldf->size(d);
                for (int istrip=0; istrip<nstrips; ++istrip) {
                    // Do the 1D FFT:
                    this->getEngine().callFFT(idim, direction, localdata);
                    // advance the data pointer
                    localdata += length;
                } // loop over 1D strips
            } // loop over all the LFields
      

        } // loop over all transformed dimensions

        // skip final assignment and compress if we used f as final temporary
        if (temp != &f) {
      
            // Now assign back into original Field, and compress last temp's storage:
            f[in_dom] = (*temp)[temp->getLayout().getDomain()];
            if (this->compressTemps()) *temp = 0;
      
        }

        // Normalize:
        if (direction == +1)
            f *= Complex_t(this->getNormFact(), 0.0);
        return;
    }
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
    Layout_t** tempLayouts_m;

    /** The array of temporary fields, one for each transform direction
        These use the corresponding tempLayouts.
    */
    ComplexField_t** tempFields_m;

};


/**
   invoke two-field transform function using direction name string
*/
template <size_t Dim, class T>
inline void
FFT<CCTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<CCTransform,Dim,T>::ComplexField_t& f,
    typename FFT<CCTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}


/**
   1D complex-to-complex FFT class
*/
template <class T>
class FFT<CCTransform,1U,T> : public FFTBase<1U,T> {

public:

    // typedefs
    typedef FieldLayout<1U> Layout_t;
    typedef std::complex<T> Complex_t;
    typedef BareField<Complex_t,1U> ComplexField_t;
    typedef LField<Complex_t,1U> ComplexLField_t;
    typedef typename FFTBase<1U,T>::Domain_t Domain_t;

    // Constructors:

    /** Create a new FFT object with the given domain for the input Field.
        Specify which dimensions to transform along.
        Optional argument compressTemps indicates whether or not to compress
        temporary Fields in between uses.
    */
    FFT(const Domain_t& cdomain, const bool transformTheseDims[1U],
        const bool& compressTemps=false);
    /** Create a new FFT object with the given domain for the input Field.
        Transform along all dimensions.
        Optional argument compressTemps indicates whether or not to compress
        temporary Fields in between uses.

    */
    FFT(const Domain_t& cdomain, const bool& compressTemps=false);

    // Destructor
    ~FFT(void);

    /** Do the FFT: specify +1 or -1 to indicate forward or inverse
        transform, or specify the user-defined name string for the direction.
        User provides separate input and output fields
        optional argument constInput indicates whether or not to treat the
        input Field argument f as const.  If not, we can use it as a temporary
        in order to avoid an additional data transpose.
    */
    void transform(int direction, ComplexField_t& f, ComplexField_t& g,
                   const bool& constInput=false);
    /**
       invoke using string for direction name
    */
    void transform(const char* directionName, ComplexField_t& f,
                   ComplexField_t& g, const bool& constInput=false);

    /**
       overloaded versions which perform the FFT "in place"
    */
    void transform(int direction, ComplexField_t& f);
    void transform(const char* directionName, ComplexField_t& f);

private:

    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    void setup(void);

    /**
       The temporary field layout
    */
    Layout_t* tempLayouts_m;

    /**
       The temporary field
    */
    ComplexField_t* tempFields_m;

};


// inline function definitions

/**
   invoke two-field transform function using direction name string
*/
template <class T>
inline void
FFT<CCTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<CCTransform,1U,T>::ComplexField_t& f,
    typename FFT<CCTransform,1U,T>::ComplexField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke in-place transform function using direction name string
*/
template <class T>
inline void
FFT<CCTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<CCTransform,1U,T>::ComplexField_t& f)
{
    int dir = this->getDirection(directionName);
    transform(dir, f);
    return;
}


/**
   real-to-complex FFT class
*/
template <size_t Dim, class T>
class FFT<RCTransform,Dim,T> : public FFTBase<Dim,T> {

private:

public:

    // typedefs
    typedef FieldLayout<Dim> Layout_t;
    typedef BareField<T,Dim> RealField_t;
    typedef LField<T,Dim> RealLField_t;
    typedef std::complex<T> Complex_t;
    typedef BareField<Complex_t,Dim> ComplexField_t;
    typedef LField<Complex_t,Dim> ComplexLField_t;
    typedef typename FFTBase<Dim, T>::Domain_t Domain_t;

    // Constructors:

    /** Create a new FFT object with the given domains for input/output Fields
        Specify which dimensions to transform along.
        Optional argument compress indicates whether or not to compress
        temporary Fields in between uses.
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool transformTheseDims[Dim], const bool& compressTemps=false);

    /**
       Same as above, but transform all dims:
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool& compressTemps=false, int serialAxes = 1);

    // Destructor
    ~FFT(void);

    /** real-to-complex FFT: specify +1 or -1 to indicate forward or inverse
        transform, or specify the user-defined name string for the direction.
        Supply a second BareField to store the output.
        optional argument constInput indicates whether or not to treat the
        input Field argument f as const.  If not, we can use it as a temporary
        in order to avoid an additional data transpose.
    */
    void transform(int direction, RealField_t& f, ComplexField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, RealField_t& f,
                   ComplexField_t& g, const bool& constInput=false);

    /** real-to-complex FFT on GPU: transfer the real field to GPU execute FFT
        return the pointer to memory on GPU where complex results are stored
    */
    /** complex-to-real FFT
        Same as above, but with input and output field types reversed.
    */
    void transform(int direction, ComplexField_t& f, RealField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, ComplexField_t& f,
                   RealField_t& g, const bool& constInput=false);

    /** complex-to-real FFT on GPU: pass pointer to GPU memory where complex field
        is stored, do the inverse FFT and transfer real field back to host memory
    */

private:

    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    void setup(void);

    /** How the temporary fields are laid out; these are computed from the
        input Field's domain. This will be allocated as an array of FieldLayouts
        with nTransformDims elements. Each is SERIAL along the zeroth dimension
        and the axes are permuted so that the transform direction is first
    */
    Layout_t** tempLayouts_m;

    /**
       extra layout for the one real Field needed
    */
    Layout_t* tempRLayout_m;

    /** The array of temporary fields, one for each transform direction
        These use the corresponding tempLayouts.
    */
    ComplexField_t** tempFields_m;

    /**
       We need one real internal Field in this case.
    */
    RealField_t* tempRField_m;

    /**
       domain of the resulting complex fields
       const Domain_t& complexDomain_m;
    */
    Domain_t complexDomain_m;

    /**
       number of axes to make serial
    */
    int serialAxes_m;
};

// Inline function definitions

/**
   invoke real-to-complex transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<RCTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<RCTransform,Dim,T>::RealField_t& f,
    typename FFT<RCTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke complex-to-real transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<RCTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<RCTransform,Dim,T>::ComplexField_t& f,
    typename FFT<RCTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}


/**
   1D real-to-complex FFT class
*/
template <class T>
class FFT<RCTransform,1U,T> : public FFTBase<1U,T> {

public:

    // typedefs
    typedef FieldLayout<1U> Layout_t;
    typedef BareField<T,1U> RealField_t;
    typedef LField<T,1U> RealLField_t;
    typedef std::complex<T> Complex_t;
    typedef BareField<Complex_t,1U> ComplexField_t;
    typedef LField<Complex_t,1U> ComplexLField_t;
    typedef typename FFTBase<1U,T>::Domain_t Domain_t;

    // Constructors:

    /**
       Create a new FFT object with the given domains for input/output Fields
       Specify which dimensions to transform along.
       Optional argument compress indicates whether or not to compress
       temporary Fields in between uses.
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool transformTheseDims[1U], const bool& compressTemps=false);
    /**
       Same as above, but transform all dims:
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool& compressTemps=false);

    /**
       Destructor
    */
    ~FFT(void);

    /**
       real-to-complex FFT: specify +1 or -1 to indicate forward or inverse
       transform, or specify the user-defined name string for the direction.
       Supply a second BareField to store the output.
       optional argument constInput indicates whether or not to treat the
       input Field argument f as const.  If not, we can use it as a temporary
       in order to avoid an additional data transpose.
    */
    void transform(int direction, RealField_t& f, ComplexField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, RealField_t& f,
                   ComplexField_t& g, const bool& constInput=false);

    /**
       complex-to-real FFT
       Same as above, but with input and output field types reversed.
    */
    void transform(int direction, ComplexField_t& f, RealField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, ComplexField_t& f,
                   RealField_t& g, const bool& constInput=false);

private:

    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    void setup(void);

    /**
       The temporary field layout
    */
    Layout_t* tempLayouts_m;

    /**
       The temporary field
    */
    ComplexField_t* tempFields_m;

    /**
       Real field layout
    */
    Layout_t* tempRLayout_m;

    /**
       We need one real internal Field in this case.
       domain of the resulting complex fields
    */
    //  const Domain_t& complexDomain_m;
    Domain_t complexDomain_m;
};

/**
   invoke real-to-complex transform using string for transform direction
*/
template <class T>
inline void
FFT<RCTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<RCTransform,1U,T>::RealField_t& f,
    typename FFT<RCTransform,1U,T>::ComplexField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke complex-to-real transform using string for transform direction
*/
template <class T>
inline void
FFT<RCTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<RCTransform,1U,T>::ComplexField_t& f,
    typename FFT<RCTransform,1U,T>::RealField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   sine transform class
*/
template <size_t Dim, class T>
class FFT<SineTransform,Dim,T> : public FFTBase<Dim,T> {

public:

    // typedefs
    typedef FieldLayout<Dim> Layout_t;
    typedef BareField<T,Dim> RealField_t;
    typedef LField<T,Dim> RealLField_t;
    typedef std::complex<T> Complex_t;
    typedef BareField<Complex_t,Dim> ComplexField_t;
    typedef LField<Complex_t,Dim> ComplexLField_t;
    typedef typename FFTBase<Dim,T>::Domain_t Domain_t;

    /** Constructor for doing sine transform(s) followed by RC FFT
        Create a new FFT object with the given domains for input/output Fields
        Specify which dimensions to transform along.
        Also specify which of these are sine transforms
        Optional argument compress indicates whether or not to compress
        temporary Fields in between uses.
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool transformTheseDims[Dim],
        const bool sineTransformDims[Dim], const bool& compressTemps=false);
    /**
       Same as above, but transform all dims:
    */
    FFT(const Domain_t& rdomain, const Domain_t& cdomain,
        const bool sineTransformDims[Dim], const bool& compressTemps=false);
    /**
       Separate constructors for doing only sine transforms
       Create a new FFT object with the given domain for input/output Field
       Specify which dimensions to transform along.
       Optional argument compress indicates whether or not to compress
       temporary Fields in between uses.
    */
    FFT(const Domain_t& rdomain, const bool sineTransformDims[Dim],
        const bool& compressTemps=false);
    /**
       Same as above, but transform all dims:
    */
    FFT(const Domain_t& rdomain, const bool& compressTemps=false);

    ~FFT(void);

    /**
       These transforms are for combinations of sine transforms and RC FFTs

       Do the FFT: specify +1 or -1 to indicate forward or inverse
       transform, or specify the user-defined name string for the direction.
       Supply a second BareField to store the output.
       optional argument constInput indicates whether or not to treat the
       input Field argument f as const.  If not, we can use it as a temporary
       in order to avoid an additional data transpose.
    */
    void transform(int direction, RealField_t& f, ComplexField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, RealField_t& f,
                   ComplexField_t& g, const bool& constInput=false);

    /**
       complex-to-real FFT, followed by sine transform(s)
       Same as above, but with input and output field types reversed.
    */
    void transform(int direction, ComplexField_t& f, RealField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, ComplexField_t& f,
                   RealField_t& g, const bool& constInput=false);

    /**
       These transforms are for doing sine transforms only
       sine transform: specify +1 or -1 to indicate forward or inverse
       transform, or specify the user-defined name string for the direction.
       Supply a second BareField to store the output.
       optional argument constInput indicates whether or not to treat the
       input Field argument f as const.  If not, we can use it as a temporary
       in order to avoid an additional data transpose.
    */
    void transform(int direction, RealField_t& f, RealField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, RealField_t& f,
                   RealField_t& g, const bool& constInput=false);

    /**
       In-place version of real-to-real transform
    */
    void transform(int direction, RealField_t& f);
    void transform(const char* directionName, RealField_t& f);

private:

    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    void setup(void);



    /**
       which dimensions are sine transformed
    */
    bool sineTransformDims_m[Dim];

    /**
       number of sine transforms to perform
    */
    size_t numSineTransforms_m;

    /**
       layouts for temporary Fields: SERIAL along the zeroth dimension, with
       the axes are permuted so that the transform direction is first

       layouts for the temporary complex Fields
    */

    Layout_t** tempLayouts_m;

    /**
       layouts for the temporary real Fields
    */
    Layout_t** tempRLayouts_m;

    /** The array of temporary complex Fields
        These use the corresponding tempLayouts.
    */
    ComplexField_t** tempFields_m;

    /** The array of temporary real Fields
        These use the corresponding tempRLayouts.
    */
    RealField_t** tempRFields_m;

    /**
       domain of the resulting complex Field for real-to-complex transform
    */
    const Domain_t* complexDomain_m;
};

/**
   invoke real-to-complex transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<SineTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,Dim,T>::RealField_t& f,
    typename FFT<SineTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke complex-to-real transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<SineTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,Dim,T>::ComplexField_t& f,
    typename FFT<SineTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke real-to-real transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<SineTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,Dim,T>::RealField_t& f,
    typename FFT<SineTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke in-place real-to-real transform using string for transform direction
*/
template <size_t Dim, class T>
inline void
FFT<SineTransform,Dim,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,Dim,T>::RealField_t& f)
{
    int dir = this->getDirection(directionName);
    transform(dir, f);
    return;
}


/**
   1D sine transform class
*/
template <class T>
class FFT<SineTransform,1U,T> : public FFTBase<1U,T> {

public:

    typedef FieldLayout<1U> Layout_t;
    typedef BareField<T,1U> RealField_t;
    typedef LField<T,1U> RealLField_t;
    typedef typename FFTBase<1U,T>::Domain_t Domain_t;

    /**
       Constructors for doing only sine transforms
       Create a new FFT object with the given domain for input/output Field
       specify which dimensions to transform along.
       Optional argument compress indicates whether or not to compress
       temporary Fields in between uses.
    */
    FFT(const Domain_t& rdomain, const bool sineTransformDims[1U],
        const bool& compressTemps=false);
    /**
       Same as above, but transform all dims:
    */
    FFT(const Domain_t& rdomain, const bool& compressTemps=false);


    ~FFT(void);

    /**
       sine transform: specify +1 or -1 to indicate forward or inverse
       transform, or specify the user-defined name string for the direction.
       Supply a second BareField to store the output.
       optional argument constInput indicates whether or not to treat the
       input Field argument f as const.  If not, we can use it as a temporary
       in order to avoid an additional data transpose.
    */
    void transform(int direction, RealField_t& f, RealField_t& g,
                   const bool& constInput=false);
    void transform(const char* directionName, RealField_t& f,
                   RealField_t& g, const bool& constInput=false);

    /**
       In-place version of real-to-real transform
    */
    void transform(int direction, RealField_t& f);
    void transform(const char* directionName, RealField_t& f);

private:

    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    void setup(void);

    /**
       The temporary real Field layout
    */
    Layout_t* tempRLayouts_m;

    /**
       The temporary real Field
    */
    RealField_t* tempRFields_m;

};

/**
   invoke real-to-real transform using string for transform direction
*/
template <class T>
inline void
FFT<SineTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,1U,T>::RealField_t& f,
    typename FFT<SineTransform,1U,T>::RealField_t& g,
    const bool& constInput)
{
    int dir = this->getDirection(directionName);
    transform(dir, f, g, constInput);
    return;
}

/**
   invoke in-place real-to-real transform using string for transform direction
*/
template <class T>
inline void
FFT<SineTransform,1U,T>::transform(
    const char* directionName,
    typename FFT<SineTransform,1U,T>::RealField_t& f)
{
    int dir = this->getDirection(directionName);
    transform(dir, f);
    return;
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
