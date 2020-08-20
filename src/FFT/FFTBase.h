//
// IPPL FFT
//
// Copyright (c) 2008-2018
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved.
//
// OPAL is licensed under GNU GPL version 3.
//

//--------------------------------------------------------------------------
// Class FFTBase
//--------------------------------------------------------------------------

#ifndef IPPL_FFT_FFTBASE_H
#define IPPL_FFT_FFTBASE_H

// include files
#include "Utility/PAssert.h"
#include "Index/NDIndex.h"
#include "Field/GuardCellSizes.h"

#include "FFT/fftpack_FFT.h"

#include <map>
#include <iostream>

// forward declarations
template <unsigned Dim, class T> class FFTBase;
template <unsigned Dim, class T>
std::ostream& operator<<(std::ostream&, const FFTBase<Dim,T>&);

/// character strings for transform types
inline 
std::string getTransformType(unsigned int i) 
{
    static const char* transformTypeString_g[4] = { "complex-to-complex FFT",
                                                    "real-to-complex FFT",
                                                    "sine transform",
                                                    "cosine transform" };

    return std::string(transformTypeString_g[i % 4]);
}

/*!
  The FFTBase class handles duties for the FFT class that do not involve
  the type of transform to be done.  FFTBase is templated on dimensionality
  of the Field to transform and the floating-point precision type of the
  Field (float or double).

  FFT Base Class to do stuff that is independent of transform type
*/
template <unsigned Dim, class T>
class FFTBase {

public: 
    // Some externally visible typedefs and enums.
    enum { dimensions = Dim };               // dimension
    typedef T Precision_t;                   // precision
    typedef NDIndex<Dim> Domain_t;           // domain type

    // Enumeration of transform types, used by derived FFT classes
    enum FFT_e { ccFFT, rcFFT, sineFFT, cosineFFT };

    // Type used for performing 1D FFTs
    typedef FFTPACK<T> InternalFFT_t;

    FFTBase() {}  
  
    /** 
     * inputs are enum of transform type, domain of input Field,
     * which dimensions to transform, and whether to compress
     * temporary Fields when not in use
     * 
     * @param transform 
     * @param domain 
     * @param transformTheseDims 
     * @param compressTemps 
     */
    
    FFTBase(FFT_e transform, const Domain_t& domain,
	    const bool transformTheseDims[Dim], bool compressTemps);
    
    /** 
     * 
     * 
     * @param transform 
     * @param domain 
     * @param compressTemps 
     */
    
    FFTBase(FFT_e transform, const Domain_t& domain, bool compressTemps);
    
    // destructor
    virtual ~FFTBase(void) { delete [] activeDims_m; }
  
    /** 
     * I/O for FFT object
     * 
     * @param out 
     */    
    void write(std::ostream& out) const;

    /** 
     * Allow the user to name the transform directions, for code clarity.
     * 
     * @param direction 
     * @param directionName 
     */
    void setDirectionName(int direction, const char* directionName);

    /** 
     * Set the FFT normalization factor (to something other than the default)
     * 
     * @param nf 
     */
    void setNormFact(Precision_t nf) { normFact_m = nf; }

    /** 
     * Utility to determine the number of vnodes to use in temporary transpose
     * fields; this is either -1, or a limited number set on the command line
     * 
     * @return 
     */
    int transVnodes() const {
	if (Ippl::maxFFTNodes() > 0 && Ippl::maxFFTNodes() <= Ippl::getNodes())
	    return Ippl::maxFFTNodes();
	else
	    return (-1);
    }

protected:

    /**! 
       These members are used by the derived FFT classes
    */

    /// null GuardCellSizes object for checking BareField arguments to transform
    static GuardCellSizes<Dim> nullGC;

    /// translate direction name string into dimension number
    int getDirection(const char* directionName) const;

    /// query whether this dimension is to be transformed
    bool transformDim(unsigned d) const;

    /// query number of transform dimensions
    unsigned numTransformDims(void) const { return nTransformDims_m; }

    /// get dimension number from list of transformed dimensions
    unsigned activeDimension(unsigned d) const;

    /// access the internal FFT Engine
    InternalFFT_t& getEngine(void) { return FFTEngine_m; }

    /// get the FFT normalization factor
    Precision_t& getNormFact(void) { return normFact_m; }

    /// get our domain
    const Domain_t& getDomain(void) const { return Domain_m; }

    /// compare indexes of two domains
    bool checkDomain(const Domain_t& dom1, const Domain_t& dom2) const;

    /// do we compress temps?
    bool compressTemps(void) const { return compressTempFields_m; }

private: 

    /// Stores user-defined names for FFT directions:
    std::map<const char*,int> directions_m;

    FFT_e transformType_m;     ///< Indicates which type of transform we do
    bool transformDims_m[Dim]; ///< Indicates which dimensions are transformed.
    unsigned nTransformDims_m; ///< Stores the number of dims to be transformed
    unsigned* activeDims_m;    ///< Stores the numbers of these dims (0,1,2).

    /// Internal FFT object for performing serial FFTs.
    InternalFFT_t FFTEngine_m;

    /// Normalization factor:
    Precision_t normFact_m;

    /// Domain of the input field, mainly used to check axis sizes and ordering, former const Domain_t& Domain_m;
    Domain_t Domain_m;

    /// Switch to turn on/off compression of intermediate Fields (tempFields) as algorithm is finished with them
    bool compressTempFields_m;
};


// Inline function definitions

/// Define operator<< to invoke write() member function:
template <unsigned Dim, class T>
inline std::ostream&
operator<<(std::ostream& out, const FFTBase<Dim,T>& fft)
{
    fft.write(out);
    return out;
}

/** 
    Allow the user to name the transform directions, for code clarity.
    Typical values might be "x_to_k", "k_to_x", "t_to_omega", "omega_to_t"
*/
template <unsigned Dim, class T>
inline void
FFTBase<Dim,T>::setDirectionName(int direction,
                                 const char* directionName) {
    PAssert_EQ(std::abs(direction), 1);
    directions_m[directionName] = direction;
    return;
}

/** 
 * Translate direction name string into dimension number
 * 
 * @param directionName 
 * 
 * @return 
 */
template <unsigned Dim, class T>
inline int
FFTBase<Dim,T>::getDirection(const char* directionName) const {
    return (*(directions_m.find(directionName))).second;
}

/** 
 * query whether this dimension is to be transformed
 * 
 * @param d 
 * 
 * @return 
 */
template <unsigned Dim, class T>
inline bool
FFTBase<Dim,T>::transformDim(unsigned d) const {
    PAssert_LT(d, Dim);
    return transformDims_m[d];
}

/** 
 * get dimension number from list of transformed dimensions
 * 
 * @param d 
 * 
 * @return 
 */
template <unsigned Dim, class T>
inline unsigned
FFTBase<Dim,T>::activeDimension(unsigned d) const {
    PAssert_LT(d, nTransformDims_m);
    return activeDims_m[d];
}

/** 
 * helper function for comparing domains
 * 
 * @param Dim 
 * @param dom1 
 * @param Dim 
 * @param dom2 
 * 
 * @return 
 */
template <unsigned Dim, class T>
inline bool
FFTBase<Dim,T>::checkDomain(const FFTBase<Dim,T>::Domain_t& dom1,
                            const FFTBase<Dim,T>::Domain_t& dom2) const {
    // check whether domains are equivalent
    // we require that some permutation of the axes gives a matching domain.
    static bool matched[Dim];
    bool found;
    unsigned d, d1;
    // initialize matched array to false
    for (d=0; d<Dim; ++d) matched[d] = false;
    d=0;
    while (d<Dim) {
	d1=0;
	found = false;
	while (!found && d1<Dim) {
	    // if we have not yet found a match for this dimension,
	    // compare length and base of Index objects
	    if (!matched[d1]) {
		found = ( dom1[d].length()==dom2[d1].length() &&
			  dom1[d].sameBase(dom2[d1]) );
		// if equivalent, mark this dimension as matched
		if (found) matched[d1] = true;
	    }
	    ++d1;
	}
	if (!found) return false;
	++d;
    }
    return true;
}

#include "FFT/FFTBase.hpp"

#endif // IPPL_FFT_FFTBASE_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
