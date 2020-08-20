//
// IPPL FFT
//
// Copyright (c) 2008-2018
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved.
//
// OPAL is licensed under GNU GPL version 3.
//

/*
  Prototypes for accessing Fortran 1D FFT routines from
  Netlib, and definitions for templated class FFTPACK, which acts as an
  FFT engine for the FFT class, providing storage for trigonometric
  information and performing the 1D FFTs as needed.
*/

#ifndef IPPL_FFT_FFTPACK_FFT_H
#define IPPL_FFT_FFTPACK_FFT_H

#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"

// FFTPACK function prototypes for Fortran routines
extern "C" {
    // double-precision CC FFT
    void cffti (size_t n, double& wsave);
    void cfftf (size_t n, double& r, double& wsave);
    void cfftb (size_t n, double& r, double& wsave);
    // double-precision RC FFT
    void rffti (size_t n, double& wsave);
    void rfftf (size_t n, double& r, double& wsave);
    void rfftb (size_t n, double& r, double& wsave);
    // double-precision sine transform
    void sinti (size_t n, double& wsave);
    void sint  (size_t n, double& r, double& wsave);
    // single-precision CC FFT
    void fcffti (size_t n, float& wsave);
    void fcfftf (size_t n, float& r, float& wsave);
    void fcfftb (size_t n, float& r, float& wsave);
    // single-precision RC FFT
    void frffti (size_t n, float& wsave);
    void frfftf (size_t n, float& r, float& wsave);
    void frfftb (size_t n, float& r, float& wsave);
    // single-precision sine transform
    void fsinti (size_t n, float& wsave);
    void fsint (size_t n, float& r, float& wsave);
}


// FFTPACK_wrap provides static functions that wrap the Fortran functions
// in a common interface.  We specialize this class on precision type.
template <class T>
class FFTPACK_wrap {};

// Specialization for float
template <>
class FFTPACK_wrap<float> {

public:
    // interface functions used by class FFTPACK

    // initialization functions for CC FFT, RC FFT, and sine transform
    static void ccffti(size_t n, float* wsave) { fcffti (n, *wsave); }
    static void rcffti(size_t n, float* wsave) { frffti (n, *wsave); }
    static void rrffti(size_t n, float* wsave) { fsinti (n, *wsave); }
    // forward and backward CC FFT
    static void ccfftf(size_t n, float* r, float* wsave) { fcfftf (n, *r, *wsave); }
    static void ccfftb(size_t n, float* r, float* wsave) { fcfftb (n, *r, *wsave); }
    // forward and backward RC FFT
    static void rcfftf(size_t n, float* r, float* wsave) { frfftf (n, *r, *wsave); }
    static void rcfftb(size_t n, float* r, float* wsave) { frfftb (n, *r, *wsave); }
    // sine transform
    static void rrfft(size_t n, float* r, float* wsave) { fsint (n, *r, *wsave); }
};

// Specialization for double
template <>
class FFTPACK_wrap<double> {

public:
    // interface functions used by class FFTPACK

    // initialization functions for CC FFT, RC FFT, and sine transform
    static void ccffti(size_t n, double* wsave) { cffti (n, *wsave); }
    static void rcffti(size_t n, double* wsave) { rffti (n, *wsave); }
    static void rrffti(size_t n, double* wsave) { sinti (n, *wsave); }
    // forward and backward CC FFT
    static void ccfftf(size_t n, double* r, double* wsave) {cfftf (n, *r, *wsave);}
    static void ccfftb(size_t n, double* r, double* wsave) {cfftb (n, *r, *wsave);}
    // forward and backward RC FFT
    static void rcfftf(size_t n, double* r, double* wsave) {rfftf (n, *r, *wsave);}
    static void rcfftb(size_t n, double* r, double* wsave) {rfftb (n, *r, *wsave);}
    // sine transform
    static void rrfft(size_t n, double* r, double* wsave) { sint (n, *r, *wsave); }
};


// Definition of FFT engine class FFTPACK
template <class T>
class FFTPACK {

public:

    // definition of complex type
    typedef std::complex<T> Complex_t;

    // Trivial constructor.  Do the real work in setup function.
    FFTPACK(void) {}

    // destructor
    ~FFTPACK(void);

    // setup internal storage and prepare to perform FFTs
    // inputs are number of dimensions to transform, the transform types,
    // and the lengths of these dimensions.
    void setup(unsigned numTransformDims, const int* transformTypes,
               const int* axisLengths);

    // invoke FFT on complex data for given dimension and direction
    void callFFT(unsigned transformDim, int direction, Complex_t* data);

    // invoke FFT on real data for given dimension and direction
    void callFFT(unsigned transformDim, int direction, T* data);

private:

    unsigned numTransformDims_m;  // number of dimensions to transform
    int* transformType_m;         // transform type for each dimension
    int* axisLength_m;            // length of each transform dimension
    T** trig_m;                   // trigonometric tables

};


// Inline member function definitions

// destructor
template <class T>
inline
FFTPACK<T>::~FFTPACK(void) {
    // delete storage
    for (unsigned d=0; d<numTransformDims_m; ++d)
        delete [] trig_m[d];
    delete [] trig_m;
    delete [] transformType_m;
    delete [] axisLength_m;
}

// setup internal storage to prepare for FFTs
template <class T>
inline void
FFTPACK<T>::setup(unsigned numTransformDims, const int* transformTypes,
                  const int* axisLengths) {

    // store transform types and lengths for each transform dim
    numTransformDims_m = numTransformDims;
    transformType_m = new int[numTransformDims_m];
    axisLength_m = new int[numTransformDims_m];
    unsigned d;
    for (d=0; d<numTransformDims_m; ++d) {
        transformType_m[d] = transformTypes[d];
        axisLength_m[d] = axisLengths[d];
    }

    // allocate and initialize trig table
    trig_m = new T*[numTransformDims_m];
    for (d=0; d<numTransformDims_m; ++d) {
        switch (transformType_m[d]) {
        case 0:  // CC FFT
            trig_m[d] = new T[4 * axisLength_m[d] + 15];
            FFTPACK_wrap<T>::ccffti(axisLength_m[d], trig_m[d]);
            break;
        case 1:  // RC FFT
            trig_m[d] = new T[2 * axisLength_m[d] + 15];
            FFTPACK_wrap<T>::rcffti(axisLength_m[d], trig_m[d]);
            break;
        case 2:  // Sine transform
            trig_m[d] = new T[static_cast<int>(2.5 * axisLength_m[d] + 0.5) + 15];
            FFTPACK_wrap<T>::rrffti(axisLength_m[d], trig_m[d]);
            break;
        default:
            ERRORMSG("Unknown transform type requested!!" << endl);
            break;
        }
    }

    return;
}

// invoke FFT on complex data for given dimension and direction
template <class T>
inline void
FFTPACK<T>::callFFT(unsigned transformDim, int direction,
                    FFTPACK<T>::Complex_t* data) {

    // check transform dimension and direction arguments
    PAssert_LT(transformDim, numTransformDims_m);
    PAssert_EQ(std::abs(direction), 1);

    // cast complex number pointer to T* for calling Fortran routines
    T* rdata = reinterpret_cast<T*>(data);

    // branch on transform type for this dimension
    switch (transformType_m[transformDim]) {
    case 0:  // CC FFT
        if (direction == +1) {
            // call forward complex-to-complex FFT
            FFTPACK_wrap<T>::ccfftf(axisLength_m[transformDim], rdata,
                                    trig_m[transformDim]);
        }
        else {
            // call backward complex-to-complex FFT
            FFTPACK_wrap<T>::ccfftb(axisLength_m[transformDim], rdata,
                                    trig_m[transformDim]);
        }
        break;
    case 1:  // RC FFT
        if (direction == +1) {
            // call forward real-to-complex FFT
            FFTPACK_wrap<T>::rcfftf(axisLength_m[transformDim], rdata,
                                    trig_m[transformDim]);
            // rearrange output to conform with SGI format for complex result
            int clen = axisLength_m[transformDim]/2 + 1;
            data[clen-1] = Complex_t(imag(data[clen-2]),0.0);
            for (int i = clen-2; i > 0; --i)
                data[i] = Complex_t(imag(data[i-1]),real(data[i]));
            data[0] = Complex_t(real(data[0]),0.0);
        }
        else {                
            // rearrange input to conform with Netlib format for complex modes
            int clen = axisLength_m[transformDim]/2 + 1;
            data[0] = Complex_t(real(data[0]),real(data[1]));
            for (int i = 1; i < clen-1; ++i)
                data[i] = Complex_t(imag(data[i]),real(data[i+1]));
            // call backward complex-to-real FFT
            FFTPACK_wrap<T>::rcfftb(axisLength_m[transformDim], rdata,
                                    trig_m[transformDim]);
        }
        break;
    case 2:  // Sine transform
        ERRORMSG("Input for real-to-real FFT should be real!!" << endl);
        break;
    default:
        ERRORMSG("Unknown transform type requested!!" << endl);
        break;
    }

    return;
}

// invoke FFT on real data for given dimension and direction
template <class T>
inline void
FFTPACK<T>::callFFT(unsigned transformDim, int direction, T* data) {

    // check transform dimension and direction arguments
    PAssert_LT(transformDim, numTransformDims_m);
    // avoid unused variable warning if we compile with Release
    // :FIXME: remove direction
    (void)direction;
    PAssert_EQ(std::abs(direction), 1);

    // branch on transform type for this dimension
    switch (transformType_m[transformDim]) {
    case 0:  // CC FFT
        ERRORMSG("Input for complex-to-complex FFT should be complex!!" << endl);
        break;
    case 1:  // RC FFT
        ERRORMSG("real-to-complex FFT uses complex input!!" << endl);
        break;
    case 2:  // Sine transform
        // invoke the real-to-real transform on the data
        FFTPACK_wrap<T>::rrfft(axisLength_m[transformDim], data,
                               trig_m[transformDim]);
        break;
    default:
        ERRORMSG("Unknown transform type requested!!" << endl);
        break;
    }

    return;
}
#endif // IPPL_FFT_FFTPACK_FFT_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:

