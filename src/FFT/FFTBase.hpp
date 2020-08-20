//
// IPPL FFT
//
// Copyright (c) 2008-2018
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved.
//
// OPAL is licensed under GNU GPL version 3.
//

#include "FFT/FFTBase.h"
#include "Utility/PAssert.h"

/***************************************************************************
  FFTBase.cpp:  constructors and write() member function for FFTBase class
***************************************************************************/

// instantiate static null GC object
template <unsigned Dim, class T>
GuardCellSizes<Dim> FFTBase<Dim,T>::nullGC = GuardCellSizes<Dim>();

//=============================================================================
// FFTBase Constructors
//=============================================================================

template <unsigned Dim, class T>
FFTBase<Dim,T>::FFTBase(FFTBase<Dim,T>::FFT_e transform,
                        const FFTBase<Dim,T>::Domain_t& domain,
		        const bool transformTheseDims[Dim], bool compressTemps)
: transformType_m(transform),               // transform type
    Domain_m(domain),                         // field domain
    compressTempFields_m(compressTemps)       // compress temp fields?
{

    // Tau profiling
  
    // Store which dims are transformed, and count up how many there are
    nTransformDims_m = 0;
    unsigned d;
    for (d=0; d<Dim; ++d) {
        transformDims_m[d] = transformTheseDims[d];
        if (transformTheseDims[d]) ++nTransformDims_m;
    }

    // check that at least one dimension is being transformed
    PInsist(nTransformDims_m>0,"Must transform at least one axis!!");

    // Store the "names" (0,1,2) of these dims in an array of computed size
    activeDims_m = new unsigned[nTransformDims_m];
    int icount = 0;
    for (d=0; d<Dim; ++d)
        if (transformDims_m[d]) activeDims_m[icount++] = d;

}

template <unsigned Dim, class T>
FFTBase<Dim,T>::FFTBase(FFTBase<Dim,T>::FFT_e transform,
                        const FFTBase<Dim,T>::Domain_t& domain,
                        bool compressTemps)
: transformType_m(transform),               // transform type
    Domain_m(domain),                         // field domain
    compressTempFields_m(compressTemps)       // compress temp fields?
{

    // Tau profiling

    // Default, transform all dims:
    nTransformDims_m = Dim;
    activeDims_m = new unsigned[Dim];
    for (unsigned d=0; d<Dim; d++) {
        transformDims_m[d] = true;
        activeDims_m[d] = d;
    }

}

//-----------------------------------------------------------------------------
// I/O: write out information about FFT object:
//-----------------------------------------------------------------------------

template <unsigned Dim, class T>
void FFTBase<Dim,T>::write(std::ostream& out) const {

    // Tau profiling
  
    // Dump contents of FFT object
    out << "---------------FFT Object Dump Begin-------------------" << std::endl;
    // Output the user-defined names for transform directions:
    out << "Map of transform direction names:" << std::endl;
    std::map<char*,unsigned>::const_iterator mi, m_end = directions_m.end();
    for (mi = directions_m.begin(); mi != m_end; ++mi)
        out << "[" << (*mi).first << "," << (*mi).second << "]" << std::endl;
    // Output type of transform
    out << "Transform type = " << getTransformType(transformType_m) << std::endl;
    // Output which dims are transformed:
    out << "Transform dimensions: ";
    for (unsigned d=0; d<Dim; ++d) {
        out << "dim " << d << " = ";
        if (transformDims_m[d])
            out << "true" << std::endl;
        else
            out << "false" << std::endl;
    }
    out << std::endl;
    out << "Input Field domain = " << Domain_m << std::endl; // Output the domain.
    out << "---------------FFT Object Dump End---------------------" << std::endl;

    return;
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
