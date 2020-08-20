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
   Implementations for FFT constructor/destructor and transforms
*/

#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/BareField.h"
#include "Utility/IpplStats.h"

//=============================================================================
// FFT CCTransform Constructors
//=============================================================================

/**
   Create a new FFT object of type CCTransform, with a
   given domain. Also specify which dimensions to transform along.
*/

template <size_t Dim, class T>
FFT<CCTransform,Dim,T>::FFT(
    const typename FFT<CCTransform,Dim,T>::Domain_t& cdomain,
    const bool transformTheseDims[Dim],
    const bool& compressTemps)
: FFTBase<Dim,T>(FFT<CCTransform,Dim,T>::ccFFT, cdomain,
                 transformTheseDims, compressTemps)
{

    // construct array of axis lengths
    size_t nTransformDims = this->numTransformDims();
    int* lengths = new int[nTransformDims];
    size_t d;
    for (d=0; d<nTransformDims; ++d)
        lengths[d] = cdomain[this->activeDimension(d)].length();

    // construct array of transform types for FFT Engine, compute normalization
    int* transformTypes = new int[nTransformDims];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    for (d=0; d<nTransformDims; ++d) {
        transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all transforms are complex-to-complex
        normFact /= lengths[d];
    }

    // set up FFT Engine
    this->getEngine().setup(nTransformDims, transformTypes, lengths);
    delete [] transformTypes;
    delete [] lengths;
    // set up the temporary fields
    setup();
}


/**
   setup performs all the initializations necessary after the transform
   directions have been specified.
*/
template <size_t Dim, class T>
void
FFT<CCTransform,Dim,T>::setup(void)
{
    // Tau profiling


    size_t d, activeDim;
    size_t nTransformDims = this->numTransformDims();
    // Set up the arrays of temporary Fields and FieldLayouts:
    e_dim_tag serialParallel[Dim];  // Specifies SERIAL, PARALLEL dims in temp
    // make zeroth dimension always SERIAL
    serialParallel[0] = SERIAL;
    // all other dimensions parallel
    for (d=1; d<Dim; ++d)
        serialParallel[d] = PARALLEL;

    tempLayouts_m = new Layout_t*[nTransformDims];
    tempFields_m = new ComplexField_t*[nTransformDims];

    // loop over transform dimensions
    for (size_t dim=0; dim<nTransformDims; ++dim) {
        // get number of dimension to be transformed
        activeDim = this->activeDimension(dim);
        // Get input Field's domain
        const Domain_t& ndic = this->getDomain();
        // make new domain with permuted Indexes, activeDim first
        Domain_t ndip;
        ndip[0] = ndic[activeDim];
        for (d=1; d<Dim; ++d) {
            size_t nextDim = activeDim + d;
            if (nextDim >= Dim) nextDim -= Dim;
            ndip[d] = ndic[nextDim];
        }
        // generate temporary field layout
        tempLayouts_m[dim] = new Layout_t(ndip, serialParallel, this->transVnodes());
        // generate temporary Field
        tempFields_m[dim] = new ComplexField_t(*tempLayouts_m[dim]);
        // If user requests no intermediate compression, uncompress right now:
        if (!this->compressTemps()) (*tempFields_m[dim]).Uncompress();
    }

    return;
}

//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<CCTransform,Dim,T>::~FFT(void) {

    // Tau profiling

    /*
      #ifdef IPPL_OPENCL
      base.ocl_cleanUp();
      #endif
    */

    // delete arrays of temporary fields and field layouts
    size_t nTransformDims = this->numTransformDims();
    for (size_t d=0; d<nTransformDims; ++d) {
        delete tempFields_m[d];
        delete tempLayouts_m[d];
    }
    delete [] tempFields_m;
    delete [] tempLayouts_m;
}


//-----------------------------------------------------------------------------
// do the CC FFT; separate input and output fields
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<CCTransform,Dim,T>::transform(
    int direction,
    typename FFT<CCTransform,Dim,T>::ComplexField_t& f,
    typename FFT<CCTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{
    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true);

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
    begdim = (direction == +1) ? 0 : static_cast<int>(nTransformDims-1);
    enddim = (direction == +1) ? static_cast<int>(nTransformDims) : -1;
    for (idim = begdim; idim != enddim; idim += direction) {

        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the transpose into the first temporary Field
        if (idim == begdim && !constInput) {
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
        // to skip the last temporary and transpose right into g
        if (idim == enddim-direction) {
            // get the domain for comparison
            const Domain_t& last_dom = tempLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (out_dom[0].sameBase(last_dom[0])) &&
                              (out_dom[0].length() == last_dom[0].length()) &&
                              (out_layout.getDistribution(0) == SERIAL) &&
                              (g.getGC() == FFT<CCTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && temp != &f) *temp = 0;
            temp = tempFields_m[idim];  // Field* management aid
        }
        else if (idim == enddim-direction && temp != &g) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using g instead

            // transpose and permute to Field with transform dim first
            g[out_dom] = (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && temp != &f) *temp = 0;
            temp = &g;  // Field* management aid
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

    // skip final assignment and compress if we used g as final temporary
    if (temp != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*temp)[temp->getLayout().getDomain()];
        if (this->compressTemps() && temp != &f) *temp = 0;

    }

    // Normalize:
    if (direction == +1)
        g *= Complex_t(this->getNormFact(), 0.0);

    return;
}

template <size_t Dim, class T>
void
FFT<CCTransform,Dim,T>::transform(
    int direction,
    typename FFT<CCTransform,Dim,T>::ComplexField_t& f)
{

    // indicate we're doing another FFT
    // INCIPPLSTAT(incFFTs);

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
    begdim = (direction == +1) ? 0 : static_cast<int>(nTransformDims-1);
    enddim = (direction == +1) ? static_cast<int>(nTransformDims) : -1;
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

//=============================================================================
// 1D FFT CCTransform Constructors
//=============================================================================

//-----------------------------------------------------------------------------
// Create a new FFT object of type CCTransform, with a
// given domain. Also specify which dimensions to transform along.
//-----------------------------------------------------------------------------

template <class T>
FFT<CCTransform,1U,T>::FFT(
    const typename FFT<CCTransform,1U,T>::Domain_t& cdomain,
    const bool transformTheseDims[1U], const bool& compressTemps)
: FFTBase<1U,T>(FFT<CCTransform,1U,T>::ccFFT, cdomain,
                transformTheseDims, compressTemps)
{

    // Tau profiling




    size_t nTransformDims = 1U;
    // get axis length
    int length;
    length = cdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::ccFFT;  // all transforms are complex-to-complex
    T& normFact = this->getNormFact();
    normFact = 1.0 / length;

    // set up FFT Engine
    this->getEngine().setup(nTransformDims, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type CCTransform, with a
// given domain. Default case of transforming along all dimensions.
//-----------------------------------------------------------------------------

template <class T>
FFT<CCTransform,1U,T>::FFT(
    const typename FFT<CCTransform,1U,T>::Domain_t& cdomain,
    const bool& compressTemps)
: FFTBase<1U,T>(FFT<CCTransform,1U,T>::ccFFT, cdomain, compressTemps)
{

    // Tau profiling

    // get axis length
    int length;
    length = cdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::ccFFT;  // all transforms are complex-to-complex
    T& normFact = this->getNormFact();
    normFact = 1.0 / length;

    // set up FFT Engine
    this->getEngine().setup(1U, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// setup performs all the initializations necessary after the transform
// directions have been specified.
//-----------------------------------------------------------------------------

template <class T>
void
FFT<CCTransform,1U,T>::setup(void)
{
    // Tau profiling

    // Get input Field's domain
    const Domain_t& ndic = this->getDomain();
    // generate temporary field layout
    tempLayouts_m = new Layout_t(ndic[0], PARALLEL, 1);
    // generate temporary Field
    tempFields_m = new ComplexField_t(*tempLayouts_m);
    // If user requests no intermediate compression, uncompress right now:
    if (!this->compressTemps()) tempFields_m->Uncompress();

    return;
}

//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <class T>
FFT<CCTransform,1U,T>::~FFT(void) {

    // Tau profiling




    // delete temporary fields and field layouts
    delete tempFields_m;
    delete tempLayouts_m;
}


//-----------------------------------------------------------------------------
// do the CC FFT; separate input and output fields
//-----------------------------------------------------------------------------

template <class T>
void
FFT<CCTransform,1U,T>::transform(
    int direction,
    typename FFT<CCTransform,1U,T>::ComplexField_t& f,
    typename FFT<CCTransform,1U,T>::ComplexField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    // INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true);

    // Field* for temp Field management:
    ComplexField_t* temp = &f;
    // Local work array passed to FFT:
    Complex_t* localdata;


    // Now do the serial transforms along this dimension:

    // get temp domain for comparison
    const Domain_t& temp_dom = tempLayouts_m->getDomain();

    bool skipTranspose = false;
    // if this is the first transform dimension, we might be able
    // to skip the transpose into the first temporary Field
    if (!constInput) {
        // check that zeroth axis is the same, has one vnode,
        // and that there are no guard cells
        skipTranspose = ( (in_dom[0].sameBase(temp_dom[0])) &&
                          (in_dom[0].length() == temp_dom[0].length()) &&
                          (in_layout.numVnodes() == 1) &&
                          (f.getGC() == FFT<CCTransform,1U,T>::nullGC) );
    }

    bool skipFinal;
    // we might be able
    // to skip the last temporary and transpose right into g

    // check that zeroth axis is the same, has one vnode
    // and that there are no guard cells
    skipFinal = ( (out_dom[0].sameBase(temp_dom[0])) &&
                  (out_dom[0].length() == temp_dom[0].length()) &&
                  (out_layout.numVnodes() == 1) &&
                  (g.getGC() == FFT<CCTransform,1U,T>::nullGC) );

    if (!skipTranspose) {
        // assign to Field with proper layout
        (*tempFields_m) = (*temp);
        temp = tempFields_m;  // Field* management aid
    }
    if (skipFinal) {
        // we can skip the last temporary field
        // so do the transpose here using g instead

        // assign to Field with proper layout
        g = (*temp);

        // Compress out previous iterate's storage:
        if (this->compressTemps() && temp != &f) *temp = 0;
        temp = &g;  // Field* management aid
    }




    // should be only one LField!
    typename ComplexField_t::const_iterator_if l_i = temp->begin_if();
    if (l_i != temp->end_if()) {
        // Get the LField
        ComplexLField_t* ldf = (*l_i).second.get();
        // make sure we are uncompressed
        ldf->Uncompress();
        // get the raw data pointer
        localdata = ldf->getP();

        // Do the 1D FFT:
        this->getEngine().callFFT(0, direction, localdata);
    }



    // skip final assignment and compress if we used g as final temporary
    if (temp != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g = (*temp);
        if (this->compressTemps() && temp != &f) *temp = 0;

    }

    // Normalize:
    if (direction == +1)
        g *= Complex_t(this->getNormFact(), 0.0);

    return;
}

//-----------------------------------------------------------------------------
// "in-place" FFT; specify +1 or -1 to indicate forward or inverse transform.
//-----------------------------------------------------------------------------

template <class T>
void
FFT<CCTransform,1U,T>::transform(
    int direction,
    typename FFT<CCTransform,1U,T>::ComplexField_t& f)
{

    // indicate we're doing another FFT
    // INCIPPLSTAT(incFFTs);

    // Check domain of incoming Field
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    PAssert_EQ(this->checkDomain(this->getDomain(),in_dom), true);

    // Field* for temp Field management:
    ComplexField_t* temp = &f;
    // Local work array passed to FFT:
    Complex_t* localdata;


    // Now do the serial transforms along this dimension:

    // get domain for comparison
    const Domain_t& temp_dom = tempLayouts_m->getDomain();

    bool skipTranspose;
    // we might be able
    // to skip the transpose into the first temporary Field

    // check that zeroth axis is the same, has one vnode,
    // and that there are no guard cells
    skipTranspose = ( (in_dom[0].sameBase(temp_dom[0])) &&
                      (in_dom[0].length() == temp_dom[0].length()) &&
                      (in_layout.numVnodes() == 1) &&
                      (f.getGC() == FFT<CCTransform,1U,T>::nullGC) );

    if (!skipTranspose) {
        // assign to Field with proper layout
        (*tempFields_m) = (*temp);
        temp = tempFields_m;  // Field* management aid
    }




    // should be only one LField!
    typename ComplexField_t::const_iterator_if l_i = temp->begin_if();
    if (l_i != temp->end_if()) {
        // Get the LField
        ComplexLField_t* ldf = (*l_i).second.get();
        // make sure we are uncompressed
        ldf->Uncompress();
        // get the raw data pointer
        localdata = ldf->getP();

        // Do the 1D FFT:
        this->getEngine().callFFT(0, direction, localdata);
    }



    // skip final assignment and compress if we used f as final temporary
    if (temp != &f) {

        // Now assign back into original Field, and compress last temp's storage:
        f = (*temp);
        if (this->compressTemps()) *temp = 0;

    }

    // Normalize:
    if (direction == +1)
        f *= Complex_t(this->getNormFact(), 0.0);

    return;
}



//=============================================================================
// FFT RCTransform Constructors
//=============================================================================

//-----------------------------------------------------------------------------
// Create a new FFT object of type RCTransform, with a
// given domain. Also specify which dimensions to transform along.
// Note that RC transform of a real array of length n results in a
// complex array of length n/2+1.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<RCTransform,Dim,T>::FFT(
    const typename FFT<RCTransform,Dim,T>::Domain_t& rdomain,
    const typename FFT<RCTransform,Dim,T>::Domain_t& cdomain,
    const bool transformTheseDims[Dim], const bool& compressTemps)
: FFTBase<Dim,T>(FFT<RCTransform,Dim,T>::rcFFT, rdomain,
                 transformTheseDims, compressTemps),
    complexDomain_m(cdomain), serialAxes_m(1)
{
    // construct array of axis lengths
    size_t nTransformDims = this->numTransformDims();
    int* lengths = new int[nTransformDims];
    size_t d;
    for (d=0; d<nTransformDims; ++d)
        lengths[d] = rdomain[this->activeDimension(d)].length();

    // construct array of transform types for FFT Engine, compute normalization
    int* transformTypes = new int[nTransformDims];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    transformTypes[0] = FFTBase<Dim,T>::rcFFT;    // first transform is real-to-complex
    normFact /= lengths[0];
    for (d=1; d<nTransformDims; ++d) {
        transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all other transforms are complex-to-complex
        normFact /= lengths[d];
    }

    // set up FFT Engine
    this->getEngine().setup(nTransformDims, transformTypes, lengths);
    delete [] transformTypes;
    delete [] lengths;

    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type RCTransform, with
// given real and complex domains. Default: transform along all dimensions.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<RCTransform,Dim,T>::FFT(
    const typename FFT<RCTransform,Dim,T>::Domain_t& rdomain,
    const typename FFT<RCTransform,Dim,T>::Domain_t& cdomain,
    const bool& compressTemps,
    int serialAxes)
: FFTBase<Dim,T>(FFT<RCTransform,Dim,T>::rcFFT, rdomain, compressTemps),
    complexDomain_m(cdomain), serialAxes_m(serialAxes)
{
    // Tau profiling

    // construct array of axis lengths
    int lengths[Dim];
    size_t d;
    for (d=0; d<Dim; ++d)
        lengths[d] = rdomain[d].length();

    // construct array of transform types for FFT Engine, compute normalization
    int transformTypes[Dim];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    transformTypes[0] = FFTBase<Dim,T>::rcFFT;    // first transform is real-to-complex
    normFact /= lengths[0];
    for (d=1; d<Dim; ++d) {
        transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all other transforms are complex-to-complex
        normFact /= lengths[d];
    }

    // set up FFT Engine
    this->getEngine().setup(Dim, transformTypes, lengths);

    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// setup performs all the initializations necessary after the transform
// directions have been specified.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<RCTransform,Dim,T>::setup(void) {

    // Tau profiling




    PAssert_GT(serialAxes_m, 0);
    PAssert_LT((size_t) serialAxes_m, Dim);

    size_t d, d2, activeDim;
    size_t nTransformDims = this->numTransformDims();

    // Set up the arrays of temporary Fields and FieldLayouts:

    // make first dimension(s) always SERIAL, all other dimensions parallel
    // for the real FFT; make first serialAxes_m axes serial for others
    e_dim_tag serialParallel[Dim];
    e_dim_tag NserialParallel[Dim];
    for (d=0; d < Dim; ++d) {
        serialParallel[d] = (d == 0 ? SERIAL : PARALLEL);
        NserialParallel[d] = (d < (size_t) serialAxes_m ? SERIAL : PARALLEL);
    }

    // check that domain lengths agree between real and complex domains
    const Domain_t& domain = this->getDomain();
    activeDim = this->activeDimension(0);
    bool match = true;
    for (d=0; d<Dim; ++d) {
        if (d == activeDim) {
            // real array length n, complex array length n/2+1
            if ( complexDomain_m[d].length() !=
                 (domain[d].length()/2 + 1) ) match = false;
        }
        else {
            // real and complex arrays should be same length for all other dims
            if (complexDomain_m[d].length() != domain[d].length()) match = false;
        }
    }
    PInsist(match,
            "Domains provided for real and complex Fields are incompatible!");

    // allocate arrays of temp fields and layouts for complex fields
    tempLayouts_m = new Layout_t*[nTransformDims];
    tempFields_m = new ComplexField_t*[nTransformDims];

    // set up the single temporary real field, with first dim serial, others par

    // make new domains with permuted Indexes, activeDim first
    Domain_t ndip;
    Domain_t ndipc;
    ndip[0] = domain[activeDim];
    ndipc[0] = complexDomain_m[activeDim];
    for (d=1; d<Dim; ++d) {
        size_t nextDim = activeDim + d;
        if (nextDim >= Dim) nextDim -= Dim;
        ndip[d] = domain[nextDim];
        ndipc[d] = complexDomain_m[nextDim];
    }

    // generate layout and object for temporary real field
    tempRLayout_m = new Layout_t(ndip, serialParallel, this->transVnodes());
    tempRField_m = new RealField_t(*tempRLayout_m);

    // generate layout and object for first temporary complex Field
    tempLayouts_m[0] = new Layout_t(ndipc, serialParallel, this->transVnodes());
    tempFields_m[0] = new ComplexField_t(*tempLayouts_m[0]);

    // determine the order in which dimensions will be transposed.  Put
    // the transposed dims first, and the others at the end.
    int fftorder[Dim], tmporder[Dim];
    int nofft = nTransformDims;
    for (d=0; d < nTransformDims; ++d)
        fftorder[d] = this->activeDimension(d);
    for (d=0; d < Dim; ++d) {
        // see if the dth dimension is one to transform
        bool active = false;
        for (d2=0; d2 < nTransformDims; ++d2) {
            if (this->activeDimension(d2) == d) {
                active = true;
                break;
            }
        }

        if (!active)
            // no it is not; put it at the bottom of list
            fftorder[nofft++] = d;
    }

    // But since the first FFT is done on a S,[P,P,...] field, permute
    // the order of this to get the first activeDimension at the end.
    nofft = fftorder[0];
    for (d=0; d < (Dim - 1); ++d)
        fftorder[d] = fftorder[d+1];
    fftorder[Dim-1] = nofft;

    // now construct the remaining temporary complex fields

    // loop through and create actual permuted layouts, and also fields
    size_t dim = 1;			// already have one temp field
    while (dim < nTransformDims) {

        int sp;
        for (sp=0; sp < serialAxes_m && dim < nTransformDims; ++sp, ++dim) {

            // make new domain with permuted Indexes
            for (d=0; d < Dim; ++d)
                ndip[d] = complexDomain_m[fftorder[d]];

            // generate layout and object for temporary complex Field
            tempLayouts_m[dim] = new Layout_t(ndip, NserialParallel, this->transVnodes());
            tempFields_m[dim] = new ComplexField_t(*tempLayouts_m[dim]);

            // permute the fft order for the first 'serialAxes_m' axes
            if (serialAxes_m > 1) {
                tmporder[0] = fftorder[0];
                for (d=0; d < (size_t) (serialAxes_m-1); ++d)
                    fftorder[d] = fftorder[d+1];
                fftorder[serialAxes_m - 1] = tmporder[0];
            }
        }

        // now, permute ALL the axes by serialAxes_m steps, to get the next
        // set of axes in the first n serial slots
        for (d=0; d < Dim; ++d)
            tmporder[d] = fftorder[d];
        for (d=0; d < Dim; ++d)
            fftorder[d] = tmporder[(d + serialAxes_m) % Dim];
    }
}


//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<RCTransform,Dim,T>::~FFT(void) {

    // Tau profiling


    // delete temporary fields and layouts
    size_t nTransformDims = this->numTransformDims();
    for (size_t d=0; d<nTransformDims; ++d) {
        delete tempFields_m[d];
        delete tempLayouts_m[d];
    }
    delete [] tempFields_m;
    delete [] tempLayouts_m;
    delete tempRField_m;
    delete tempRLayout_m;

}

template <size_t Dim, class T>
void
FFT<RCTransform,Dim,T>::transform(
    int direction,
    typename FFT<RCTransform,Dim,T>::RealField_t& f,
    typename FFT<RCTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{
    // indicate we're doing another fft
    // incipplstat(incffts);

    // check domain of incoming fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();


    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(complexDomain_m,out_dom), true);

    // common loop iterate and other vars:
    size_t d;
    size_t idim;      // idim loops over the number of transform dims.
    size_t nTransformDims = this->numTransformDims();

    // handle first rc transform separately
    idim = 0;

    RealField_t* tempR = tempRField_m;  // field* management aid
    if (!constInput) {
        // see if we can use input field f as a temporary
        bool skipTemp = true;

        // more rigorous match required here; check that layouts are identical
        if ( !(in_layout == *tempRLayout_m) ) {
            skipTemp = false;
        } else {
            // make sure distributions match
            for (d=0; d<Dim; ++d)
                if (in_layout.getDistribution(d) != tempRLayout_m->getDistribution(d))
                    skipTemp = false;

            // make sure vnode counts match
            if (in_layout.numVnodes() != tempRLayout_m->numVnodes())
                skipTemp = false;

            // also make sure there are no guard cells
            if (!(f.getGC() == FFT<RCTransform,Dim,T>::nullGC))
                skipTemp = false;
        }

        // if we can skip using this temporary, set the tempr pointer to the
        // original incoming field.  otherwise, it will stay pointing at the
        // temporary real field, and we'll need to do a transpose of the data
        // from the original into the temporary.
        if (skipTemp)
            tempR = &f;
    }

    // if we're not using input as a temporary ...
    if (tempR != &f) {


        // transpose AND PERMUTE TO REAL FIELD WITH TRANSFORM DIM FIRST
        (*tempR)[tempR->getDomain()] = f[in_dom];

    }

    // field* for temp field management:
    ComplexField_t* temp = tempFields_m[0];

    // see if we can put final result directly into g.  this is useful if
    // we're doing just a 1d fft of one dimension of a multi-dimensional field.
    if (nTransformDims == 1) {  // only a single rc transform
        bool skipTemp = true;

        // more rigorous match required here; check that layouts are identical
        if (!(out_layout == *tempLayouts_m[0])) {
            skipTemp = false;
        } else {
            for (d=0; d<Dim; ++d)
                if (out_layout.getDistribution(d) !=
                    tempLayouts_m[0]->getDistribution(d))
                    skipTemp = false;

            if ( out_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
                skipTemp = false;

            // also make sure there are no guard cells
            if (!(g.getGC() == FFT<RCTransform,Dim,T>::nullGC))
                skipTemp = false;

            // if we can skip using the temporary, set the pointer to the output
            // field for the first fft to the second provided field (g)
            if (skipTemp)
                temp = &g;
        }
    }

    // loop over all the vnodes, working on the lfield in each.
    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
        // get the lfields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();

        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();

        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        // number of strips should be the same for real and complex lfields!
        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
        for (d=1; d<Dim; ++d)
            nstrips *= rldf->size(d);


        for (int istrip=0; istrip<nstrips; ++istrip) {
            // move the data into the complex strip, which is two reals longer
            for (int ilen=0; ilen<lengthreal; ilen+=2) {
                localcomp[ilen/2] = Complex_t(localreal[ilen],localreal[ilen+1]);
            }

            // do the 1d real-to-complex fft:
            // note that real-to-complex fft direction is always +1
            this->getEngine().callFFT(idim, +1, localcomp);

            // advance the data pointers
            localreal += lengthreal;
            localcomp += lengthcomp;
        } // loop over 1d strips

    } // loop over all the lfields

    // compress temporary storage
    if (this->compressTemps() && tempR != &f)
        *tempR = 0;

    // now proceed with the other complex-to-complex transforms

    // local work array passed to fft:
    Complex_t* localdata;

    // loop over the remaining dimensions to be transformed:
    for (idim = 1; idim < nTransformDims; ++idim) {

        bool skipTranspose = false;

        // if this is the last transform dimension, we might be able
        // to skip the last temporary and transpose right into g
        if (idim == nTransformDims-1) {
            // get the domain for comparison
            const Domain_t& last_dom = tempLayouts_m[idim]->getDomain();

            // make sure there are no guard cells, and that the first
            // axis matches what we expect and is serial.  only need to
            // check first axis since we're just fft'ing that one dimension.
            skipTranspose = (g.getGC() == FFT<RCTransform,Dim,T>::nullGC &&
                             out_dom[0].sameBase(last_dom[0]) &&
                             out_dom[0].length() == last_dom[0].length() &&
                             out_layout.getDistribution(0) == SERIAL);
        }

        if (!skipTranspose) {
            // transpose and permute to field with transform dim first
            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                (*temp)[temp->getLayout().getDomain()];

            // compress out previous iterate's storage:
            if (this->compressTemps())
                *temp = 0;
            temp = tempFields_m[idim];  // field* management aid

        } else if (idim == nTransformDims-1) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using g instead

            // transpose and permute to field with transform dim first

            g[out_dom] = (*temp)[temp->getLayout().getDomain()];

            // compress out previous iterate's storage:
            if (this->compressTemps())
                *temp = 0;
            temp = &g;  // field* management aid

        }

        // loop over all the vnodes, working on the lfield in each.
        typename ComplexField_t::const_iterator_if l_i, l_end = temp->end_if();
        for (l_i = temp->begin_if(); l_i != l_end; ++l_i) {
            // get the lfield
            ComplexLField_t* ldf = (*l_i).second.get();

            // make sure we are uncompressed
            ldf->Uncompress();

            // get the raw data pointer
            localdata = ldf->getP();

            // do 1d complex-to-complex fft's on all the strips in the lfield:
            int nstrips = 1, length = ldf->size(0);
            for (d=1; d<Dim; ++d)
                nstrips *= ldf->size(d);

            for (int istrip=0; istrip<nstrips; ++istrip) {
                // do the 1D FFT:
                //this->getEngine().callFFT(idim, direction, localdata);
                this->getEngine().callFFT(idim, +1, localdata);

                // advance the data pointer
                localdata += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions


    // skip final assignment and compress if we used g as final temporary
    if (temp != &g) {


        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*temp)[temp->getLayout().getDomain()];

        if (this->compressTemps()) *temp = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    // finish timing the whole mess

}
//#endif

//-----------------------------------------------------------------------------
// RC FFT; opposite direction, from complex to real
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<RCTransform,Dim,T>::transform(
    int direction,
    typename FFT<RCTransform,Dim,T>::ComplexField_t& f,
    typename FFT<RCTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{
    // indicate we're doing another FFT
    // INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(complexDomain_m,in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true);

    // Common loop iterate and other vars:
    size_t d;
    size_t idim;      // idim loops over the number of transform dims.
    size_t nTransformDims = this->numTransformDims();

    // proceed with the complex-to-complex transforms

    // Field* for temp Field management:
    ComplexField_t* temp = &f;

    // Local work array passed to FFT:
    Complex_t* localdata;

    // Loop over all dimensions to be transformed except last one:
    for (idim = nTransformDims-1; idim != 0; --idim) {

        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the transpose into the first temporary Field
        if (idim == nTransformDims-1 && !constInput) {
            // get domain for comparison
            const Domain_t& first_dom = tempLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                              (in_dom[0].length() == first_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<RCTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && temp != &f)
                *temp = 0;
            temp = tempFields_m[idim];  // Field* management aid
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
            for (d=1; d<Dim; ++d)
                nstrips *= ldf->size(d);

            for (int istrip=0; istrip<nstrips; ++istrip) {
                // Do the 1D FFT:
                //this->getEngine().callFFT(idim, direction, localdata);
                this->getEngine().callFFT(idim, -1, localdata);

                // advance the data pointer
                localdata += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // handle last CR transform separately
    idim = 0;

    // see if we can put final result directly into g
    RealField_t* tempR;
    bool skipTemp = true;

    // more rigorous match required here; check that layouts are identical
    if (!(out_layout == *tempRLayout_m)) {
        skipTemp = false;
    } else {
        for (d=0; d<Dim; ++d)
            if (out_layout.getDistribution(d) != tempRLayout_m->getDistribution(d))
                skipTemp = false;

        if ( out_layout.numVnodes() != tempRLayout_m->numVnodes() )
            skipTemp = false;

        // also make sure there are no guard cells
        if (!(g.getGC() == FFT<RCTransform,Dim,T>::nullGC))
            skipTemp = false;
    }

    if (skipTemp)
        tempR = &g;
    else
        tempR = tempRField_m;

    skipTemp = true;
    if (nTransformDims == 1 && !constInput) {
        // only one CR transform
        // see if we really need to transpose input data
        // more rigorous match required here; check that layouts are identical
        if (!(in_layout == *tempLayouts_m[0])) {
            skipTemp = false;
        } else {
            for (d=0; d<Dim; ++d)
                if (in_layout.getDistribution(d) !=
                    tempLayouts_m[0]->getDistribution(d))
                    skipTemp = false;

            if ( in_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
                skipTemp = false;

            // also make sure there are no guard cells
            if (!(f.getGC() == FFT<RCTransform,Dim,T>::nullGC))
                skipTemp = false;
        }
    } else {  // cannot skip transpose
        skipTemp = false;
    }


    if (!skipTemp) {
        // transpose and permute to complex Field with transform dim first
        (*tempFields_m[0])[tempLayouts_m[0]->getDomain()] =
            (*temp)[temp->getLayout().getDomain()];

        // compress previous iterates storage
        if (this->compressTemps() && temp != &f)
            *temp = 0;
        temp = tempFields_m[0];
    }

    // Loop over all the Vnodes, working on the LField in each.
    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
        // Get the LFields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();

        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();

        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        // number of strips should be the same for real and complex LFields!
        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
        for (d=1; d<Dim; ++d)
            nstrips *= rldf->size(d);

        for (int istrip=0; istrip<nstrips; ++istrip) {
            // Do the 1D complex-to-real FFT:
            // note that complex-to-real FFT direction is always -1
            this->getEngine().callFFT(idim, -1, localcomp);

            // move the data into the real strip, which is two reals shorter
            for (int ilen=0; ilen<lengthreal; ilen+=2) {
                localreal[ilen] = real(localcomp[ilen/2]);
                localreal[ilen+1] = imag(localcomp[ilen/2]);
            }

            // advance the data pointers
            localreal += lengthreal;
            localcomp += lengthcomp;
        } // loop over 1D strips
    } // loop over all the LFields

    // compress previous iterates storage
    if (this->compressTemps() && temp != &f)
        *temp = 0;

    // skip final assignment and compress if we used g as final temporary
    if (tempR != &g) {


        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];

        if (this->compressTemps())
            *tempR = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    // finish up timing the whole mess
}

//=============================================================================
// 1D FFT RCTransform Constructors
//=============================================================================

//-----------------------------------------------------------------------------
// Create a new FFT object of type RCTransform, with a
// given domain. Also specify which dimensions to transform along.
// Note that RC transform of a real array of length n results in a
// complex array of length n/2+1.
//-----------------------------------------------------------------------------

template <class T>
FFT<RCTransform,1U,T>::FFT(
    const typename FFT<RCTransform,1U,T>::Domain_t& rdomain,
    const typename FFT<RCTransform,1U,T>::Domain_t& cdomain,
    const bool transformTheseDims[1U], const bool& compressTemps)
: FFTBase<1U,T>(FFT<RCTransform,1U,T>::rcFFT, rdomain,
                transformTheseDims, compressTemps),
    complexDomain_m(cdomain)
{
    size_t nTransformDims = 1U;
    // get axis length
    int length;
    length = rdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::rcFFT;    // first transform is real-to-complex
    T& normFact = this->getNormFact();
    normFact = 1.0 / length;

    // set up FFT Engine
    this->getEngine().setup(nTransformDims, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type RCTransform, with
// given real and complex domains. Default: transform along all dimensions.
//-----------------------------------------------------------------------------

template <class T>
FFT<RCTransform,1U,T>::FFT(
    const typename FFT<RCTransform,1U,T>::Domain_t& rdomain,
    const typename FFT<RCTransform,1U,T>::Domain_t& cdomain,
    const bool& compressTemps)
: FFTBase<1U,T>(FFT<RCTransform,1U,T>::rcFFT, rdomain, compressTemps),
    complexDomain_m(cdomain)
{
    // Tau profiling




    // get axis length
    int length;
    length = rdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::rcFFT;    // first transform is real-to-complex
    T& normFact = this->getNormFact();
    normFact = 1.0 / length;

    // set up FFT Engine
    this->getEngine().setup(1U, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// setup performs all the initializations necessary after the transform
// directions have been specified.
//-----------------------------------------------------------------------------

template <class T>
void
FFT<RCTransform,1U,T>::setup(void) {

    // Tau profiling




    // check that domain lengths agree between real and complex domains
    const Domain_t& domain = this->getDomain();
    bool match = true;
    // real array length n, complex array length n/2+1
    if ( complexDomain_m[0].length() !=
         (domain[0].length()/2 + 1) ) match = false;
    PInsist(match,
            "Domains provided for real and complex Fields are incompatible!");

    // generate layout for temporary real Field
    tempRLayout_m = new Layout_t(domain[0], PARALLEL, 1);
    // create temporary real Field
    this->tempRField_m = new RealField_t(*tempRLayout_m);
    // If user requests no intermediate compression, uncompress right now:
    if (!this->compressTemps()) this->tempRField_m->Uncompress();

    // generate temporary field layout
    tempLayouts_m = new Layout_t(complexDomain_m[0], PARALLEL, 1);
    // create temporary complex Field
    tempFields_m = new ComplexField_t(*tempLayouts_m);
    // If user requests no intermediate compression, uncompress right now:
    if (!this->compressTemps()) this->tempFields_m->Uncompress();

    return;
}

//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <class T>
FFT<RCTransform,1U,T>::~FFT(void) {

    // Tau profiling




    // delete temporary fields and layouts
    delete tempFields_m;
    delete tempLayouts_m;
    delete this->tempRField_m;
    delete tempRLayout_m;
}

//-----------------------------------------------------------------------------
// real-to-complex FFT; direction is +1 or -1
//-----------------------------------------------------------------------------

template <class T>
void
FFT<RCTransform,1U,T>::transform(
    int direction,
    typename FFT<RCTransform,1U,T>::RealField_t& f,
    typename FFT<RCTransform,1U,T>::ComplexField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(complexDomain_m,out_dom), true);

    // Common loop iterate and other vars:
    RealField_t* tempR = this->tempRField_m;  // Field* management aid
    if (!constInput) {
        // see if we can use input field f as a temporary
        bool skipTemp = true;
        // more rigorous match required here; check that layouts are identical
        if ( !(in_layout == *tempRLayout_m) ) skipTemp = false;
        if ( in_layout.getDistribution(0) !=
             tempRLayout_m->getDistribution(0) ) skipTemp = false;
        if ( in_layout.numVnodes() != tempRLayout_m->numVnodes() )
            skipTemp = false;
        // also make sure there are no guard cells
        if (!(f.getGC() == FFT<RCTransform,1U,T>::nullGC)) skipTemp = false;
        if (skipTemp) tempR = &f;
    }

    if (tempR != &f) {  // not using input as a temporary

        // assign to real Field with proper layout
        (*tempR) = f;

    }

    // Field* for temp Field management:
    ComplexField_t* temp = tempFields_m;
    // see if we can put final result directly into g
    bool skipFinal = true;
    // more rigorous match required here; check that layouts are identical
    if ( !(out_layout == *tempLayouts_m) ) skipFinal = false;
    if ( out_layout.getDistribution(0) !=
         tempLayouts_m->getDistribution(0) ) skipFinal = false;
    if ( out_layout.numVnodes() != tempLayouts_m->numVnodes() )
        skipFinal = false;
    // also make sure there are no guard cells
    if (!(g.getGC() == FFT<RCTransform,1U,T>::nullGC)) skipFinal = false;
    if (skipFinal) temp = &g;


    // There should be just one vnode!
    typename RealField_t::const_iterator_if rl_i = tempR->begin_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    if (rl_i != tempR->end_if() && cl_i != temp->end_if()) {
        // Get the LFields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();
        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();
        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        int lengthreal = rldf->size(0);
        // move the data into the complex strip, which is two reals longer
        for (int ilen=0; ilen<lengthreal; ilen+=2)
            localcomp[ilen/2] = Complex_t(localreal[ilen],localreal[ilen+1]);
        // Do the 1D real-to-complex FFT:
        // note that real-to-complex FFT direction is always +1
        this->getEngine().callFFT(0, +1, localcomp);
    }



    // compress temporary storage
    if (this->compressTemps() && tempR != &f) *tempR = 0;


    // skip final assignment and compress if we used g as final temporary
    if (temp != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g = (*temp);
        if (this->compressTemps()) *temp = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}

//-----------------------------------------------------------------------------
// RC FFT; opposite direction, from complex to real
//-----------------------------------------------------------------------------

template <class T>
void
FFT<RCTransform,1U,T>::transform(
    int direction,
    typename FFT<RCTransform,1U,T>::ComplexField_t& f,
    typename FFT<RCTransform,1U,T>::RealField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(complexDomain_m,in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true);

    // Field* for temp Field management:
    ComplexField_t* temp = &f;

    // see if we can put final result directly into g
    RealField_t* tempR;
    bool skipFinal = true;
    // more rigorous match required here; check that layouts are identical
    if ( !(out_layout == *tempRLayout_m) ) skipFinal = false;
    if ( out_layout.getDistribution(0) !=
         tempRLayout_m->getDistribution(0) ) skipFinal = false;
    if ( out_layout.numVnodes() != tempRLayout_m->numVnodes() )
        skipFinal = false;
    // also make sure there are no guard cells
    if (!(g.getGC() == FFT<RCTransform,1U,T>::nullGC)) skipFinal = false;
    if (skipFinal)
        tempR = &g;
    else
        tempR = this->tempRField_m;

    bool skipTemp = true;
    if (!constInput) {
        // only one CR transform
        // see if we really need to transpose input data
        // more rigorous match required here; check that layouts are identical
        if ( !(in_layout == *tempLayouts_m) ) skipTemp = false;
        if ( in_layout.getDistribution(0) !=
             tempLayouts_m->getDistribution(0) ) skipTemp = false;
        if ( in_layout.numVnodes() != tempLayouts_m->numVnodes() )
            skipTemp = false;
        // also make sure there are no guard cells
        if (!(f.getGC() == FFT<RCTransform,1U,T>::nullGC)) skipTemp = false;
    }
    else {  // cannot skip transpose
        skipTemp = false;
    }


    if (!skipTemp) {
        // assign to complex Field with proper layout
        (*tempFields_m) = (*temp);
        // compress previous iterates storage
        if (this->compressTemps() && temp != &f) *temp = 0;
        temp = tempFields_m;
    }



    // There should be just one vnode!
    typename RealField_t::const_iterator_if rl_i = tempR->begin_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    if (rl_i != tempR->end_if() && cl_i != temp->end_if()) {
        // Get the LFields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();
        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();
        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        int lengthreal = rldf->size(0);
        // Do the 1D complex-to-real FFT:
        // note that complex-to-real FFT direction is always -1
        this->getEngine().callFFT(0, -1, localcomp);
        // move the data into the real strip, which is two reals shorter
        for (int ilen=0; ilen<lengthreal; ilen+=2) {
            localreal[ilen] = real(localcomp[ilen/2]);
            localreal[ilen+1] = imag(localcomp[ilen/2]);
        }
    }



    // compress previous iterates storage
    if (this->compressTemps() && temp != &f) *temp = 0;


    // skip final assignment and compress if we used g as final temporary
    if (tempR != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g = (*tempR);
        if (this->compressTemps()) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}


//=============================================================================
// FFT SineTransform Constructors
//=============================================================================

//-----------------------------------------------------------------------------
// Create a new FFT object of type SineTransform, with given real and
// complex field domains. Also specify which dimensions to transform along,
// and which of these are sine transforms.
// Note that RC transform of a real array of length n results in a
// complex array of length n/2+1.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<SineTransform,Dim,T>::FFT(
    const typename FFT<SineTransform,Dim,T>::Domain_t& rdomain,
    const typename FFT<SineTransform,Dim,T>::Domain_t& cdomain,
    const bool transformTheseDims[Dim], const bool sineTransformDims[Dim],
    const bool& compressTemps)
: FFTBase<Dim,T>(FFT<SineTransform,Dim,T>::sineFFT, rdomain,
                 transformTheseDims, compressTemps),
    complexDomain_m(&cdomain)
{

    size_t d;
    // store which dimensions get sine transforms and count how many
    numSineTransforms_m = 0;
    for (d=0; d<Dim; ++d) {
        sineTransformDims_m[d] = sineTransformDims[d];
        if (sineTransformDims[d]) {
            PAssert_EQ(transformTheseDims[d], true);  // should be marked as a transform dim
            ++numSineTransforms_m;
        }
    }

    // construct array of axis lengths for all transform dims
    size_t nTransformDims = this->numTransformDims();
    int* lengths = new int[nTransformDims];
    for (d=0; d<nTransformDims; ++d)
        lengths[d] = rdomain[this->activeDimension(d)].length();

    // construct array of transform types for FFT Engine, compute normalization
    int* transformTypes = new int[nTransformDims];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    bool foundRC = false;
    for (d=0; d<nTransformDims; ++d) {
        if (sineTransformDims_m[this->activeDimension(d)]) {
            transformTypes[d] = FFTBase<Dim,T>::sineFFT;  // sine transform
            normFact /= (2.0 * (lengths[d] + 1));
        }
        else if (!foundRC) {
            transformTypes[d] = FFTBase<Dim,T>::rcFFT;    // real-to-complex FFT
            normFact /= lengths[d];
            foundRC = true;
        }
        else {
            transformTypes[d] = FFTBase<Dim,T>::ccFFT;    // complex-to-complex FFT
            normFact /= lengths[d];
        }
    }

    // set up FFT Engine
    this->getEngine().setup(nTransformDims, transformTypes, lengths);
    delete [] transformTypes;
    delete [] lengths;
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type SineTransform, with
// given real and complex domains. Default: transform along all dimensions.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<SineTransform,Dim,T>::FFT(
    const typename FFT<SineTransform,Dim,T>::Domain_t& rdomain,
    const typename FFT<SineTransform,Dim,T>::Domain_t& cdomain,
    const bool sineTransformDims[Dim], const bool& compressTemps)
: FFTBase<Dim,T>(FFT<SineTransform,Dim,T>::sineFFT, rdomain, compressTemps),
    complexDomain_m(&cdomain)
{

    size_t d;
    // store which dimensions get sine transforms and count how many
    numSineTransforms_m = 0;
    for (d=0; d<Dim; ++d) {
        sineTransformDims_m[d] = sineTransformDims[d];
        if (sineTransformDims[d]) ++numSineTransforms_m;
    }

    // construct array of axis lengths for all transform dims
    int lengths[Dim];
    for (d=0; d<Dim; ++d)
        lengths[d] = rdomain[d].length();

    // construct array of transform types for FFT Engine, compute normalization
    int transformTypes[Dim];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    bool foundRC = false;
    for (d=0; d<Dim; ++d) {
        if (sineTransformDims_m[d]) {
            transformTypes[d] = FFTBase<Dim,T>::sineFFT;  // sine transform
            normFact /= (2.0 * (lengths[d] + 1));
        }
        else if (!foundRC) {
            transformTypes[d] = FFTBase<Dim,T>::rcFFT;    // real-to-complex FFT
            normFact /= lengths[d];
            foundRC = true;
        }
        else {
            transformTypes[d] = FFTBase<Dim,T>::ccFFT;    // complex-to-complex FFT
            normFact /= lengths[d];
        }
    }

    // set up FFT Engine
    this->getEngine().setup(Dim, transformTypes, lengths);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Constructor for doing only sine transforms.
// Create a new FFT object of type SineTransform, with given real field
// domain. Also specify which dimensions to sine transform along.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<SineTransform,Dim,T>::FFT(
    const typename FFT<SineTransform,Dim,T>::Domain_t& rdomain,
    const bool sineTransformDims[Dim], const bool& compressTemps)
: FFTBase<Dim,T>(FFT<SineTransform,Dim,T>::sineFFT, rdomain,
                 sineTransformDims, compressTemps)
{
    // Tau profiling




    // store which dimensions get sine transforms and how many
    numSineTransforms_m = this->numTransformDims();
    size_t d;
    for (d=0; d<Dim; ++d)
        sineTransformDims_m[d] = sineTransformDims[d];

    // construct array of axis lengths
    int* lengths = new int[numSineTransforms_m];
    for (d=0; d<numSineTransforms_m; ++d)
        lengths[d] = rdomain[this->activeDimension(d)].length();

    // construct array of transform types for FFT Engine, compute normalization
    int* transformTypes = new int[numSineTransforms_m];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    for (d=0; d<numSineTransforms_m; ++d) {
        transformTypes[d] = FFTBase<Dim,T>::sineFFT;  // sine transform
        normFact /= (2.0 * (lengths[d] + 1));
    }

    // set up FFT Engine
    this->getEngine().setup(numSineTransforms_m, transformTypes, lengths);
    delete [] transformTypes;
    delete [] lengths;
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type SineTransform, with
// given real domain. Default: sine transform along all dimensions.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<SineTransform,Dim,T>::FFT(
    const typename FFT<SineTransform,Dim,T>::Domain_t& rdomain, const bool& compressTemps)
: FFTBase<Dim,T>(FFT<SineTransform,Dim,T>::sineFFT, rdomain, compressTemps)
{
    // Tau profiling




    size_t d;
    // store which dimensions get sine transforms and how many
    numSineTransforms_m = this->numTransformDims();
    for (d=0; d<Dim; ++d)
        sineTransformDims_m[d] = true;

    // construct array of axis lengths
    int lengths[Dim];
    for (d=0; d<Dim; ++d)
        lengths[d] = rdomain[d].length();

    // construct array of transform types for FFT Engine, compute normalization
    int transformTypes[Dim];
    T& normFact = this->getNormFact();
    normFact = 1.0;
    for (d=0; d<Dim; ++d) {
        transformTypes[d] = FFTBase<Dim,T>::sineFFT;  // sine transform
        normFact /= (2.0 * (lengths[d] + 1));
    }

    // set up FFT Engine
    this->getEngine().setup(Dim, transformTypes, lengths);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// setup performs all the initializations necessary after the transform
// directions have been specified.
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<SineTransform,Dim,T>::setup(void) {

    // Tau profiling




    size_t d, dim, activeDim = 0;
    size_t icount;
    Domain_t ndip;
    size_t nTransformDims = this->numTransformDims();  // total number of transforms
    const Domain_t& domain = this->getDomain();          // get real Field domain

    // Set up the arrays of temporary Fields and FieldLayouts:
    e_dim_tag serialParallel[Dim];  // Specifies SERIAL, PARALLEL dims in temp
    // make zeroth dimension always SERIAL
    serialParallel[0] = SERIAL;
    // all other dimensions parallel
    for (d=1; d<Dim; ++d)
        serialParallel[d] = PARALLEL;

    // do we have a real-to-complex transform to do or not?
    if (nTransformDims > numSineTransforms_m) {  // have RC transform

        PAssert(complexDomain_m);  // This pointer should be initialized!
        // find first non-sine transform dimension; this is rc transform
        bool match = false;
        d=0;
        while (d<Dim && !match) {
            if (this->transformDim(d) && !sineTransformDims_m[d]) {
                activeDim = this->activeDimension(d);
                match = true;
            }
            ++d;
        }
        PAssert_EQ(match, true);  // check that we found rc transform dimension
        // compare lengths of real and complex Field domains
        for (d=0; d<Dim; ++d) {
            if (d == activeDim) {
                // real array length n, complex array length n/2+1
                if ( (*complexDomain_m)[d].length() !=
                     (domain[d].length()/2 + 1) ) match = false;
            }
            else {
                // real and complex arrays should be same length for all other dims
                if ( (*complexDomain_m)[d].length() !=
                     domain[d].length() ) match = false;
            }
        }
        PInsist(match,
                "Domains provided for real and complex Fields are incompatible!");

        // set up the real Fields first
        // we will have one for each sine transform, plus one for the rc transform
        tempRLayouts_m = new Layout_t*[numSineTransforms_m+1];
        tempRFields_m = new RealField_t*[numSineTransforms_m+1];
        // loop over the sine transform dimensions
        icount=0;
        for (dim=0; dim<numSineTransforms_m; ++dim, ++icount) {
            // get next dimension to be sine transformed
            while (!sineTransformDims_m[icount]) ++icount;
            PAssert_LT(icount, Dim);  // check that icount is valid dimension
            // make new domain with permuted Indexes, icount first
            ndip[0] = domain[icount];
            for (d=1; d<Dim; ++d) {
                size_t nextDim = icount + d;
                if (nextDim >= Dim) nextDim -= Dim;
                ndip[d] = domain[nextDim];
            }
            // generate temporary field layout
            tempRLayouts_m[dim] = new Layout_t(ndip, serialParallel, this->transVnodes());
            // create temporary real Field
            tempRFields_m[dim] = new RealField_t(*tempRLayouts_m[dim]);
            // If user requests no intermediate compression, uncompress right now:
            if (!this->compressTemps()) (*tempRFields_m[dim]).Uncompress();
        }

        // build final real Field for rc transform along activeDim
        ndip[0] = domain[activeDim];
        for (d=1; d<Dim; ++d) {
            size_t nextDim = activeDim + d;
            if (nextDim >= Dim) nextDim -= Dim;
            ndip[d] = domain[nextDim];
        }
        // generate temporary field layout
        tempRLayouts_m[numSineTransforms_m] = new Layout_t(ndip, serialParallel,
                                                           this->transVnodes());
        // create temporary real Field
        tempRFields_m[numSineTransforms_m] =
            new RealField_t(*tempRLayouts_m[numSineTransforms_m]);
        // If user requests no intermediate compression, uncompress right now:
        if (!this->compressTemps()) (*tempRFields_m[numSineTransforms_m]).Uncompress();

        // now create the temporary complex Fields
        size_t numComplex = nTransformDims - numSineTransforms_m;
        // allocate arrays of temp fields and layouts
        tempLayouts_m = new Layout_t*[numComplex];
        tempFields_m = new ComplexField_t*[numComplex];
        icount=0;  // reset counter
        for (dim=0; dim<numComplex; ++dim, ++icount) {
            // get next non-sine transform dimension
            while (!this->transformDim(icount) || sineTransformDims_m[icount]) ++icount;
            PAssert_LT(icount, Dim);  // check that this is a valid dimension
            // make new domain with permuted Indexes, icount first
            ndip[0] = (*complexDomain_m)[icount];
            for (d=1; d<Dim; ++d) {
                size_t nextDim = icount + d;
                if (nextDim >= Dim) nextDim -= Dim;
                ndip[d] = (*complexDomain_m)[nextDim];
            }
            // generate temporary field layout
            tempLayouts_m[dim] = new Layout_t(ndip, serialParallel, this->transVnodes());
            // create temporary complex Field
            tempFields_m[dim] = new ComplexField_t(*tempLayouts_m[dim]);
            // If user requests no intermediate compression, uncompress right now:
            if (!this->compressTemps()) (*tempFields_m[dim]).Uncompress();
        }

    }
    else {  // sine transforms only

        // set up the real Fields
        // we will have one for each sine transform
        tempRLayouts_m = new Layout_t*[numSineTransforms_m];
        tempRFields_m = new RealField_t*[numSineTransforms_m];
        // loop over the sine transform dimensions
        for (dim=0; dim<numSineTransforms_m; ++dim) {
            // get next dimension to be sine transformed
            activeDim = this->activeDimension(dim);
            // make new domain with permuted Indexes, activeDim first
            ndip[0] = domain[activeDim];
            for (d=1; d<Dim; ++d) {
                size_t nextDim = activeDim + d;
                if (nextDim >= Dim) nextDim -= Dim;
                ndip[d] = domain[nextDim];
            }
            // generate temporary field layout
            tempRLayouts_m[dim] = new Layout_t(ndip, serialParallel, this->transVnodes());
            // create temporary real Field
            tempRFields_m[dim] = new RealField_t(*tempRLayouts_m[dim]);
            // If user requests no intermediate compression, uncompress right now:
            if (!this->compressTemps()) (*tempRFields_m[dim]).Uncompress();
        }

    }

    return;
}

//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
FFT<SineTransform,Dim,T>::~FFT(void) {

    // Tau profiling




    // delete temporary fields and layouts
    size_t d;
    size_t nTransformDims = this->numTransformDims();
    if (nTransformDims > numSineTransforms_m) {
        for (d=0; d<numSineTransforms_m+1; ++d) {
            delete tempRFields_m[d];
            delete tempRLayouts_m[d];
        }
        size_t numComplex = nTransformDims - numSineTransforms_m;
        for (d=0; d<numComplex; ++d) {
            delete tempFields_m[d];
            delete tempLayouts_m[d];
        }
        delete [] tempFields_m;
        delete [] tempLayouts_m;
    }
    else {
        for (d=0; d<numSineTransforms_m; ++d) {
            delete tempRFields_m[d];
            delete tempRLayouts_m[d];
        }
    }
    delete [] tempRFields_m;
    delete [] tempRLayouts_m;
}

//-----------------------------------------------------------------------------
// Sine and RC FFT; separate input and output fields, direction is +1 or -1
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<SineTransform,Dim,T>::transform(
    int direction,
    FFT<SineTransform,Dim,T>::RealField_t& f,
    FFT<SineTransform,Dim,T>::ComplexField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(*complexDomain_m,out_dom), true );

    // Common loop iterate and other vars:
    size_t d;
    int icount, activeDim;
    int idim;      // idim loops over the number of transform dims.
    size_t nTransformDims = this->numTransformDims();
    // check that there is a real-to-complex transform to do
    PInsist(nTransformDims>numSineTransforms_m,
            "Wrong output Field type for real-to-real transform!!");

    // first do all the sine transforms

    // Field* management aid
    RealField_t* tempR = &f;
    // Local work array passed to FFT:
    T* localdataR;

    // Loop over the dimensions to be sine transformed:
    icount = 0;
    activeDim = 0;
    for (idim = 0; idim != numSineTransforms_m; ++idim, ++icount, ++activeDim) {

        // find next sine transform dim
        while (!sineTransformDims_m[icount]) {
            if (this->transformDim(icount)) ++activeDim;
            ++icount;
        }
        PAssert_LT(activeDim, Dim);  // check that this is a valid dimension!
        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the transpose into the first temporary Field
        if (idim == 0 && !constInput) {
            // get domain for comparison
            const Domain_t& first_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                              (in_dom[0].length() == first_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempRFields_m[idim])[tempRLayouts_m[idim]->getDomain()] =
                (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && tempR != &f) *tempR = 0;
            tempR = tempRFields_m[idim];  // Field* management aid
        }



        // Loop over all the Vnodes, working on the LField in each.
        typename RealField_t::const_iterator_if l_i, l_end = tempR->end_if();
        for (l_i = tempR->begin_if(); l_i != l_end; ++l_i) {

            // Get the LField
            RealLField_t* ldf = (*l_i).second.get();
            // make sure we are uncompressed
            ldf->Uncompress();
            // get the raw data pointer
            localdataR = ldf->getP();

            // Do 1D real-to-real FFT's on all the strips in the LField:
            int nstrips = 1, length = ldf->size(0);
            for (d=1; d<Dim; ++d) nstrips *= ldf->size(d);
            for (int istrip=0; istrip<nstrips; ++istrip) {
                // Do the 1D FFT:
                this->getEngine().callFFT(activeDim, direction, localdataR);
                // advance the data pointer
                localdataR += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // now handle the RC transform separately

    // find first non-sine transform dimension
    icount = 0;
    activeDim = 0;
    while (!this->transformDim(icount) || sineTransformDims_m[icount]) {
        if (sineTransformDims_m[icount]) ++activeDim;
        ++icount;
    }
    PAssert_LT(activeDim, Dim);  // check that this is a valid dimension!


    // transpose and permute to final real Field with transform dim first
    int last = numSineTransforms_m;
    (*tempRFields_m[last])[tempRLayouts_m[last]->getDomain()] =
        (*tempR)[tempR->getLayout().getDomain()];

    // Compress out previous iterate's storage:
    if (this->compressTemps() && tempR != &f) *tempR = 0;
    tempR = tempRFields_m[last];  // Field* management aid


    // Field* for temp Field management:
    ComplexField_t* temp = tempFields_m[0];
    // see if we can put final result directly into g
    int numComplex = nTransformDims-numSineTransforms_m;
    if (numComplex == 1) {  // only a single RC transform
        bool skipTemp = true;
        // more rigorous match required here; check that layouts are identical
        if ( !(out_layout == *tempLayouts_m[0]) ) skipTemp = false;
        for (d=0; d<Dim; ++d) {
            if ( out_layout.getDistribution(d) !=
                 tempLayouts_m[0]->getDistribution(d) ) skipTemp = false;
        }
        if ( out_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
            skipTemp = false;
        // also make sure there are no guard cells
        if (!(g.getGC() == FFT<SineTransform,Dim,T>::nullGC)) skipTemp = false;
        if (skipTemp) temp = &g;
    }



    // Loop over all the Vnodes, working on the LField in each.
    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
        // Get the LFields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();
        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();
        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
        // number of strips should be the same for real and complex LFields!
        for (d=1; d<Dim; ++d) nstrips *= rldf->size(d);
        for (int istrip=0; istrip<nstrips; ++istrip) {
            // move the data into the complex strip, which is two reals longer
            for (int ilen=0; ilen<lengthreal; ilen+=2)
                localcomp[ilen/2] = Complex_t(localreal[ilen],localreal[ilen+1]);
            // Do the 1D real-to-complex FFT:
            // note that real-to-complex FFT direction is always +1
            this->getEngine().callFFT(activeDim, +1, localcomp);
            // advance the data pointers
            localreal += lengthreal;
            localcomp += lengthcomp;
        } // loop over 1D strips
    } // loop over all the LFields


    // now proceed with the other complex-to-complex transforms

    // Local work array passed to FFT:
    Complex_t* localdata;

    // Loop over the remaining dimensions to be transformed:
    ++icount;
    ++activeDim;
    for (idim = 1; idim != numComplex; ++idim, ++icount, ++activeDim) {

        // find the next non-sine transform dimension
        while (!this->transformDim(icount) || sineTransformDims_m[icount]) {
            if (sineTransformDims_m[icount]) ++activeDim;
            ++icount;
        }
        PAssert_LT(activeDim, Dim);  // check that this is a valid dimension!
        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the last transform dimension, we might be able
        // to skip the last temporary and transpose right into g
        if (idim == numComplex-1) {
            // get the domain for comparison
            const Domain_t& last_dom = tempLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (out_dom[0].sameBase(last_dom[0])) &&
                              (out_dom[0].length() == last_dom[0].length()) &&
                              (out_layout.getDistribution(0) == SERIAL) &&
                              (g.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps()) *temp = 0;
            temp = tempFields_m[idim];  // Field* management aid
        }
        else if (idim == numComplex-1) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using g instead

            // transpose and permute to Field with transform dim first
            g[out_dom] = (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps()) *temp = 0;
            temp = &g;  // Field* management aid
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
                this->getEngine().callFFT(activeDim, direction, localdata);
                // advance the data pointer
                localdata += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // skip final assignment and compress if we used g as final temporary
    if (temp != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*temp)[temp->getLayout().getDomain()];
        if (this->compressTemps()) *temp = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}

//-----------------------------------------------------------------------------
// Sine and RC FFT; opposite direction, from complex to real
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<SineTransform,Dim,T>::transform(
    int direction,
    FFT<SineTransform,Dim,T>::ComplexField_t& f,
    FFT<SineTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    // INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(*complexDomain_m,in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true );

    // Common loop iterate and other vars:
    size_t d;
    int icount, activeDim;
    int idim;      // idim loops over the number of transform dims.
    size_t nTransformDims = this->numTransformDims();

    // proceed with the complex-to-complex transforms

    // Field* for temp Field management:
    ComplexField_t* temp = &f;
    // Local work array passed to FFT:
    Complex_t* localdata;

    // Loop over all dimensions to be non-sine transformed except last one:
    int numComplex = nTransformDims - numSineTransforms_m;
    icount = Dim-1;  // start with last dimension
    activeDim = nTransformDims-1;
    for (idim = numComplex-1; idim != 0; --idim, --icount, --activeDim) {

        // find next non-sine transform dim
        while (!this->transformDim(icount) || sineTransformDims_m[icount]) {
            if (sineTransformDims_m[icount]) --activeDim;
            --icount;
        }
        PAssert_GE(activeDim, 0);  // check that this is a valid dimension!
        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the transpose into the first temporary Field
        if (idim == numComplex-1 && !constInput) {
            // get domain for comparison
            const Domain_t& first_dom = tempLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                              (in_dom[0].length() == first_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
                (*temp)[temp->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && temp != &f) *temp = 0;
            temp = tempFields_m[idim];  // Field* management aid
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
                this->getEngine().callFFT(activeDim, direction, localdata);
                // advance the data pointer
                localdata += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // handle the CR transform separately

    // find next non-sine transform dim
    while (!this->transformDim(icount) || sineTransformDims_m[icount]) {
        if (sineTransformDims_m[icount]) --activeDim;
        --icount;
    }
    PAssert_GE(activeDim, 0);  // check that this is a valid dimension!

    // Temp Field* management aid
    RealField_t* tempR = tempRFields_m[numSineTransforms_m];

    bool skipTemp = true;
    if (numComplex == 1 && !constInput) {
        // only one CR transform
        // see if we really need to transpose input data
        // more rigorous match required here; check that layouts are identical
        if ( !(in_layout == *tempLayouts_m[0]) ) skipTemp = false;
        for (d=0; d<Dim; ++d) {
            if ( in_layout.getDistribution(d) !=
                 tempLayouts_m[0]->getDistribution(d) ) skipTemp = false;
        }
        if ( in_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
            skipTemp = false;
        // also make sure there are no guard cells
        if (!(f.getGC() == FFT<SineTransform,Dim,T>::nullGC)) skipTemp = false;
    }
    else {  // cannot skip transpose
        skipTemp = false;
    }

    if (!skipTemp) {

        // transpose and permute to complex Field with transform dim first
        (*tempFields_m[0])[tempLayouts_m[0]->getDomain()] =
            (*temp)[temp->getLayout().getDomain()];
        // compress previous iterates storage
        if (this->compressTemps() && temp != &f) *temp = 0;
        temp = tempFields_m[0];

    }


    // Loop over all the Vnodes, working on the LField in each.
    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
        // Get the LFields
        RealLField_t* rldf = (*rl_i).second.get();
        ComplexLField_t* cldf = (*cl_i).second.get();
        // make sure we are uncompressed
        rldf->Uncompress();
        cldf->Uncompress();
        // get the raw data pointers
        T* localreal = rldf->getP();
        Complex_t* localcomp = cldf->getP();

        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
        // number of strips should be the same for real and complex LFields!
        for (d=1; d<Dim; ++d) nstrips *= rldf->size(d);
        for (int istrip=0; istrip<nstrips; ++istrip) {
            // Do the 1D complex-to-real FFT:
            // note that complex-to-real FFT direction is always -1
            this->getEngine().callFFT(activeDim, -1, localcomp);
            // move the data into the real strip, which is two reals shorter
            for (int ilen=0; ilen<lengthreal; ilen+=2) {
                localreal[ilen] = real(localcomp[ilen/2]);
                localreal[ilen+1] = imag(localcomp[ilen/2]);
            }
            // advance the data pointers
            localreal += lengthreal;
            localcomp += lengthcomp;
        } // loop over 1D strips
    } // loop over all the LFields



    // compress previous iterates storage
    if (this->compressTemps() && temp != &f) *temp = 0;


    // now do the real-to-real FFTs

    // Local work array passed to FFT:
    T* localdataR;

    // Loop over the remaining dimensions to be transformed:
    icount = Dim-1;  // start with last dimension
    activeDim = nTransformDims - 1;
    for (idim = numSineTransforms_m-1; idim != -1;
         --idim, --icount, --activeDim) {

        // find next sine transform dim
        while (!sineTransformDims_m[icount]) {
            if (this->transformDim(icount)) --activeDim;
            --icount;
        }
        PAssert_GE(activeDim, 0);  // check that this is a valid dimension!
        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the last transform dimension, we might be able
        // to skip the last temporary and transpose right into g
        if (idim == 0) {
            // get the domain for comparison
            const Domain_t& last_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (out_dom[0].sameBase(last_dom[0])) &&
                              (out_dom[0].length() == last_dom[0].length()) &&
                              (out_layout.getDistribution(0) == SERIAL) &&
                              (g.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempRFields_m[idim])[tempRLayouts_m[idim]->getDomain()] =
                (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps()) *tempR = 0;
            tempR = tempRFields_m[idim];  // Field* management aid
        }
        else if (idim == 0) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using g instead

            // transpose and permute to Field with transform dim first
            g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps()) *tempR = 0;
            tempR = &g;  // Field* management aid
        }



        // Loop over all the Vnodes, working on the LField in each.
        typename RealField_t::const_iterator_if l_i, l_end = tempR->end_if();
        for (l_i = tempR->begin_if(); l_i != l_end; ++l_i) {

            // Get the LField
            RealLField_t* ldf = (*l_i).second.get();
            // make sure we are uncompressed
            ldf->Uncompress();
            // get the raw data pointer
            localdataR = ldf->getP();

            // Do 1D complex-to-complex FFT's on all the strips in the LField:
            int nstrips = 1, length = ldf->size(0);
            for (d=1; d<Dim; ++d) nstrips *= ldf->size(d);
            for (int istrip=0; istrip<nstrips; ++istrip) {
                // Do the 1D FFT:
                this->getEngine().callFFT(activeDim, direction, localdataR);
                // advance the data pointer
                localdataR += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // skip final assignment and compress if we used g as final temporary
    if (tempR != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];
        if (this->compressTemps()) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}

//-----------------------------------------------------------------------------
// Sine FFT only; separate input and output fields, direction is +1 or -1
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<SineTransform,Dim,T>::transform(
    int direction,
    FFT<SineTransform,Dim,T>::RealField_t& f,
    FFT<SineTransform,Dim,T>::RealField_t& g,
    const bool& constInput)
{

    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true );

    // Common loop iterate and other vars:
    size_t d;
    int idim;      // idim loops over the number of transform dims.
    int begdim, enddim;
    size_t nTransformDims = this->numTransformDims();
    // check that there is no real-to-complex transform to do
    PInsist(nTransformDims==numSineTransforms_m,
            "Wrong output Field type for real-to-complex transform!!");

    // do all the sine transforms

    // Field* management aid
    RealField_t* tempR = &f;
    // Local work array passed to FFT:
    T* localdataR;

    // Loop over the dimensions to be sine transformed:
    begdim = (direction == +1) ? 0 : static_cast<int>(nTransformDims-1);
    enddim = (direction == +1) ? static_cast<int>(nTransformDims) : -1;
    for (idim = begdim; idim != enddim; idim+=direction) {

        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the first temporary and just use f
        if (idim == begdim && !constInput) {
            // get the domain for comparison
            const Domain_t& first_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                              (in_dom[0].length() == first_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        // if this is the last transform dimension, we might be able
        // to skip the last temporary and transpose right into g
        if (idim == enddim-direction) {
            // get the domain for comparison
            const Domain_t& last_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (out_dom[0].sameBase(last_dom[0])) &&
                              (out_dom[0].length() == last_dom[0].length()) &&
                              (out_layout.getDistribution(0) == SERIAL) &&
                              (g.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempRFields_m[idim])[tempRLayouts_m[idim]->getDomain()] =
                (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && tempR != &f) *tempR = 0;
            tempR = tempRFields_m[idim];  // Field* management aid
        }
        else if (idim == enddim-direction && tempR != &g) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using g instead

            // transpose and permute to Field with transform dim first
            g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && tempR != &f) *tempR = 0;
            tempR = &g;  // Field* management aid
        }



        // Loop over all the Vnodes, working on the LField in each.
        typename RealField_t::const_iterator_if l_i, l_end = tempR->end_if();
        for (l_i = tempR->begin_if(); l_i != l_end; ++l_i) {

            // Get the LField
            RealLField_t* ldf = (*l_i).second.get();
            // make sure we are uncompressed
            ldf->Uncompress();
            // get the raw data pointer
            localdataR = ldf->getP();

            // Do 1D real-to-real FFT's on all the strips in the LField:
            int nstrips = 1, length = ldf->size(0);
            for (d=1; d<Dim; ++d) nstrips *= ldf->size(d);
            for (int istrip=0; istrip<nstrips; ++istrip) {
                // Do the 1D FFT:
                this->getEngine().callFFT(idim, direction, localdataR);
                // advance the data pointer
                localdataR += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // skip final assignment and compress if we used g as final temporary
    if (tempR != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];
        if (this->compressTemps() && tempR != &f) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}

//-----------------------------------------------------------------------------
// Sine FFT only; in-place transform, direction is +1 or -1
//-----------------------------------------------------------------------------

template <size_t Dim, class T>
void
FFT<SineTransform,Dim,T>::transform(
    int direction,
    FFT<SineTransform,Dim,T>::RealField_t& f)
{
    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Field
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    PAssert_EQ(this->checkDomain(this->getDomain(),in_dom), true);

    // Common loop iterate and other vars:
    size_t d;
    int idim;      // idim loops over the number of transform dims.
    int begdim, enddim;
    size_t nTransformDims = this->numTransformDims();
    // check that there is no real-to-complex transform to do
    PInsist(nTransformDims==numSineTransforms_m,
            "Cannot perform real-to-complex transform in-place!!");

    // do all the sine transforms

    // Field* management aid
    RealField_t* tempR = &f;
    // Local work array passed to FFT:
    T* localdataR;

    // Loop over the dimensions to be sine transformed:
    begdim = (direction == +1) ? 0 : static_cast<int>(nTransformDims-1);
    enddim = (direction == +1) ? static_cast<int>(nTransformDims) : -1;
    for (idim = begdim; idim != enddim; idim+=direction) {

        // Now do the serial transforms along this dimension:

        bool skipTranspose = false;
        // if this is the first transform dimension, we might be able
        // to skip the transpose into the first temporary Field
        if (idim == begdim) {
            // get domain for comparison
            const Domain_t& first_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(first_dom[0])) &&
                              (in_dom[0].length() == first_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        // if this is the last transform dimension, we might be able
        // to skip the last temporary and transpose right into f
        if (idim == enddim-direction) {
            // get the domain for comparison
            const Domain_t& last_dom = tempRLayouts_m[idim]->getDomain();
            // check that zeroth axis is the same and is serial
            // and that there are no guard cells
            skipTranspose = ( (in_dom[0].sameBase(last_dom[0])) &&
                              (in_dom[0].length() == last_dom[0].length()) &&
                              (in_layout.getDistribution(0) == SERIAL) &&
                              (f.getGC() == FFT<SineTransform,Dim,T>::nullGC) );
        }

        if (!skipTranspose) {
            // transpose and permute to Field with transform dim first
            (*tempRFields_m[idim])[tempRLayouts_m[idim]->getDomain()] =
                (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && tempR != &f) *tempR = 0;
            tempR = tempRFields_m[idim];  // Field* management aid
        }
        else if (idim == enddim-direction && tempR != &f) {
            // last transform and we can skip the last temporary field
            // so do the transpose here using f instead

            // transpose and permute to Field with transform dim first
            f[in_dom] = (*tempR)[tempR->getLayout().getDomain()];

            // Compress out previous iterate's storage:
            if (this->compressTemps() && tempR != &f) *tempR = 0;
            tempR = &f;  // Field* management aid
        }



        // Loop over all the Vnodes, working on the LField in each.
        typename RealField_t::const_iterator_if l_i, l_end = tempR->end_if();
        for (l_i = tempR->begin_if(); l_i != l_end; ++l_i) {

            // Get the LField
            RealLField_t* ldf = (*l_i).second.get();
            // make sure we are uncompressed
            ldf->Uncompress();
            // get the raw data pointer
            localdataR = ldf->getP();

            // Do 1D real-to-real FFT's on all the strips in the LField:
            int nstrips = 1, length = ldf->size(0);
            for (d=1; d<Dim; ++d) nstrips *= ldf->size(d);
            for (int istrip=0; istrip<nstrips; ++istrip) {
                // Do the 1D FFT:
                this->getEngine().callFFT(idim, direction, localdataR);
                // advance the data pointer
                localdataR += length;
            } // loop over 1D strips
        } // loop over all the LFields

    } // loop over all transformed dimensions

    // skip final assignment and compress if we used g as final temporary
    if (tempR != &f) {

        // Now assign into output Field, and compress last temp's storage:
        f[in_dom] = (*tempR)[tempR->getLayout().getDomain()];
        if (this->compressTemps()) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) f = f * this->getNormFact();

    return;
}


//=============================================================================
// 1D FFT SineTransform Constructors
//=============================================================================

//-----------------------------------------------------------------------------
// Constructor for doing only sine transforms.
// Create a new FFT object of type SineTransform, with given real field
// domain. Also specify which dimensions to sine transform along.
//-----------------------------------------------------------------------------

template <class T>
FFT<SineTransform,1U,T>::FFT(
    const typename FFT<SineTransform,1U,T>::Domain_t& rdomain,
    const bool sineTransformDims[1U], const bool& compressTemps)
: FFTBase<1U,T>(FFT<SineTransform,1U,T>::sineFFT, rdomain,
                sineTransformDims, compressTemps)
{
    // Tau profiling




    // get axis length
    int length;
    length = rdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::sineFFT;  // sine transform
    T& normFact = this->getNormFact();
    normFact = 1.0 / (2.0 * (length + 1));

    // set up FFT Engine
    this->getEngine().setup(1U, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// Create a new FFT object of type SineTransform, with
// given real domain. Default: sine transform along all dimensions.
//-----------------------------------------------------------------------------

template <class T>
FFT<SineTransform,1U,T>::FFT(
    const typename FFT<SineTransform,1U,T>::Domain_t& rdomain,
    const bool& compressTemps)
: FFTBase<1U,T>(FFT<SineTransform,1U,T>::sineFFT, rdomain, compressTemps)
{
    // Tau profiling




    // get axis length
    int length;
    length = rdomain[0].length();

    // get transform type for FFT Engine, compute normalization
    int transformType;
    transformType = FFTBase<1U,T>::sineFFT;  // sine transform
    T& normFact = this->getNormFact();
    normFact = 1.0 / (2.0 * (length + 1));

    // set up FFT Engine
    this->getEngine().setup(1U, &transformType, &length);
    // set up the temporary fields
    setup();
}

//-----------------------------------------------------------------------------
// setup performs all the initializations necessary after the transform
// directions have been specified.
//-----------------------------------------------------------------------------

template <class T>
void
FFT<SineTransform,1U,T>::setup(void) {

    // Tau profiling




    const Domain_t& domain = this->getDomain();          // get real Field domain

    // generate temporary field layout
    tempRLayouts_m = new Layout_t(domain[0], PARALLEL, 1);
    // create temporary real Field
    tempRFields_m = new RealField_t(*tempRLayouts_m);
    // If user requests no intermediate compression, uncompress right now:
    if (!this->compressTemps()) tempRFields_m->Uncompress();

    return;
}

//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------

template <class T>
FFT<SineTransform,1U,T>::~FFT(void) {

    // Tau profiling




    // delete temporary field and layout
    delete tempRFields_m;
    delete tempRLayouts_m;
}

//-----------------------------------------------------------------------------
// Sine FFT only; separate input and output fields, direction is +1 or -1
//-----------------------------------------------------------------------------

template <class T>
void
FFT<SineTransform,1U,T>::transform(
    int direction,
    FFT<SineTransform,1U,T>::RealField_t& f,
    FFT<SineTransform,1U,T>::RealField_t& g,
    const bool& constInput)
{
    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Fields
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    const Layout_t& out_layout = g.getLayout();
    const Domain_t& out_dom = out_layout.getDomain();
    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
                this->checkDomain(this->getDomain(),out_dom), true);

    // Field* management aid
    RealField_t* tempR = &f;
    // Local work array passed to FFT:
    T* localdataR;


    // Now do the serial transform along this dimension:

    // get the domain for comparison
    const Domain_t& temp_dom = tempRLayouts_m->getDomain();
    bool skipTranspose = false;
    // we might be able
    // to skip the first temporary and just use f
    if (!constInput) {
        // check that zeroth axis is the same, has one vnode
        // and that there are no guard cells
        skipTranspose = ( (in_dom[0].sameBase(temp_dom[0])) &&
                          (in_dom[0].length() == temp_dom[0].length()) &&
                          (in_layout.numVnodes() == 1) &&
                          (f.getGC() == FFT<SineTransform,1U,T>::nullGC) );
    }

    bool skipFinal = false;
    // we might be able
    // to skip the last temporary and transpose right into g

    // check that zeroth axis is the same, has one vnode
    // and that there are no guard cells
    skipFinal = ( (out_dom[0].sameBase(temp_dom[0])) &&
                  (out_dom[0].length() == temp_dom[0].length()) &&
                  (out_layout.numVnodes() == 1) &&
                  (g.getGC() == FFT<SineTransform,1U,T>::nullGC) );

    if (!skipTranspose) {
        // assign to Field with proper layout
        (*tempRFields_m) = (*tempR);
        tempR = tempRFields_m;  // Field* management aid
    }
    if (skipFinal) {
        // we can skip the last temporary field
        // so do the transpose here using g instead

        // assign to Field with proper layout
        g = (*tempR);

        // Compress out previous iterate's storage:
        if (this->compressTemps() && tempR != &f) *tempR = 0;
        tempR = &g;  // Field* management aid
    }



    // There should be just one LField.
    typename RealField_t::const_iterator_if l_i = tempR->begin_if();
    if (l_i != tempR->end_if()) {

        // Get the LField
        RealLField_t* ldf = (*l_i).second.get();
        // make sure we are uncompressed
        ldf->Uncompress();
        // get the raw data pointer
        localdataR = ldf->getP();

        // Do the 1D FFT:
        this->getEngine().callFFT(0, direction, localdataR);
    }


    // skip final assignment and compress if we used g as final temporary
    if (tempR != &g) {

        // Now assign into output Field, and compress last temp's storage:
        g = (*tempR);
        if (this->compressTemps() && tempR != &f) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) g = g * this->getNormFact();

    return;
}

//-----------------------------------------------------------------------------
// Sine FFT only; in-place transform, direction is +1 or -1
//-----------------------------------------------------------------------------

template <class T>
void
FFT<SineTransform,1U,T>::transform(
    int direction,
    FFT<SineTransform,1U,T>::RealField_t& f)
{
    // indicate we're doing another FFT
    //INCIPPLSTAT(incFFTs);

    // Check domain of incoming Field
    const Layout_t& in_layout = f.getLayout();
    const Domain_t& in_dom = in_layout.getDomain();
    PAssert_EQ(this->checkDomain(this->getDomain(),in_dom), true);

    // Field* management aid
    RealField_t* tempR = &f;
    // Local work array passed to FFT:
    T* localdataR;


    // Now do the serial transform along this dimension:

    // get domain for comparison
    const Domain_t& temp_dom = tempRLayouts_m->getDomain();
    bool skipTranspose = false;
    // we might be able
    // to skip the transpose into the first temporary Field

    // check that zeroth axis is the same and is serial
    // and that there are no guard cells
    skipTranspose = ( (in_dom[0].sameBase(temp_dom[0])) &&
                      (in_dom[0].length() == temp_dom[0].length()) &&
                      (in_layout.numVnodes() == 1) &&
                      (f.getGC() == FFT<SineTransform,1U,T>::nullGC) );

    bool skipFinal = false;
    // we might be able
    // to skip the last temporary and transpose right into f

    // check that zeroth axis is the same, has one vnode
    // and that there are no guard cells
    skipFinal = ( (in_dom[0].sameBase(temp_dom[0])) &&
                  (in_dom[0].length() == temp_dom[0].length()) &&
                  (in_layout.numVnodes() == 1) &&
                  (f.getGC() == FFT<SineTransform,1U,T>::nullGC) );

    if (!skipTranspose) {
        // assign to Field with proper layout
        (*tempRFields_m) = (*tempR);

        tempR = tempRFields_m;  // Field* management aid
    }
    if (skipFinal) {
        // we can skip the last temporary field
        // so do the transpose here using f instead

        // assign to Field with proper layout
        f = (*tempR);

        // Compress out previous iterate's storage:
        if (this->compressTemps() && tempR != &f) *tempR = 0;
        tempR = &f;  // Field* management aid
    }



    // There should be just one LField.
    typename RealField_t::const_iterator_if l_i = tempR->begin_if();
    if (l_i != tempR->end_if()) {

        // Get the LField
        RealLField_t* ldf = (*l_i).second.get();
        // make sure we are uncompressed
        ldf->Uncompress();
        // get the raw data pointer
        localdataR = ldf->getP();

        // Do the 1D FFT:
        this->getEngine().callFFT(0, direction, localdataR);
    }


    // skip final assignment and compress if we used g as final temporary
    if (tempR != &f) {

        // Now assign into output Field, and compress last temp's storage:
        f = (*tempR);
        if (this->compressTemps()) *tempR = 0;

    }

    // Normalize:
    if (direction == +1) f = f * this->getNormFact();

    return;
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End: