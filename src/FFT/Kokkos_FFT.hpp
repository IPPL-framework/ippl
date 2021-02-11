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

#include "FFT/Kokkos_FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/BareField.h"


namespace ippl {

    //=============================================================================
    // FFT CCTransform Constructors
    //=============================================================================
    
    /**
       Create a new FFT object of type CCTransform, with a
       given domain. Also specify which dimensions to transform along.
    */
    
    template <size_t Dim, class T>
    FFT<CCTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const HeffteParams& params)
//    : FFTBase<Dim,T>(FFT<CCTransform,Dim,T>::ccFFT, cdomain)
    {
    
        std::array<int, Dim> low; 
        std::array<int, Dim> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low = {(int)lDom[0].first(), (int)lDom[1].first(), (int)lDom[2].first()};
        high = {(int)lDom[0].length() + (int)lDom[0].first() - 1,
                (int)lDom[1].length() + (int)lDom[1].first() - 1,
                (int)lDom[2].length() + (int)lDom[2].first() - 1};

        setup(low, high, params);
    }
    
    
    /**
       setup performs all the initializations necessary after the transform
       directions have been specified.
    */
    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::setup(const std::array<int, Dim>& low, 
                                  const std::array<int, Dim>& high,
                                  const HeffteParams& params)
    {
   
         heffte::box3d inbox = { low, high };
         heffte::box3d outbox = { low, high };

         heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();
         heffteOptions.use_alltoall = params.getAllToAll();
         heffteOptions.use_pencils = params.getPencils();
         heffteOptions.use_reorder = params.getReorder();

         heffte_m = std::make_shared<heffte::fft3d<heffteBackend>>(inbox, outbox, Ippl::getComm(), heffteOptions);
         
         //int fftsize = std::max( heffte_m->size_outbox(), heffte_m->size_inbox() );
         //tempField_m = Kokkos::View<heffteComplex_t*>(Kokkos::ViewAllocateWithoutInitializing( "tempField_m" ), fftsize );
  
        //return;
    }
    
 
    
    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::transform(
        int direction,
        typename FFT<CCTransform,Dim,T>::ComplexField_t& f)
        //ComplexField_t& f)
    {
       auto fview = f.getView();
       const int nghost = f.getNghost();
       //std::array<int, Dim> length;

       //length = {(int)fview.extent(0) - nghost, 
       //          (int)fview.extent(1) - nghost,
       //          (int)fview.extent(2) - nghost};
     
       //auto viewtempField = createView<heffteComplex_t, Kokkos::LayoutRight>(length, tempField_m.data());
       
       Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight> tempField("tempField", fview.extent(0) - 2*nghost,
                                                                                   fview.extent(1) - 2*nghost,
                                                                                   fview.extent(2) - 2*nghost);
       //Kokkos::resize(tempField, fview.extent(0) - nghost, 
       //                          fview.extent(1) - nghost,
       //                          fview.extent(2) - nghost);

       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

       Kokkos::parallel_for("copy from Kokkos FFT",
                            mdrange_type({nghost, nghost, nghost},
                                         {fview.extent(0) - nghost, 
                                          fview.extent(1) - nghost,
                                          fview.extent(2) - nghost
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                                const size_t j,
                                                const size_t k)
                            {
                              //tempField_m(i, j, k) = this->copyFromKokkosComplex(fview(i, j, k), 
                              //                                               tempField_m(i, j, k));  
                              //this->copyFromKokkosComplex(fview(i, j, k), tempField(i, j, k));  
#ifdef KOKKOS_ENABLE_CUDA
                              tempField(i-nghost, j-nghost, k-nghost).x = fview(i, j, k).real();
                              tempField(i-nghost, j-nghost, k-nghost).y = fview(i, j, k).imag();
                              //viewtempField(i, j, k).x = fview(i, j, k).real();
                              //viewtempField(i, j, k).y = fview(i, j, k).imag();
#else
                              tempField(i, j, k).real() = fview(i, j, k).real();
                              tempField(i, j, k).imag() = fview(i, j, k).imag();
#endif
                            });
       if ( direction == 1 )
       {
           heffte_m->forward( tempField.data(), tempField.data(), heffte::scale::full );
           //heffte_m->forward( viewtempField.data(), viewtempField.data(), heffte::scale::full );
       }
       else if ( direction == -1 )
       {
           heffte_m->backward( tempField.data(), tempField.data(), heffte::scale::none );
           //heffte_m->backward( viewtempField.data(), viewtempField.data(), heffte::scale::none );
       }
       else
       {
           throw std::logic_error( "Only 1:forward and -1:backward are allowed as directions" );
       }

    
       Kokkos::parallel_for("copy to Kokkos FFT",
                            mdrange_type({nghost, nghost, nghost},
                                         {fview.extent(0) - nghost, 
                                          fview.extent(1) - nghost,
                                          fview.extent(2) - nghost
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
                            {
                              //fview(i, j, k) = this->copyToKokkosComplex(tempField_m(i, j, k), 
                              //                                     fview(i, j, k));  
                              //this->copyToKokkosComplex(tempField(i, j, k), fview(i, j, k));  
#ifdef KOKKOS_ENABLE_CUDA
                              fview(i, j, k).real() = tempField(i-nghost, j-nghost, k-nghost).x;
                              fview(i, j, k).imag() = tempField(i-nghost, j-nghost, k-nghost).y;
                              //fview(i, j, k).real() = viewtempField(i, j, k).x;
                              //fview(i, j, k).imag() = viewtempField(i, j, k).y;
#else
                              fview(i, j, k).real() = tempField(i, j, k).real();
                              fview(i, j, k).imag() = tempField(i, j, k).imag();
#endif
                            });
    
    }
    
    template <class T, class... Params>
    Kokkos::View<T***, Params..., Kokkos::MemoryUnmanaged>
    createView( const std::array<int, 3>& length, T* data )
    {
        return Kokkos::View<T***, Params..., Kokkos::MemoryUnmanaged>(
               data, length[0], length[1], length[2]);
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
    
    //template <size_t Dim, class T>
    //FFT<RCTransform,Dim,T>::FFT(
    //    const typename FFT<RCTransform,Dim,T>::Domain_t& rdomain,
    //    const typename FFT<RCTransform,Dim,T>::Domain_t& cdomain,
    //    const bool transformTheseDims[Dim], const bool& compressTemps)
    //: FFTBase<Dim,T>(FFT<RCTransform,Dim,T>::rcFFT, rdomain,
    //                 transformTheseDims, compressTemps),
    //    complexDomain_m(cdomain), serialAxes_m(1)
    //{
    //    // construct array of axis lengths
    //    size_t nTransformDims = this->numTransformDims();
    //    int* lengths = new int[nTransformDims];
    //    size_t d;
    //    for (d=0; d<nTransformDims; ++d)
    //        lengths[d] = rdomain[this->activeDimension(d)].length();
    //
    //    // construct array of transform types for FFT Engine, compute normalization
    //    int* transformTypes = new int[nTransformDims];
    //    T& normFact = this->getNormFact();
    //    normFact = 1.0;
    //    transformTypes[0] = FFTBase<Dim,T>::rcFFT;    // first transform is real-to-complex
    //    normFact /= lengths[0];
    //    for (d=1; d<nTransformDims; ++d) {
    //        transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all other transforms are complex-to-complex
    //        normFact /= lengths[d];
    //    }
    //
    //    // set up FFT Engine
    //    this->getEngine().setup(nTransformDims, transformTypes, lengths);
    //    delete [] transformTypes;
    //    delete [] lengths;
    //
    //    // set up the temporary fields
    //    setup();
    //}
    //
    ////-----------------------------------------------------------------------------
    //// Create a new FFT object of type RCTransform, with
    //// given real and complex domains. Default: transform along all dimensions.
    ////-----------------------------------------------------------------------------
    //
    //template <size_t Dim, class T>
    //FFT<RCTransform,Dim,T>::FFT(
    //    const typename FFT<RCTransform,Dim,T>::Domain_t& rdomain,
    //    const typename FFT<RCTransform,Dim,T>::Domain_t& cdomain,
    //    const bool& compressTemps,
    //    int serialAxes)
    //: FFTBase<Dim,T>(FFT<RCTransform,Dim,T>::rcFFT, rdomain, compressTemps),
    //    complexDomain_m(cdomain), serialAxes_m(serialAxes)
    //{
    //    // Tau profiling
    //
    //    // construct array of axis lengths
    //    int lengths[Dim];
    //    size_t d;
    //    for (d=0; d<Dim; ++d)
    //        lengths[d] = rdomain[d].length();
    //
    //    // construct array of transform types for FFT Engine, compute normalization
    //    int transformTypes[Dim];
    //    T& normFact = this->getNormFact();
    //    normFact = 1.0;
    //    transformTypes[0] = FFTBase<Dim,T>::rcFFT;    // first transform is real-to-complex
    //    normFact /= lengths[0];
    //    for (d=1; d<Dim; ++d) {
    //        transformTypes[d] = FFTBase<Dim,T>::ccFFT;  // all other transforms are complex-to-complex
    //        normFact /= lengths[d];
    //    }
    //
    //    // set up FFT Engine
    //    this->getEngine().setup(Dim, transformTypes, lengths);
    //
    //    // set up the temporary fields
    //    setup();
    //}
    //
    ////-----------------------------------------------------------------------------
    //// setup performs all the initializations necessary after the transform
    //// directions have been specified.
    ////-----------------------------------------------------------------------------
    //
    //template <size_t Dim, class T>
    //void
    //FFT<RCTransform,Dim,T>::setup(void) {
    //
    //    // Tau profiling
    //
    //
    //
    //
    //    PAssert_GT(serialAxes_m, 0);
    //    PAssert_LT((size_t) serialAxes_m, Dim);
    //
    //    size_t d, d2, activeDim;
    //    size_t nTransformDims = this->numTransformDims();
    //
    //    // Set up the arrays of temporary Fields and FieldLayouts:
    //
    //    // make first dimension(s) always SERIAL, all other dimensions parallel
    //    // for the real FFT; make first serialAxes_m axes serial for others
    //    e_dim_tag serialParallel[Dim];
    //    e_dim_tag NserialParallel[Dim];
    //    for (d=0; d < Dim; ++d) {
    //        serialParallel[d] = (d == 0 ? SERIAL : PARALLEL);
    //        NserialParallel[d] = (d < (size_t) serialAxes_m ? SERIAL : PARALLEL);
    //    }
    //
    //    // check that domain lengths agree between real and complex domains
    //    const Domain_t& domain = this->getDomain();
    //    activeDim = this->activeDimension(0);
    //    bool match = true;
    //    for (d=0; d<Dim; ++d) {
    //        if (d == activeDim) {
    //            // real array length n, complex array length n/2+1
    //            if ( complexDomain_m[d].length() !=
    //                 (domain[d].length()/2 + 1) ) match = false;
    //        }
    //        else {
    //            // real and complex arrays should be same length for all other dims
    //            if (complexDomain_m[d].length() != domain[d].length()) match = false;
    //        }
    //    }
    //    PInsist(match,
    //            "Domains provided for real and complex Fields are incompatible!");
    //
    //    // allocate arrays of temp fields and layouts for complex fields
    //    tempLayouts_m = new Layout_t*[nTransformDims];
    //    tempFields_m = new ComplexField_t*[nTransformDims];
    //
    //    // set up the single temporary real field, with first dim serial, others par
    //
    //    // make new domains with permuted Indexes, activeDim first
    //    Domain_t ndip;
    //    Domain_t ndipc;
    //    ndip[0] = domain[activeDim];
    //    ndipc[0] = complexDomain_m[activeDim];
    //    for (d=1; d<Dim; ++d) {
    //        size_t nextDim = activeDim + d;
    //        if (nextDim >= Dim) nextDim -= Dim;
    //        ndip[d] = domain[nextDim];
    //        ndipc[d] = complexDomain_m[nextDim];
    //    }
    //
    //    // generate layout and object for temporary real field
    //    tempRLayout_m = new Layout_t(ndip, serialParallel, this->transVnodes());
    //    tempRField_m = new RealField_t(*tempRLayout_m);
    //
    //    // generate layout and object for first temporary complex Field
    //    tempLayouts_m[0] = new Layout_t(ndipc, serialParallel, this->transVnodes());
    //    tempFields_m[0] = new ComplexField_t(*tempLayouts_m[0]);
    //
    //    // determine the order in which dimensions will be transposed.  Put
    //    // the transposed dims first, and the others at the end.
    //    int fftorder[Dim], tmporder[Dim];
    //    int nofft = nTransformDims;
    //    for (d=0; d < nTransformDims; ++d)
    //        fftorder[d] = this->activeDimension(d);
    //    for (d=0; d < Dim; ++d) {
    //        // see if the dth dimension is one to transform
    //        bool active = false;
    //        for (d2=0; d2 < nTransformDims; ++d2) {
    //            if (this->activeDimension(d2) == d) {
    //                active = true;
    //                break;
    //            }
    //        }
    //
    //        if (!active)
    //            // no it is not; put it at the bottom of list
    //            fftorder[nofft++] = d;
    //    }
    //
    //    // But since the first FFT is done on a S,[P,P,...] field, permute
    //    // the order of this to get the first activeDimension at the end.
    //    nofft = fftorder[0];
    //    for (d=0; d < (Dim - 1); ++d)
    //        fftorder[d] = fftorder[d+1];
    //    fftorder[Dim-1] = nofft;
    //
    //    // now construct the remaining temporary complex fields
    //
    //    // loop through and create actual permuted layouts, and also fields
    //    size_t dim = 1;			// already have one temp field
    //    while (dim < nTransformDims) {
    //
    //        int sp;
    //        for (sp=0; sp < serialAxes_m && dim < nTransformDims; ++sp, ++dim) {
    //
    //            // make new domain with permuted Indexes
    //            for (d=0; d < Dim; ++d)
    //                ndip[d] = complexDomain_m[fftorder[d]];
    //
    //            // generate layout and object for temporary complex Field
    //            tempLayouts_m[dim] = new Layout_t(ndip, NserialParallel, this->transVnodes());
    //            tempFields_m[dim] = new ComplexField_t(*tempLayouts_m[dim]);
    //
    //            // permute the fft order for the first 'serialAxes_m' axes
    //            if (serialAxes_m > 1) {
    //                tmporder[0] = fftorder[0];
    //                for (d=0; d < (size_t) (serialAxes_m-1); ++d)
    //                    fftorder[d] = fftorder[d+1];
    //                fftorder[serialAxes_m - 1] = tmporder[0];
    //            }
    //        }
    //
    //        // now, permute ALL the axes by serialAxes_m steps, to get the next
    //        // set of axes in the first n serial slots
    //        for (d=0; d < Dim; ++d)
    //            tmporder[d] = fftorder[d];
    //        for (d=0; d < Dim; ++d)
    //            fftorder[d] = tmporder[(d + serialAxes_m) % Dim];
    //    }
    //}
    //
    //
    ////-----------------------------------------------------------------------------
    //// Destructor
    ////-----------------------------------------------------------------------------
    //
    //template <size_t Dim, class T>
    //FFT<RCTransform,Dim,T>::~FFT(void) {
    //
    //    // Tau profiling
    //
    //
    //    // delete temporary fields and layouts
    //    size_t nTransformDims = this->numTransformDims();
    //    for (size_t d=0; d<nTransformDims; ++d) {
    //        delete tempFields_m[d];
    //        delete tempLayouts_m[d];
    //    }
    //    delete [] tempFields_m;
    //    delete [] tempLayouts_m;
    //    delete tempRField_m;
    //    delete tempRLayout_m;
    //
    //}
    //
    //template <size_t Dim, class T>
    //void
    //FFT<RCTransform,Dim,T>::transform(
    //    int direction,
    //    typename FFT<RCTransform,Dim,T>::RealField_t& f,
    //    typename FFT<RCTransform,Dim,T>::ComplexField_t& g,
    //    const bool& constInput)
    //{
    //    // check domain of incoming fields
    //    const Layout_t& in_layout = f.getLayout();
    //    const Domain_t& in_dom = in_layout.getDomain();
    //    const Layout_t& out_layout = g.getLayout();
    //    const Domain_t& out_dom = out_layout.getDomain();
    //
    //
    //    PAssert_EQ( this->checkDomain(this->getDomain(),in_dom) &&
    //                this->checkDomain(complexDomain_m,out_dom), true);
    //
    //    // common loop iterate and other vars:
    //    size_t d;
    //    size_t idim;      // idim loops over the number of transform dims.
    //    size_t nTransformDims = this->numTransformDims();
    //
    //    // handle first rc transform separately
    //    idim = 0;
    //
    //    RealField_t* tempR = tempRField_m;  // field* management aid
    //    if (!constInput) {
    //        // see if we can use input field f as a temporary
    //        bool skipTemp = true;
    //
    //        // more rigorous match required here; check that layouts are identical
    //        if ( !(in_layout == *tempRLayout_m) ) {
    //            skipTemp = false;
    //        } else {
    //            // make sure distributions match
    //            for (d=0; d<Dim; ++d)
    //                if (in_layout.getDistribution(d) != tempRLayout_m->getDistribution(d))
    //                    skipTemp = false;
    //
    //            // make sure vnode counts match
    //            if (in_layout.numVnodes() != tempRLayout_m->numVnodes())
    //                skipTemp = false;
    //
    //            // also make sure there are no guard cells
    //            if (!(f.getGC() == FFT<RCTransform,Dim,T>::nullGC))
    //                skipTemp = false;
    //        }
    //
    //        // if we can skip using this temporary, set the tempr pointer to the
    //        // original incoming field.  otherwise, it will stay pointing at the
    //        // temporary real field, and we'll need to do a transpose of the data
    //        // from the original into the temporary.
    //        if (skipTemp)
    //            tempR = &f;
    //    }
    //
    //    // if we're not using input as a temporary ...
    //    if (tempR != &f) {
    //
    //
    //        // transpose AND PERMUTE TO REAL FIELD WITH TRANSFORM DIM FIRST
    //        (*tempR)[tempR->getDomain()] = f[in_dom];
    //
    //    }
    //
    //    // field* for temp field management:
    //    ComplexField_t* temp = tempFields_m[0];
    //
    //    // see if we can put final result directly into g.  this is useful if
    //    // we're doing just a 1d fft of one dimension of a multi-dimensional field.
    //    if (nTransformDims == 1) {  // only a single rc transform
    //        bool skipTemp = true;
    //
    //        // more rigorous match required here; check that layouts are identical
    //        if (!(out_layout == *tempLayouts_m[0])) {
    //            skipTemp = false;
    //        } else {
    //            for (d=0; d<Dim; ++d)
    //                if (out_layout.getDistribution(d) !=
    //                    tempLayouts_m[0]->getDistribution(d))
    //                    skipTemp = false;
    //
    //            if ( out_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
    //                skipTemp = false;
    //
    //            // also make sure there are no guard cells
    //            if (!(g.getGC() == FFT<RCTransform,Dim,T>::nullGC))
    //                skipTemp = false;
    //
    //            // if we can skip using the temporary, set the pointer to the output
    //            // field for the first fft to the second provided field (g)
    //            if (skipTemp)
    //                temp = &g;
    //        }
    //    }
    //
    //    // loop over all the vnodes, working on the lfield in each.
    //    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    //    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    //    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
    //        // get the lfields
    //        RealLField_t* rldf = (*rl_i).second.get();
    //        ComplexLField_t* cldf = (*cl_i).second.get();
    //
    //        // make sure we are uncompressed
    //        rldf->Uncompress();
    //        cldf->Uncompress();
    //
    //        // get the raw data pointers
    //        T* localreal = rldf->getP();
    //        Complex_t* localcomp = cldf->getP();
    //
    //        // number of strips should be the same for real and complex lfields!
    //        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
    //        for (d=1; d<Dim; ++d)
    //            nstrips *= rldf->size(d);
    //
    //
    //        for (int istrip=0; istrip<nstrips; ++istrip) {
    //            // move the data into the complex strip, which is two reals longer
    //            for (int ilen=0; ilen<lengthreal; ilen+=2) {
    //                localcomp[ilen/2] = Complex_t(localreal[ilen],localreal[ilen+1]);
    //            }
    //
    //            // do the 1d real-to-complex fft:
    //            // note that real-to-complex fft direction is always +1
    //            this->getEngine().callFFT(idim, +1, localcomp);
    //
    //            // advance the data pointers
    //            localreal += lengthreal;
    //            localcomp += lengthcomp;
    //        } // loop over 1d strips
    //
    //    } // loop over all the lfields
    //
    //    // compress temporary storage
    //    if (this->compressTemps() && tempR != &f)
    //        *tempR = 0;
    //
    //    // now proceed with the other complex-to-complex transforms
    //
    //    // local work array passed to fft:
    //    Complex_t* localdata;
    //
    //    // loop over the remaining dimensions to be transformed:
    //    for (idim = 1; idim < nTransformDims; ++idim) {
    //
    //        bool skipTranspose = false;
    //
    //        // if this is the last transform dimension, we might be able
    //        // to skip the last temporary and transpose right into g
    //        if (idim == nTransformDims-1) {
    //            // get the domain for comparison
    //            const Domain_t& last_dom = tempLayouts_m[idim]->getDomain();
    //
    //            // make sure there are no guard cells, and that the first
    //            // axis matches what we expect and is serial.  only need to
    //            // check first axis since we're just fft'ing that one dimension.
    //            skipTranspose = (g.getGC() == FFT<RCTransform,Dim,T>::nullGC &&
    //                             out_dom[0].length() == last_dom[0].length() &&
    //                             out_layout.getDistribution(0) == SERIAL);
    //        }
    //
    //        if (!skipTranspose) {
    //            // transpose and permute to field with transform dim first
    //            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
    //                (*temp)[temp->getLayout().getDomain()];
    //
    //            // compress out previous iterate's storage:
    //            if (this->compressTemps())
    //                *temp = 0;
    //            temp = tempFields_m[idim];  // field* management aid
    //
    //        } else if (idim == nTransformDims-1) {
    //            // last transform and we can skip the last temporary field
    //            // so do the transpose here using g instead
    //
    //            // transpose and permute to field with transform dim first
    //
    //            g[out_dom] = (*temp)[temp->getLayout().getDomain()];
    //
    //            // compress out previous iterate's storage:
    //            if (this->compressTemps())
    //                *temp = 0;
    //            temp = &g;  // field* management aid
    //
    //        }
    //
    //        // loop over all the vnodes, working on the lfield in each.
    //        typename ComplexField_t::const_iterator_if l_i, l_end = temp->end_if();
    //        for (l_i = temp->begin_if(); l_i != l_end; ++l_i) {
    //            // get the lfield
    //            ComplexLField_t* ldf = (*l_i).second.get();
    //
    //            // make sure we are uncompressed
    //            ldf->Uncompress();
    //
    //            // get the raw data pointer
    //            localdata = ldf->getP();
    //
    //            // do 1d complex-to-complex fft's on all the strips in the lfield:
    //            int nstrips = 1, length = ldf->size(0);
    //            for (d=1; d<Dim; ++d)
    //                nstrips *= ldf->size(d);
    //
    //            for (int istrip=0; istrip<nstrips; ++istrip) {
    //                // do the 1D FFT:
    //                //this->getEngine().callFFT(idim, direction, localdata);
    //                this->getEngine().callFFT(idim, +1, localdata);
    //
    //                // advance the data pointer
    //                localdata += length;
    //            } // loop over 1D strips
    //        } // loop over all the LFields
    //
    //    } // loop over all transformed dimensions
    //
    //
    //    // skip final assignment and compress if we used g as final temporary
    //    if (temp != &g) {
    //
    //
    //        // Now assign into output Field, and compress last temp's storage:
    //        g[out_dom] = (*temp)[temp->getLayout().getDomain()];
    //
    //        if (this->compressTemps()) *temp = 0;
    //
    //    }
    //
    //    // Normalize:
    //    if (direction == +1) g = g * this->getNormFact();
    //
    //    // finish timing the whole mess
    //
    //}
    ////#endif
    //
    ////-----------------------------------------------------------------------------
    //// RC FFT; opposite direction, from complex to real
    ////-----------------------------------------------------------------------------
    //
    //template <size_t Dim, class T>
    //void
    //FFT<RCTransform,Dim,T>::transform(
    //    int direction,
    //    typename FFT<RCTransform,Dim,T>::ComplexField_t& f,
    //    typename FFT<RCTransform,Dim,T>::RealField_t& g,
    //    const bool& constInput)
    //{
    //    // Check domain of incoming Fields
    //    const Layout_t& in_layout = f.getLayout();
    //    const Domain_t& in_dom = in_layout.getDomain();
    //    const Layout_t& out_layout = g.getLayout();
    //    const Domain_t& out_dom = out_layout.getDomain();
    //    PAssert_EQ( this->checkDomain(complexDomain_m,in_dom) &&
    //                this->checkDomain(this->getDomain(),out_dom), true);
    //
    //    // Common loop iterate and other vars:
    //    size_t d;
    //    size_t idim;      // idim loops over the number of transform dims.
    //    size_t nTransformDims = this->numTransformDims();
    //
    //    // proceed with the complex-to-complex transforms
    //
    //    // Field* for temp Field management:
    //    ComplexField_t* temp = &f;
    //
    //    // Local work array passed to FFT:
    //    Complex_t* localdata;
    //
    //    // Loop over all dimensions to be transformed except last one:
    //    for (idim = nTransformDims-1; idim != 0; --idim) {
    //
    //        // Now do the serial transforms along this dimension:
    //
    //        bool skipTranspose = false;
    //        // if this is the first transform dimension, we might be able
    //        // to skip the transpose into the first temporary Field
    //        if (idim == nTransformDims-1 && !constInput) {
    //            // get domain for comparison
    //            const Domain_t& first_dom = tempLayouts_m[idim]->getDomain();
    //            // check that zeroth axis is the same and is serial
    //            // and that there are no guard cells
    //            skipTranspose = ( (in_dom[0].length() == first_dom[0].length()) &&
    //                              (in_layout.getDistribution(0) == SERIAL) &&
    //                              (f.getGC() == FFT<RCTransform,Dim,T>::nullGC) );
    //        }
    //
    //        if (!skipTranspose) {
    //            // transpose and permute to Field with transform dim first
    //            (*tempFields_m[idim])[tempLayouts_m[idim]->getDomain()] =
    //                (*temp)[temp->getLayout().getDomain()];
    //
    //            // Compress out previous iterate's storage:
    //            if (this->compressTemps() && temp != &f)
    //                *temp = 0;
    //            temp = tempFields_m[idim];  // Field* management aid
    //        }
    //
    //        // Loop over all the Vnodes, working on the LField in each.
    //        typename ComplexField_t::const_iterator_if l_i, l_end = temp->end_if();
    //        for (l_i = temp->begin_if(); l_i != l_end; ++l_i) {
    //
    //            // Get the LField
    //            ComplexLField_t* ldf = (*l_i).second.get();
    //
    //            // make sure we are uncompressed
    //            ldf->Uncompress();
    //
    //            // get the raw data pointer
    //            localdata = ldf->getP();
    //
    //            // Do 1D complex-to-complex FFT's on all the strips in the LField:
    //            int nstrips = 1, length = ldf->size(0);
    //            for (d=1; d<Dim; ++d)
    //                nstrips *= ldf->size(d);
    //
    //            for (int istrip=0; istrip<nstrips; ++istrip) {
    //                // Do the 1D FFT:
    //                //this->getEngine().callFFT(idim, direction, localdata);
    //                this->getEngine().callFFT(idim, -1, localdata);
    //
    //                // advance the data pointer
    //                localdata += length;
    //            } // loop over 1D strips
    //        } // loop over all the LFields
    //
    //    } // loop over all transformed dimensions
    //
    //    // handle last CR transform separately
    //    idim = 0;
    //
    //    // see if we can put final result directly into g
    //    RealField_t* tempR;
    //    bool skipTemp = true;
    //
    //    // more rigorous match required here; check that layouts are identical
    //    if (!(out_layout == *tempRLayout_m)) {
    //        skipTemp = false;
    //    } else {
    //        for (d=0; d<Dim; ++d)
    //            if (out_layout.getDistribution(d) != tempRLayout_m->getDistribution(d))
    //                skipTemp = false;
    //
    //        if ( out_layout.numVnodes() != tempRLayout_m->numVnodes() )
    //            skipTemp = false;
    //
    //        // also make sure there are no guard cells
    //        if (!(g.getGC() == FFT<RCTransform,Dim,T>::nullGC))
    //            skipTemp = false;
    //    }
    //
    //    if (skipTemp)
    //        tempR = &g;
    //    else
    //        tempR = tempRField_m;
    //
    //    skipTemp = true;
    //    if (nTransformDims == 1 && !constInput) {
    //        // only one CR transform
    //        // see if we really need to transpose input data
    //        // more rigorous match required here; check that layouts are identical
    //        if (!(in_layout == *tempLayouts_m[0])) {
    //            skipTemp = false;
    //        } else {
    //            for (d=0; d<Dim; ++d)
    //                if (in_layout.getDistribution(d) !=
    //                    tempLayouts_m[0]->getDistribution(d))
    //                    skipTemp = false;
    //
    //            if ( in_layout.numVnodes() != tempLayouts_m[0]->numVnodes() )
    //                skipTemp = false;
    //
    //            // also make sure there are no guard cells
    //            if (!(f.getGC() == FFT<RCTransform,Dim,T>::nullGC))
    //                skipTemp = false;
    //        }
    //    } else {  // cannot skip transpose
    //        skipTemp = false;
    //    }
    //
    //
    //    if (!skipTemp) {
    //        // transpose and permute to complex Field with transform dim first
    //        (*tempFields_m[0])[tempLayouts_m[0]->getDomain()] =
    //            (*temp)[temp->getLayout().getDomain()];
    //
    //        // compress previous iterates storage
    //        if (this->compressTemps() && temp != &f)
    //            *temp = 0;
    //        temp = tempFields_m[0];
    //    }
    //
    //    // Loop over all the Vnodes, working on the LField in each.
    //    typename RealField_t::const_iterator_if rl_i, rl_end = tempR->end_if();
    //    typename ComplexField_t::const_iterator_if cl_i = temp->begin_if();
    //    for (rl_i = tempR->begin_if(); rl_i != rl_end; ++rl_i, ++cl_i) {
    //        // Get the LFields
    //        RealLField_t* rldf = (*rl_i).second.get();
    //        ComplexLField_t* cldf = (*cl_i).second.get();
    //
    //        // make sure we are uncompressed
    //        rldf->Uncompress();
    //        cldf->Uncompress();
    //
    //        // get the raw data pointers
    //        T* localreal = rldf->getP();
    //        Complex_t* localcomp = cldf->getP();
    //
    //        // number of strips should be the same for real and complex LFields!
    //        int nstrips = 1, lengthreal = rldf->size(0), lengthcomp = cldf->size(0);
    //        for (d=1; d<Dim; ++d)
    //            nstrips *= rldf->size(d);
    //
    //        for (int istrip=0; istrip<nstrips; ++istrip) {
    //            // Do the 1D complex-to-real FFT:
    //            // note that complex-to-real FFT direction is always -1
    //            this->getEngine().callFFT(idim, -1, localcomp);
    //
    //            // move the data into the real strip, which is two reals shorter
    //            for (int ilen=0; ilen<lengthreal; ilen+=2) {
    //                localreal[ilen] = real(localcomp[ilen/2]);
    //                localreal[ilen+1] = imag(localcomp[ilen/2]);
    //            }
    //
    //            // advance the data pointers
    //            localreal += lengthreal;
    //            localcomp += lengthcomp;
    //        } // loop over 1D strips
    //    } // loop over all the LFields
    //
    //    // compress previous iterates storage
    //    if (this->compressTemps() && temp != &f)
    //        *temp = 0;
    //
    //    // skip final assignment and compress if we used g as final temporary
    //    if (tempR != &g) {
    //
    //
    //        // Now assign into output Field, and compress last temp's storage:
    //        g[out_dom] = (*tempR)[tempR->getLayout().getDomain()];
    //
    //        if (this->compressTemps())
    //            *tempR = 0;
    //
    //    }
    //
    //    // Normalize:
    //    if (direction == +1) g = g * this->getNormFact();
    //
    //    // finish up timing the whole mess
    //}
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
