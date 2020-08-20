// -*- C++ -*-

/***************************************************************************
 * PwrSpec.hh 
 * 
 * Periodic BC in x,y and z. 
 ***************************************************************************/

#ifndef PWR_SPEC_H_
#define PWR_SPEC_H_

//////////////////////////////////////////////////////////////
#include "Ippl.h"
//////////////////////////////////////////////////////////////

template <class T, unsigned int Dim>
class PwrSpec 
{
public:
    // some useful typedefs
    typedef typename ChargedParticles<T,Dim>::Center_t      Center_t;
    typedef typename ChargedParticles<T,Dim>::Mesh_t        Mesh_t;
    typedef typename ChargedParticles<T,Dim>::FieldLayout_t FieldLayout_t;
    typedef typename ChargedParticles<T,Dim>::Vector_t      Vector_t;
    typedef typename ChargedParticles<T,Dim>::IntrplCIC_t   IntrplCIC_t;

    typedef Field<std::complex<double>, Dim, Mesh_t, Center_t> CxField_t;
    typedef Field<T, Dim, Mesh_t, Center_t>                 RxField_t;
    typedef FFT<CCTransform, Dim, T>                        FFT_t;

    // constructor and destructor
    PwrSpec(Mesh_t *mesh, FieldLayout_t *FL);

    ~PwrSpec();
    
    /// CIC forward move (deposits current on rho_m)
    void CICforward(ChargedParticles<T,Dim> *univ);

    /// dumps power spectra in file fn
    void calcPwrSpecAndSave(ChargedParticles<T,Dim> *univ, string fn) ;

private:    

    void saveField(string fn, CxField_t &f, int n );
    void saveField(string fn, RxField_t &f, int n );

    /// fortrans nint function                                                                                            
    inline T nint(T x)
    {
	return ceil(x + 0.5) - (fmod(x*0.5 + 0.25, 1.0) != 0);
    }
    
    FFT_t *fft_m;

    // mesh and layout objects for rho_m
    Mesh_t *mesh_m;
    FieldLayout_t *layout_m;

    // bigger mesh (length+1)
    FieldLayout_t *FLI_m;
    Mesh_t *meshI_m;
      
    /// global domain for the various fields
    NDIndex<Dim> gDomain_m;  
    /// local domain for the various fields
    NDIndex<Dim> lDomain_m;             

    /// global domain for the enlarged fields 
    NDIndex<Dim> gDomainL_m;  
  
    BConds<T,Dim,Mesh_t,Center_t> bc_m;
    BConds<T,Dim,Mesh_t,Center_t> zerobc_m;

    e_dim_tag dcomp_m[Dim];
    Vektor<int,Dim> nr_m;

    /// Fourier transformed density field
    CxField_t rho_m;
    
    /// power spectra kmax
    int kmax_m;

    /// 1D power spectra
    T *spectra1D_m;
    /// Nk power spectra
    int *Nk_m;

};

// needed if we're not using implicit inclusion
#include "PwrSpec.cc"

#endif

/***************************************************************************
 * $RCSfile: PwrSpec.hh,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2001/08/08 11:21:48 $
 ***************************************************************************/
