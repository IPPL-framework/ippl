template <class T, unsigned int Dim>
PwrSpec<T,Dim>::PwrSpec(Mesh_t *mesh, FieldLayout_t *FL):
    layout_m(FL),
    mesh_m(mesh)
{
    Inform msg ("FFT doInit");

    gDomain_m = layout_m->getDomain();       // global domain
    lDomain_m = layout_m->getLocalNDIndex(); // local domain


    gDomainL_m = NDIndex<Dim>(Index(gDomain_m[0].length()+1),
			      Index(gDomain_m[1].length()+1),
			      Index(gDomain_m[2].length()+1));

    msg << "GDomain " << gDomain_m << " GDomainL " << gDomainL_m << endl;


    for (int i=0; i < 2*Dim; ++i) {
        bc_m[i] = new ParallelPeriodicFace<T,Dim,Mesh_t,Center_t>(i);
        zerobc_m[i] = new ZeroFace<T,Dim,Mesh_t,Center_t>(i);
    }

    for(int d=0; d<Dim; d++) {
        dcomp_m[d]=layout_m->getRequestedDistribution(d);
        nr_m[d] = gDomain_m[d].length();
    }

    // create additional objects for the use in the FFT's
    rho_m.initialize(*mesh_m, *layout_m, GuardCellSizes<3>(1));

    // create the FFT object
    bool compressTemps = true;
    fft_m = new FFT_t(layout_m->getDomain(), compressTemps);
    fft_m->setDirectionName(+1, "forward");

    meshI_m = new Mesh_t(gDomainL_m);
    FLI_m = new FieldLayout_t(*meshI_m,dcomp_m);

    //XXX: At the moment we know that the scatter operator works correctly
    //with 2 guard cells, but we do not have fully understood why!?
    rhocic_m.initialize(*meshI_m,*FLI_m,GuardCellSizes<3>(2));

    // power spectra calc
    kmax_m = 1; // fixme ada nint(sqrt(3*(simData_m.ng_comp*simData_m.ng_comp/4))) + 1 ;
    spectra1D_m = (T *) malloc (kmax_m*sizeof(T));
    Nk_m = (int *) malloc (kmax_m*sizeof(int));


}

template <class T, unsigned int Dim>
PwrSpec<T,Dim>::~PwrSpec() {


}

//FIXME: There is something going wrong when having ng_comp != np
template <class T, unsigned int Dim>
void PwrSpec<T,Dim>::calcPwrSpecAndSave(ChargedParticles<T,Dim> *univ, string fn) 
{
    Inform m("calcPwrSpecAndSave ");

    unsigned int kk;
    const unsigned int ng_m = simData_m.ng_comp;

    CICforward(univ);

    T rho_0 = real(sum(rho_m))/nr_m[0]/nr_m[1]/nr_m[2];
    rho_m = (rho_m - rho_0)/rho_0;

    fft_m->transform("forward", rho_m);

    //FIXME: DO DECONVOLUTION ANTI-CIC FIX?

    rho_m = real(rho_m*conj(rho_m));

    for (int i=0;i<kmax_m;i++) {
        Nk_m[i]=0;
        spectra1D_m[i] = 0.0;
    }

    m << "Sum psp=real( ... " << sum(real(rho_m)) << " kmax= " << kmax_m << endl;

    NDIndex<Dim> loop;

    // This computes the 1-D power spectrum of a 3-D field (rho_m)
    // by binning values
    for (int i=lDomain_m[0].first(); i<=lDomain_m[0].last(); i++) {
        loop[0]=Index(i,i);
        for (int j=lDomain_m[1].first(); j<=lDomain_m[1].last(); j++) {
            loop[1]=Index(j,j);
            for (int k=lDomain_m[2].first(); k<=lDomain_m[2].last(); k++) {

                loop[2]=Index(k,k);
                int ii = i;
                int jj = j;
                int k2 = k;
                //FIXME: this seems to have no change
                //if(i >= (gDomain_m[0].max()+1)/2)
                if(i >= gDomain_m[0].max()/2)
                    ii -= ng_m;
                if(j >= gDomain_m[1].max()/2)
                    jj -= ng_m;
                if(k >= gDomain_m[2].max()/2)
                    k2 -= ng_m;

                kk=(int)nint(sqrt(ii*ii+jj*jj+k2*k2));
                kk = min(kmax_m, (int)kk);
                spectra1D_m[kk] += real(rho_m.localElement(loop));
                Nk_m[kk]++;
            }
        }
    }

    /*
       Error above 2k with OLD CODE:

       Ippl{0}> rhocic_m= 6.87725e+10 sum(M)= 6.86225e+10 rho_m= (6.8753e+10,0)
       calcPwrSpecAndSave {0}> Sum psp=real( ... 0.294218 kmax= 3548
       Ippl{0}> Loops done
       [0] MPICH has run out of unexpected buffer space.
       Try increasing the value of env var MPICH_UNEX_BUFFER_SIZE (cur value is 62914560),
       and/or reducing the size of MPICH_MAX_SHORT_MSG_SIZE (cur value is 50000).
       aborting job:
       out of unexpected buffer space
       [0] MPICH has run out of unexpected buffer space.
       Try increasing the value of env var MPICH_UNEX_BUFFER_SIZE (cur value is 62914560),
       and/or reducing the size of MPICH_MAX_SHORT_MSG_SIZE (cur value is 50000).
       aborting job:
       out of unexpected buffer space
    */

    INFOMSG("Loops done" << endl);
    reduce( &(Nk_m[0]), &(Nk_m[0]) + kmax_m , &(Nk_m[0]) ,OpAddAssign());
    reduce( &(spectra1D_m[0]), &(spectra1D_m[0]) + kmax_m, &(spectra1D_m[0]) ,OpAddAssign());

    Inform* fdip = new Inform(NULL,fn.c_str(),Inform::OVERWRITE,0);
    Inform& fdi = *fdip;
    setInform(fdi);
    setFormat(9,1,0);

    T tpiL = 8.0*atan(1.0)/simData_m.rL;   // k = 2 pi / rL

    // Renormalize power spectrum to match mc2.
    T scale = std::pow((T)(1.0*simData_m.ng_comp),(T)3.0);
    int sumNk = 0;
    for (int i=0; i<kmax_m;i++) {
        sumNk += Nk_m[i];
        spectra1D_m[i] /= 1.0*Nk_m[i];
        spectra1D_m[i] *= scale;
    }

    fdi << "# " ;
    for (int i=0; i < kmax_m; i++)
        fdi << "Nk[" << i << "]= " << Nk_m[i] << " "; 
    fdi << " sum Nk= " << sumNk << endl;

    scale = std::pow((T)(simData_m.rL/simData_m.ng_comp),(T)3.0);
    //FIXME: Why start at 1? Why only half?
    //kmax = ng/2
    //but should start at 0!
    for (int i=1; i<=simData_m.ng_comp/2; i++) {
        fdi << (i)*tpiL << "\t" << spectra1D_m[i]*scale << endl;
    }

    delete fdip;
}


template<class T, unsigned int Dim>
void PwrSpec<T,Dim>::CICforward(ChargedParticles<T,Dim> *univ)
{
    // deposit charge/mass on grid.  We first zero out the rho field, then
    // scatter particles using our selected interpolater type.  In this
    // scatter routine, we cache the values used to calculate where the
    // particles go in the field, and use these values later on in this
    // spaceCharge function to gather elements back from the field to the
    // particle positions.  We can do that because the particles do not
    // change their position between the scatter and gather phases.
    Inform msg ("CICforward ");
    Inform msg2all ("FF ",INFORM_ALL_NODES);

    rhocic_m = 0.0;

    univ->M.scatter(rhocic_m,univ->R,IntCIC());

    Index M = gDomainL_m[0];
    Index J = gDomainL_m[1];
    Index K = gDomainL_m[2];

    rhocic_m[0][J][K] += rhocic_m[nr_m[0]][J][K];
    rhocic_m[M][0][K] += rhocic_m[M][nr_m[1]][K];
    rhocic_m[M][J][0] += rhocic_m[M][J][nr_m[2]];

    rho_m[gDomainL_m]  = rhocic_m[gDomainL_m];

    INFOMSG("rhocic_m= " << sum(rhocic_m) << " sum(M)= " << sum(univ->M) << " rho_m= " << sum(rho_m) << endl;);
}

/***************************************************************************
 * $RCSfile: PwrSpec.cc,v $   $Author: adelmann $
 * $Revision: 1.3 $   $Date: 2001/08/16 09:36:09 $
 ***************************************************************************/
