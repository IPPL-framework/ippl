//
// Class FFT
//   The FFT class performs complex-to-complex, 
//   real-to-complex on IPPL Fields. 
//   FFT is templated on the type of transform to be performed, 
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte.
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan, 
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
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
       given layout and heffte parameters.
    */
    
    template <size_t Dim, class T>
    FFT<CCTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const HeffteParams& params)
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
           setup performs the initialization necessary.
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
         
    }
    
 
    
    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::transform(
        int direction,
        typename FFT<CCTransform,Dim,T>::ComplexField_t& f)
    {
       auto fview = f.getView();
       const int nghost = f.getNghost();

       /**
        *This copy to a temporary Kokkos view is needed because heffte accepts input and output data
        *in layout right (usual C++) format, whereas default Kokkos views can be layout left or right
        *depending on whether the device is gpu or cpu.
       */
       Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight> tempField("tempField", fview.extent(0) - 2*nghost,
                                                                                   fview.extent(1) - 2*nghost,
                                                                                   fview.extent(2) - 2*nghost);

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
#ifdef KOKKOS_ENABLE_CUDA
                              tempField(i-nghost, j-nghost, k-nghost).x = fview(i, j, k).real();
                              tempField(i-nghost, j-nghost, k-nghost).y = fview(i, j, k).imag();
#else
                              tempField(i-nghost, j-nghost, k-nghost).real( fview(i, j, k).real() );
                              tempField(i-nghost, j-nghost, k-nghost).imag( fview(i, j, k).imag() );
#endif
                            });
       if ( direction == 1 )
       {
           heffte_m->forward( tempField.data(), tempField.data(), heffte::scale::full );
       }
       else if ( direction == -1 )
       {
           heffte_m->backward( tempField.data(), tempField.data(), heffte::scale::none );
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
#ifdef KOKKOS_ENABLE_CUDA
                              fview(i, j, k).real() = tempField(i-nghost, j-nghost, k-nghost).x;
                              fview(i, j, k).imag() = tempField(i-nghost, j-nghost, k-nghost).y;
#else
                              fview(i, j, k).real() = tempField(i-nghost, j-nghost, k-nghost).real();
                              fview(i, j, k).imag() = tempField(i-nghost, j-nghost, k-nghost).imag();
#endif
                            });
    
    }
    
    
    //=============================================================================
    // FFT RCTransform Constructors
    //=============================================================================
    
    /**
       Create a new FFT object of type RCTransform, with given input and output layouts and heffte parameters. 
    */
    
    template <size_t Dim, class T>
    FFT<RCTransform,Dim,T>::FFT(
        const Layout_t& layoutInput,
        const Layout_t& layoutOutput,
        const HeffteParams& params)
    {
    
        std::array<int, Dim> lowInput; 
        std::array<int, Dim> highInput;
        std::array<int, Dim> lowOutput; 
        std::array<int, Dim> highOutput;

        const NDIndex<Dim>& lDomInput = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();

        lowInput = {(int)lDomInput[0].first(), (int)lDomInput[1].first(), (int)lDomInput[2].first()};
        highInput = {(int)lDomInput[0].length() + (int)lDomInput[0].first() - 1,
                     (int)lDomInput[1].length() + (int)lDomInput[1].first() - 1,
                     (int)lDomInput[2].length() + (int)lDomInput[2].first() - 1};
   
        lowOutput = {(int)lDomOutput[0].first(), (int)lDomOutput[1].first(), (int)lDomOutput[2].first()};
        highOutput = {(int)lDomOutput[0].length() + (int)lDomOutput[0].first() - 1,
                     (int)lDomOutput[1].length() + (int)lDomOutput[1].first() - 1,
                     (int)lDomOutput[2].length() + (int)lDomOutput[2].first() - 1};

        setup(lowInput, highInput, lowOutput, highOutput, params);
    }
    
    
    /**
       setup performs the initialization.
    */
    template <size_t Dim, class T>
    void
    FFT<RCTransform,Dim,T>::setup(const std::array<int, Dim>& lowInput, 
                                  const std::array<int, Dim>& highInput,
                                  const std::array<int, Dim>& lowOutput, 
                                  const std::array<int, Dim>& highOutput,
                                  const HeffteParams& params)
    {
   
         heffte::box3d inbox = { lowInput, highInput };
         heffte::box3d outbox = { lowOutput, highOutput };

         heffte::plan_options heffteOptions = heffte::default_options<heffteBackend>();
         heffteOptions.use_alltoall = params.getAllToAll();
         heffteOptions.use_pencils = params.getPencils();
         heffteOptions.use_reorder = params.getReorder();

         heffte_m = std::make_shared<heffte::fft3d_r2c<heffteBackend>>(inbox, outbox, params.getRCDirection(), 
                                                                       Ippl::getComm(), heffteOptions);
         
    }
    
    template <size_t Dim, class T>
    void
    FFT<RCTransform,Dim,T>::transform(
        int direction,
        typename FFT<RCTransform,Dim,T>::RealField_t& f,
        typename FFT<RCTransform,Dim,T>::ComplexField_t& g)
    {
       auto fview = f.getView();
       auto gview = g.getView();
       const int nghostf = f.getNghost();
       const int nghostg = g.getNghost();

       /**
        *This copy to a temporary Kokkos view is needed because heffte accepts input and output data
        *in layout right (usual C++) format, whereas default Kokkos views can be layout left or right
        *depending on whether the device is gpu or cpu.
       */
       Kokkos::View<T***,Kokkos::LayoutRight> tempFieldf("tempFieldf", fview.extent(0) - 2*nghostf,
                                                                       fview.extent(1) - 2*nghostf,
                                                                       fview.extent(2) - 2*nghostf);

       Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight> tempFieldg("tempFieldg", gview.extent(0) - 2*nghostg,
                                                                                   gview.extent(1) - 2*nghostg,
                                                                                   gview.extent(2) - 2*nghostg);

       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

       Kokkos::parallel_for("copy from Kokkos f field in FFT",
                            mdrange_type({nghostf, nghostf, nghostf},
                                         {fview.extent(0) - nghostf, 
                                          fview.extent(1) - nghostf,
                                          fview.extent(2) - nghostf
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
                            {
                              tempFieldf(i-nghostf, j-nghostf, k-nghostf) = fview(i, j, k);
                            });
       Kokkos::parallel_for("copy from Kokkos g field in FFT",
                            mdrange_type({nghostg, nghostg, nghostg},
                                         {gview.extent(0) - nghostg, 
                                          gview.extent(1) - nghostg,
                                          gview.extent(2) - nghostg
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
                            {
#ifdef KOKKOS_ENABLE_CUDA
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).x = gview(i, j, k).real();
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).y = gview(i, j, k).imag();
#else
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).real( gview(i, j, k).real() );
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag( gview(i, j, k).imag() );
#endif
                            });
       if ( direction == 1 )
       {
           heffte_m->forward( tempFieldf.data(), tempFieldg.data(), heffte::scale::full );
       }
       else if ( direction == -1 )
       {
           heffte_m->backward( tempFieldg.data(), tempFieldf.data(), heffte::scale::none );
       }
       else
       {
           throw std::logic_error( "Only 1:forward and -1:backward are allowed as directions" );
       }

       Kokkos::parallel_for("copy to Kokkos f field FFT",
                            mdrange_type({nghostf, nghostf, nghostf},
                                         {fview.extent(0) - nghostf, 
                                          fview.extent(1) - nghostf,
                                          fview.extent(2) - nghostf
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
                            {
                              fview(i, j, k) = tempFieldf(i-nghostf, j-nghostf, k-nghostf);
                            });
       
       Kokkos::parallel_for("copy to Kokkos g field FFT",
                            mdrange_type({nghostg, nghostg, nghostg},
                                         {gview.extent(0) - nghostg, 
                                          gview.extent(1) - nghostg,
                                          gview.extent(2) - nghostg
                                         }),
                            KOKKOS_LAMBDA(const size_t i,
                                          const size_t j,
                                          const size_t k)
                            {
#ifdef KOKKOS_ENABLE_CUDA
                              gview(i, j, k).real() = tempFieldg(i-nghostg, j-nghostg, k-nghostg).x;
                              gview(i, j, k).imag() = tempFieldg(i-nghostg, j-nghostg, k-nghostg).y;
#else
                              gview(i, j, k).real() = tempFieldg(i-nghostg, j-nghostg, k-nghostg).real();
                              gview(i, j, k).imag() = tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag();
#endif
                            });
    
    }
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
