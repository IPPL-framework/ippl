//
// Class FFT
//   The FFT class performs complex-to-complex,
//   real-to-complex on IPPL Fields.
//   FFT is templated on the type of transform to be performed,
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte. In making this interface,
//   we have utilized ideas from Cabana library
//   https://github.com/ECP-copa/Cabana especially for the temporary
//   field with layout right for passing into heffte.
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

#include "FFT/FFT.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/BareField.h"


namespace ippl {

    //=========================================================================
    // FFT CCTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CCTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<CCTransform,Dim,T>::FFT(
        const Layout_t& layout,
        const FFTParams& params)
    {


        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<int, 3> low;
        std::array<int, 3> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to int is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<int>(lDom[d].first());
            high[d] = static_cast<int>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T>
    void
    FFT<CCTransform,Dim,T>::setup(const std::array<int, Dim>& low,
                                  const std::array<int, Dim>& high,
                                  const FFTParams& params)
    {

         heffte::box3d inbox = {low, high};
         heffte::box3d outbox = {low, high};

         heffte::plan_options heffteOptions =
             heffte::default_options<heffteBackend>();
         heffteOptions.use_alltoall = params.getAllToAll();
         heffteOptions.use_pencils = params.getPencils();
         heffteOptions.use_reorder = params.getReorder();

         heffte_m = std::make_shared<heffte::fft3d<heffteBackend>>
                    (inbox, outbox, Ippl::getComm(), heffteOptions);

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
        *This copy to a temporary Kokkos view is needed because heffte accepts
        *input and output data in layout right (usual C++) format, whereas
        *default Kokkos views can be layout left or right depending on whether
        *the device is gpu or cpu.
       */
       Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight>
           tempField("tempField", fview.extent(0) - 2*nghost,
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
                              tempField(i-nghost, j-nghost, k-nghost).x =
                              fview(i, j, k).real();
                              tempField(i-nghost, j-nghost, k-nghost).y =
                              fview(i, j, k).imag();
#else
                              tempField(i-nghost, j-nghost, k-nghost).real(
                                      fview(i, j, k).real());
                              tempField(i-nghost, j-nghost, k-nghost).imag(
                                      fview(i, j, k).imag());
#endif
                            });
       if ( direction == 1 )
       {
           heffte_m->forward(tempField.data(), tempField.data(),
                             heffte::scale::full);
       }
       else if ( direction == -1 )
       {
           heffte_m->backward(tempField.data(), tempField.data(),
                              heffte::scale::none);
       }
       else
       {
           throw std::logic_error(
                "Only 1:forward and -1:backward are allowed as directions");
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
                              fview(i, j, k).real() =
                              tempField(i-nghost, j-nghost, k-nghost).x;
                              fview(i, j, k).imag() =
                              tempField(i-nghost, j-nghost, k-nghost).y;
#else
                              fview(i, j, k).real() =
                              tempField(i-nghost, j-nghost, k-nghost).real();
                              fview(i, j, k).imag() =
                              tempField(i-nghost, j-nghost, k-nghost).imag();
#endif
                            });

    }


    //========================================================================
    // FFT RCTransform Constructors
    //========================================================================

    /**
       *Create a new FFT object of type RCTransform, with given input and output
       *layouts and heffte parameters.
    */

    template <size_t Dim, class T>
    FFT<RCTransform,Dim,T>::FFT(
        const Layout_t& layoutInput,
        const Layout_t& layoutOutput,
        const FFTParams& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<int, 3> lowInput;
        std::array<int, 3> highInput;
        std::array<int, 3> lowOutput;
        std::array<int, 3> highOutput;

        const NDIndex<Dim>& lDomInput = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();

        lowInput.fill(0);
        highInput.fill(0);
        lowOutput.fill(0);
        highOutput.fill(0);

        /**
         * Static cast to int is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            lowInput[d] = static_cast<int>(lDomInput[d].first());
            highInput[d] = static_cast<int>(lDomInput[d].length() +
                           lDomInput[d].first() - 1);

            lowOutput[d] = static_cast<int>(lDomOutput[d].first());
            highOutput[d] = static_cast<int>(lDomOutput[d].length() +
                            lDomOutput[d].first() - 1);
        }


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
                                  const FFTParams& params)
    {

         heffte::box3d inbox = {lowInput, highInput};
         heffte::box3d outbox = {lowOutput, highOutput};

         heffte::plan_options heffteOptions =
             heffte::default_options<heffteBackend>();
         heffteOptions.use_alltoall = params.getAllToAll();
         heffteOptions.use_pencils = params.getPencils();
         heffteOptions.use_reorder = params.getReorder();

         heffte_m = std::make_shared<heffte::fft3d_r2c<heffteBackend>>
                    (inbox, outbox, params.getRCDirection(), Ippl::getComm(),
                     heffteOptions);

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
        *This copy to a temporary Kokkos view is needed because heffte
        *accepts input and output data in layout right (usual C++) format,
        *whereas default Kokkos views can be layout left or right
        *depending on whether the device is gpu or cpu.
       */
       Kokkos::View<T***,Kokkos::LayoutRight>
           tempFieldf("tempFieldf", fview.extent(0) - 2*nghostf,
                                    fview.extent(1) - 2*nghostf,
                                    fview.extent(2) - 2*nghostf);

       Kokkos::View<heffteComplex_t***,Kokkos::LayoutRight>
           tempFieldg("tempFieldg", gview.extent(0) - 2*nghostg,
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
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).x =
                              gview(i, j, k).real();
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).y =
                              gview(i, j, k).imag();
#else
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).real(
                                      gview(i, j, k).real());
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag(
                                      gview(i, j, k).imag());
#endif
                            });
       if ( direction == 1 )
       {
           heffte_m->forward( tempFieldf.data(), tempFieldg.data(),
                              heffte::scale::full );
       }
       else if ( direction == -1 )
       {
           heffte_m->backward( tempFieldg.data(), tempFieldf.data(),
                               heffte::scale::none );
       }
       else
       {
           throw std::logic_error(
                "Only 1:forward and -1:backward are allowed as directions");
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
                              gview(i, j, k).real() =
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).x;
                              gview(i, j, k).imag() =
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).y;
#else
                              gview(i, j, k).real() =
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).real();
                              gview(i, j, k).imag() =
                              tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag();
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
