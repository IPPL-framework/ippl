//
// Class FFT
//   The FFT class performs complex-to-complex,
//   real-to-complex on IPPL Fields.
//   FFT is templated on the type of transform to be performed,
//   the dimensionality of the Field to transform, and the
//   floating-point precision type of the Field (float or double).
//   Currently, we use heffte for taking the transforms and the class FFT
//   serves as an interface between IPPL and heffte. In making this interface,
//   we have referred Cabana library.
//   https://github.com/ECP-copa/Cabana.
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
#include "Utility/IpplTimings.h"


namespace ippl {

    //=========================================================================
    // FFT CCTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CCTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class M>
    FFT<CCTransform,Dim,T,M>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {


        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, Dim> low;
        std::array<long long, Dim> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class M>
    void
    FFT<CCTransform,Dim,T,M>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

         heffte::box3d<long long> inbox  = {low, high};
         heffte::box3d<long long> outbox = {low, high};

         heffte::plan_options heffteOptions =
             heffte::default_options<heffteBackend>();

         if(!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

            switch (params.get<int>("comm")) {
            
               case a2a:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                   break;
               case a2av:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                   break;
               case p2p:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                   break;
               case p2p_pl:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                   break;
               default:
                   throw IpplException("FFT::setup",
                                       "Unrecognized heffte communication type");
            }
         }

         heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                    (inbox, outbox, Ippl::getComm(), heffteOptions);

         //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
         if(workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());

    }



    template <size_t Dim, class T, class M>
    void
    FFT<CCTransform,Dim,T,M>::transform(
        int direction,
        typename FFT<CCTransform,Dim,T,M>::ComplexField_t& f)
    {
       auto fview = f.getView();
       const int nghost = f.getNghost();

       /**
        *This copy to a temporary Kokkos view is needed because of following
        *reasons:
        *1) heffte wants the input and output fields without ghost layers
        *2) heffte accepts data in layout left (by default) eventhough this
        *can be changed during heffte box creation
       */
       Kokkos::View<Complex_t***,Kokkos::LayoutLeft>
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
                                tempField(i-nghost, j-nghost, k-nghost).real(
                                      fview(i, j, k).real());
                                tempField(i-nghost, j-nghost, k-nghost).imag(
                                      fview(i, j, k).imag());
                            });




       if ( direction == 1 )
       {
           heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                             heffte::scale::full);
       }
       else if ( direction == -1 )
       {
           heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
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
                                fview(i, j, k).real() =
                                tempField(i-nghost, j-nghost, k-nghost).real();
                                fview(i, j, k).imag() =
                                tempField(i-nghost, j-nghost, k-nghost).imag();
                            });

    }


    //========================================================================
    // FFT RCTransform Constructors
    //========================================================================

    /**
       *Create a new FFT object of type RCTransform, with given input and output
       *layouts and heffte parameters.
    */

    template <size_t Dim, class T, class M>
    FFT<RCTransform,Dim,T,M>::FFT(
        const Layout_t& layoutInput,
        const Layout_t& layoutOutput,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, 3> lowInput;
        std::array<long long, 3> highInput;
        std::array<long long, 3> lowOutput;
        std::array<long long, 3> highOutput;

        const NDIndex<Dim>& lDomInput = layoutInput.getLocalNDIndex();
        const NDIndex<Dim>& lDomOutput = layoutOutput.getLocalNDIndex();


        lowInput.fill(0);
        highInput.fill(0);
        lowOutput.fill(0);
        highOutput.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            lowInput[d] = static_cast<long long>(lDomInput[d].first());
            highInput[d] = static_cast<long long>(lDomInput[d].length() +
                           lDomInput[d].first() - 1);

            lowOutput[d] = static_cast<long long>(lDomOutput[d].first());
            highOutput[d] = static_cast<long long>(lDomOutput[d].length() +
                            lDomOutput[d].first() - 1);
        }

        setup(lowInput, highInput, lowOutput, highOutput, params);
    }


    /**
       setup performs the initialization.
    */
    template <size_t Dim, class T, class M>
    void
    FFT<RCTransform,Dim,T,M>::setup(const std::array<long long, Dim>& lowInput,
                                  const std::array<long long, Dim>& highInput,
                                  const std::array<long long, Dim>& lowOutput,
                                  const std::array<long long, Dim>& highOutput,
                                  const ParameterList& params)
    {

         heffte::box3d<long long> inbox  = {lowInput, highInput};
         heffte::box3d<long long> outbox = {lowOutput, highOutput};

         heffte::plan_options heffteOptions = 
             heffte::default_options<heffteBackend>();
        
         if(!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif

            switch (params.get<int>("comm")) {
            
               case a2a:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                   break;
               case a2av:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                   break;
               case p2p:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                   break;
               case p2p_pl:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                   break;
               default:
                   throw IpplException("FFT::setup",
                                       "Unrecognized heffte communication type");
            }
         }

         heffte_m = std::make_shared<heffte::fft3d_r2c<heffteBackend, long long>>
                    (inbox, outbox, params.get<int>("r2c_direction"), Ippl::getComm(),
                     heffteOptions);
        
         //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
         if(workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());

    }

    template <size_t Dim, class T, class M>
    void
    FFT<RCTransform,Dim,T,M>::transform(
        int direction,
        typename FFT<RCTransform,Dim,T,M>::RealField_t& f,
        typename FFT<RCTransform,Dim,T,M>::ComplexField_t& g)
    {
       auto fview = f.getView();
       auto gview = g.getView();
       const int nghostf = f.getNghost();
       const int nghostg = g.getNghost();

       /**
        *This copy to a temporary Kokkos view is needed because of following
        *reasons:
        *1) heffte wants the input and output fields without ghost layers
        *2) heffte accepts data in layout left (by default) eventhough this
        *can be changed during heffte box creation
       */
       Kokkos::View<T***, Kokkos::LayoutLeft>
           tempFieldf("tempFieldf", fview.extent(0) - 2*nghostf,
                                    fview.extent(1) - 2*nghostf,
                                    fview.extent(2) - 2*nghostf);

       Kokkos::View<Complex_t***, Kokkos::LayoutLeft>
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
                                tempFieldg(i-nghostg, j-nghostg, k-nghostg).real(
                                      gview(i, j, k).real());
                                tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag(
                                      gview(i, j, k).imag());
                            });

      
       if ( direction == 1 )
       {
           heffte_m->forward( tempFieldf.data(), tempFieldg.data(), workspace_m.data(),
                              heffte::scale::full );
       }
       else if ( direction == -1 )
       {
           heffte_m->backward( tempFieldg.data(), tempFieldf.data(), workspace_m.data(),
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
                                gview(i, j, k).real() =
                                tempFieldg(i-nghostg, j-nghostg, k-nghostg).real();
                                gview(i, j, k).imag() =
                                tempFieldg(i-nghostg, j-nghostg, k-nghostg).imag();
                            });

    }

    //=========================================================================
    // FFT SineTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type SineTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class M>
    FFT<SineTransform,Dim,T,M>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, Dim> low;
        std::array<long long, Dim> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class M>
    void
    FFT<SineTransform,Dim,T,M>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

         heffte::box3d<long long> inbox  = {low, high};
         heffte::box3d<long long> outbox = {low, high};

         heffte::plan_options heffteOptions =
             heffte::default_options<heffteBackend>();

         if(!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif
            switch (params.get<int>("comm")) {
            
               case a2a:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                   break;
               case a2av:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                   break;
               case p2p:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                   break;
               case p2p_pl:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                   break;
               default:
                   throw IpplException("FFT::setup",
                                       "Unrecognized heffte communication type");
            }
         }

         heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                    (inbox, outbox, Ippl::getComm(), heffteOptions);

         //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
         if(workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());

    }

    template <size_t Dim, class T, class M>
    void
    FFT<SineTransform,Dim,T,M>::transform(
        int direction,
        typename FFT<SineTransform,Dim,T,M>::Field_t& f)
    {
       auto fview = f.getView();
       const int nghost = f.getNghost();
       
       /**
        *This copy to a temporary Kokkos view is needed because of following
        *reasons:
        *1) heffte wants the input and output fields without ghost layers
        *2) heffte accepts data in layout left (by default) eventhough this
        *can be changed during heffte box creation
       */
       Kokkos::View<T***,Kokkos::LayoutLeft>
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
                              tempField(i-nghost, j-nghost, k-nghost) = 
                              fview(i, j, k);
                            });

       if ( direction == 1 )
       {
           heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                             heffte::scale::full);
       }
       else if ( direction == -1 )
       {
           heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
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
                              fview(i, j, k) =
                              tempField(i-nghost, j-nghost, k-nghost);
                            });

    }

    //=========================================================================
    // FFT CosTransform Constructors
    //=========================================================================

    /**
       Create a new FFT object of type CosTransform, with a
       given layout and heffte parameters.
    */

    template <size_t Dim, class T, class M>
    FFT<CosTransform,Dim,T,M>::FFT(
        const Layout_t& layout,
        const ParameterList& params)
    {

        /**
         * Heffte requires to pass a 3D array even for 2D and
         * 1D FFTs we just have to make the length in other
         * dimensions to be 1.
         */
        std::array<long long, Dim> low;
        std::array<long long, Dim> high;

        const NDIndex<Dim>& lDom = layout.getLocalNDIndex();

        low.fill(0);
        high.fill(0);

        /**
         * Static cast to detail::long long (uint64_t) is necessary, as heffte::box3d requires it
         * like that.
         */
        for(size_t d = 0; d < Dim; ++d) {
            low[d] = static_cast<long long>(lDom[d].first());
            high[d] = static_cast<long long>(lDom[d].length() + lDom[d].first() - 1);
        }

        setup(low, high, params);
    }


    /**
           setup performs the initialization necessary.
    */
    template <size_t Dim, class T, class M>
    void
    FFT<CosTransform,Dim,T,M>::setup(const std::array<long long, Dim>& low,
                                  const std::array<long long, Dim>& high,
                                  const ParameterList& params)
    {

         heffte::box3d<long long> inbox  = {low, high};
         heffte::box3d<long long> outbox = {low, high};

         heffte::plan_options heffteOptions =
             heffte::default_options<heffteBackend>();

         if(!params.get<bool>("use_heffte_defaults")) {
            heffteOptions.use_pencils = params.get<bool>("use_pencils");
            heffteOptions.use_reorder = params.get<bool>("use_reorder");
#ifdef Heffte_ENABLE_GPU
            heffteOptions.use_gpu_aware = params.get<bool>("use_gpu_aware");
#endif
            switch (params.get<int>("comm")) {
            
               case a2a:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoall;
                   break;
               case a2av:
                   heffteOptions.algorithm = heffte::reshape_algorithm::alltoallv;
                   break;
               case p2p:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p;
                   break;
               case p2p_pl:
                   heffteOptions.algorithm = heffte::reshape_algorithm::p2p_plined;
                   break;
               default:
                   throw IpplException("FFT::setup",
                                       "Unrecognized heffte communication type");
            }
         }

         heffte_m = std::make_shared<heffte::fft3d<heffteBackend, long long>>
                    (inbox, outbox, Ippl::getComm(), heffteOptions);

         //heffte::gpu::device_set(Ippl::Comm->rank() % heffte::gpu::device_count());
         if(workspace_m.size() < heffte_m->size_workspace())
            workspace_m = workspace_t(heffte_m->size_workspace());

    }


    template <size_t Dim, class T, class M>
    void
    FFT<CosTransform,Dim,T,M>::transform(
        int direction,
        typename FFT<CosTransform,Dim,T,M>::Field_t& f)
    {
       auto fview = f.getView();
       const int nghost = f.getNghost();

       /**
        *This copy to a temporary Kokkos view is needed because of following
        *reasons:
        *1) heffte wants the input and output fields without ghost layers
        *2) heffte accepts data in layout left (by default) eventhough this
        *can be changed during heffte box creation
       */
       Kokkos::View<T***,Kokkos::LayoutLeft>
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
                              tempField(i-nghost, j-nghost, k-nghost) = 
                              fview(i, j, k);
                            });

       if ( direction == 1 )
       {
           heffte_m->forward(tempField.data(), tempField.data(), workspace_m.data(),
                             heffte::scale::full);
       }
       else if ( direction == -1 )
       {
           heffte_m->backward(tempField.data(), tempField.data(), workspace_m.data(),
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
                              fview(i, j, k) =
                              tempField(i-nghost, j-nghost, k-nghost);
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
