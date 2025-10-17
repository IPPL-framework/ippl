#pragma once
#include "Stream/InSitu/CatalystAdaptor.h"

/* instead of maps storing kokkos view in scope we use the ViewRegistry to keep everything in frame .... and be totally type indepedent
we can set with name (but since we likely will not have the need to ever retrieve we can just stire nameless
to redzcede unncessary computin type ...) */

namespace ippl{
    
/*  sets a file path to a certain node, first tries to fetch from environment, afterwards uses the dafault path passed  */
void CatalystAdaptor::set_node_script(
    conduit_cpp::Node node_path,
    const char* env_var,
    const std::filesystem::path default_file_path
)
{
        Inform m("CatalystAdaptor::set_node_scripts(): ");
            
        const char* file_path_env = std::getenv(env_var);
        std::filesystem::path file_path;

        
       if (file_path_env && std::filesystem::exists(file_path_env)) {
           m << "Using " << env_var << " from environment: " << file_path_env << endl;
           file_path = file_path_env;
       } else {
           m << "No valid " << env_var <<" set. Using default: " << default_file_path << endl;
           file_path = default_file_path;
       }

       node_path.set(file_path.string());
}




// init visualisation for SCALAR FIELDS 
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                    [[maybe_unused]]  
                      const Field<T, Dim, ViewArgs...>& entry
                    , const std::string label
                )
{
        std::cout << "      init_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << std::endl;
        // const Field_t<Dim>* field = &entry;
        const std::string channelName = "ippl_sField_" + label; 
        if(png_extracts){

            const std::string script = "catalyst/scripts/" + label;
            
            set_node_script( node[script + "/filename"],
                            "CATALYST_EXTRACTOR_SCRIPT_P",
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_sfield.py"
                        );
            conduit_cpp::Node args = node[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
        }

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

    }


// init visualisation for VECTOR FIELDS  
// == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                  const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                , const std::string label
                            )
{
        std::cout << "      init_entry(ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Dim_v << ">," << Dim << ">) called" << std::endl;
        // const VField_t<T, Dim>* field = &entry;
        const std::string channelName = "ippl_vField_" + label;

        if(png_extracts){
            const std::string script = "catalyst/scripts/" + label;

            set_node_script( node[script + "/filename"],
                            "CATALYST_EXTRACTOR_SCRIPT_P",
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_vfield.py"
                            
                        );
            conduit_cpp::Node args = node[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
        }

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

}

// init visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
 void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                  const T& entry
                                , const std::string label
                            )
{
        std::cout   << "      init_entry(ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass]) called" << std::endl;
                    
            const std::string channelName = "ippl_particles_" + label;
            if(png_extracts){
                const std::string script = "catalyst/scripts/"+ label;

                set_node_script( 
                                node[script + "/filename"],
                                "CATALYST_EXTRACTOR_SCRIPT_P",
                                source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_particle.py"
                                
                            );
                conduit_cpp::Node args = node[script + "/args"];
                args.append().set_string("--channel_name");
                args.append().set_string(channelName);
            }

            conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
            script_args.append().set_string(channelName);
}
  
/* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
 void CatalystAdaptor::init_entry( 
              const std::shared_ptr<T>&   entry
            , const std::string           label
)
{
        if (entry) {
            // std::cout << "  dereferencing shared pointer and reattempting execute..." << std::endl;
            init_entry(  *entry
                        , label
            );
        } else {
            Inform m("init_entry():");
            m << "Null shared_ptr passed as entry. ID: "<< label  << endl;
            m << "Channel will not be registered in Conduit Node passed to catalyst." << endl;
        }
}


// BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
 void CatalystAdaptor::init_entry(
                    [[maybe_unused]]
                    // T&& entry
                    const T& entry
                ,   const std::string label
)
{
        Inform m("init_entry():");
        m << "Entry type can't be processed. ID: "<< label <<", Type: "<< typeid(std::decay_t<T>).name() <<  endl;
        m << "Channel will not be registered in Conduit Node passed to catalyst." << endl;
}
  


template <typename T, typename = void>
struct has_getRegionLayout : std::false_type {};

template <typename T>
struct has_getRegionLayout<T, std::void_t<decltype(std::declval<T>().getRegionLayout())>> 
    : std::true_type {};

template <typename T>
constexpr bool has_getRegionLayout_v = has_getRegionLayout<T>::value;



// execute visualisation for SCALAR FIELDS 
// == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
 void CatalystAdaptor::execute_entry(
                            const Field<T, Dim, ViewArgs...>& entry, 
                            const std::string label
    ) {

        const Field<T, Dim, ViewArgs...>* field = &entry;


        std::string channelName;
        if constexpr (std::is_scalar_v<T>) {
            channelName = "ippl_sField_" + label;
        } else if constexpr (is_vector_v<T>) {
            channelName = "ippl_vField_" + label;
        }else{
            channelName = "ippl_errorField_" + label;
        }


        /* 
        options can diverge in the following levels:
                          /type =="mesh" or multimesh
            channels/ *** /data/fields/     *** -> changes
                               /coordsets/  *** -> stays the same
                               /topologies/ *** -> stays the same for al
        
        Currently we are using the same topology and mesh we can
         technically reuse the same channel ->
         but for advanced uses might not be the case
         the labeling og coordsets and topologies in accordances with 
         the layouts ids of ippl would be interesting
         
         
         */
        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");
        // channel of type mesh adheres to conduits mesh blueprint
        
        
        auto data   = channel["data"];
        auto fields = channel["data/fields"];
        auto field_node = fields[label];

        // add topology anmed fmesh_topo of type uniform
        data["topologies/fmesh_topo/type"].set_string("uniform");
        // define which coordinates to use 
        data["topologies/fmesh_topo/coordset"].set_string("cart_uniform_coords");
        // add a coordinate set named cart_uniform_coords  of type uniform
        data["coordsets/cart_uniform_coords/type"].set_string("uniform");

        const auto Layout_        = field->getLayout();
        // check for BareField maybe???....
        const auto Mesh_          = field->get_mesh();

        const auto LocalNDIndex_  = Layout_.getLocalNDIndex();
        const auto Origin_        = Mesh_.getOrigin();
        const auto Spacing_       = Mesh_.getMeshSpacing();

        {
            data["coordsets/cart_uniform_coords/dims/i"].set(LocalNDIndex_[0].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dx"].set(Spacing_[0]);
            data["coordsets/cart_uniform_coords/origin/x"].set( Origin_[0] + LocalNDIndex_[0].first() * Spacing_[0] );
            data["topologies/fmesh_topo/origin/x"].set(  Origin_[0] + LocalNDIndex_[0].first() * Spacing_[0] );
        }
        if constexpr(Dim >= 2){
            data["coordsets/cart_uniform_coords/dims/j"].set(LocalNDIndex_[1].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dy"].set(Spacing_[1]);
            data["coordsets/cart_uniform_coords/origin/y"].set( Origin_[1] + LocalNDIndex_[1].first() * Spacing_[1] );
            data["topologies/fmesh_topo/origin/y"].set(  Origin_[1] + LocalNDIndex_[1].first() * Spacing_[1] );
        }
        if constexpr(Dim >= 3){
            data["coordsets/cart_uniform_coords/dims/k"].set(LocalNDIndex_[2].length() + 1);
            data["coordsets/cart_uniform_coords/spacing/dz"].set(Spacing_[2]);
            data["coordsets/cart_uniform_coords/origin/z"].set( Origin_[2] + LocalNDIndex_[2].first() * Spacing_[2] );
            data["topologies/fmesh_topo/origin/z"].set(  Origin_[2] + LocalNDIndex_[2].first() * Spacing_[2] );
        }

        Kokkos::View<typename Field<T, Dim, ViewArgs...>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> host_view_layout_left;
        host_view_layout_left = Kokkos::View<typename Field<T, Dim, ViewArgs...>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>(
           "host_view_layout_left",
           LocalNDIndex_[0].length(),
           LocalNDIndex_[1].length(),
           LocalNDIndex_[2].length()
        );

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        auto host_view =   Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());

        /* optimize */
        // Copy data from field to the memory+style which will be passed to Catalyst
        auto nGhost = field->getNghost();
        for (size_t i = 0; i < field->getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field->getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field->getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

        if constexpr (std::is_scalar_v<T>) {
            // --- SCALAR FIELD CASE ---
            std::cout << "      execute_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << std::endl;

            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
            field_node["values"].set_external(host_view_layout_left.data(), host_view_layout_left.size());

        } else if constexpr (is_vector_v<T>) {
            // --- VECTOR FIELD CASE ---
            std::cout << "      execute_entry(ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Field<T, Dim, ViewArgs...>::dim << ">," << Dim << ">) called" << std::endl;
       

            field_node["association"].set_string("element");
            field_node["topology"].set_string("fmesh_topo");
            field_node["volume_dependent"].set_string("false");
            auto length = std::size(host_view_layout_left);
            // offset is zero as we start without the ghost cells
            // stride is 1 as we have every index of the array
                                     field_node["values/x"].set_external(&host_view_layout_left.data()[0][0], length, 0, 1);
            if constexpr (T::dim>=2) field_node["values/y"].set_external(&host_view_layout_left.data()[0][1], length, 0, 1);
            if constexpr (T::dim>=3) field_node["values/z"].set_external(&host_view_layout_left.data()[0][2], length, 0, 1);        
                
        } else {
            // --- INVALID CASE ---
            std::cout << "execute_entry(Field<"<<typeid(T).name()<< ">)" << std::endl;
            std::cout   << "For this type of Field the Conduit Blueprint description wasnt \n" 
                        << "implemented in ippl. Therefore this type of field is not \n"
                        << "supported for visualisation." << std::endl;
        }

        viewRegistry.set(host_view_layout_left);
    }




// execute visualisation for PARTICLECONTAINERS derived from ParticleBaseBase:
// == ippl::ParticleBase<PLayout<T, dim>,...>,...>
template<typename T>
requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
void CatalystAdaptor::execute_entry(
                      const T& entry
                    , const std::string label
) {
        std::cout   << "      execute_entry(ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass]) called" << std::endl;
        const std::string channelName = "ippl_particles_" + label;

        auto particleContainer = &entry;

        
 
        assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

        
        // channel for this particleContainer
        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");
        
        auto data = channel["data"];
        auto fields = channel["data/fields"];
        
        ParticleAttrib<std::int64_t>::HostMirror      ID_host;
        ParticleAttrib<Vector<double, 3>>::HostMirror R_host;
        ID_host = particleContainer->ID.getHostMirror();
        R_host  = particleContainer->R.getHostMirror();
        Kokkos::deep_copy(ID_host,  particleContainer->ID.getView());
        Kokkos::deep_copy(R_host ,  particleContainer->R.getView());
        viewRegistry.set(ID_host);
        viewRegistry.set(R_host);





        
        /* CHECK IF pLAYOUT IS SATIAL LAYOUT OR PURE LAYOUT */
        using PLayout_t = T::Layout_t;
        if constexpr (has_getRegionLayout_v<PLayout_t>){

            using RLayout_t  = PLayout_t::RegionLayout_t;
            using NDRegion_t = RLayout_t::NDRegion_t;
            constexpr unsigned dim_ = PLayout_t::dim;
            // using value_type = PLayout_t::value_type;
            const NDRegion_t ndr = particleContainer->getLayout().getRegionLayout().getDomain();
            
            /* HELPER COORDINATES TO PASS THE BOUNDING BOX */
            data["coordsets/bound_helper_coords/type"].set_string("uniform");
            /* HELPER TOPOLOGY TO PASS THE BOUNDING BOX (????)*/
            data["topologies/bound_helper_topo/coordset"].set_string("bound_helper_coords");
            data["topologies/bound_helper_topo/type"].set_string("uniform");
            /* create unfirom coordinate mesh only consisting of the corner points of the domain */ 
            {
                data["coordsets/bound_helper_coords/dims/i"].set(2);
                data["coordsets/bound_helper_coords/spacing/dx"].set( ndr[0].max()  - ndr[0].min() );
                data["coordsets/bound_helper_coords/origin/x"].set(   ndr[0].min() );
                data["topologies/bound_helper_topo/origin/x"].set(    ndr[0].min() );
            }
            if constexpr(dim_ >= 2){
                data["coordsets/bound_helper_coords/dims/j"].set(2);
                data["coordsets/bound_helper_coords/spacing/dy"].set( ndr[1].max()- ndr[1].min() );
                data["coordsets/bound_helper_coords/origin/y"].set(   ndr[1].min()               );
                data["topologies/bound_helper_topo/origin/y"].set(    ndr[1].min()               );
            }
            if constexpr(dim_ >= 3){
                data["coordsets/bound_helper_coords/dims/k"].set(2);
                data["coordsets/bound_helper_coords/spacing/dz"].set( ndr[2].max()- ndr[1].min() );
                data["coordsets/bound_helper_coords/origin/z"].set(   ndr[2].min()               );
                data["topologies/bound_helper_topo/origin/z"].set(    ndr[2].min()               );
            }
    }
        /* might need to pass an actual feel that uses this topology ... */

        
        
        
        //* EXPLICIT COORDINATES -> EACH PARTICLE POSITION */
        data["coordsets/p_coords/type"].set_string("explicit");
        data["coordsets/p_coords/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        data["coordsets/p_coords/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        data["coordsets/p_coords/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        
        data["topologies/pMesh_topo/type"].set_string("unstructured");
        data["topologies/pMesh_topo/coordset"].set_string("p_coords");
        data["topologies/pMesh_topo/elements/shape"].set_string("point");
        data["topologies/pMesh_topo/elements/connectivity"].set_external(ID_host.data(),particleContainer->getLocalNum());
        /* no copy in situ vis:... */
        //mesh["topologies/pmesh_topo/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        
        /* ATTRIBUTES HARDCODED IN PARTICELBASE */
        /* POSITION ATTRIBUTE */
        // this can be left hardcodeed or made part of the for loop, but more efficient to do it right here and take out of loop...
        fields["position/association"].set_string("vertex");
        fields["position/topology"].set_string("pMesh_topo");
        fields["position/volume_dependent"].set_string("false");
        fields["position/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
            
        
        // "no way" to know what dimensions and types particle attributes will have
        // from pointer instance since is base pointer make execute member method instead to 
        // use virtual functions. Is most pragmatic approach.

        std::cout << "============================================" << std::endl;
        const size_t n_attributes =  entry.getAttributeNum();
        for(size_t i = 2; i < n_attributes; ++i){

            const auto attribute = entry.getAttribute(i);
            const std::string  attribute_name = attribute->get_name();
            std::cout << "Execute attribute: " << attribute_name << std::endl;
            attribute->signConduitBlueprintNode_rememberHostCopy(particleContainer->getLocalNum(), fields, viewRegistry);
            
        }
        std::cout << "============================================" << std::endl;
        
        // entry.template forAllAttributes<void>(
        //     [&]<typename Attributes>(const Attributes& atts) {
        //         for (auto* attribute : atts) {
        //             const std::string attribute_name = attribute->get_name();
        //             std::cout << "Execute attribute: " << attribute_name << std::endl;

        //             if( attribute_name  != "ID"){
        //                 attribute->signConduitBlueprintNode_rememberHostCopy(   particleContainer->getLocalNum(), 
        //                                                                         node, 
        //                                                                         viewRegistry
        //                                                                 );
        //             }
        //         }
        //     }
        // );
        // std::cout << "============================================" << std::endl;
}

    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
 void CatalystAdaptor::execute_entry(   [[maybe_unused]] T&& entry, 
                                        const std::string label
) {
        std::cout << "  Entry type can't be processed: ID "<< label <<" "<< typeid(std::decay_t<T>).name() << std::endl;
    }

    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
 void CatalystAdaptor::execute_entry(   const std::shared_ptr<T>& entry,
                                        const std::string  label
                                    ) {
        if (entry) {
            std::cout << "  dereferencing shared pointer and reattempting execute..." << std::endl;
            execute_entry(*entry, label
                // ,  node, vr
            );  // Dereference and dispatch to reference version
        } else {
            std::cout << "  Null shared_ptr encountered" << std::endl;
        }
    }

template<typename T>
void CatalystAdaptor::AddSteerableChannel( T steerable_scalar_forwardpass, 
                                            std::string steerable_suffix
                                        ) {
        std::cout << "      AddSteerableChanelValue( " << steerable_suffix << "); | Type: " << typeid(T).name() << std::endl;
        
        
        auto steerable_channel = node["catalyst/channels/steerable_channel_forward_" + steerable_suffix];

        steerable_channel["type"].set("mesh");
        auto steerable_data = steerable_channel["data"];
        steerable_data["coordsets/coords/type"].set_string("explicit");
        steerable_data["coordsets/coords/values/x"].set_float64_vector({ 1 });
        steerable_data["coordsets/coords/values/y"].set_float64_vector({ 2 });
        steerable_data["coordsets/coords/values/z"].set_float64_vector({ 3 });

        steerable_data["topologies/sMesh_topo/type"].set("unstructured");
        steerable_data["topologies/sMesh_topo/coordset"].set("coords");
        steerable_data["topologies/sMesh_topo/elements/shape"].set("point");
        steerable_data["topologies/sMesh_topo/elements/connectivity"].set_int32_vector({ 0 });


        conduit_cpp::Node steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
        steerable_field["association"].set("vertex");
        steerable_field["topology"].set("sMesh_topo");
        steerable_field["volume_dependent"].set("false");

        conduit_cpp::Node values = steerable_field["values"];


        if constexpr (std::is_same_v<T, double>) {
            values.set_float64_vector({steerable_scalar_forwardpass});
        } else if constexpr (std::is_same_v<T, float>) {
            values.set_float32_vector({steerable_scalar_forwardpass});
        } else if constexpr (std::is_same_v<T, int>) {
            values.set_int64_vector({steerable_scalar_forwardpass});
        } else if constexpr (std::is_same_v<T, unsigned int>) {
            values.set_uint64_vector({steerable_scalar_forwardpass});
        } else {
            throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported steerable type for channel: " + steerable_suffix);
        }
        
    }

    /* maybe use function overloading instead ... */
        // const std::string value_path = ... 
        // steerable_scalar_backwardpass = static_cast<T>(results[value_path].value());
        // steerable_scalar_backwardpass = results[value_path].value()[0];
        /* ????? this should work?? */
        // if constexpr (std::is_same_v<std::remove_cvref_t<T>, double>) {

template<typename T>
 void CatalystAdaptor::FetchSteerableChannelValue( T& steerable_scalar_backwardpass, 
                                                    std::string steerable_suffix
                                                ) {
        std::cout << "      FetchSteerableChanelValue(" << steerable_suffix  << ") | Type: " << typeid(T).name() << std::endl;

            
            conduit_cpp::Node steerable_channel     = results["catalyst/steerable_channel_backward_" + steerable_suffix];
            conduit_cpp::Node steerable_field       = steerable_channel[ "fields/steerable_field_b_" + steerable_suffix];
            conduit_cpp::Node values = steerable_field["values"];

        if constexpr (std::is_same_v<T, double>) {
            if (steerable_field["values"].dtype().is_number()) {
                steerable_scalar_backwardpass = steerable_field["values"].to_double();
                // std::cout << "value scalar fetched" << std::endl;
            }
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr){
            //         steerable_scalar_backwardpass = ptr[0];
            //         std::cout << "value vector fetched ..." << std::endl;
            //     }
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + steerable_suffix);
            // }
            else {
                throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable value type for channel: " + steerable_suffix);
            }
        }
         else if constexpr (std::is_same_v<T, float>) {
            if (steerable_field["values"].dtype().is_number()) {
                steerable_scalar_backwardpass = steerable_field["values"].to_double();
            }
            // else if (steerable_field["values"].dtype().is_float32()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + steerable_suffix);
            // }
            else {
                throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable value type for channel: " + steerable_suffix);
            }
        }
        else if constexpr (std::is_same_v<T, int>) {
            if (steerable_field["values"].dtype().is_number()) {
                steerable_scalar_backwardpass = steerable_field["values"].to_int32();
            }
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + steerable_suffix);
            // }
            else {
                throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable value type for channel: " + steerable_suffix);
            }
        }
        else if constexpr (std::is_same_v<T, unsigned int>) {
            if (steerable_field["values"].dtype().is_number()) {
                steerable_scalar_backwardpass = steerable_field["values"].to_uint32();
            }
            // else if (steerable_field["values"].dtype().is_float64()) {
            //     auto ptr = steerable_field["values"].as_float64_ptr();
            //     if (ptr) steerable_scalar_backwardpass = ptr[0];
            //     else throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Null pointer for steerable value: " + steerable_suffix);
            // }
            else {
                throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable value type for channel: " + steerable_suffix);
            }
        }
        
        else {
            std::cout << "failed to fetch value" << std::endl;
            throw IpplException("CatalystAdaptor::FetchSteerableChannelValue", "Unsupported steerable type for channel: " + steerable_suffix);
        } 
    }




void CatalystAdaptor::fetchResults() {
        
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
        // else
        // {
        //     std::cout << "Result Node dump:" << std::endl;
        //     results.print();
        // }   
    }


// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry)
// =====================================================================================

void CatalystAdaptor::InitializeRuntime(
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg
                        ) {
    Inform m("Catalyst::InitializeRuntime()");

        // if ( !(catalyst_steer && std::string(catalyst_steer) == "OFF") ){
        // m << "Catalyst Visualisation was deactivated via setting env variable IPPL_CATALYST_VIS=OFF"

    visRegistry   = visReg;
    steerRegistry = steerReg;

    // conduit_cpp::Node node;
    // std::filesystem::path source_dir = source_dir_in;
    // if (source_dir.empty()) {
    // }


    source_dir = std::filesystem::path(CATALYST_ADAPTOR_ABS_DIR) / "Stream" / "InSitu";
    m << "using source_dir = " << source_dir.string() << endl;

    // Pipeline script (allow override by environment)
    set_node_script(node["catalyst/scripts/script/filename"],
                    "CATALYST_PIPELINE_PATH",
                    source_dir / "catalyst_scripts" / "pipeline_default.py");
    conduit_cpp::Node args = node["catalyst/scripts/script/args"];
    args.append().set_string("--channel_names");

    const char* catalyst_png = std::getenv("IPPL_CATALYST_PNG");
    const char* catalyst_vtk = std::getenv("IPPL_CATALYST_VTK");
    const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");

    // If PNG extraction requested, run init visitor over visualization registry
    // const bool class member 
    png_extracts = (catalyst_png && std::string(catalyst_png) == "ON");


    if (png_extracts) {
        m << "PNG extraction ACTIVATED" << endl;
    } else {
        m << "PNG extraction DEACTIVATED" << endl;
    }
    
    InitVisitor initV{*this};
    visRegistry->for_each(initV);

    if (catalyst_vtk && std::string(catalyst_vtk) == "ON") {
        m << "VTK extraction ACTIVATED" << endl;
        args.append().set_string("--VTKextract");
        args.append().set_string("True");
    } else {
        m << "VTK extraction DEACTIVATED" << endl;
    }

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        m << "Steering ACTIVATED" << endl;
        args.append().set_string("--steer");
        args.append().set_string("ON");

        set_node_script(node["catalyst/proxies/proxy_e/filename"],
                        "CATALYST_PROXYS_PATH_E",
                        source_dir / "catalyst_scripts" / "proxy_default_electric.xml");
        set_node_script(node["catalyst/proxies/proxy_m/filename"],
                        "CATALYST_PROXYS_PATH_M",
                        source_dir / "catalyst_scripts" / "proxy_default_magnetic.xml");



        // TODO:
        // InitVisitor steerInit{node, source_dir};
        // steerReg.for_each(steerInit);


    } else {
        m << "Steering DEACTIVATED" << endl;
        args.append().set_string("--steer");
        args.append().set_string("OFF");
    }

    m << "ippl: catalyst_initialize() =>" << endl;
    catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        m << "\n Catalyst initialization failed.\n" << endl;
        throw IpplException("Stream::InSitu::CatalystAdaptor", "Failed to initialize Catalyst (runtime path)");
    } else {
        m << "\n Catalyst initialized successfully (runtime path).\n" << endl;
    }
}

void CatalystAdaptor::ExecuteRuntime( int cycle, double time, int rank /* default = ippl::Comm->rank() */) {
    Inform m("Catalyst::ExecuteRuntime()");
 
    const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");
    const char* catalyst_vis = std::getenv("IPPL_CATALYST_VIS");

    auto state = node["catalyst/state"];
    state["cycle"].set(cycle);
    state["time"].set(time);
    state["domain_id"].set(rank);

    // m << "Catalyst Visualisation was deactivated via setting env variable IPPL_CATALYST_VIS=OFF" << endl;
    if ( !(catalyst_vis && std::string(catalyst_vis) == "OFF") ){
        // forward Node: add visualisation channels
        ExecuteVisitor execV{*this};
        visRegistry->for_each(execV); 

    }

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        // forward Node: add steering channels
        SteerForwardVisitor steerV{*this};
        steerRegistry->for_each(steerV); 
    }
    if( cycle == 0)  node.print();
    // Conduit Node Forward pass to Catalyst
    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        std::cerr << "Failed to execute Catalyst (runtime path): " << err << std::endl;
    }

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        // Conduit Node Backward pass from Catalyst
        conduit_cpp::Node results;
        fetchResults();
        // backward Node: fetch updated steering values
        SteerFetchVisitor fetchV{*this};
        steerRegistry->for_each(fetchV);
    }
        Kokkos::fence();
        viewRegistry.clear();
        node.reset();
        results.reset();
}


void CatalystAdaptor::Finalize() {
    conduit_cpp::Node node;
    catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
    }
}



}//ippl