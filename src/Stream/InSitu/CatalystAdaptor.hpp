#pragma once
#include "Stream/InSitu/CatalystAdaptor.h"

// Function implementations moved from CatalystAdaptor.h

inline void CatalystAdaptor::setData(conduit_cpp::Node& node, const View_vector& view) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string("mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

inline void CatalystAdaptor::setData(conduit_cpp::Node& node, const View_scalar& view) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string("mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }

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




    /* SCALAR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                [[maybe_unused]]  
                const ippl::Field<T, Dim, ViewArgs...>& entry
                , const std::string label
                ,       conduit_cpp::Node& node
                , const std::filesystem::path source_dir
    // , ViewRegistry& vr
)
{
        std::cout << "      init_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << std::endl;
        // const Field_t<Dim>* field = &entry;
        const std::string script = "catalyst/scripts/" + label;
        const std::string channelName = "ippl_sField_" + label;
        
        set_node_script( node[script + "/filename"],
                        "CATALYST_EXTRACTOR_SCRIPT_P",
                        source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_sfield.py"
                    );
        conduit_cpp::Node args = node[script + "/args"];
        args.append().set_string("--channel_name");
        args.append().set_string(channelName);
        args.append().set_string("--experiment_name");
        args.append().set_string(TestName);

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

    }

    /* VECTOR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                    const ippl::Field<ippl::Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                , const std::string label
                                ,       conduit_cpp::Node& node
                                , const std::filesystem::path source_dir
                                // , ViewRegistry& vr
)
{
        std::cout << "      init_entry(ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Dim_v << ">," << Dim << ">) called" << std::endl;
        // const VField_t<T, Dim>* field = &entry;
        const std::string script = "catalyst/scripts/" + label;
        const std::string channelName = "ippl_vField_" + label;

        set_node_script( node[script + "/filename"],
                        "CATALYST_EXTRACTOR_SCRIPT_P",
                        source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_vfield.py"
                        
                    );
        conduit_cpp::Node args = node[script + "/args"];
        args.append().set_string("--channel_name");
        args.append().set_string(channelName);
        args.append().set_string("--experiment_name");
        args.append().set_string(TestName);

        conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
        script_args.append().set_string(channelName);

    }

    // PARTICLECONTAINERS DERIVED FROM PARTICLEBASE:
    // == ippl::ParticleBaseBase -> ParticleBase<PLayout<T, Dim>, ... , ... >
template<typename T>
requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
void CatalystAdaptor::init_entry( 
                                [[maybe_unused]]  
                                    const T& entry
                                , const std::string label
                                ,       conduit_cpp::Node& node
                                , const std::filesystem::path source_dir
                                // , ViewRegistry& vr
)
{
        std::cout   << "      init_entry(ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass]) called" << std::endl;
                    
            const std::string script = "catalyst/scripts/"+ label;
            const std::string channelName = "ippl_particles_" + label;
            
            set_node_script( 
                            node[script + "/filename"],
                            // file
                            "CATALYST_EXTRACTOR_SCRIPT_P",
                            source_dir /"catalyst_scripts" / "catalyst_extractors" /"png_ext_particle.py"
                            
                        );
            conduit_cpp::Node args = node[script + "/args"];
            args.append().set_string("--channel_name");
            args.append().set_string(channelName);
            args.append().set_string("--experiment_name");
            args.append().set_string(TestName);


            conduit_cpp::Node script_args = node["catalyst/scripts/script/args"];
            script_args.append().set_string(channelName);

    }

    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>)
void CatalystAdaptor::init_entry(
            [[maybe_unused]]         T&& entry
            ,                  const std::string label
            , [[maybe_unused]]       conduit_cpp::Node& node
            , [[maybe_unused]] const std::filesystem::path source_dir
    // , ViewRegistry& vr
)
{
        Inform m("init_entry():");
        m << "Entry type can't be processed. ID: "<< label <<", Type: "<< typeid(std::decay_t<T>).name() <<  endl;
        m << "Channel will not be registered in Conduit Node passed to catalyst." << endl;
    }

    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
void CatalystAdaptor::init_entry( 
            const std::shared_ptr<T>&   entry
        , const std::string           label
        ,       conduit_cpp::Node&    node
        , const std::filesystem::path source_dir
        // , ViewRegistry& vr 
)
{
        if (entry) {
            // std::cout << "  dereferencing shared pointer and reattempting execute..." << std::endl;
            init_entry(  *entry
                        , label
                        , node
                        , source_dir
                        // , vr
            );          // Dereference and dispatch to reference version
        } else {

            Inform m("init_entry():");
            m << "Null shared_ptr passed as entry. ID: "<< label  << endl;
            m << "Channel will not be registered in Conduit Node passed to catalyst." << endl;
        }
    }

void CatalystAdaptor::Execute_Particle(
    const std::string& channelName ,
    const auto& particleContainer
    , const auto& R_host
    , const auto& P_host
    , const auto& q_host
    , const auto& ID_host,
    conduit_cpp::Node& node
)
{

        // channel for particles
        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("explicit");

        //mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        mesh["coordsets/coords/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        mesh["topologies/mesh/type"].set_string("unstructured");
        mesh["topologies/mesh/coordset"].set_string("coords");
        mesh["topologies/mesh/elements/shape"].set_string("point");
        //mesh["topologies/mesh/elements/connectivity"].set_external(particleContainer->ID.getView().data(),particleContainer->getLocalNum());
        mesh["topologies/mesh/elements/connectivity"].set_external(ID_host.data(),particleContainer->getLocalNum());

        //auto charge_view = particleContainer->getQ().getView();






        // add values for scalar charge field
        auto fields = mesh["fields"];
        fields["charge/association"].set_string("vertex");
        fields["charge/topology"].set_string("mesh");
        fields["charge/volume_dependent"].set_string("false");

        //fields["charge/values"].set_external(particleContainer->q.getView().data(), particleContainer->getLocalNum());
        fields["charge/values"].set_external(q_host.data(), particleContainer->getLocalNum());




        

        // add values for vector velocity field
        //auto velocity_view = particleContainer->P.getView();
        fields["velocity/association"].set_string("vertex");
        fields["velocity/topology"].set_string("mesh");
        fields["velocity/volume_dependent"].set_string("false");

        //fields["velocity/values/x"].set_external(&velocity_view.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/y"].set_external(&velocity_view.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        //fields["velocity/values/z"].set_external(&velocity_view.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/x"].set_external(&P_host.data()[0][0], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/y"].set_external(&P_host.data()[0][1], particleContainer->getLocalNum(),0 ,sizeof(double)*3);
        fields["velocity/values/z"].set_external(&P_host.data()[0][2], particleContainer->getLocalNum(),0 ,sizeof(double)*3);






        fields["position/association"].set_string("vertex");
        fields["position/topology"].set_string("mesh");
        fields["position/volume_dependent"].set_string("false");

        //fields["position/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //fields["position/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/x"].set_external(&R_host.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/y"].set_external(&R_host.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        fields["position/values/z"].set_external(&R_host.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);
    }

    /* this needs to be overworked ... */
template <class Field>  // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
void CatalystAdaptor::Execute_Field(const std::string& channelName, Field* field, 
    Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
    conduit_cpp::Node& node)
{
        static_assert(Field::dim == 3, "CatalystAdaptor only supports 3D");

        // A) define mesh

        auto channel = node["catalyst/channels/"+ channelName];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/coords/dims/i"};
        std::string field_node_origin{"coordsets/coords/origin/x"};
        std::string field_node_spacing{"coordsets/coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // add dimension
            mesh[field_node_dim].set(field->getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            mesh[field_node_origin].set(
                field->get_mesh().getOrigin()[iDim] + field->getLayout().getLocalNDIndex()[iDim].first()
                      * field->get_mesh().getMeshSpacing(iDim));

            // add spacing
            mesh[field_node_spacing].set(field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set_string("uniform");
        mesh["topologies/mesh/coordset"].set_string("coords");
        std::string field_node_origin_topo{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < field->get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh[field_node_origin_topo].set(field->get_mesh().getOrigin()[iDim]
                                             + field->getLayout().getLocalNDIndex()[iDim].first()
                                                   * field->get_mesh().getMeshSpacing(iDim));

            // increment last char in string ('x' becomes 'y' becomes 'z')
            ++field_node_origin_topo.back();
        }

        // B) Set the field values

        // Initialize the existing Kokkos::View
        host_view_layout_left = Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>(
           "host_view_layout_left",
           field->getLayout().getLocalNDIndex()[0].length(),
           field->getLayout().getLocalNDIndex()[1].length(),
           field->getLayout().getLocalNDIndex()[2].length());

        // Creates a host-accessible mirror view and copies the data from the device view to the host.
        auto host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field->getView());

        // Copy data from field to the memory+style which will be passed to Catalyst
        auto nGhost = field->getNghost();
        for (size_t i = 0; i < field->getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < field->getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < field->getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }

        // Add values and subscribe to data
        auto fields = mesh["fields"];
        setData(fields, host_view_layout_left);
    }

        // Handle fields
        // // Map of all Kokkos::Views. This keeps a reference on all Kokkos::Views
        // // which ensures that Kokkos does not free the memory before the end of this function.


        /* SCALAR FIELDS - handles both reference and shared_ptr */
        // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, class... ViewArgs>
void CatalystAdaptor::execute_entry(const ippl::Field<T, Dim, ViewArgs...>& entry, const std::string label, conduit_cpp::Node& node, ViewRegistry& vr) {
        std::cout << "      execute_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << std::endl;
        
        
        const std::string channelName = "ippl_sField_" + label;
        // std::map<std::string, Kokkos::View<typename Field_t<Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > scalar_host_views;
        
        Kokkos::View<typename Field_t<Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> scalar_host_view;
        
        const Field_t<Dim>* field = &entry;
        /* creates amp entry with default initialized values ... */
        // Execute_Field(label, field, scalar_host_views[label], node);
        Execute_Field(channelName, field, scalar_host_view, node);

        vr.set(scalar_host_view);
    }

        /* VECTOR FIELDS - handles both reference and shared_ptr */
        // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void CatalystAdaptor::execute_entry(const ippl::Field<ippl::Vector<T, Dim_v>, Dim, ViewArgs...>& entry, const std::string label,conduit_cpp::Node& node, ViewRegistry& vr) {
        std::cout << "      execute_entry(ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Dim_v << ">," << Dim << ">) called" << std::endl;
        
        const std::string channelName = "ippl_vField_" + label;
        const VField_t<T, Dim>* field = &entry;

        
        // std::map<std::string, Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > vector_host_views;
        // Execute_Field(label,field, vector_host_views[label], node); 


        /* use make shaed so we dont pass adereference to a nullptr .... */
        /* for us  more conveneient approach */
        // std::shared<Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > vector_host_views =
        // std::make_shared<Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> >;

        Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_host_view;

        
        Execute_Field(channelName, field, vector_host_view, node); 



        
        vr.set(vector_host_view);
        /* save a copy to shaed pointer, keeps dynamco reference to the View alive
        saving a reference would make sense since referenced object will not exist when exiting this scope ... */
        
    }

        // const std::string& fieldName = "E";
    // You remove const for particles in your execute_entry function
    // because you need to call non-const member functions 
    // (like getHostMirror()) on the particle attributes, which
    // are not marked as const in their class definition. For 
    // fields, you can keep const because the field-related
    // functions you call (like getView(), getLayout(), etc.) 
    // are either const or do not modify the object.




    /* instead of maps storing kokkos view in scope we use the registry to keep everything in frame .... and be totally type indepedent
    we can set with name (but since we likely will not have the need to ever retrieve we can just stire nameless
    to redzcede unncessary computin type ...) */


    // PARTICLECONTAINERS DERIVED FROM PARTICLEBASE:
template<typename T>
requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
void CatalystAdaptor::execute_entry(const T& entry
    , const std::string label
    , conduit_cpp::Node& node
    , ViewRegistry& vr) {
        std::cout   << "      execute_entry(ParticleBase<PLayout<" 
                    << typeid(particle_value_t<T>).name() 
                    << ","
                    << particle_dim_v<T> 
                    << ",...>...> [or subclass]) called" << std::endl;


            const std::string channelName = "ippl_particles_" + label;


            ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror    R_host_view;
            ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror    P_host_view;
            ippl::ParticleAttrib<double>::HostMirror                     q_host_view;
            ippl::ParticleAttrib<std::int64_t>::HostMirror              ID_host_view;

            auto particleContainer = &entry;
            assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

            R_host_view  = particleContainer->R.getHostMirror();
            P_host_view  = particleContainer->P.getHostMirror();
            q_host_view  = particleContainer->q.getHostMirror();
            ID_host_view = particleContainer->ID.getHostMirror();

            Kokkos::deep_copy(R_host_view ,  particleContainer->R.getView());
            Kokkos::deep_copy(P_host_view ,  particleContainer->P.getView());
            Kokkos::deep_copy(q_host_view ,  particleContainer->q.getView());
            Kokkos::deep_copy(ID_host_view, particleContainer->ID.getView());





            Execute_Particle(
                    channelName,
                    particleContainer,
                    R_host_view,
                    P_host_view,
                    q_host_view,
                    ID_host_view,
                  node
            );



            vr.set(R_host_view);
            vr.set(P_host_view);
            vr.set(q_host_view);
            vr.set(ID_host_view);
    }

    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>)
void CatalystAdaptor::execute_entry(const std::string label, [[maybe_unused]] T&& entry, conduit_cpp::Node& node, ViewRegistry& vr) {
        std::cout << "  Entry type can't be processed: ID "<< label <<" "<< typeid(std::decay_t<T>).name() << std::endl;
    }

    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
void CatalystAdaptor::execute_entry( const std::shared_ptr<T>& entry,const std::string  label, conduit_cpp::Node& node, ViewRegistry& vr ) {
        if (entry) {
            // std::cout << "  dereferencing shared pointer and reattempting execute..." << std::endl;
            execute_entry(*entry, label,  node, vr);  // Dereference and dispatch to reference version
        } else {
            std::cout << "  Null shared_ptr encountered" << std::endl;
        }
    }

template<typename T>
void CatalystAdaptor::AddSteerableChannel( T steerable_scalar_forwardpass, std::string steerable_suffix, conduit_cpp::Node& node) {
        std::cout << "      AddSteerableChanelValue( " << steerable_suffix << "); | Type: " << typeid(T).name() << std::endl;
        
        
        auto steerable_channel = node["catalyst/channels/steerable_channel_forward_" + steerable_suffix];

        steerable_channel["type"].set("mesh");
        auto steerable_data = steerable_channel["data"];
        steerable_data["coordsets/coords/type"].set_string("explicit");
        steerable_data["coordsets/coords/values/x"].set_float64_vector({ 1 });
        steerable_data["coordsets/coords/values/y"].set_float64_vector({ 2 });
        steerable_data["coordsets/coords/values/z"].set_float64_vector({ 3 });
        steerable_data["topologies/mesh/type"].set("unstructured");
        steerable_data["topologies/mesh/coordset"].set("coords");
        steerable_data["topologies/mesh/elements/shape"].set("point");
        steerable_data["topologies/mesh/elements/connectivity"].set_int32_vector({ 0 });


        conduit_cpp::Node steerable_field = steerable_data["fields/steerable_field_f_" + steerable_suffix];
        steerable_field["association"].set("vertex");
        steerable_field["topology"].set("mesh");
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
void CatalystAdaptor::FetchSteerableChannelValue( T& steerable_scalar_backwardpass, std::string steerable_suffix, conduit_cpp::Node& results) {
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

void CatalystAdaptor::Results(conduit_cpp::Node& results) {
        
        // conduit_cpp::Node results;
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));
        // catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));
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

/* might not even need references to registries since a copy of s shared pointer still points to the 
right place... */
void CatalystAdaptor::Execute(
    auto& registry_vis, auto& registry_steer,
    int cycle, double time, int rank
){
        
        // add time/cycle information
        conduit_cpp::Node node;
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);     

        /* catch view registry by referrence and pass it to execute by refernece 
        viewregistry with shared pointer will be delted by registry running out of scope,
         shared pointers being deleted and deallocating the allocated copies for memories ...*/
        ViewRegistry vr;


        /* ideally avoid this ... */
        const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");



        if(catalyst_steer && std::string(catalyst_steer)=="ON"){
            
            registry_steer.for_each(
                [&node](std::string_view label, const auto& entry) {
                    // std::cout << "   Entry ID: " << label << "\n";
                    AddSteerableChannel(entry, std::string(label), node);
                }
            );
        }

        registry_vis.for_each(
            [&node, &vr](std::string_view label, const auto& entry){
                // std::cout << "  Entry ID: " << label << "\n";
                execute_entry(entry, std::string(label),  node, vr);
            }
        );
       
        /* std::cout << "dump Conduit Catalyst Node pass forward "< std::endl; */
        // node.print();


        // Pass Conduit node to Catalyst and execute extraction and visualisation
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
        /* Catch steerables in results node */
        conduit_cpp::Node results;
        Results(results);  
        


        if(catalyst_steer && std::string(catalyst_steer)=="ON"){
        
            // /* transfer steearble scalars back to original locaton via registry*/
            registry_steer.for_each(
                [&results](std::string_view label, auto& entry) {
                    // std::cout << "   Entry ID: " << label << "\n";
                    FetchSteerableChannelValue(entry, std::string(label), results);
                }
            );
        }





    }

void CatalystAdaptor::Initialize([[maybe_unused]] auto& registry_vis, [[maybe_unused]] auto& registry_steer) {
        Inform m("Catalyst::Initialize()");

        conduit_cpp::Node node;
        
        
        const std::filesystem::path source_dir =  std::filesystem::path(CATALYST_ADAPTOR_ABS_DIR) / "Stream" / "InSitu";       
        std::cout << "using source_dir = " << source_dir.string() << std::endl;
        
        
        set_node_script( node["catalyst/scripts/script/filename"],
                        "CATALYST_PIPELINE_PATH",
                        source_dir / "catalyst_scripts" / "pipeline_default.py"
                    );
        conduit_cpp::Node args = node["catalyst/scripts/script/args"];
        /* empty arguments cause erros?? pass something useful maybe ^*/
        args.append().set_string("--channel_names");

        // args.append().set_string("noname");
                    


        // TODO: create steering poxy file ... for each steering channel for versatility
        // if(catalyst_steer && std::string(catalyst_steer)=="ON"){
        //     registry_steer.for_each(
        //         [&node](std::string_view label, const auto& entry) {
        //             CreateProxySteerableChannel(entry, std::string(label), node);
        //         }
        //     );
        // }
        const char* catalyst_png = std::getenv("IPPL_CATALYST_PNG");
        if(catalyst_png && std::string(catalyst_png)=="ON"){

            m << "TRYING TO SET CATALYST PNG EXTRACTS" << endl;

            registry_vis.for_each(
                [&](std::string_view label, const auto& entry){
                    init_entry(   entry
                                , std::string(label)
                                , node
                                , source_dir
                            );
                }
            );


        }
        else{
            m << "catalyst PNG extract DEACTIVATED" << endl;
        }

        args.append().set_string("--experiment_name");
        args.append().set_string(TestName);

        const char* catalyst_vtk = std::getenv("IPPL_CATALYST_VTK");

        if(catalyst_vtk && std::string(catalyst_vtk)=="ON"){
            m << "catalyst vtk extraction ACTIVATED" << endl;
            
            args.append().set_string("--VTKextract");
            args.append().set_string("True");
        }
        else{
            m << "catalyst vtk extraction DEACTIVATED" << endl;
            /*  dont set since default will be false ... (and this doesnt work ...) */
            // args.append().set_string("--VTKextract");
            // args.append().set_string("0");
        }

        /* TWO HARDCODED STEERING OPTIONS .... >  */
        /* ideally, this needs to be saved .. */
        const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");

        if(catalyst_steer && std::string(catalyst_steer)=="ON"){
            m << "catalyst steering ACTIVATED." << endl;
            
            args.append().set_string("--steer");
            args.append().set_string("ON");

            set_node_script( node["catalyst/proxies/proxy_e/filename"],
                            "CATALYST_PROXYS_PATH_E",
                            source_dir / "catalyst_scripts" / "proxy_default_electric.xml"
            );
            set_node_script( node["catalyst/proxies/proxy_m/filename"],
                            "CATALYST_PROXYS_PATH_M",
                            source_dir / "catalyst_scripts" / "proxy_default_magnetic.xml"
            );

        }
        else{
            m << "catalyst steering:      DEACTIVATED." << endl;
            /*  dont set since default will be false ... (and this doesnt work ...) */
            args.append().set_string("--steer");
            // args.append().set_int16("0");
            args.append().set_string("OFF");
        }

        // args.print();
        // node.print();


        m << "ippl: catalyst_initialize() =>" << endl;
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {

            m << "\n Catalyst initialized fail.....\n" << endl;
            throw IpplException("Stream::InSitu::CatalystAdaptor", "Failed to initialize Catalyst");
            // std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
        else{
            m << "\n Catalyst initialized successfully.\n" << endl;
        }
    }

void CatalystAdaptor::Finalize() {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }

// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry path)
// =====================================================================================

void CatalystAdaptor::InitializeRuntime(visreg::VisRegistryRuntime& visReg,
                                        visreg::VisRegistryRuntime& steerReg,
                                        const std::filesystem::path& source_dir_in) {
    Inform m("Catalyst::InitializeRuntime()");

    conduit_cpp::Node node;
    std::filesystem::path source_dir = source_dir_in;
    if (source_dir.empty()) {
        source_dir = std::filesystem::path(CATALYST_ADAPTOR_ABS_DIR) / "Stream" / "InSitu";
    }
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
    if (catalyst_png && std::string(catalyst_png) == "ON") {
        m << "PNG extraction ACTIVATED" << endl;
        InitVisitor initV{node, source_dir};
        visReg.for_each(initV);
    } else {
        m << "PNG extraction DEACTIVATED" << endl;
    }

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

        // Optionally initialize steerable placeholders (not strictly required here)
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

void CatalystAdaptor::ExecuteRuntime(visreg::VisRegistryRuntime& visReg,
                                     visreg::VisRegistryRuntime& steerReg,
                                     int cycle, double time, int rank) {
    conduit_cpp::Node node;
    auto state = node["catalyst/state"];
    state["cycle"].set(cycle);
    state["time"].set(time);
    state["domain_id"].set(rank);
    ViewRegistry viewReg;

    const char* catalyst_steer = std::getenv("IPPL_CATALYST_STEER");

    // Forward pass: add channels
    ExecuteVisitor execV{node, viewReg};
    visReg.for_each(execV); // fields + particles only

    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        SteerForwardVisitor steerV{node};
        steerReg.for_each(steerV); // add only steerable scalar channels
    }

    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) {
        std::cerr << "Failed to execute Catalyst (runtime path): " << err << std::endl;
    }

    // Backward pass: fetch updated steering values
    if (catalyst_steer && std::string(catalyst_steer) == "ON") {
        conduit_cpp::Node results;
        Results(results);
        SteerFetchVisitor fetchV{results};
        steerReg.for_each(fetchV);
    }
}


