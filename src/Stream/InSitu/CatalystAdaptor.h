#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <catalyst.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <utility>
#include <type_traits>
#include "Utility/IpplException.h"

#include<filesystem>

/* catalyst header defined the following for free use ... */
//   CATALYST_EXPORT enum catalyst_status catalyst_initialize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_finalize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_about(conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_results(conduit_node* params);


namespace CatalystAdaptor {

    template <typename T, unsigned Dim>
    using FieldVariant = std::variant<Field_t<Dim>*, VField_t<T, Dim>*>;



    template <typename T, unsigned Dim>
    using FieldPair = std::pair<std::string, FieldVariant<T, Dim>>;

    template <typename T, unsigned Dim>
    using ParticlePair = std::pair<std::string, std::shared_ptr<ParticleContainer<T, Dim> > >;


    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_vector& view) {
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


    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_scalar& view) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string("mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }



    void Initialize() {
        conduit_cpp::Node node;
        

       const char* catalyst_pipeline_path_env = std::getenv("CATALYST_PIPELINE_PATH");
       const char* catalyst_proxy_path_env    = std::getenv("CATALYST_PROXY_PATH");

       std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();

       std::filesystem::path pipeline_file_path;
       if (catalyst_pipeline_path_env && std::filesystem::exists(catalyst_pipeline_path_env)) {
           pipeline_file_path = catalyst_pipeline_path_env;
           std::cout << "Using CATALYST_PIPELINE_PATH from environment: " << pipeline_file_path << std::endl;
       } else {
           pipeline_file_path = source_dir / "catalyst_scripts" / "pipeline_default.py";
           std::cout << "No valid CATALYST_PIPELINE_PATH. Using default: " << pipeline_file_path << std::endl;
       }

       std::filesystem::path proxy_file_path;
       if (catalyst_proxy_path_env && std::filesystem::exists(catalyst_proxy_path_env)) {
           proxy_file_path = catalyst_proxy_path_env;
           std::cout << "Using CATALYST_PROXY_PATH from environment: " << proxy_file_path << std::endl;
       } else {
           proxy_file_path = source_dir / "catalyst_scripts" / "proxy_default.xml";
           std::cout << "No valid CATALYST_PROXY_PATH. Using default: " << proxy_file_path << std::endl;
       }

       // Apply resolved paths to Catalyst node
       node["catalyst/scripts/script/filename"].set(pipeline_file_path.string());
       node["catalyst/proxies/proxy"].set(proxy_file_path.string());
       

        std::cout << "ippl: catalyst_initialize() call" << std::endl;
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            throw IpplException("Stream::InSitu::CatalystAdaptor", "Failed to initialize Catalyst");
            // std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
        else{
            std::cout << "\n Catalyst initialized successfully.\n" << std::endl;
        }
    }

    // void Initialize_Adios(int argc, char* argv[])
    // {
    //     conduit_cpp::Node node;
    //     for (int cc = 1; cc < argc; ++cc)
    //     {
    //         if (strstr(argv[cc], "xml"))
    //         {
    //             node["adios/config_filepath"].set_string(argv[cc]);
    //         }
    //         else
    //         {
    //             node["catalyst/scripts/script" +std::to_string(cc - 1)].set_string(argv[cc]);
    //         }
    //     }
    //     node["catalyst_load/implementation"] = getenv("CATALYST_IMPLEMENTATION_NAME");
    //     catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
    //     if (err != catalyst_status_ok)
    //     {
    //         std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
    //     }
    // }

    void Finalize() {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }


    void Execute_Particle(
         const auto& particleContainer,
         const auto& R_host, const auto& P_host, const auto& q_host, const auto& ID_host,
         const std::string& particlesName,
         conduit_cpp::Node& node) {

        // channel for particles
        auto channel = node["catalyst/channels/ippl_" + particlesName];
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
    




    template <class Field>  // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    void Execute_Field(Field* field, const std::string& fieldName,
         Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
         conduit_cpp::Node& node) {
        static_assert(Field::dim == 3, "CatalystAdaptor only supports 3D");

        // A) define mesh

        // add catalyst channel named ippl_"field", as fields is reserved
        auto channel = node["catalyst/channels/ippl_" + fieldName];
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

    void AddSteerableChannel(conduit_cpp::Node& node, double scaleFactor) {
        auto steerable = node["catalyst/channels/steerable"];
        steerable["type"].set("mesh");

        auto steerable_data = steerable["data"];
        steerable_data["coordsets/coords/type"].set_string("explicit");

        steerable_data["coordsets/coords/values/x"].set_float64_vector({ 1 });
        steerable_data["coordsets/coords/values/y"].set_float64_vector({ 2 });
        steerable_data["coordsets/coords/values/z"].set_float64_vector({ 3 });

        steerable_data["topologies/mesh/type"].set("unstructured");
        steerable_data["topologies/mesh/coordset"].set("coords");
        steerable_data["topologies/mesh/elements/shape"].set("point");
        steerable_data["topologies/mesh/elements/connectivity"].set_int32_vector({ 0 });

        steerable_data["fields/steerable/association"].set("vertex");
        steerable_data["fields/steerable/topology"].set("mesh");
        steerable_data["fields/steerable/volume_dependent"].set("false");
        steerable_data["fields/steerable/values"].set_float64_vector({scaleFactor});
    }

    void Results( double& scaleFactor) {
        
        conduit_cpp::Node results;
        catalyst_status err = catalyst_results(conduit_cpp::c_node(&results));

        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to execute Catalyst-results: " << err << std::endl;
        }
        else
        {
            std::cout << "Result Node dump:" << std::endl;
            const std::string value_path = "catalyst/steerable/fields/scalefactor/values";
            scaleFactor = results[value_path].to_double();
        }   
    }




    
            // catalyst blueprint definition
            // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
            //
            // conduit blueprint definition (v.8.3)
            // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
            
            
    template<typename entry_t>
    void execute_steer(entry_t entry){

        using EntryType = std::decay_t<decltype(entry)>;
        // std::cout << "   Processing entry of type: " << typeid(EntryType).name() << "\n";
        std::cout << "   Processing Steering Entries: " <<std::endl;
        //  typeid(EntryType).name() << "\n";
        
        // Type-specific processing
        if constexpr (std::is_same_v<EntryType, double>) {
            std::cout << "     -> double steering parameter: " << "\n";
        } else if constexpr (std::is_same_v<EntryType, float>) {
            std::cout << "     -> float steering parameter" <<  "\n";
        } else if constexpr (std::is_same_v<EntryType, int>) {
            std::cout << "     -> int  steering parameter\n" << std::endl;;
        } else {
            std::cout << "   Unclear steeerig " << "\n";
        }
    }






/* SCALAR FIELDS - handles both reference and shared_ptr */
template<typename T, unsigned Dim, class... ViewArgs>
void execute_entry([[maybe_unused]]  const ippl::Field<T, Dim, ViewArgs...>& entry) {
    std::cout << " execute_entry(ippl::Field<" << typeid(T).name() << "," << Dim << ">) called" << std::endl;
    // Add your field processing logic here
}

/* VECTOR FIELDS - handles both reference and shared_ptr */
template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
void execute_entry([[maybe_unused]]  const ippl::Field<ippl::Vector<T, Dim_v>, Dim, ViewArgs...>& entry) {
    std::cout << " execute_entry(ippl::Field<ippl::Vector<" << typeid(T).name() << "," << Dim_v << ">," << Dim << ">) called" << std::endl;
    // Add your vector field processing logic here
}



        // using vector_type            = typename PLayout::vector_type;
        // using index_type             = typename PLayout::index_type;
        // using particle_position_type = typename PLayout::particle_position_type;
        // using particle_index_type    = ParticleAttrib<index_type, IDProperties...>;

        // using Layout_t = PLayout;

        // template <typename... Properties>
        // using attribute_type = typename detail::ParticleAttribBase<Properties...>;

        // template <typename MemorySpace>
        // using container_type = std::vector<attribute_type<MemorySpace>*>;

        // using attribute_container_type =
        //     typename detail::ContainerForAllSpaces<container_type>::type;

        // using bc_container_type = typename PLayout::bc_container_type;

        // using hash_container_type = typename detail::ContainerForAllSpaces<detail::hash_type>::type;

        // using size_type = detail::size_type;
// C++20: function template with requires
template<typename T>
requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
void execute_entry([[maybe_unused]] const T& entry) {
    std::cout << "DD execute_entry(ParticleBaseBase or subclass) called" << std::endl;
    std::cout << "Particle Space Dimension:"<< particle_dim_v<T> << std::endl;
    std::cout << "Particle Data Scalar Type:"  <<  typeid(particle_value_t<T>).name() << std::endl;
}




// Base case: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
template<typename T>
requires (!std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>)
void execute_entry([[maybe_unused]] T&& entry) {
    std::cout << "AA  Entry type can't be processed: " << typeid(std::decay_t<T>).name() << std::endl;
}





//     // using Base = ippl::ParticleBase<typename T::Layout_t>;
//     using Layout = typename T::Layout_t;
//     using value_type = typename Layout::vector_type::value_type;












/* WE DONT  USE BASE CASE BECAUSE WE CANT AVOID TRIGGERING IT FOR PARTICLECONTAINERS AAAAAAAAAA WIESO.....!!!! */

/* BASE CASE - handles unknown types 
withouth enable_if this will be preferred for in derived classes from particle base
which is not ideal, use sfinae to make base case fail, if addAttribute method exists,
 since then there shoudl be another specialisation*/
// template<typename EntryT>
// std::enable_if_t<!has_addAttribute<EntryT>::value,void>  execute_entry([[maybe_unused]] EntryT&& entry) {
//     std::cout << "AA  Entry type can't be processed: " << typeid(std::decay_t<EntryT>).name() << std::endl;   
// }

// or
// template<typename EntryT, std::enable_if_t<!has_addAttribute<EntryT>::value, int> = 0>
// void execute_entry([[maybe_unused]] EntryT&& entry) {
//    std::cout << "AA  Entry type can't be processed: " << typeid(std::decay_t<EntryT>).name() << std::endl;   

//    // ...
// }






/* Doesn't work ... */

// // Detection idiom for addAttribute
// template<typename, typename = void>
// struct has_addAttribute : std::false_type {};

// template<typename T>
// struct has_addAttribute<T, std::void_t<decltype(&T::addAttribute)>> : std::true_type {};


/* Also does't work ... */


// Primary template: false by default
// template<typename T, typename = void>
// struct is_any_particle_base : std::false_type {};

// // Specialization: only enabled if T has Layout_t
// template<typename T>
// struct is_any_particle_base<T, std::void_t<typename T::Layout_t>>
//     : std::bool_constant<std::is_base_of_v<ippl::ParticleBase<typename T::Layout_t>, std::decay_t<T>>> {};




/* PARTICLE CONTAINERS; fails for derived types .. and is limiting later on maybe ....
using ParticleBase with trait extraction via specifie Particle Layout */
// template<typename T, unsigned Dim, typename... PositionProperties, typename... IDProperties>
// void execute_entry([[maybe_unused]]  const ippl::ParticleBase<ippl::detail::ParticleLayout<T, Dim, PositionProperties...>, IDProperties...>& entry) {
//     std::cout   << "CC execute_entry(ippl::ParticleBase<ippl::detail::ParticleLayou<"
//                 << typeid(T).name() << "," << Dim << ">) called" << std::endl;
// }





// template<unsigend Dim,  typename T>
// template<typename ippl::Field<ippl::Vector<T, Dim>, Dim>
// void execute_entry(entry_t entry){;
// }



/* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
template<typename T>
void execute_entry(const std::shared_ptr<T>& entry) {
    if (entry) {
        std::cout << "shared pointer ..." << std::endl;
        execute_entry(*entry);  // Dereference and dispatch to reference version
    } else {
        std::cout << "  Null shared_ptr encountered" << std::endl;
    }
}






    void Execute(
        /* template instead of auto ... */
        auto registry_vis, auto registry_steer,
        int cycle, double time, int rank
    ){
        
        // add time/cycle information
        conduit_cpp::Node node;
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);



        // std::cout << "All declared IDs:\n";
        // auto all_ids = registry_vis.getAllIds();
        // std::cout << "   ";
        // for (const auto& id : all_ids) {
        //     std::cout << "\"" << id << "\" ";
        // }
        // std::cout << "\n\n";
        

        registry_vis.forEach([](std::string_view id, const auto& entry) {
            std::cout << "   Entry ID: " << id << "\n";
            execute_entry(entry);
        });


        registry_steer.forEach([](std::string_view id, const auto& entry) {
            std::cout << "   Entry ID: " << id << "\n";
            execute_steer(entry);
        });






    // Handle particles{
/* 
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> R_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror> P_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<double>::HostMirror> q_host_map;
        std::map<std::string, typename ippl::ParticleAttrib<std::int64_t>::HostMirror> ID_host_map;

        // Loop over all particle container
        for (const auto& particlesPair : particles)
        {
            const std::string& particlesName = particlesPair.first;
            const auto particleContainer = particlesPair.second;

            assert((particleContainer->ID.getView().data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");

            R_host_map[particlesName]  = particleContainer->R.getHostMirror();
            P_host_map[particlesName]  = particleContainer->P.getHostMirror();
            q_host_map[particlesName]  = particleContainer->q.getHostMirror();
            ID_host_map[particlesName] = particleContainer->ID.getHostMirror();

            Kokkos::deep_copy(R_host_map[particlesName],  particleContainer->R.getView());
            Kokkos::deep_copy(P_host_map[particlesName],  particleContainer->P.getView());
            Kokkos::deep_copy(q_host_map[particlesName],  particleContainer->q.getView());
            Kokkos::deep_copy(ID_host_map[particlesName], particleContainer->ID.getView());

            Execute_Particle(
              particleContainer,
              R_host_map[particlesName], P_host_map[particlesName], q_host_map[particlesName], ID_host_map[particlesName],
              particlesName,
              node);
        }
        

 */
/* 
    // Handle fields

        // Map of all Kokkos::Views. This keeps a reference on all Kokkos::Views
        // which ensures that Kokkos does not free the memory before the end of this function.
        std::map<std::string, Kokkos::View<typename Field_t<Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > scalar_host_views;
        std::map<std::string, Kokkos::View<typename VField_t<T, Dim>::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace> > vector_host_views;

        // Loop over all fields
        for (const auto& fieldPair : fields)
        {
            const std::string& fieldName = fieldPair.first;
            const auto& fieldVariant = fieldPair.second;

            // If field is a _scalar_ field
            if (std::holds_alternative<Field_t<Dim>*>(fieldVariant)) {
                Field_t<Dim>* field = std::get<Field_t<Dim>*>(fieldVariant);
                // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, scalar_host_views[fieldName], node);
            }
            // If field is a _vector_ field
            else if (std::holds_alternative<VField_t<T, Dim>*>(fieldVariant)) {
                VField_t<T, Dim>* field = std::get<VField_t<T, Dim>*>(fieldVariant);
                // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*

                Execute_Field(field, fieldName, vector_host_views[fieldName], node);     
            }
        }
*/







        // AddSteerableChannel(node, scaleFactor);

        // // Pass Conduit node to Catalyst
        // catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        // if (err != catalyst_status_ok) {
        //     std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        // }

        // Results(scaleFactor);
    }

}  // namespace CatalystAdaptor

#endif


