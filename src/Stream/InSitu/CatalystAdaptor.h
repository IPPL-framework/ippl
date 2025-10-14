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
#include <list>
#include<filesystem>

#include "Utility/IpplException.h"

#include "Stream/Registry/ViewRegistry.h"

#include "Stream/Registry/RegistryHelper.h"
#include "Stream/Registry/VisRegistry.h"
#include "Stream/Registry/VisRegistry_mini.h"
// #include "Stream/InSitu/VisBaseAdaptor.h"


/* catalyst header defined the following for free use ... */
//   CATALYST_EXPORT enum catalyst_status catalyst_initialize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_finalize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_about(conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_results(conduit_node* params);


// ############################################
// Possible TODO:
// ability to initialize new objects or reinitialize 
// with completely new set of objects
// 
// 
// CatalystAdaptor.h needs VisRegistryRuntime.h
// VisRegistryRuntime.h needs CataylstVistors.h (only Visitors)
// CataylstVistors.h need CatalystAdaptors.h
// 
// 
// 
// A -> forward declare VisRegistryRuntime in Catalyst Adaptors
//   -> will will only allow registry to be member via smartpointer
// B -> (?)move Visitor structs out of the cATALYST Catalyst Adaptor,
//       (might nto work easily)
// 
// 
// 
// ->A
// 
// 
// ############################################




namespace ippl{

    /* FORWARD DECLARATION */
class VisRegistryRuntime;


// namespace CatalystAdaptor {
class CatalystAdaptor {
    std::shared_ptr<ippl::VisRegistryRuntime> visRegistry;
    std::shared_ptr<ippl::VisRegistryRuntime> steerRegistry;

    public:

    using View_vector =
        Kokkos::View<Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    /**
     * @brief Sets electrostatic vector field data in a Conduit node.
     *
     * @param node The Conduit node to populate.
     * @param view The Kokkos vector view containing field data.
     */
    static inline void setData(conduit_cpp::Node& node, const View_vector& view);

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    /**
     * @brief Sets scalar field data in a Conduit node.
     *
     * @param node The Conduit node to populate.
     * @param view The Kokkos scalar view containing field data.
     */
    static inline void setData(conduit_cpp::Node& node, const View_scalar& view);



    /*  sets a file path to a certain node, first tries to fetch from environment, afterwards uses the dafault path passed  */
    /**
     * @brief Sets a file path to a node, using an environment variable if available, otherwise a default path.
     *
     * @param node_path The Conduit node to set the file path in.
     * @param env_var The name of the environment variable to check.
     * @param default_file_path The default file path to use if the environment variable is not set or invalid.
     */
    static void set_node_script(
        conduit_cpp::Node node_path,
        const char* env_var,
        const std::filesystem::path default_file_path
    );


    // ==========================================================
    // CHANNEL INITIALIZERS =====================================
    // ==========================================================



    /* SCALAR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    /**
     * @brief Initializes a Conduit node entry for a scalar field.
     *
     * @tparam T Field value type.
     * @tparam Dim Field dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The scalar field to initialize.
     * @param label The label for the field/channel.
     * @param node The Conduit node to populate.
     * @param source_dir The source directory for script lookup.
     */
    template<typename T, unsigned Dim, class... ViewArgs>
    static void init_entry( 
                    [[maybe_unused]]  
                    const Field<T, Dim, ViewArgs...>& entry
                    , const std::string label
                    ,       conduit_cpp::Node& node
                    , const std::filesystem::path source_dir
                    , const bool png_extracts
        // , ViewRegistry& vr
    );


    /* VECTOR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
        /**
         * @brief Initializes a Conduit node entry for a vector field.
         *
         * @tparam T Vector value type.
         * @tparam Dim Field dimension.
         * @tparam Dim_v Vector dimension.
         * @tparam ViewArgs Additional template arguments for the field.
         * @param entry The vector field to initialize.
         * @param label The label for the field/channel.
         * @param node The Conduit node to populate.
         * @param source_dir The source directory for script lookup.
         */
        template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
        static void init_entry( 
                                        [[maybe_unused]]  
                                            const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                        , const std::string label
                                        ,       conduit_cpp::Node& node
                                        , const std::filesystem::path source_dir
                                        , const bool png_extracts
                                        // , ViewRegistry& vr
        );



    // PARTICLECONTAINERS DERIVED FROM PARTICLEBASE:
    // == ippl::ParticleBaseBase -> ParticleBase<PLayout<T, Dim>, ... , ... >
        /**
         * @brief Initializes a Conduit node entry for a particle container derived from ParticleBaseBase.
         *
         * @tparam T Particle container type (must derive from ippl::ParticleBaseBase).
         * @param entry The particle container to initialize.
         * @param label The label for the container/channel.
         * @param node The Conduit node to populate.
         * @param source_dir The source directory for script lookup.
         */
        template<typename T>
        requires std::derived_from<std::decay_t<T>, ParticleBaseBase>
        static void init_entry( 
                                        [[maybe_unused]]  
                                            const T& entry
                                        , const std::string label
                                        ,       conduit_cpp::Node& node
                                        , const std::filesystem::path source_dir
                                        , const bool png_extracts
                                        // , ViewRegistry& vr
        );


    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
    /**
     * @brief Fallback for init_entry: handles types not derived from ParticleBaseBase.
     *
     * @tparam T Entry type.
     * @param entry The entry to initialize (not a particle container).
     * @param label The label for the entry/channel.
     * @param node The Conduit node to populate.
     * @param source_dir The source directory for script lookup.
     */
    template<typename T>
    requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    static void init_entry(
                [[maybe_unused]]         T&& entry
                ,                  const std::string label
                , [[maybe_unused]]       conduit_cpp::Node& node
                , [[maybe_unused]] const std::filesystem::path source_dir
                , [[maybe_unused]] const bool png_extracts
        // , ViewRegistry& vr
    );


    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
        /**
         * @brief Dispatcher for init_entry: unwraps shared_ptr and dispatches to the appropriate overload.
         *
         * @tparam T Entry type.
         * @param entry Shared pointer to the entry.
         * @param label The label for the entry/channel.
         * @param node The Conduit node to populate.
         * @param source_dir The source directory for script lookup.
         */
        template<typename T>
    static void init_entry( 
            const std::shared_ptr<T>&   entry
        , const std::string           label
        ,       conduit_cpp::Node&    node
        , const std::filesystem::path source_dir
        , const bool                  png_extracts
        // , ViewRegistry& vr 
    );



    // ==========================================================
    // CHANNEL EXECUTIONERS =====================================
    // ==========================================================


    /**
     * @brief Populates a Conduit node with particle container data for Catalyst.
     *
     * @param channelName The name of the channel.
     * @param particleContainer The particle container.
     * @param R_host Host mirror of position attribute.
     * @param P_host Host mirror of velocity attribute.
     * @param q_host Host mirror of charge attribute.
     * @param ID_host Host mirror of ID attribute.
     * @param node The Conduit node to populate.
     */
    static void Execute_Particle(
         const std::string& channelName ,
         const auto& particleContainer
         , const auto& R_host
         , const auto& P_host
         , const auto& q_host
         , const auto& ID_host,
         conduit_cpp::Node& node
        );
    



    /* this needs to be overworked ... */
    /**
    * @brief Populates a Conduit node with field data for Catalyst (3D fields only).
    *
    * @tparam Field Field type (must have dim == 3).
    * @param channelName The name of the channel.
    * @param field Pointer to the field.
    * @param host_view_layout_left Host mirror view for field data.
    * @param node The Conduit node to populate.
    */
    template <class Field>  // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    static void Execute_Field(const std::string& channelName, Field* field, 
        Kokkos::View<typename Field::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>& host_view_layout_left,
        conduit_cpp::Node& node);



        // Handle fields
        // // Map of all Kokkos::Views. This keeps a reference on all Kokkos::Views
        // // which ensures that Kokkos does not free the memory before the end of this function.


        /* SCALAR FIELDS - handles both reference and shared_ptr */
        // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    /**
     * @brief Executes a scalar field entry, populating the Conduit node and updating the view registry.
     *
     * @tparam T Field value type.
     * @tparam Dim Field dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The scalar field to execute.
     * @param label The label for the field/channel.
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     */
    template<typename T, unsigned Dim, class... ViewArgs>
    static void execute_entry(const Field<T, Dim, ViewArgs...>& entry, const std::string label, conduit_cpp::Node& node, ViewRegistry& vr);



        /* VECTOR FIELDS - handles both reference and shared_ptr */
        // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
    /**
     * @brief Executes a vector field entry, populating the Conduit node and updating the view registry.
     *
     * @tparam T Vector value type.
     * @tparam Dim Field dimension.
     * @tparam Dim_v Vector dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The vector field to execute.
     * @param label The label for the field/channel.
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     */
    template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
    static void execute_entry(const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry, const std::string label,conduit_cpp::Node& node, ViewRegistry& vr);


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
    /**
     * @brief Executes a particle container entry (derived from ParticleBaseBase), populating the Conduit node and updating the view registry.
     *
     * @tparam T Particle container type (must derive from ippl::ParticleBaseBase).
     * @param entry The particle container to execute.
     * @param label The label for the container/channel.
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     */
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ParticleBaseBase>
    static void execute_entry(const T& entry
        , const std::string label
        , conduit_cpp::Node& node
        , ViewRegistry& vr);


    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
    /**
     * @brief Fallback for execute_entry: handles types not derived from ParticleBaseBase.
     *
     * @tparam T Entry type.
     * @param label The label for the entry/channel.
     * @param entry The entry to execute (not a particle container).
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     */
    template<typename T>
    requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    static void execute_entry(const std::string label, [[maybe_unused]] T&& entry, conduit_cpp::Node& node, ViewRegistry& vr);


    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
    /**
     * @brief Dispatcher for execute_entry: unwraps shared_ptr and dispatches to the appropriate overload.
     *
     * @tparam T Entry type.
     * @param entry Shared pointer to the entry.
     * @param label The label for the entry/channel.
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     */
    template<typename T>
    static void execute_entry( const std::shared_ptr<T>& entry,const std::string  label, conduit_cpp::Node& node, ViewRegistry& vr );





    /**
     * @brief Adds a steerable channel to the Conduit node for runtime parameter adjustment.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerable_scalar_forwardpass The value to pass forward.
     * @param steerable_suffix Suffix for the steerable channel name.
     * @param node The Conduit node to update.
     */
    template<typename T>
    static void AddSteerableChannel( T steerable_scalar_forwardpass, std::string steerable_suffix, conduit_cpp::Node& node);


    /* maybe use function overloading instead ... */
        // const std::string value_path = ... 
        // steerable_scalar_backwardpass = static_cast<T>(results[value_path].value());
        // steerable_scalar_backwardpass = results[value_path].value()[0];
        /* ????? this should work?? */
        // if constexpr (std::is_same_v<std::remove_cvref_t<T>, double>) {

    /**
     * @brief Fetches the value of a steerable channel from Catalyst results.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerable_scalar_backwardpass Reference to store the fetched value.
     * @param steerable_suffix Suffix for the steerable channel name.
     * @param results The Conduit node containing results.
     */
    template<typename T>
    static void FetchSteerableChannelValue( T& steerable_scalar_backwardpass, std::string steerable_suffix, conduit_cpp::Node& results);
        
    /**
     * @brief Initializes Catalyst with the provided visualization and steering registries.
     *
     * @param registry_vis Visualization registry.
     * @param registry_steer Steering registry.
     */
    static void Initialize([[maybe_unused]] auto& registry_vis, [[maybe_unused]] auto& registry_steer);




    /**
     * @brief Retrieves results from Catalyst and populates the given Conduit node.
     *
     * @param results The Conduit node to populate with results.
     */
    static void Results(conduit_cpp::Node& results);


/* might not even need references to registries since a copy of s shared pointer still points to the 
right place... */
    /**
     * @brief Executes Catalyst with the current visualization and steering registries.
     *
     * @param registry_vis Visualization registry.
     * @param registry_steer Steering registry.
     * @param cycle Current simulation cycle.
     * @param time Current simulation time.
     * @param rank Domain/process rank.
     */
    static void Execute(
            auto& registry_vis, auto& registry_steer,
            int cycle, double time, int rank
        );



    /**
     * @brief Finalizes Catalyst and releases resources.
     */
    static void Finalize();

// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry)
// =====================================================================================


    // Runtime (non-templated) API additions -------------------------------------------------
    // Initialize Catalyst using a runtime registry (vis + steer)
    void InitializeRuntime(
                        //    VisRegistryRuntime& visReg,
                        //    VisRegistryRuntime& steerReg,
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg,
                           const std::filesystem::path& source_dir = {});

    // Execute Catalyst for a given timestep using runtime registry.
    // Populates forward steerable values and fetches back updated ones.
    void ExecuteRuntime(
                        // VisRegistryRuntime& visReg,
                        // VisRegistryRuntime& steerReg,
                        int cycle, double time, 
                        int rank = ippl::Comm->rank());


    // Base Adaptor
    struct InitVisitor;
    struct ExecuteVisitor;
    struct SteerForwardVisitor;
    struct SteerFetchVisitor;




};//class CatalystAdaptor
} //namespace ippl
  
#include "Stream/InSitu/CatalystVisitors.h"
  // runtime non-templated flexible (slow?) registry
#include "Stream/Registry/VisRegistryRuntime.h"     // visitor structs
#include "CatalystAdaptor.hpp"



#endif


// # protocol: initialize
// # Currently, the initialize protocol defines how to load the catalyst implementation library and how to pass scripts to load for analysis.

// # catalyst_load/*: (optional) Catalyst will attempt to use the metadata under this node to find the implementation name and location. If it is missing the environmental variables CATALYST_IMPLEMENTATION_NAME and CATALYST_IMPLEMENTATION_PATH will be queried. See the catalyst_initialize documentation of the Catalyst API.

// # catalyst/scripts: (optional) if present must either be a list or object node with child nodes that provides paths to the Python scripts to load for in situ analysis.

// # catalyst/scripts/[name]: (optional) if present can be a string or object. If it is a string it is interpreted as path to the Python script. If it is an object it can have the following attributes.

// # catalyst/scripts/[name]/filename: path to the Python script

// # catalyst/scripts/[name]/args: (optional) if present must be of type list with each child node of type string. To retrieve these arguments from the script itself use the get_args() method of the paraview catalyst module.





// The rule is simple: Values defined programmatically in your C++ Conduit node will always overwrite values loaded from the JSON config file.

// Here's the sequence of operations Catalyst performs during catalyst_initialize:

// Loads JSON: If catalyst/config_file is present in the node you pass from C++, Catalyst loads that JSON file's contents into an internal Conduit node.

// Merges C++ Node: Catalyst then merges the node you passed from your C++ code on top of the one it just loaded from the file.

// Result:

// If a parameter exists in both the JSON file and your C++ node, the value from the C++ node is used.

// If a parameter exists only in the JSON file, its value is kept.

// If a parameter exists only in your C++ node, it is added to the final configuration.





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







    
            // catalyst blueprint definition
            // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
            //
            // conduit blueprint definition (v.8.3)
            // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html




/* Doesn't work ... */
/* SFINAEEEEE */

// // Detection idiom for addAttribute
// template<typename, typename = void>
// struct has_addAttribute : std::false_type {};

// template<typename T>
// struct has_addAttribute<T, std::void_t<decltype(&T::addAttribute)>> : std::true_type {};

// std::enable_if_t<!has_addAttribute<EntryT>::value,void>  execute_entry([[maybe_unused]] EntryT&& entry) {
// or
// template<typename EntryT, std::enable_if_t<!has_addAttribute<EntryT>::value, int> = 0>
// void execute_entry([[maybe_unused]] EntryT&& entry) {
    //    std::cout << "AA  Entry type can't be processed: " << typeid(std::decay_t<EntryT>).name() << std::endl;   


            
//     // using Base = ippl::ParticleBase<typename T::Layout_t>;
//     using Layout = typename T::Layout_t;
//     using value_type = typename Layout::vector_type::value_type;





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

        // std::cout << "All declared IDs:\n";
        // auto all_ids = registry_vis.getAllIds();
        // std::cout << "   ";
        // for (const auto& id : all_ids) {
        //     std::cout << "\"" << id << "\" ";
        // }
        // std::cout << "\n\n";
