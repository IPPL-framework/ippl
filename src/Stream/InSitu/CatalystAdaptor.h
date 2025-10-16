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
// - keep node structure and only reset data ...
// - remember function
// - figure out why the field and particles work differetnly
// - reduce virtual function calls get rid of execute_FIeld set_data and execute particle this
// - test no copy visualisation
// 
// how do we get attribute identifier
// - enable pure attribute not make sense in location inf is missing...
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


/* are passing const references through functions faster than having 
member variables .. */

namespace ippl{

    /* FORWARD DECLARATION */
class VisRegistryRuntime;


// namespace CatalystAdaptor {
class CatalystAdaptor {
    std::shared_ptr<ippl::VisRegistryRuntime> visRegistry;
    std::shared_ptr<ippl::VisRegistryRuntime> steerRegistry;
    
    ViewRegistry viewRegistry;
    conduit_cpp::Node node;
    conduit_cpp::Node results;


    /* taken from environemnt can be const... */
    std::filesystem::path source_dir;
    bool png_extracts;


    // conduit_cpp::Node node_results;
    
    // conduit_cpp::Node node_forward;
    // conduit_cpp::Node node_backward;


    public:

    using View_vector =
        Kokkos::View<Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;



    /*  sets a file path to a certain node, first tries to fetch from environment, afterwards uses the dafault path passed  */
    /**
     * @brief Sets a file path to a node, using an environment variable if available, otherwise a default path.
     *
     * @param node_path The Conduit node to set the file path in.
     * @param env_var The name of the environment variable to check.
     * @param default_file_path The default file path to use if the environment variable is not set or invalid.
     */
    void set_node_script(
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
    void init_entry( 
                    [[maybe_unused]]  
                    const Field<T, Dim, ViewArgs...>& entry
                    , const std::string label
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
        void init_entry( 
                                          [[maybe_unused]]  
                                          const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                        , const std::string label
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
    void init_entry( 
                      [[maybe_unused]]  
                      const T& entry
                    , const std::string label
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
    void init_entry( 
                  const std::shared_ptr<T>&   entry
                , const std::string           label
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
    void init_entry(
                    [[maybe_unused]]
                    // T&& entry
                    const T& entry
                ,   const std::string label
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
    void Execute_Particle(
         const std::string& channelName ,
         const auto& particleContainer
         , const auto& R_host
         , const auto& P_host
         , const auto& q_host
         , const auto& ID_host
    );
    

    /**
     * @brief Sets electrostatic vector field data in a Conduit node.
     *
     * @param node The Conduit node to populate.
     * @param view The Kokkos vector view containing field data.
     */
    inline void setData(conduit_cpp::Node& field_node, const View_vector& view);

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    /**
     * @brief Sets scalar field data in a Conduit node.
     *
     * @param node The Conduit node to populate.
     * @param view The Kokkos scalar view containing field data.
     */
    inline void setData(conduit_cpp::Node& field_node, const View_scalar& view);


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
    template<typename T, unsigned Dim, class... ViewArgs>
    void Execute_Field(
        const Field<T, Dim, ViewArgs...>& entry, 
        const std::string& label
    );



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
    void execute_entry(    
                        const Field<T, Dim, ViewArgs...>& entry
                     ,  const std::string label
    );



    /* could catch this for overall case distinction
    but this isnt re */
       
// execute visualisation for VECTOR FIELDS  
// // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
//         /* VECTOR FIELDS - handles both reference and shared_ptr */
//         // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
//     /**
//      * @brief Executes a vector field entry, populating the Conduit node and updating the view registry.
//      *
//      * @tparam T Vector value type.
//      * @tparam Dim Field dimension.
//      * @tparam Dim_v Vector dimension.
//      * @tparam ViewArgs Additional template arguments for the field.
//      * @param entry The vector field to execute.
//      * @param label The label for the field/channel.
//      * @param node The Conduit node to populate.
//      * @param vr The view registry to update.
//      */
//     template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
//     void execute_entry(  
//                         const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry 
//                       , const std::string label
//     );


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


    template<typename T>
    void execute_attribute(const ParticleAttrib<T> & pa);

    template<typename T, unsigned Dim_v>
    void execute_attribute(const ParticleAttrib<Vector<T, Dim_v>> & pa);



    // template<typename T, unsigned Dim_v, typename memspace>
    
    // template <typename memspace>
    // template <typename T>
    template<typename T, typename memspace>
    void execute_attribute(const detail::ParticleAttribBase<memspace>& pa);
    



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
    requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    void execute_entry(
                    const T& entry
                  , const std::string label
    );


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
    void execute_entry(
                [[maybe_unused]] T&& entry
                , const std::string label
    );


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
    void execute_entry(  
                        const std::shared_ptr<T>& entry
                      , const std::string  label
    );





    /**
     * @brief Adds a steerable channel to the Conduit node for runtime parameter adjustment.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerable_scalar_forwardpass The value to pass forward.
     * @param steerable_suffix Suffix for the steerable channel name.
     * @param node The Conduit node to update.
     */
    template<typename T>
    void AddSteerableChannel( 
                           T steerable_scalar_forwardpass
                         , std::string steerable_suffix
    );


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
    void FetchSteerableChannelValue( 
                           T& steerable_scalar_backwardpass
                         , std::string steerable_suffix
    );
        

    /**
     * @brief Retrieves results from Catalyst and populates the given Conduit node.
     *
     * @param results The Conduit node to populate with results.
     */
    void fetchResults();




    /**
     * @brief Finalizes Catalyst and releases resources.
     */
    void Finalize();

// =====================================================================================
// Runtime registry based Initialize / Execute (non-templated registry)
// =====================================================================================


    // Runtime (non-templated) API additions -------------------------------------------------
    // Initialize Catalyst using a runtime registry (vis + steer)
    void InitializeRuntime(
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg
    );

    // Execute Catalyst for a given timestep using runtime registry.
    // Populates forward steerable values and fetches back updated ones.
    void ExecuteRuntime(
                        int cycle, double time, 
                        int rank = ippl::Comm->rank()
    );


    // put into Base Adaptor
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
