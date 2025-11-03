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

#if defined(MPI_VERSION)
#include <mpi.h>
#endif

#include "Utility/IpplException.h"

#include "Stream/Registry/ViewRegistry.h"

#include "Stream/InSitu/ProxyWriter.h"
#include "Stream/Registry/RegistryHelper.h"

// #include "Stream/InSitu/VisBaseAdaptor.h"


/* catalyst header defined the following for free use ... */
//   CATALYST_EXPORT enum catalyst_status catalyst_initialize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_finalize(const conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_about(conduit_node* params);
//   CATALYST_EXPORT enum catalyst_status catalyst_results(conduit_node* params);


// ############################################
// Possible TODO:
// 
// DONE:
// - figure out why the field and particles work differetnly adapt vis frame for particles
// - iterate over all attributes for visualisation
// - reduce virtual function calls get rid of execute_FIeld set_data and execute particle this
// - added attribute names to ParticleAttribBase
// - improve avoidance of ghost duplicates 
// - remember functionality to allow visualisation for potetnial and density at the same time ..
// - full versatile steering ...
// 
// NEXT: 
// - test 2D
// - use same topology and mesh for all fields?
// - at least avoid regenerating the same ghost data by using unordered map ....or similar
// - use same topology and mesh for all steerable channels??
// - inquire about improced versatile sttering
// 
// 
// MAYBE:
// - keep node structure and only reset data ... 
//   but more than initialially thought needs to eb reset every iteration anew...
// - move exece_entry purely to the visitor structure???
// - test no copy visualisation
// 
//
// UNLIKELY:
// 
// 
// - ability to initialize new objects or reinitialize 
//   with completely new set of objects
//
// 
// ############################################


/* are passing const references through functions faster than having 
member variables ... maybe not ...*/

namespace ippl{

    /* FORWARD DECLARATION */
class VisRegistryRuntime;


    struct Button {
        Button() = default;
        
        // explicit not explicit allows: if(my_btn) to work  
        operator bool() const { return value; }
        
        // explicit Button(bool v) : value(v) {}
        explicit Button(bool v){

            if(v){ // Button is initialized pressd
                value = true;
                prior_state=false;
            }
            else /* if(!v) */{ // Button is initialized  unpressd
                value=false;
                prior_state = false;
            }

        }


        // Assignment operator: Button = bool
        Button& operator=(bool v) {
            if(v) { // Button is being pressed
                if(!value && !prior_state) { // True unpressed state
                    value = true;
                    prior_state = false;  // Fixed typo: was "priot_state"
                }
                else if(value && !prior_state) { // Button was pressed last iteration and is still "being pushed down"
                    // Internal button needs to snap back! And mark it as "freshly snapped"
                    value = false;
                    prior_state = true;
                }
                else if(!value && prior_state) { // Button was pressed at some previous iteration and is still "being pushed down"
                    // Button needs to stay in freshly snapped back state
                    value = false;
                    prior_state = true;
                }
                else if(value && prior_state) { // Impossible state
                    throw IpplException("CatalystAdaptor::Button Assignment", "Impossible State: Button is malfunctioning!!!");
                }
            }
            else { // Button is unpressed
                value = false;
                prior_state = false;
            }
            return *this;  // CRITICAL: Return reference to this object for chaining
        }

        // Friend function to overload << operator for output streams
        friend std::ostream& operator<<(std::ostream& os, const Button& btn) {
            os << (btn.value ? "PUSHED" : "not PUSHED");
            return os;
        }

        private: 
            bool value = false;
            bool prior_state = false;
    }; 


using HostMaskView1D_t = Kokkos::View<unsigned char*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
// 1. Define an alias for your key type for cleanliness
using GhostKey_t = std::tuple<const void*, const void*, size_t>;

// 2. Define the custom hash function struct
struct GhostKeyHash {
    std::size_t operator()(const GhostKey_t& k) const {
        // Get the hash for each element in the tuple
        auto h1 = std::hash<const void*>{}(std::get<0>(k));
        auto h2 = std::hash<const void*>{}(std::get<1>(k));
        auto h3 = std::hash<size_t>{}(std::get<2>(k));

        // Combine the hashes. This is a common pattern (based on boost::hash_combine)
        // It xors and bit-shifts to mix the bits well.
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

// namespace CatalystAdaptor {
class CatalystAdaptor {
    std::shared_ptr<ippl::VisRegistryRuntime> visRegistry;
    std::shared_ptr<ippl::VisRegistryRuntime> steerRegistry;
    
    ViewRegistry viewRegistry;
    // conduit_cpp::Node node_init;
    conduit_cpp::Node node;
    conduit_cpp::Node results;
    public:
    Inform ca_m;
    Inform ca_warn;
    private:

    ProxyWriter proxyWriter;
    // Optional enum metadata: label -> list of (text,value) choices
    std::unordered_map<std::string, std::vector<std::pair<std::string,int>>> enumChoices_;


    // conduit_cpp::Node node_forward;
    // conduit_cpp::Node node_backward;



    /* taken from environemnt can be const... */
    
    
    const char* catalyst_vis  ;
    const char* catalyst_steer;
    const char* catalyst_png  ;
    const char* catalyst_vtk  ;
    const char* ghost_mask  ;
    const char* proxy_option;
    const char* associate_;
    std::string associate;

    const bool vis_enabled;
    const bool steer_enabled;
    const bool png_extracts ;
    const bool vtk_extracts ;
    const bool use_ghost_masks;

    const std::filesystem::path source_dir;

    std::unordered_map<std::string, bool> forceHostCopy;    
    
    // std::unordered_map<std::tuple<const void*, const void*,const size_t>, HostMaskView1D_t> ghostMaskCache;
    std::unordered_map<GhostKey_t, HostMaskView1D_t, GhostKeyHash> ghostMaskCache;

    public:

    CatalystAdaptor() : CatalystAdaptor(ippl::Info->getOutputLevel()){}

    CatalystAdaptor(int outputLevel_) : 
    // CatalystAdaptor() : 
                ca_m("CatalystAdaptor::", 0),  // Only print on rank 0
                ca_warn("CatalystAdaptor_WARNING", std::cerr, INFORM_ALL_NODES), 
                catalyst_vis(std::getenv("IPPL_CATALYST_VIS")),
                catalyst_steer(std::getenv("IPPL_CATALYST_STEER")),
                catalyst_png(std::getenv("IPPL_CATALYST_PNG")),
                catalyst_vtk(std::getenv("IPPL_CATALYST_VTK")),
                ghost_mask(std::getenv("IPPL_CATALYST_GHOST_MASKS")),
                proxy_option(std::getenv("IPPL_CATALYST_PROXY_OPTION")),
                associate_(std::getenv("IPPL_CATALYST_ASSOCIATE")),
                vis_enabled(catalyst_vis && std::string(catalyst_vis) == "ON"),
                steer_enabled(catalyst_steer && std::string(catalyst_steer) == "ON"),
                png_extracts(catalyst_png && std::string(catalyst_png) == "ON"),
                vtk_extracts(catalyst_vtk && std::string(catalyst_vtk) == "ON"),
                use_ghost_masks(ghost_mask && std::string(ghost_mask) == "ON"),
                source_dir(std::filesystem::path(CATALYST_ADAPTOR_ABS_DIR) / "Stream" / "InSitu")
    {
        
        
        associate = std::string(associate_);

        // if(!associate) associate = "element";
        if(! (associate==std::string("vertex"))   )   associate="element";

        ca_m << associate << endl;


        ca_m.setOutputLevel(outputLevel_);
        
        #if defined(MPI_VERSION)
        MPI_Barrier(MPI_COMM_WORLD);
        if(ippl::Comm->rank()==0) ca_warn << "[rank = 0 size=" << ippl::Comm->size() << "]" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        if(ippl::Comm->rank()==1) ca_warn << "[rank= 1 size=" << ippl::Comm->size() << "]" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        #endif


        // ca_warn << "[rank=" << ippl::Comm->rank() << " size=" << ippl::Comm->size() << "]" << endl;
        // ca_warn.setOutputLevel(5);
        // ca_m.setMessageLevel(2);
        // ca_warn.setMessageLevel(5);


        /* 
            Default Message and Output Level are set to global Message and Output Level.

            When Inform Output level are set to e.g 3, messages from this inform with
            level bigger than 3 {4,5} are no longer printed because MessageLevel>OutPutLevel.

            Message Level will always be 1 and Output according to setting...


            so if the output level is fixed e.g via global output level and not changed,
            I can give my message a low level eg 2 so my message will be printed for most verboity levels
            globalOutputLevel >=  informMessageLevel 2-5 and only not printed for very low verbosity levels 
            0,1 = globalOutputLevel < informMessageLevel = 2 wont likely be pri
            But Message Level is reset to minimum after message has been sent, so this is useless for me atm
            why ?? why no alternatve
            and why is [2] in print statment -_-

            so i need to manually level every message which, then let the user overwrite with setOutpt
            if he wants to overwrite the global verbosits option for visualisation.

            Currently Message Level are forced to 1 so any output level other than 0 will print everything ...
        */

        ca_m << "::CatalystAdaptor()   Global        Output  Level setting: " << ippl::Info->getOutputLevel() << endl;
        ca_m << "::CatalystAdaptor()   Catalyst Info Output  Level setting: " << ca_m.getOutputLevel() << endl;
        ca_m << "::CatalystAdaptor()   Catalyst Warn Output  Level setting: " << ca_warn.getOutputLevel() << endl;
        ca_m << "::CatalystAdaptor()   using source_dir = " << source_dir.string() << endl;
        if  (png_extracts) 
            { ca_m << "::CatalystAdaptor()   PNG extraction ACTIVATED"   << endl;} 
        else{ ca_m << "::CatalystAdaptor()   PNG extraction DEACTIVATED" << endl;}
        if  (vtk_extracts) 
            { ca_m << "::CatalystAdaptor()   VTK extraction ACTIVATED"   << endl;}
        else{ ca_m << "::CatalystAdaptor()   VTK extraction DEACTIVATED" << endl;}
        if  (steer_enabled) 
            { ca_m << "::CatalystAdaptor()   Steering       ACTIVATED"   << endl;}
        else{ ca_m << "::CatalystAdaptor()   Steering       DEACTIVATED" << endl;}

    }


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
        // const char* env_var,
        const std::string env_var,
        const std::filesystem::path default_file_path
    );



    bool replace_in_file(const std::string& input_filename,
                     const std::string& output_filename,
                     const std::string& search_string,
                     const std::string& replace_string);

    bool create_new_proxy_file(const std::string & label);



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
               
        /* VECTOR FIELDS - handles both reference and shared_ptr */
     
        /* SCALAR FIELDS - handles both reference and shared_ptr */
        // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
        // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
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
    
     * @brief Executes a scalar field entry, populating the Conduit node and updating the view registry.
     *
     * @tparam T Field value type.
     * @tparam Dim Field dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The scalar field to execute.
     * @param label The label for the field/channel.
     * @param node The Conduit node to populate.
     * @param vr The view registry to update.
     *
     * @brief Populates a Conduit node with field data for Catalyst (3D fields only).
     *
     * @tparam Field Field type (must have dim == 3).
     * @param channelName The name of the channel.
     * @param field Pointer to the field.
     * @param host_view_layout_left Host mirror view for field data.
     * @param node The Conduit node to populate.
    */
    template<typename T, unsigned Dim, class... ViewArgs>
    void execute_entry(    
                        const Field<T, Dim, ViewArgs...>& entry
                     ,  const std::string label
    );



    /* instead of maps storing kokkos view in scope we use the registry to keep everything in frame .... and be totally type indepedent
    we can set with name (but since we likely will not have the need to ever retrieve we can just stire nameless
    to redzcede unncessary computin type ...) */




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


    template<typename T>
    requires (!std::is_enum_v<std::decay_t<T>>)
    void InitSteerableChannel( const T& steerable_scalar_forwardpass,  const std::string& label );

    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void InitSteerableChannel( const E& e, const std::string& label );

    // Bool-like switch overload
    void InitSteerableChannel( const bool& sw, const std::string& label );

    // Vector overloads for steerable channels
    template<typename T, unsigned Dim_v>
    void InitSteerableChannel( const ippl::Vector<T, Dim_v>& steerable_vec_forwardpass, const std::string& label );



    /**
     * @brief Adds a steerable channel to the Conduit node for runtime parameter adjustment.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerable_scalar_forwardpass The value to pass forward.
     * @param steerable_suffix Suffix for the steerable channel name.
     * @param node The Conduit node to update.
     */
    template<typename T>
    requires (!std::is_enum_v<std::decay_t<T>>)
    void AddSteerableChannel(const T& steerable_scalar_forwardpass,  const std::string& steerable_suffix);

    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void AddSteerableChannel(const E& e, const std::string& steerable_suffix);

    // Bool-like switch overload
    void AddSteerableChannel(const bool& sw, const std::string& steerable_suffix);
    // Button-like push overloads
    void InitSteerableChannel( const ippl::Button& btn, const std::string& label );
    void AddSteerableChannel(const ippl::Button& btn, const std::string& steerable_suffix);

    // Vector overloads for steerable channels
    template<typename T, unsigned Dim_v>
    void AddSteerableChannel(const ippl::Vector<T, Dim_v>& steerable_vec_forwardpass, const std::string& steerable_suffix);


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
    requires (!std::is_enum_v<std::decay_t<T>>)
    void FetchSteerableChannelValue( T& steerable_scalar_backwardpass, const std::string& steerable_suffix);
    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void FetchSteerableChannelValue( E& e, const std::string& steerable_suffix);
    
    // Vector overloads for steerable channels
    template<typename T, unsigned Dim_v>
    void FetchSteerableChannelValue( ippl::Vector<T, Dim_v>& steerable_vec_backwardpass, const std::string& steerable_suffix);
        

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

    // Provide enum choices metadata before InitializeRuntime, so the GUI shows a dropdown.
    // Example: RegisterEnumChoices("mode", {{"Off",0},{"Basic",1},{"Advanced",2}});
    void RegisterEnumChoices(const std::string& label,
                             const std::vector<std::pair<std::string,int>>& entries) {
        enumChoices_[label] = entries;
    }

    void Remember_now(const std::string);




    // Execute Catalyst for a given timestep using runtime registry.
    // Populates forward steerable values and fetches back updated ones.
    void ExecuteRuntime(
                        int cycle, double time, 
                        int rank = ippl::Comm->rank()
    );


    // put into Base Adaptor
    struct InitVisitor;
    struct ExecuteVisitor;
    struct SteerInitVisitor;
    struct SteerForwardVisitor;
    struct SteerFetchVisitor;




};//class CatalystAdaptor
} //namespace ippl
  
#include "Stream/InSitu/CatalystVisitors.h"
  // runtime non-templated flexible (slow?) registry
#include "Stream/Registry/VisRegistryRuntime.h"     // visitor structs
#include "CatalystAdaptor.hpp"



#endif
