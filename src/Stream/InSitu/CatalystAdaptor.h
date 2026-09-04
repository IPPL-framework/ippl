/**
 * @file CatalystAdaptor.h
 * @brief Declarations and lightweight types for ParaView Catalyst in-situ integration.
 *
 * This header provides helper types, forward declarations, and includes used by the
 * Catalyst adaptor. Heavy implementation details live in `CatalystAdaptor.hpp` and 
 * `CatalystAdaptorSteering.hpp`.
 */
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <catalyst.hpp>
#include <catalyst_conduit.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <utility>
#include <type_traits>
#include <list>
#include <filesystem>
#include <typeindex>

#if defined(MPI_VERSION)
#include <mpi.h>
#endif

#include "Utility/IpplException.h"

#include "Stream/Registry/ViewRegistry.h"

#include "Stream/InSitu/ProxyWriter.h"
#include "Stream/Registry/RegistryHelper.h"


namespace ippl{

/* FORWARD DECLARATION */
class VisRegistryRuntime;



/**
 * @struct Button
 * @brief Momentary-action control for steering.
 *
 * A Button acts like an edge-trigger: it reports a single transition when
 * pressed and then resets. Useful for one-shot actions triggered from the
 * Catalyst GUI without latching state.
 */
struct Button {
        Button() = default;
        
        // explicit not explicit allows: if(my_btn) to work  
        operator bool() const { return value_m; }
        
        // explicit Button(bool v) : value_m(v) {}
        explicit Button(bool v){

            if(v){ // Button is initialized pressd
                value_m = true;
                priorState_m=false;
            }
            else /* if(!v) */{ // Button is initialized  unpressd
                value_m=false;
                priorState_m = false;
            }

        }


        // Assignment operator: Button = bool
        Button& operator=(bool v) {
            if(v) { // Button is being pressed
                if(!value_m && !priorState_m) { // True unpressed state
                    value_m = true;
                    priorState_m = false;  // Fixed typo: was "priot_state"
                }
                else if(value_m && !priorState_m) { // Button was pressed last iteration and is still "being pushed down"
                    // Internal button needs to snap back! And mark it as "freshly snapped"
                    value_m = false;
                    priorState_m = true;
                }
                else if(!value_m && priorState_m) { // Button was pressed at some previous iteration and is still "being pushed down"
                    // Button needs to stay in freshly snapped back state
                    value_m = false;
                    priorState_m = true;
                }
                else if(value_m && priorState_m) { // Impossible state
                    throw IpplException("CatalystAdaptor::Button Assignment", "Impossible State: Button is malfunctioning!!!");
                }
            }
            else { // Button is unpressed
                value_m = false;
                priorState_m = false;
            }
            return *this;  // CRITICAL: Return reference to this object for chaining
        }

        // Friend function to overload << operator for output streams
        friend std::ostream& operator<<(std::ostream& os, const Button& btn) {
            os << (btn.value_m ? "PUSHED" : "not PUSHED");
            return os;
        }

        private: 
            bool value_m = false;
            bool priorState_m = false;
};// Button


/**
 * @brief Host-side 1D mask view (0/1) used for ghost/halo flags.
 */
using HostMaskView1D_t = Kokkos::View<unsigned char*, Kokkos::LayoutLeft, Kokkos::HostSpace>;


/**
 * @brief Cache key for ghosted data.
 *
 * Tuple components:
 *  - pointer identifying the mesh/topology
 *  - pointer identifying the owning view/container
 *  - size/extent of the ghost region
 */
using GhostKey_t = std::tuple<const void*, const void*, size_t>;


/**
 * @brief Hash functor for GhostKey_t used in unordered caches.
 */
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


/**
 * @class CatalystAdaptor
 * @brief High-level orchestrator for Catalyst initialization, execution, and steering.
 *
 * The adaptor wires up registered fields/particles and steerables to Conduit `node_m` instances,
 * invokes Catalyst scripts, and forwards/fetches steering values between the
 * simulation and the GUI. Use the runtime registry API (Initialize/Execute)
 * for flexible, non-templated integration.
 */
class CatalystAdaptor {
public:
    struct InitVisitor;
    struct ExecVisitor;
    struct SteerInitVisitor;
    struct SteerForwardVisitor;
    struct SteerFetchVisitor;

    using VisVisitorVariant_t = std::variant<InitVisitor*, ExecVisitor*>;
    using SteerVisitorVariant_t = std::variant<SteerInitVisitor*, SteerForwardVisitor*, SteerFetchVisitor*>;

private:
    std::shared_ptr<ippl::VisRegistryRuntime> visRegistry_m;
    std::shared_ptr<ippl::VisRegistryRuntime> steerRegistry_m;
    
    ViewRegistry viewRegistry_m;
    conduit_cpp::Node node_m;
    conduit_cpp::Node results_m;
    public:
    Inform catalystInfo_m;
    Inform catalystWarn_m;
    private:

    ProxyWriter proxyWriter_m;
    std::unordered_map<std::type_index, std::vector<std::pair<std::string,int>>> enumChoicesByType_m;
    std::unordered_map<std::string, std::vector<std::pair<std::string,int>>> enumChoices_m;

    
    const char* catalystVis_m  ;
    const char* catalystLive_m  ;
    const char* catalystSteer_m;
    const char* catalystPng_m  ;
    const char* catalystVtk_m  ;
    const char* catalystVerbosity_m;
    const char* catalystGhostMask_m  ;
    const char* proxyOption_m;
    
    const bool visEnabled_m;
    const bool liveEnabled_m;
    const bool steerEnabled_m;
    const bool pngExtracts_m ;
    const bool vtkExtracts_m ;
    const int  outputLevel_m  ; 
    const bool useGhostMasks_m;
    
    // std::string associate_m;
    const std::filesystem::path sourceDir_m;

    std::unordered_map<std::string, bool> forceHostCopy_m;    
    
    std::unordered_map<GhostKey_t, HostMaskView1D_t, GhostKeyHash> ghostMaskCache_m;

public:


    

    CatalystAdaptor() : 
                catalystInfo_m("CatalystAdaptor::", 0),  // Only print on rank 0
                catalystWarn_m("CatalystAdaptor_WARNING", std::cerr, INFORM_ALL_NODES), 
                catalystVis_m(std::getenv("IPPL_CATALYST_VIS")),
                catalystLive_m(std::getenv("IPPL_CATALYST_LIVE")),
                catalystSteer_m(std::getenv("IPPL_CATALYST_STEER")),
                catalystPng_m(std::getenv("IPPL_CATALYST_PNG")),
                catalystVtk_m(std::getenv("IPPL_CATALYST_VTK")),
                catalystVerbosity_m(std::getenv("IPPL_CATALYST_VERBOSITY")),
                catalystGhostMask_m(std::getenv("IPPL_CATALYST_GHOST_MASKS")),
                proxyOption_m(std::getenv("IPPL_CATALYST_PROXY_OPTION")),
                visEnabled_m( ! (catalystVis_m        && std::string(catalystVis_m)    == "OFF") ),
                liveEnabled_m(   catalystLive_m       && std::string(catalystLive_m)   == "ON"),
                steerEnabled_m(  catalystSteer_m      && std::string(catalystSteer_m)  == "ON"),
                pngExtracts_m(   catalystPng_m        && std::string(catalystPng_m)    == "ON"),
                vtkExtracts_m(   catalystVtk_m        && std::string(catalystVtk_m)    == "ON"),
                outputLevel_m(   catalystVerbosity_m  ? std::stoi(catalystVerbosity_m) : ippl::Info->getOutputLevel()),
                useGhostMasks_m(catalystGhostMask_m && std::string(catalystGhostMask_m) == "ON"),
                sourceDir_m(std::filesystem::path(CATALYST_ADAPTOR_ABS_DIR) / "Stream" / "InSitu")
    {
        // associate_m="element";

        if(!liveEnabled_m){
            catalystLive_m ="OFF";
        }
        
        catalystInfo_m.setOutputLevel(outputLevel_m);
        
        // #if defined(MPI_VERSION)
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(ippl::Comm->rank()==0) catalystWarn_m << "[rank = 0 size=" << ippl::Comm->size() << "]" << endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(ippl::Comm->rank()==1) catalystWarn_m << "[rank= 1 size=" << ippl::Comm->size() << "]" << endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        // #endif

        catalystInfo_m << "::CatalystAdaptor()   Global        Output  Level setting: " << ippl::Info->getOutputLevel() << endl;
        catalystInfo_m << "::CatalystAdaptor()   Catalyst Info Output  Level setting: " << catalystInfo_m.getOutputLevel() << endl;
        catalystInfo_m << "::CatalystAdaptor()   Catalyst Warn Output  Level setting: " << catalystWarn_m.getOutputLevel() << endl;
        catalystInfo_m << "::CatalystAdaptor()   using sourceDir_m = " << sourceDir_m.string() << endl;
        if  (pngExtracts_m) 
            { catalystInfo_m << "::CatalystAdaptor()   PNG extraction ACTIVATED"   << endl;} 
        else{ catalystInfo_m << "::CatalystAdaptor()   PNG extraction DEACTIVATED" << endl;}
        if  (vtkExtracts_m) 
            { catalystInfo_m << "::CatalystAdaptor()   VTK extraction ACTIVATED"   << endl;}
        else{ catalystInfo_m << "::CatalystAdaptor()   VTK extraction DEACTIVATED" << endl;}
        if  (steerEnabled_m) 
            { catalystInfo_m << "::CatalystAdaptor()   Steering       ACTIVATED"   << endl;}
        else{ catalystInfo_m << "::CatalystAdaptor()   Steering       DEACTIVATED" << endl;}

    }

    private:

    // ==============================================================================================
    //  HELPERS =====================================================================================
    // ==============================================================================================
    /**
     * @brief Sets a file path to a node, using an environment variable if available, otherwise a default path.
     *
     * @param nodePath The Conduit `node_m` instance to set the file path in.
     * @param envVar The name of the environment variable to check.
     * @param defaultFilePath The default file path to use if the environment variable is not set or invalid.
     */
    void setNodeScript(
        conduit_cpp::Node nodePath,
        // const char* envVar,
        const std::string envVar,
        const std::filesystem::path defaultFilePath
    );


    // ==========================================================
    // VISUALIZATION CHANNEL INITIALIZERS =======================
    // ==========================================================



    /* SCALAR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>*
    /**
     * @brief Initializes a Conduit `node_m` instance entry for a scalar field.
     *
     * @tparam T Field value type.
     * @tparam Dim Field dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The scalar field to initialize.
     * @param label The label for the field/channel.
     */
    template<typename T, unsigned Dim, class... ViewArgs>
    void InitVizChannel( 
                    [[maybe_unused]]  
                    const Field<T, Dim, ViewArgs...>& entry
                    , const std::string label
    );


    /* VECTOR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>*
    /**
     * @brief Initializes a Conduit `node_m` instance entry for a vector field.
     *
     * @tparam T Vector value type.
     * @tparam Dim Field dimension.
     * @tparam Dim_v Vector dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The vector field to initialize.
     * @param label The label for the field/channel.
     */
    template<typename T, unsigned Dim, unsigned Dim_v, class... ViewArgs>
    void InitVizChannel( 
                                      [[maybe_unused]]  
                                      const Field<Vector<T, Dim_v>, Dim, ViewArgs...>& entry
                                    , const std::string label
    );



    // PARTICLECONTAINERS DERIVED FROM PARTICLEBASE:
    // == ippl::ParticleBaseBase -> ParticleBase<PLayout<T, Dim>, ... , ... >
    /**
     * @brief Initializes a Conduit `node_m` instance entry for a particle container derived from ParticleBaseBase.
     *
     * @tparam T Particle container type (must derive from ippl::ParticleBaseBase).
     * @param entry The particle container to initialize.
     * @param label The label for the container/channel.
     */
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ParticleBaseBase>
    void InitVizChannel( 
                      [[maybe_unused]]  
                      const T& entry
                    , const std::string label
    );


    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
    /**
     * @brief Dispatcher for InitVizChannel: unwraps shared_ptr and dispatches to the appropriate overload.
     *
     * @tparam T Entry type.
     * @param entry Shared pointer to the entry.
     * @param label The label for the entry/channel.
     */
    template<typename T>
    void InitVizChannel( 
                  const std::shared_ptr<T>&   entry
                , const std::string           label
    );



    // BASE CASE: 
    // only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
    /**
     * @brief Fallback for InitVizChannel: handles types not derived from ParticleBaseBase 
     * and not having specific overloads. Should never be called since these types are already 
     * filtered out inside a Visitor struct with AllowedVisType_v.
     *
     * @tparam T Entry type.
     * @param entry The entry to initialize (not a particle container).
     * @param label The label for the entry/channel.
     */
    template<typename T>
    requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    void InitVizChannel(
                    [[maybe_unused]]
                    // T&& entry
                    const T& entry
                ,   const std::string label
    );



    // ==========================================================
    // VISUALIZATION CHANNEL EXECUTIONERS =======================
    // ==========================================================
               
    
    /* SCALAR FIELDS - handles both reference and shared_ptr */
    /* VECTOR FIELDS - handles both reference and shared_ptr */
    // == ippl::Field<double, 3, ippl::UniformCartesian<double, 3>, Cell>
    // == ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>, Cell>
    /**
     * @brief Executes a scalar/vector field entry, populating the Conduit `node_m` instance and updating the view registry.
     *
     * @tparam T Field value type.
     * @tparam Dim Field dimension.
     * @tparam ViewArgs Additional template arguments for the field.
     * @param entry The scalar/vector field to execute.
     * @param label The label for the field/channel.
     *
    */
    template<typename T, unsigned Dim, class... ViewArgs>
    void ExecVizChannel(    
                        const Field<T, Dim, ViewArgs...>& entry
                     ,  const std::string label
    );


    // PARTICLECONTAINERS DERIVED FROM PARTICLEBASE:
    /**
     * @brief Executes a particle container entry (derived from ParticleBaseBase), populating the Conduit `node_m` instance and updating the view registry.
     *
     * @tparam T Particle container type (must derive from ippl::ParticleBaseBase).
     * @param entry The particle container to execute.
     * @param label The label for the container/channel.
     */
    template<typename T>
    requires (std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    void ExecVizChannel(
                    const T& entry
                  , const std::string label
    );


    // BASE CASE: only enabled if EntryT is NOT derived from ippl::ParticleBaseBase
    /**
     * @brief Fallback for ExecVizChannel: handles types not derived from ParticleBaseBase.
     *
     * @tparam T Entry type.
     * @param label The label for the entry/channel.
     * @param entry The entry to execute (not a particle container).
     */
    template<typename T>
    requires (!std::derived_from<std::decay_t<T>, ParticleBaseBase>)
    void ExecVizChannel(
                [[maybe_unused]] T&& entry
                , const std::string label
    );


    /* SHARED_PTR DISPATCHER - automatically unwraps and dispatches to appropriate overload */
    /**
     * @brief Dispatcher for ExecVizChannel: unwraps shared_ptr and dispatches to the appropriate overload.
     *
     * @tparam T Entry type.
     * @param entry Shared pointer to the entry.
     * @param label The label for the entry/channel.
     */
    template<typename T>
    void ExecVizChannel(  
                        const std::shared_ptr<T>& entry
                      , const std::string  label
    );



    // ==========================================================
    // STEERING CHANNEL INITIALIZERS===== =======================
    // ==========================================================

    /**
     * @brief Initializes a steerable channel in the Conduit `node_m` instance for runtime parameter adjustment.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerableScalarForwardpass The initial value to set.
     * @param label The label for the steerable channel.
     */
    template<typename T>
    requires (!std::is_enum_v<std::decay_t<T>>)
    void InitSteerChannel( const T& steerableScalarForwardpass,  const std::string& label );

    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void InitSteerChannel( const E& e, const std::string& label );

    // Bool-like switch overload
    void InitSteerChannel( const bool& sw, const std::string& label );

    // Vector overloads for steerable channels
    template<typename T, unsigned Dim_v>
    void InitSteerChannel( const ippl::Vector<T, Dim_v>& steerableVecForwardpass, const std::string& label );


    void InitSteerChannel( const ippl::Button& btn, const std::string& label );

    // Generic std::vector elements (arithmetic/bool/Button) for array steerables
    template<typename Elem>
    requires (std::is_arithmetic_v<std::decay_t<Elem>> || std::is_enum_v<std::decay_t<Elem>> || std::is_same_v<std::decay_t<Elem>, bool> || std::is_same_v<std::decay_t<Elem>, ippl::Button>)
    void InitSteerChannel( const std::vector<Elem>& arr, const std::string& label );

    // std::vector of ippl::Vector<T,Dim> steerables
    template<typename T, unsigned Dim_v>
    void InitSteerChannel( const std::vector<ippl::Vector<T, Dim_v>>& arr, const std::string& label );


    // ==========================================================
    // STEERING CHANNEL EXECUTIONERS=============================
    // ==========================================================

    /**
     * @brief Adds a steerable channel to the Conduit `node_m` instance for runtime parameter adjustment.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerableScalarForwardpass The value to pass forward.
     * @param steerableSuffix Suffix for the steerable channel name.
     */
    template<typename T>
    requires (!std::is_enum_v<std::decay_t<T>>)
    void ForwardSteerChannel(const T& steerableScalarForwardpass,  const std::string& steerableSuffix);

    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void ForwardSteerChannel(const E& e, const std::string& steerableSuffix);

    // Bool-like switch overload
    void ForwardSteerChannel(const bool& sw, const std::string& steerableSuffix);

    // Button-like push overloads
    void ForwardSteerChannel(const ippl::Button& btn, const std::string& steerableSuffix);

    // Vector overloads for steerable channels
    template<typename T, unsigned Dim_v>
    void ForwardSteerChannel(const ippl::Vector<T, Dim_v>& steerableVecForwardpass, const std::string& steerableSuffix);

    // Generic std::vector elements (arithmetic/bool/Button) for array steerables
    template<typename Elem>
    requires (std::is_arithmetic_v<std::decay_t<Elem>> || std::is_enum_v<std::decay_t<Elem>> || std::is_same_v<std::decay_t<Elem>, bool> || std::is_same_v<std::decay_t<Elem>, ippl::Button>)
    void ForwardSteerChannel(const std::vector<Elem>& arr, const std::string& label);

    // std::vector of ippl::Vector<T,Dim> arrays forward
    template<typename T, unsigned Dim_v>
    void ForwardSteerChannel(const std::vector<ippl::Vector<T, Dim_v>>& arr, const std::string& label);



    // ==========================================================
    // STEERING CHANNEL FETCHERS    =============================
    // ==========================================================

    /**
     * @brief Fetches the value of a steerable channel from Catalyst results.
     *
     * @tparam T Type of the steerable parameter.
     * @param steerableScalarBackwardpass Reference to store the fetched value.
     * @param steerableSuffix Suffix for the steerable channel name.
     */
    template<typename T>
    requires (!std::is_enum_v<std::decay_t<T>>)
    void FetchSteerChannel( T& steerableScalarBackwardpass, const std::string& steerableSuffix);

    // Enum overloads (arbitrary enum types)
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void FetchSteerChannel( E& e, const std::string& steerableSuffix);
    
    // ippl Vector overload
    template<typename T, unsigned Dim_v>
    void FetchSteerChannel( ippl::Vector<T, Dim_v>& steerableVecBackwardpass, const std::string& steerableSuffix);

    // standard vector overload for basic types
    template<typename Elem>
    requires (std::is_arithmetic_v<std::decay_t<Elem>> || std::is_enum_v<std::decay_t<Elem>> || std::is_same_v<std::decay_t<Elem>, bool> || std::is_same_v<std::decay_t<Elem>, ippl::Button>)
    void FetchSteerChannel( std::vector<Elem>& out, const std::string& label);

    // std::vector of ippl::Vector<T,Dim> arrays backward
    template<typename T, unsigned Dim_v>
    void FetchSteerChannel( std::vector<ippl::Vector<T, Dim_v>>& out, const std::string& label);
        

    /**
     * @brief Retrieves results from Catalyst and populates the given Conduit `node_m` instance.
     *
     */
    void fetchResults();
    

    // =====================================================================================
    //  CatalystAdaptor Public Methods
    // =====================================================================================
    public:

    /**
     * @brief Struct steering registration.
     *
     * Expose any user struct composed of already supported steerable member
     * types (arithmetic, bool, ippl::Button, ippl::Vector<>, enums). Nested structs are not
     * supported yet. Must be called before adding the struct instance to a
     * runtime registry. Args must be an even-sized pack of: name(string-like), pointer-to-member.
     * Validates each member type at registration (throws IpplException if invalid).
     * Stores three lambdas which expand the pack and delegate to visitor overloads.
     * 
     * @tparam T The struct type to register.
     * @tparam Args Types of the member pointers.
     * @param args Pointers to the struct members.
     */
    template<typename T, typename... Args>
    static void RegisterStructMembers(Args&&... args);



    /* DEPRECATED atm ... */
    // Optional enum metadata: label -> list of (text,value) choices
    // std::unordered_map<std::string, std::vector<std::pair<std::string,int>>> enumChoices_m;
    // /**
    //  * @brief Provides enum choices metadata so the GUI shows a dropdown.
    //  *
    //  * Use this before InitializeRuntime. 
    //  * Example: RegisterEnumChoices("mode", {{"Off",0},{"Basic",1},{"Advanced",2}});
    //  * 
    //  * @param label The label for the enum choice.
    //  * @param entries A vector of pairs containing the display name and integer value.
    //  */
    // void RegisterEnumChoices(const std::string& label,
    //                          const std::vector<std::pair<std::string,int>>& entries) {
    //     enumChoices_m[label] = entries;
    // }

    // /**
    //  * @brief Provides typed enum choices metadata mapped to a specific label for the GUI dropdown.
    //  *
    //  * @tparam E The enumeration type.
    //  * @param label The specific label of the steerable channel to associate the choices with.
    //  * @param entries A vector of pairs containing the display name and the enumeration value.
    //  */
    // template<typename E>
    // requires (std::is_enum_v<std::decay_t<E>>)
    // void RegisterEnumChoicesTyped(const std::string& label, const std::vector<std::pair<std::string, E>>& entries);

    /**
     * @brief Provides typed enum choices metadata registered by the enum type, to be reused across labels.
     *
     * @tparam E The enumeration type.
     * @param entries A vector of pairs containing the display name and the enumeration value.
     */
    template<typename E>
    requires (std::is_enum_v<std::decay_t<E>>)
    void RegisterEnumChoicesTyped(const std::vector<std::pair<std::string, E>>& entries);




    /**
     * @brief Initializes Catalyst using runtime registries (visualization and steering).
     *
     * @param visReg Shared pointer to the visualization runtime registry.
     * @param steerReg Shared pointer to the steering runtime registry.
     */
    void Initialize(
                           const std::shared_ptr<VisRegistryRuntime>& visReg,
                           const std::shared_ptr<VisRegistryRuntime>& steerReg
    );

    /**
     * @brief Explicitly forces a host copy for a specifically labelled channel right now.
     * 
     * @param label The label specifying which field or particle to remember currently.
     */
    void rememberNow(const std::string label);

    /**
     * @brief Executes Catalyst for a given timestep using the runtime registry.
     *
     * Populates forward steerable values and fetches back updated ones.
     * 
     * @param cycle The current simulation cycle or timestep index.
     * @param time The current simulation time.
     * @param rank The MPI rank of the executing process (defaults to ippl::Comm->rank()).
     */
    void Execute(
                        int cycle, double time, 
                        int rank = ippl::Comm->rank()
    );


    /**
     * @brief Finalizes Catalyst and releases resources.
     */
    void Finalize();



};//class CatalystAdaptor
} //namespace ippl
  
#include "Stream/InSitu/CatalystVisitors.h"
#include "Stream/Registry/VisRegistryRuntime.h"     // visitor structs
#include "CatalystAdaptor.hpp"



#endif

