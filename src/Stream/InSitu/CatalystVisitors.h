/**
 * @file CatalystVisitors.h
 * @brief Visitor functors used by CatalystAdaptor for initialization, execution, and steering.
 */
#pragma once
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>
#include <memory>

#include <Stream/InSitu/CatalystAdaptor.h>
#include <catalyst.hpp>




namespace ippl{

// =========================================
// METAPROGRAMMING HELPER TRAITS
// =========================================

template<class T>
inline constexpr bool is_particle_v = std::is_base_of<ippl::ParticleBaseBase, typename std::decay<T>::type>::value;

template<class T>
struct is_field : std::false_type {};

template<class V, unsigned Dim, class... Rest>
struct is_field<ippl::Field<V, Dim, Rest...>> : std::true_type {};

template<class T>
inline constexpr bool is_field_v = is_field<std::decay_t<T>>::value;



// Detect ippl::Vector<T,Dim>
template<typename U>
struct is_ippl_vector : std::false_type {};
template<typename V, unsigned Dim>
struct is_ippl_vector<ippl::Vector<V, Dim>> : std::true_type {};
template<typename T>
inline constexpr bool is_ippl_vector_v = is_ippl_vector<std::decay_t<T>>::value;

// detect std::vector of supported scalar/button types
template<typename U>
struct is_std_vector_of_allowed_scalar : std::false_type {};
template<typename U>
struct is_std_vector_of_allowed_scalar<std::vector<U>> : std::bool_constant<
    std::is_arithmetic_v<std::decay_t<U>>
    || std::is_enum_v<std::decay_t<U>>
    || std::is_same_v<std::decay_t<U>, bool>
    || std::is_same_v<std::decay_t<U>, Button>> {};

// detect std::vector of ippl::Vector<...>
template<typename U>
struct is_std_vector_of_ippl_vector : std::false_type {};
template<typename U>
struct is_std_vector_of_ippl_vector<std::vector<U>> : std::bool_constant<is_ippl_vector_v<U>> {};




// detect any std::vector<...> (used to route struct arrays via StructMeta)
template<typename U>
struct is_std_vector_any : std::false_type {};
template<typename U>
struct is_std_vector_any<std::vector<U>> : std::true_type {};

template<class T>
inline constexpr bool AllowedSteerType_v =
    std::is_arithmetic_v<std::decay_t<T>>
    || std::is_enum_v<std::decay_t<T>>
    || std::is_same_v<std::decay_t<T>, Button>
    || is_ippl_vector_v<T> // ippl::Vector<>
    || is_std_vector_of_allowed_scalar<std::decay_t<T>>::value
    || is_std_vector_of_ippl_vector<std::decay_t<T>>::value
    ; 


template<class T>
inline constexpr bool AllowedVisType_v = is_particle_v<T> || is_field_v<T>;

template<class T>
inline constexpr bool AllowedRegistryType_v =      AllowedVisType_v<T> 
                                                || AllowedSteerType_v<T> 
                                                || is_std_vector_any<std::decay_t<T>>::value
                                                ;
    

// Accept shared_ptr<U> when U is allowed
template<class T>
struct is_allowed_shared_ptr : std::false_type {};
template<class U>
struct is_allowed_shared_ptr<std::shared_ptr<U>> : std::bool_constant<AllowedRegistryType_v<U>> {};


template<class T>
inline constexpr bool AllowedRegistryTypeOrShared_v =
    AllowedRegistryType_v<T> || is_allowed_shared_ptr<std::decay_t<T>>::value;


// Checks for region Layout indicating Particle Spatial Layout vs Pure Layout
template <typename T, typename = void>
struct has_getRegionLayout : std::false_type {};

template <typename T>
struct has_getRegionLayout<T, std::void_t<decltype(std::declval<T>().getRegionLayout())>> 
    : std::true_type {};

template <typename T>
constexpr bool has_getRegionLayout_v = has_getRegionLayout<T>::value;





// =========================================
// VISITORS
// =========================================
///////////////////////////////////////////////////////////////////////////////////
// Note: 
// Cant overload identical function with requires conditions
// since requires does not affect signature so they end up with identical singature
// but we can switch reference or constness of arguments like the label, changiing 
// function signature allowing for early throwing of exception (in this file) 
// instead of the additional function overloading in the hpp files.
///////////////////////////////////////////////////////////////////////////////////


/**
 * @struct CatalystAdaptor::InitVisitor
 * @brief Builds Conduit channel metadata for registered fields/particles.
 */
struct CatalystAdaptor::InitVisitor {
    CatalystAdaptor& ca;

    template<typename T>
    requires(AllowedVisType_v<T>)
    void operator()(const std::string& label, const T& entry) const {
        ca.InitVizChannel(entry, label);
    }

    template<typename T>
    requires(!AllowedVisType_v<T>)
    void operator()(const std::string label, const T& entry) const {
       throw IpplException("CatalystAdaptor::ExecVisitor", "Unsupported VIS type for channel: " + label);
    }
    
};




/**
 * @struct CatalystAdaptor::ExecVisitor
 * @brief Publishes data views each step for registered fields/particles.
 */
struct CatalystAdaptor::ExecVisitor {
    CatalystAdaptor& ca;

    template<typename T>
    requires(AllowedVisType_v<T>)
    void operator()(const std::string& label, const T& entry) const {
        ca.ExecVizChannel(entry, label );
    }

    template<typename T>
    requires(!AllowedVisType_v<T>)
    void operator()(const std::string label, const T& entry) const {
       throw IpplException("CatalystAdaptor::ExecVisitor", "Unsupported VIS type for channel: " + label);
    }

};







/**
 * @struct CatalystAdaptor::SteerInitVisitor
 * @brief Declares steerable properties (scalars/vectors/bools/buttons/enums) for the GUI.
 */
struct CatalystAdaptor::SteerInitVisitor {
    CatalystAdaptor& ca;

    template<class S> 
    requires(AllowedSteerType_v<S>)
    void operator()(const std::string& label, const S& entry) const {
        ca.InitSteerChannel(entry, label);
    }

    template<class S>
    requires(!AllowedSteerType_v<S>)
    void operator()(const std::string label , const S&) const {
        throw IpplException("CatalystAdaptor::ForwardSteerChannel", "Unsupported steerable type for channel: " + label);
    }
};



/**
 * @struct CatalystAdaptor::SteerForwardVisitor
 * @brief Sends current simulation values forward into Catalyst for display.
 */
struct CatalystAdaptor::SteerForwardVisitor {
    CatalystAdaptor& ca;

    template<class S>
    requires(AllowedSteerType_v<S>)
    void operator()(const std::string& label, const S& entry) const {
        ca.ForwardSteerChannel(entry, label);
    }

    template<class S>
    requires(!AllowedSteerType_v<S>)
    void operator()(const std::string label , const S&) const {
        throw IpplException("CatalystAdaptor::ForwardSteerChannel", "Unsupported steerable type for channel: " + label);
    }
};

/**
 * @struct CatalystAdaptor::SteerFetchVisitor
 * @brief Retrieves updated values from Catalyst and writes back into the simulation.
 */
struct CatalystAdaptor::SteerFetchVisitor {
    CatalystAdaptor& ca;

    template<class S> 
    requires( AllowedSteerType_v<S> )
    void operator()(const std::string& label, S& value) const {
        ca.FetchSteerChannel(value, label);
    }

    template<class S>
    requires(!AllowedSteerType_v<S>)
    void operator()(const std::string label, S&) const { 
        throw IpplException("CatalystAdaptor::FetchSteerChannel", "Unsupported steerable type for channel: " + label);
    }
};



}