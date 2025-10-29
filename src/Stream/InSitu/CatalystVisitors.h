#pragma once
#include <filesystem>
#include <string>
#include <type_traits>

#include <Stream/InSitu/CatalystAdaptor.h>
#include <catalyst.hpp>

// template<class T>
// struct is_scalar : std::integral_constant<bool, std::is_arithmetic<T>::value
//                                              || std::is_enum<T>::value
//                                              || std::is_pointer<T>::value
//                                              || std::is_member_pointer<T>::value
//                                              || std::is_null_pointer<T>::value>



namespace ippl{

/* in the advanced version we might want to get rid of this virtual function call */
struct CatalystAdaptor::InitVisitor {
    CatalystAdaptor& ca;

    template<class V, unsigned Dim, class... Rest>
    void operator()(const std::string& label, const ippl::Field<V, Dim, Rest...>& sf) const {
        ca.init_entry(sf, label);
    }
            
    template<class T, unsigned Dim, unsigned Dim_v, class... Rest>
    void operator()(const std::string& label, const ippl::Field<ippl::Vector<T, Dim_v>, Dim, Rest...>& vf) const {
        ca.init_entry(vf, label);
    }
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
    void operator()(const std::string& label, const T& pc) const {
        ca.init_entry(pc, label);
    }
    // template<class S> requires std::is_arithmetic_v<S>
    // void operator()(const std::string& label, S value) const {
        // Optional: create steerable channel already at init time
        // AddSteerableChannel(value, label, node);
    //     (void)label; (void)value;
    // }
};

struct CatalystAdaptor::ExecuteVisitor {
    CatalystAdaptor& ca;

    template<class V, unsigned Dim, class... Rest>
    void operator()(const std::string& label, const ippl::Field<V, Dim, Rest...>& sf) const {
        ca.execute_entry(sf, label);
    }
    template<class T, unsigned Dim, unsigned Dim_v, class... Rest>
    void operator()(const std::string& label, const ippl::Field<ippl::Vector<T, Dim_v>, Dim, Rest...>& vf) const {
        ca.execute_entry(vf, label);
    }
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
    void operator()(const std::string& label, const T& pc) const {
        ca.execute_entry(pc, label );
    }
};

// Initialize steerable channel
struct CatalystAdaptor::SteerInitVisitor {
    CatalystAdaptor& ca;

    template<class S> requires std::is_arithmetic_v<std::decay_t<S>>
    void operator()(const std::string& label, const S& value) const {
        ca.InitSteerableChannel(value, label);
    }

    // Bool-like Switch overload
    void operator()(const std::string& label, const bool& value) const {
        ca.InitSteerableChannel(value, label);
    }

    // Button-like overload
    void operator()(const std::string& label, const ippl::Button& value) const {
        ca.InitSteerableChannel(value, label);
    }

    // // Vector overload: initialize steerable channel for vectors
    // template<class V> requires is_vector_v<std::decay_t<V>>
    // void operator()(const std::string& label, const V& value) const {
    template<typename T, unsigned Dim_v>
    void operator()( const std::string& label, const ippl::Vector<T, Dim_v>& value){
            ca.InitSteerableChannel(value, label);
    }

    template<class T>
    requires (!std::is_arithmetic_v<std::decay_t<T>> && !is_vector_v<std::decay_t<T>>)
    void operator()(const std::string& label , const T&) const {

        throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported steerable type for channel: " + label);
        
    }


};


// Forward steering: add steerable scalar channels only
struct CatalystAdaptor::SteerForwardVisitor {
    CatalystAdaptor& ca;

    template<class S> requires std::is_arithmetic_v<std::decay_t<S>>
    void operator()(const std::string& label, const S& value) const {
        ca.AddSteerableChannel(value, label);
    }

    // Bool-like Switch overload (diverts to scalar)
    void operator()(const std::string& label, const bool& value) const {
        ca.AddSteerableChannel(value, label);
    }

    // Button-like overload (diverts to scalar version)
    void operator()(const std::string& label, const ippl::Button& value) const {
        ca.AddSteerableChannel(value, label);
    }

    // Vector overload: forward steerable vector values
    template<class V> requires is_vector_v<std::decay_t<V>>
    void operator()(const std::string& label, const V& value) const {
        ca.AddSteerableChannel(value, label);
    }

    template<class T>
    requires (!std::is_scalar_v<std::decay_t<T>> && !is_vector_v<std::decay_t<T>>)
    void operator()(const std::string& label , const T&) const {

        throw IpplException("CatalystAdaptor::AddSteerableChannel", "Unsupported steerable type for channel: " + label);
        
    }


};

// Backward steering fetch (mutates external scalars)
struct CatalystAdaptor::SteerFetchVisitor {
    CatalystAdaptor& ca;

    template<class S> requires std::is_arithmetic_v<std::decay_t<S>>
    void operator()(const std::string& label, S& value) const {
        ca.FetchSteerableChannelValue(value, label);
    }

    // Vector overload: fetch steerable vector values (if present)
    template<class V> requires is_vector_v<std::decay_t<V>>
    void operator()(const std::string& label, V& value) const {
        ca.FetchSteerableChannelValue(value, label);
    }

    template<class T>
    requires (!std::is_arithmetic_v<std::decay_t<T>>)
    void operator()(const std::string&, const T&) const {
        ca.ca_m << "INVALID FETCH CALLEd" << endl;    }

    // Optional: fetch for Switch via bool bridge
    void operator()(const std::string& label, bool& value) const {
        ca.FetchSteerableChannelValue(value, label);
    }

    // Optional: fetch for Button via bool bridge (momentary action)
    void operator()(const std::string& label, ippl::Button& value) const {
        /* diverge to normal scalar overload ... */
        // bool iv = bool(value);
        // ca.FetchSteerableChannelValue(value, label);
        ca.FetchSteerableChannelValue(value, label);
        // value = ippl::Button(iv);
    }
};

}



// NOTE: runtime registry currently stores by value; to mutate you need stored reference or pointer.
// If registry adjusted to store reference_wrapper<S>, update dispatch accordingly.