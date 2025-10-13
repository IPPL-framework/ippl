#pragma once
#include <filesystem>
#include <string>
#include <type_traits>

#include <Stream/InSitu/CatalystAdaptor.h>

// #include "Stream/Registry/VisRegistryRuntime.h"
// #include "Stream/Registry/ViewRegistry.h"
// #include "Stream/Registry/RegistryHelper.h"




namespace conduit_cpp { class Node; }



struct CatalystAdaptor::InitVisitor {
    conduit_cpp::Node& node;
    std::filesystem::path source_dir;
    bool png_extracts{false};
    template<class V, unsigned Dim, class... Rest>
    void operator()(const std::string& label, const ippl::Field<V, Dim, Rest...>& f) const {
        init_entry(f, label, node, source_dir, png_extracts);
    }
    template<class T, unsigned Dim, unsigned Dim_v, class... Rest>
    void operator()(const std::string& label, const ippl::Field<ippl::Vector<T, Dim_v>, Dim, Rest...>& f) const {
        init_entry(f, label, node, source_dir, png_extracts);
    }
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
    void operator()(const std::string& label, const T& pc) const {
        init_entry(pc, label, node, source_dir, png_extracts);
    }
    template<class S> requires std::is_arithmetic_v<S>
    void operator()(const std::string& label, S value) const {
        // Optional: create steerable channel already at init time
        // AddSteerableChannel(value, label, node);
        (void)label; (void)value;
    }
};

struct CatalystAdaptor::ExecuteVisitor {
    conduit_cpp::Node& node;
    ViewRegistry& vreg;
    template<class V, unsigned Dim, class... Rest>
    void operator()(const std::string& label, const ippl::Field<V, Dim, Rest...>& f) const {
        execute_entry(f, label, node, vreg);
    }
    template<class T, unsigned Dim, unsigned Dim_v, class... Rest>
    void operator()(const std::string& label, const ippl::Field<ippl::Vector<T, Dim_v>, Dim, Rest...>& f) const {
        execute_entry(f, label, node, vreg);
    }
    template<typename T>
    requires std::derived_from<std::decay_t<T>, ippl::ParticleBaseBase>
    void operator()(const std::string& label, const T& pc) const {
        execute_entry(pc, label, node, vreg);
    }
};

// Forward steering: add steerable scalar channels only
struct CatalystAdaptor::SteerForwardVisitor {
    conduit_cpp::Node& node;
    template<class S> requires std::is_arithmetic_v<std::decay_t<S>>
    void operator()(const std::string& label, S value) const {
        AddSteerableChannel(value, label, node);
    }
    template<class T>
    requires (!std::is_arithmetic_v<std::decay_t<T>>)
    void operator()(const std::string&, const T&) const { /* ignore non-scalars */ }
};

// Backward steering fetch (mutates external scalars)
struct CatalystAdaptor::SteerFetchVisitor {
    conduit_cpp::Node& results;
    template<class S> requires std::is_arithmetic_v<std::decay_t<S>>
    void operator()(const std::string& label, S& value) const {
        // NOTE: runtime registry currently stores by value; to mutate you need stored reference or pointer.
        // If registry adjusted to store reference_wrapper<S>, update dispatch accordingly.
        FetchSteerableChannelValue(value, label, results);
    }
    template<class T>
    requires (!std::is_arithmetic_v<std::decay_t<T>>)
    void operator()(const std::string&, const T&) const { /* ignore non-scalars */ }
};

// } // namespace CatalystAdaptor
