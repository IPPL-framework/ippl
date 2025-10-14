#pragma once
// VisRegistryRuntime: Non-templated heterogeneous registry for visualization & steering.
// Stores only three allowed categories:
//  * Scalars   (arithmetic types) – for steerable parameters or simple values
//  * Particles (deriving from ParticleBaseBase)
//  * Fields    (ippl::Field<V,Dim,...>)
// Provides visitor-based for_each with overload resolution on real concrete type.
// NOTE: To allow mutation of scalars through steering, store reference_wrapper<T>.

#include <memory>
#include "Ippl.h"
#include <string>
#include <string_view>
#include <vector>
#include <type_traits>
#include <utility>
#include <functional>
#include <cassert>

#include <Stream/InSitu/CatalystAdaptor.h>


// Category traits
namespace ippl {

template<class T>
inline constexpr bool is_scalar_v = std::is_arithmetic_v<typename std::decay<T>::type>;

template<class T>
inline constexpr bool is_particle_v = std::is_base_of<ippl::ParticleBaseBase, typename std::decay<T>::type>::value;

template<class T>
struct is_field : std::false_type {};

template<class V, unsigned Dim, class... Rest>
struct is_field<ippl::Field<V, Dim, Rest...>> : std::true_type {};

template<class T>
inline constexpr bool is_field_v = is_field<typename std::decay<T>::type>::value;
template<class T>
inline constexpr bool AllowedVisType_v = is_scalar_v<T> || is_particle_v<T> || is_field_v<T>;

// Accept shared_ptr<U> when U is allowed
template<class T>
struct is_allowed_shared_ptr : std::false_type {};
template<class U>
struct is_allowed_shared_ptr<std::shared_ptr<U>> : std::bool_constant<AllowedVisType_v<U>> {};
template<class T>
inline constexpr bool AllowedVisTypeOrShared_v = AllowedVisType_v<T> || is_allowed_shared_ptr<typename std::decay<T>::type>::value;

// Helper to unwrap reference_wrapper/pointers uniformly if later needed
template<class T>
struct access_traits {
    using value_type = T;
    static const T& get(const T& v) { return v; }
};

template<class T>
struct access_traits<std::reference_wrapper<T>> {
    using value_type = T;
    static T& get(std::reference_wrapper<T> r) { return r.get(); }
};

template<class T>
struct access_traits<T*> {
    using value_type = T;
    static T& get(T* p) { return *p; }
};

/**
 * @brief Non-templated heterogeneous registry for visualization & steering.
 *
 * Stores only three allowed categories:
 *  - Scalars   (arithmetic types) – for steerable parameters or simple values
 *  - Particles (deriving from ParticleBaseBase)
 *  - Fields    (ippl::Field<V,Dim,...>)
 *
 * Provides visitor-based for_each with overload resolution on real concrete type.
 * NOTE: To allow mutation of scalars through steering, store reference_wrapper<T>.
 */
class VisRegistryRuntime {
    using InitVisitor_t          = CatalystAdaptor::InitVisitor;
    using ExecuteVisitor_t       = CatalystAdaptor::ExecuteVisitor;
    using SteerForwardVisitor_t  = CatalystAdaptor::SteerForwardVisitor;
    using SteerFetchVisitor_t    = CatalystAdaptor::SteerFetchVisitor;

    /**
     * @brief Internal struct representing a registry entry.
     *
     * Holds the label and per-visitor callbacks for each registered object.
     */
    struct Entry {
        std::string label;
        // Per-visitor callbacks; only relevant ones are set per entry
        std::function<void(InitVisitor_t&)> do_init;
        std::function<void(ExecuteVisitor_t&)> do_exec;
        std::function<void(SteerForwardVisitor_t&)> do_steer_fwd;
        std::function<void(SteerFetchVisitor_t&)> do_steer_fetch;
    };

    /**
     * @brief Container for all registered entries.
     */
    std::vector<Entry> entries_;

public:

    /**
     * @brief Apply a visitor to all entries with an init callback.
     * @param v The visitor to apply.
     */
    void for_each(InitVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_init) e.do_init(v);
    }
    /**
     * @brief Apply a visitor to all entries with an execute callback.
     * @param v The visitor to apply.
     */
    void for_each(ExecuteVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_exec) e.do_exec(v);
    }
    /**
     * @brief Apply a visitor to all entries with a steer forward callback.
     * @param v The visitor to apply.
     */
    void for_each(SteerForwardVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_steer_fwd) e.do_steer_fwd(v);
    }
    /**
     * @brief Apply a visitor to all entries with a steer fetch callback.
     * @param v The visitor to apply.
     */
    void for_each(SteerFetchVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_steer_fetch) e.do_steer_fetch(v);
    }

public:
    /**
     * @brief Default constructor.
     */
    VisRegistryRuntime() = default;
    /**
     * @brief Deleted copy constructor.
     */
    VisRegistryRuntime(const VisRegistryRuntime&) = delete;
    /**
     * @brief Deleted copy assignment operator.
     */
    VisRegistryRuntime& operator=(const VisRegistryRuntime&) = delete;
    /**
     * @brief Move constructor.
     */
    VisRegistryRuntime(VisRegistryRuntime&&) noexcept = default;
    /**
     * @brief Move assignment operator.
     */
    VisRegistryRuntime& operator=(VisRegistryRuntime&&) noexcept = default;

    
    
    // template<class T>
    // std::enable_if_t<!is_allowed_shared_ptr<typename std::decay<T>::type>::value, void>

    /**
     * @brief Add an entry by reference (lvalue overload).
     *
     * Registers a scalar, particle, or field by reference. The object must outlive the registry.
     * LVALUE overload: bind callbacks to referenced object
     *
     * @tparam T The type of the object to register.
     * @param label The label for the entry.
     * @param value The object to register (by reference).
     */
    template<class T>
    requires (!is_allowed_shared_ptr<typename std::decay<T>::type>::value)
    void add(const std::string& label, T& value);

    /**
     * @brief Add an entry by shared pointer (shared_ptr overload).
     *
     * Registers a scalar, particle, or field by shared pointer. The registry keeps a copy of the shared_ptr to ensure the object remains alive.
     * Overload: add shared_ptr<U> by binding to referenced object and keeping lifetime
     *
     * @tparam U The type of the object to register.
     * @param label The label for the entry.
     * @param ptr The shared_ptr to the object.
     */
    template<class U>
    void add(const std::string& label, const std::shared_ptr<U>& ptr);

    /**
     * @brief Get the number of registered entries.
     * @return The number of entries.
     */
    std::size_t size() const noexcept { return entries_.size(); }


    /**
     * @brief Check if the registry is empty.
     * @return True if empty, false otherwise.
     */
    bool empty() const noexcept { return entries_.empty(); }
};


} // namespace ippl

#include "Stream/Registry/VisRegistryRuntime.hpp"