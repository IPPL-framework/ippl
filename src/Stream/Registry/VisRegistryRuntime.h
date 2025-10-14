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

class VisRegistryRuntime {
    using InitVisitor_t          = CatalystAdaptor::InitVisitor;
    using ExecuteVisitor_t       = CatalystAdaptor::ExecuteVisitor;
    using SteerForwardVisitor_t  = CatalystAdaptor::SteerForwardVisitor;
    using SteerFetchVisitor_t    = CatalystAdaptor::SteerFetchVisitor;

    struct Entry {
        std::string label;
        // Per-visitor callbacks; only relevant ones are set per entry
        std::function<void(InitVisitor_t&)> do_init;
        std::function<void(ExecuteVisitor_t&)> do_exec;
        std::function<void(SteerForwardVisitor_t&)> do_steer_fwd;
        std::function<void(SteerFetchVisitor_t&)> do_steer_fetch;
    };

    std::vector<Entry> entries_;

public:
    VisRegistryRuntime() = default;
    VisRegistryRuntime(const VisRegistryRuntime&) = delete;
    VisRegistryRuntime& operator=(const VisRegistryRuntime&) = delete;
    VisRegistryRuntime(VisRegistryRuntime&&) noexcept = default;
    VisRegistryRuntime& operator=(VisRegistryRuntime&&) noexcept = default;

    // LVALUE overload: bind callbacks to referenced object
    template<class T>
    std::enable_if_t<!is_allowed_shared_ptr<typename std::decay<T>::type>::value, void>
    add(const std::string& label, T& value) {
        static_assert(AllowedVisType_v<T>, "VisRegistryRuntime: unsupported value type (lvalue)");
        Entry e;
        e.label = label;
        if constexpr (is_field_v<T>) {
            e.do_init = [&value, label](InitVisitor_t& v) { v(label, value); };
            e.do_exec = [&value, label](ExecuteVisitor_t& v) { v(label, value); };
        } else if constexpr (is_particle_v<T>) {
            e.do_init = [&value, label](InitVisitor_t& v) { v(label, value); };
            e.do_exec = [&value, label](ExecuteVisitor_t& v) { v(label, value); };
        } else if constexpr (is_scalar_v<T>) {
            e.do_steer_fwd  = [&value, label](SteerForwardVisitor_t& v) { v(label, value); };
            e.do_steer_fetch= [&value, label](SteerFetchVisitor_t& v) { v(label, value); };
        }
        entries_.push_back(std::move(e));
    }

    // Overload: add shared_ptr<U> by binding to referenced object and keeping lifetime
    template<class U>
    void add(const std::string& label, const std::shared_ptr<U>& ptr) {
        static_assert(AllowedVisType_v<U>, "VisRegistryRuntime: unsupported shared_ptr<U> type");
        if (!ptr) return;
        add(label, *ptr);
        // keep alive by capturing shared_ptr in a no-op callback
        auto& e = entries_.back();
        auto keep = ptr; // copy
        // augment one of the callbacks or create a dummy fetch to hold lifetime
        if (!e.do_init) {
            e.do_init = [keep](InitVisitor_t&) {};
        } else {
            auto fn = e.do_init;
            e.do_init = [keep, fn](InitVisitor_t& v) { fn(v); };
        }
    }

    // Overloads for known visitors
    void for_each(InitVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_init) e.do_init(v);
    }
    void for_each(ExecuteVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_exec) e.do_exec(v);
    }
    void for_each(SteerForwardVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_steer_fwd) e.do_steer_fwd(v);
    }
    void for_each(SteerFetchVisitor_t& v) const {
        for (auto const& e : entries_) if (e.do_steer_fetch) e.do_steer_fetch(v);
    }

    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
};

// =====================================================================================
// Factory helpers
// Build a runtime registry from alternating (label, value) arguments.
// Usage:
//   auto reg = MakeVisRegistryRuntime(
//                 "particles", pcontainer,
//                 "E", std::ref(fieldE),
//                 "rho", std::ref(fieldRho),
//                 "magnetic_scale", std::ref(magnetic_scale));
// A shared_ptr-returning variant is also provided.
namespace detail {
    inline void add_pairs(VisRegistryRuntime&) {}

    template<class L, class V, class... Rest>
    void add_pairs(VisRegistryRuntime& r, L&& label, V&& value, Rest&&... rest) {
        static_assert(AllowedVisTypeOrShared_v<typename std::decay<V>::type>,
                      "VisRegistryRuntime: unsupported value type in factory");

        // Materialize label as std::string to avoid ambiguous overloads
        std::string lbl{std::forward<L>(label)};

        if constexpr (std::is_lvalue_reference_v<V&&>) {
            // lvalue path: keep reference semantics for non-owning entries
            r.add(lbl, value); // prefers const std::string& overloads
        } 
        // else {
        //     // rvalue path: allow ownership-taking overload
        //     r.add(std::move(lbl), std::forward<V>(value));
        // }

        if constexpr (sizeof...(Rest) > 0) {
            static_assert(sizeof...(Rest) % 2 == 0, "Factory arguments must be label-value pairs");
            add_pairs(r, std::forward<Rest>(rest)...);
        }
    }
} // namespace detail

template<class... Args>
VisRegistryRuntime MakeVisRegistryRuntime(Args&&... args) {
    static_assert(sizeof...(Args) % 2 == 0, "MakeVisRegistryRuntime requires label-value pairs");
    VisRegistryRuntime reg;
    if constexpr (sizeof...(Args) > 0) {
        detail::add_pairs(reg, std::forward<Args>(args)...);
    }
    return reg;
}

template<class... Args>
std::shared_ptr<VisRegistryRuntime> MakeVisRegistryRuntimePtr(Args&&... args) {
    static_assert(sizeof...(Args) % 2 == 0, "MakeVisRegistryRuntimePtr requires label-value pairs");
    auto reg = std::make_shared<VisRegistryRuntime>();
    if constexpr (sizeof...(Args) > 0) {
        detail::add_pairs(*reg, std::forward<Args>(args)...);
    }
    return reg;

} 

} // namespace ippl
