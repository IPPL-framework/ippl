#pragma once
#include "Stream/Registry/VisRegistryRuntime.h"


namespace ippl {

    // LVALUE overload: bind callbacks to referenced object
    template<class T>
    requires(!is_allowed_shared_ptr<typename std::decay<T>::type>::value)
    void VisRegistryRuntime::add(const std::string& label, T& value) {
        static_assert(AllowedVisType_v<T>, "VisRegistryRuntime: unsupported value type (lvalue)");
        Entry e;
        e.label = label;
        if constexpr (is_field_v<T>) {
            e.do_init = [&value, label](InitVisitor_t& v) { v(label, value); };
            e.do_exec = [&value, label](ExecuteVisitor_t& v) { v(label, value); };
        } else if constexpr (is_particle_v<T>) {
            e.do_init = [&value, label](InitVisitor_t& v) { v(label, value); };
            e.do_exec = [&value, label](ExecuteVisitor_t& v) { v(label, value); };
        } 
        /*  dont need scalar vis?... */
        else if constexpr (is_scalar_v<T>) {
            e.do_steer_init = [&value, label](SteerInitVisitor_t& v)    { v(label, value); };
            e.do_steer_fwd  = [&value, label](SteerForwardVisitor_t& v) { v(label, value); };
            e.do_steer_fetch= [&value, label](SteerFetchVisitor_t& v)   { v(label, value); };
        }
        entries_.push_back(std::move(e));
    }


    // Overload: add shared_ptr<U> by binding to referenced object and keeping lifetime
    template<class U>
    void VisRegistryRuntime::add(const std::string& label, const std::shared_ptr<U>& ptr) {
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

    /**
     * @brief Base case for add_pairs: does nothing (end of recursion).
     * @param r The registry (unused).
     */
    inline void add_pairs(VisRegistryRuntime&) {}

    /**
     * @brief Recursively adds label-value pairs to a VisRegistryRuntime.
     *
     * This helper is used by MakeVisRegistryRuntime and MakeVisRegistryRuntimePtr to build a registry from alternating label-value arguments.
     *
     * @tparam L The type of the label (should be convertible to std::string).
     * @tparam V The type of the value (must satisfy AllowedVisTypeOrShared_v).
     * @tparam Rest The remaining label-value argument types.
     * @param r The registry to add to.
     * @param label The label for the entry.
     * @param value The value to register.
     * @param rest The remaining label-value arguments.
     */
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

} // detail
} //ippl