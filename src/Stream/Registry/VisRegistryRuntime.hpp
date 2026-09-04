#pragma once
#include "Stream/Registry/VisRegistryRuntime.h"

#include <type_traits>
#include <vector>


namespace ippl {

    // Small trait to detect exactly std::vector<T, A>
    template<typename T>
    struct is_std_vector : std::false_type {};
    template<typename T, typename A>
    struct is_std_vector<std::vector<T, A>> : std::true_type {};
    template<typename T>
    inline constexpr bool is_std_vector_v = is_std_vector<std::decay_t<T>>::value;

    // LVALUE overload: bind callbacks to referenced object
    template<class T>
    requires(!is_allowed_shared_ptr<typename std::decay<T>::type>::value)
    void VisRegistryRuntime::add(const std::string& label, T& value) {
        using DecayT = std::decay_t<T>;
        // Canonicalize label for std::vector entries: ensure 'array:' prefix
        std::string effectiveLabel = label;
        if constexpr (is_std_vector_v<DecayT>) {
            if (effectiveLabel.rfind("array:", 0) != 0) {
                effectiveLabel = std::string("array:") + effectiveLabel;
            }
        }
        // Materialize label as an owned copy for safe capture in lambdas
        std::string L = effectiveLabel;
        Entry e; e.label_m = L;

        if constexpr (AllowedVisType_v<T>) {
            e.dispatchVis_m = [&value, L](CatalystAdaptor::VisVisitorVariant_t var) {
                std::visit([&](auto* active_ptr) {
                    if (active_ptr) (*active_ptr)(L, value);
                }, var);
            };
        } 
        else if constexpr (AllowedSteerType_v<T>) {
            e.dispatchSteer_m = [&value, L](CatalystAdaptor::SteerVisitorVariant_t var) {
                std::visit([&](auto* active_ptr) {
                    if (active_ptr) (*active_ptr)(L, value);
                }, var);
            };
        }
        else if (ippl::detail::StructMeta<DecayT>::registered) {
            e.dispatchSteer_m = [&value, L](CatalystAdaptor::SteerVisitorVariant_t var) {
                ippl::detail::StructMeta<DecayT>::dispatch(var, value, L);
            };
        }
        else {
            if constexpr (is_std_vector_v<DecayT>) {
                using ElemT = typename DecayT::value_type;
                if (ippl::detail::StructMeta<ElemT>::registered) {
                    e.dispatchSteer_m = [&value, L](CatalystAdaptor::SteerVisitorVariant_t var) {
                        ippl::detail::StructMeta<ElemT>::dispatch_vec(var, value, L);
                    };
                }
            }
        }
        entries_m.push_back(std::move(e));
        if constexpr (AllowedVisType_v<T>) {
            indexExec_m[L] = entries_m.size() - 1;
        }

        // Check if type is completely unsupported
        bool supported = false;
        if constexpr (AllowedVisType_v<T> || AllowedSteerType_v<T>) {
            supported = true;
        } else if (ippl::detail::StructMeta<DecayT>::registered) {
            supported = true;
        } else {
            if constexpr (is_std_vector_v<DecayT>) {
                using ElemT = typename DecayT::value_type;
                if (ippl::detail::StructMeta<ElemT>::registered) {
                    supported = true;
                }
            }
        }
        
        if (!supported) {
            throw IpplException("VisRegistryRuntime::add", std::string("Unsupported value type for registry entry '") + label + "' (type=" + typeid(T).name() + ")");
        }
    }


    // Overload: add shared_ptr<U> by binding to referenced object and keeping lifetime
    template<class U>
    void VisRegistryRuntime::add(const std::string& label, const std::shared_ptr<U>& ptr) {
        if (!ptr) return;

        add(label, *ptr); // delegates to lvalue path (already handles struct/meta)
        // keep alive by capturing shared_ptr in a wrapper callback
        auto& e = entries_m.back();
        auto keep = ptr; // copy
        if (e.dispatchVis_m) {
            auto fn = e.dispatchVis_m;
            e.dispatchVis_m = [keep, fn](CatalystAdaptor::VisVisitorVariant_t v) { fn(v); };
        }
        if (e.dispatchSteer_m) {
            auto fn = e.dispatchSteer_m;
            e.dispatchSteer_m = [keep, fn](CatalystAdaptor::SteerVisitorVariant_t v) { fn(v); };
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
     * @tparam V The type of the value (must satisfy AllowedRegistryTypeOrShared_v).
     * @tparam Rest The remaining label-value argument types.
     * @param r The registry to add to.
     * @param label The label for the entry.
     * @param value The value to register.
     * @param rest The remaining label-value arguments.
     */
    template<class L, class V, class... Rest>
    void add_pairs(VisRegistryRuntime& r, L&& label, V&& value, Rest&&... rest) {

        
        // using DecayV = typename std::decay<V>::type;
        // static_assert(AllowedRegistryTypeOrShared_v<DecayV> || std::is_enum_v<DecayV>,
                    //   "VisRegistryRuntime: unsupported value type in factory");

        // Materialize label as std::string to avoid ambiguous overloads
        std::string lbl{std::forward<L>(label)};

        if constexpr (std::is_lvalue_reference_v<V&&>) {
            // lvalue path: keep reference semantics for non-owning entries
            r.add(lbl, value); // prefers const std::string& overloads
        } 

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