/**
 * @file VisRegistryRuntime.h
 * @brief Non-templated heterogeneous registry for steering parameters.
 */
#pragma once
// VisRegistryRuntime: Non-templated heterogeneous registry for steering.
// Stores steerable parameters (arithmetic scalars, bools, Buttons, ippl::Vector,
// enums, and registered struct types via StructMeta).
// Visualization data is exported through the CatalystAdaptor channel API
// (addMeshChannel/addMeshBlock/addParticleChannel) instead of this registry.
// Provides visitor-based forEach with overload resolution on real concrete type.
// NOTE: To allow mutation of scalars through steering, store reference_wrapper<T>.

#include "Ippl.h"

#include <Stream/InSitu/CatalystAdaptor.h>
#include <Stream/InSitu/CatalystAdaptorSteering.hpp>  // ensure StructMeta available for registered structs
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Yes, you can use if constexpr to generate a single function body that
//  results in different return types based on a compile-time condition on
//  a template parameter.

// // However, the function's declared return type must be able to
//  resolve to the correct type for every possible branch. This
//  is typically achieved using auto for the return type (for C++14/17
//     return type deduction) or by using decltype(auto) or a
//  specific type trait that resolves conditionally.

#include <variant>

// Category traits
namespace ippl {

    // Helper to unwrap reference_wrapper/pointers uniformly if later needed
    template <class T>
    struct access_traits {
        using value_type = T;
        static const T& get(const T& v) { return v; }
    };

    template <class T>
    struct access_traits<std::reference_wrapper<T>> {
        using value_type = T;
        static T& get(std::reference_wrapper<T> r) { return r.get(); }
    };

    template <class T>
    struct access_traits<T*> {
        using value_type = T;
        static T& get(T* p) { return *p; }
    };

    /**
     * @brief Non-templated heterogeneous registry for steering parameters.
     *
     * Stores steerable parameters:
     *  - Scalars   (arithmetic types, bools, Buttons, ippl::Vector, enums)
     *  - Registered struct types (via StructMeta) and vectors thereof
     *
     * Provides visitor-based forEach with overload resolution on real concrete type.
     * NOTE: To allow mutation of scalars through steering, store reference_wrapper<T>.
     */
    class VisRegistryRuntime {
    private:
        /**
         * @brief Internal struct representing a registry entry.
         *
         * Holds the label and per-visitor callbacks for each registered object.
         */
        struct Entry {
            std::string label_m;
            std::function<void(CatalystAdaptor::SteerVisitorVariant_t)> dispatchSteer_m;
        };

        /**
         * @brief Container for all registered entries.
         */
        std::vector<Entry> entries_m;

        // Fast index for every registered entry. Labels are unique registry-wide.
        std::unordered_map<std::string, std::size_t> indexExec_m;

    public:
        /**
         * @brief Apply a visitor to all registered entries.
         * @param v The visitor to apply.
         */
        template <typename VisitorT>
        void forEach(VisitorT& v) const {
            if constexpr (std::is_constructible_v<CatalystAdaptor::SteerVisitorVariant_t,
                                                  VisitorT*>) {
                CatalystAdaptor::SteerVisitorVariant_t var = &v;
                for (const auto& e : entries_m) {
                    if (e.dispatchSteer_m)
                        e.dispatchSteer_m(var);
                }
            }
        }

        /**
         * @brief Apply a visualization or steering visitor to one entry identified by label.
         * @return true if found and executed, false otherwise.
         */
        template <typename VisitorT>
        bool forOne(const std::string& label, VisitorT& v) const {
            auto it = indexExec_m.find(label);
            if (it == indexExec_m.end()) {
                std::cerr << "VisRegistryRuntime::forOne: label not found: '" << label << "'\n";
                std::cerr << "  Available registry labels (" << indexExec_m.size()
                          << "):" << std::endl;
                for (const auto& kv : indexExec_m)
                    std::cerr << " " << kv.first << std::endl;
                std::cerr << std::endl;
                return false;
            }
            const auto& e = entries_m[it->second];

            if constexpr (std::is_constructible_v<CatalystAdaptor::SteerVisitorVariant_t,
                                                  VisitorT*>) {
                if (!e.dispatchSteer_m) {
                    std::cout << "forOne: No steer dispatch CallBack" << std::endl;
                    return false;
                }
                CatalystAdaptor::SteerVisitorVariant_t var = &v;
                e.dispatchSteer_m(var);
                return true;
            } else {
                return false;
            }
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
         * Registers a steerable parameter by reference. The object must outlive the registry.
         * LVALUE overload: bind callbacks to referenced object
         *
         * @tparam T The type of the object to register.
         * @param label The label for the entry.
         * @param value The object to register (by reference).
         */
        template <class T>
            requires(!is_allowed_shared_ptr<typename std::decay<T>::type>::value)
        void add(const std::string& label, T& value);

        /**
         * @brief Add an entry by shared pointer (shared_ptr overload).
         *
         * Registers a steerable parameter by shared pointer. The registry keeps a copy of the
         * shared_ptr to ensure the object remains alive. Overload: add shared_ptr<U> by binding to
         * referenced object and keeping lifetime
         *
         * @tparam U The type of the object to register.
         * @param label The label for the entry.
         * @param ptr The shared_ptr to the object.
         */
        template <class U>
        void add(const std::string& label, const std::shared_ptr<U>& ptr);

        /**
         * @brief Get the number of registered entries.
         * @return The number of entries.
         */
        std::size_t size() const noexcept { return entries_m.size(); }

        /**
         * @brief Check if the registry is empty.
         * @return True if empty, false otherwise.
         */
        bool empty() const noexcept { return entries_m.empty(); }
    };

}  // namespace ippl

#include "Stream/Registry/VisRegistryRuntime.hpp"
