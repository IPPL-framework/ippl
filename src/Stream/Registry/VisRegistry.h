/**
 * @file VisRegistry.h
 * @brief Compile-time and runtime registry for visualization objects and data.
 *
 * Provides a type-safe, extensible registry for storing and accessing objects by compile-time string IDs.
 * Supports late binding, iteration, and factories for easy construction.
 */
#pragma once

#include "Types/Vector.h"
#include "Utility/IpplException.h"
#include "Stream/Registry/RegistryHelper.h"
#include <any>
#include <string>
#include <unordered_map>
#include <memory>
#include <tuple>
#include <vector>
#include <iostream>

/**
 * @brief Base registry entry container for storing named pointers (type-erased).
 */
class RegistryBase{
public:
    virtual ~RegistryBase() = default;
    /**
     * @brief Entry struct for holding a name and a type-erased pointer.
     */
    struct Entry {
        std::string name; ///< Name of the entry
        std::any ptr_any; ///< Type-erased pointer to the object
        /**
         * @brief Construct an entry from a name and pointer.
         * @tparam T Type of the pointer
         * @param entry_name Name for the entry
         * @param ptr Pointer to the object
         */
        template <typename T>
        Entry(const std::string& entry_name, T* ptr) : name(entry_name), ptr_any(ptr) {}
        // For internal use (initializer_list)
        Entry(const std::string& entry_name, std::any ptr) : name(entry_name), ptr_any(ptr) {}
    };
};

/**
 * @brief Typed registry keyed by compile-time IDs; supports late binding and iteration.
 *
 * @tparam Slots List of Slot<Id, Type> types, where Id is a fixed_string and Type is the object type.
 */
template <typename... Slots>
class RegistryFluent : public RegistryBase {
    static_assert(ids_unique<Slots...>::value, "Duplicate ID in RegistryFluent (compile-time)");

    template <std::size_t I>
    using SlotAt = typename nth<I, Slots...>::type;
    template <std::size_t I> using TypeAt = typename SlotAt<I>::type;

    template <std::size_t I>
    static constexpr auto IdAt = SlotAt<I>::Id;

    template <auto IdV>
    static constexpr std::size_t index_of_v = find_index_rec<IdV, 0, Slots...>::value;

    std::tuple<typename Slots::type*...> m_ptrs{};

public:
    using Entry = typename RegistryBase::Entry;

    /**
     * @brief Default constructor. All entries are null.
     */
    RegistryFluent();
    /**
     * @brief Construct from an initializer list of entries.
     * @param entries List of entries to initialize the registry.
     */
    RegistryFluent(std::initializer_list<Entry> entries);

    /**
     * @brief Get a reference to the object for a given compile-time ID.
     * @tparam IdV Compile-time string ID
     * @return Reference to the object
     * @throws std::runtime_error if the entry is null
     */
    template <fixed_string IdV>
    auto& Get() const;

    /**
     * @brief Get a reference to the object for a given tag.
     * @tparam IdV Compile-time string ID
     * @param tag Tag for the ID
     * @return Reference to the object
     */
    template <fixed_string IdV>
    auto& Get(id_tag<IdV>);

    /**
     * @brief Get a const reference to the object for a given tag.
     * @tparam IdV Compile-time string ID
     * @param tag Tag for the ID
     * @return Const reference to the object
     */
    template <fixed_string IdV>
    const auto& Get(id_tag<IdV>) const;

    /**
     * @brief Set the object for a given compile-time ID by reference.
     * @tparam IdV Compile-time string ID
     * @tparam U Type of the object
     * @param object Reference to the object
     */
    template <fixed_string IdV, typename U>
    void Set(U& object);

    /**
     * @brief Set the object for a given tag by reference.
     * @tparam IdV Compile-time string ID
     * @tparam U Type of the object
     * @param tag Tag for the ID
     * @param object Reference to the object
     */
    template <fixed_string IdV, typename U>
    void Set(id_tag<IdV>, U& object);

    /**
     * @brief Set the object for a given compile-time ID by pointer.
     * @tparam IdV Compile-time string ID
     * @tparam U Type of the object
     * @param ptr Pointer to the object
     */
    template <fixed_string IdV, typename U>
    void SetPtr(U* ptr);

    /**
     * @brief Set the object for a given tag by pointer.
     * @tparam IdV Compile-time string ID
     * @tparam U Type of the object
     * @param tag Tag for the ID
     * @param ptr Pointer to the object
     */
    template <fixed_string IdV, typename U>
    void SetPtr(id_tag<IdV>, U* ptr);

    /**
     * @brief Check if an entry for a given compile-time ID is set (non-null).
     * @tparam IdV Compile-time string ID
     * @return True if the entry is set, false otherwise
     */
    template <fixed_string IdV>
    bool Contains() const;

    /**
     * @brief Check if an entry for a given tag is set (non-null).
     * @tparam IdV Compile-time string ID
     * @param tag Tag for the ID
     * @return True if the entry is set, false otherwise
     */
    template <fixed_string IdV>
    bool Contains(id_tag<IdV>) const;

    /**
     * @brief Unset (nullify) the entry for a given compile-time ID.
     * @tparam IdV Compile-time string ID
     */
    template <fixed_string IdV>
    void Unset();

    /**
     * @brief Unset (nullify) the entry for a given tag.
     * @tparam IdV Compile-time string ID
     * @param tag Tag for the ID
     */
    template <fixed_string IdV>
    void Unset(id_tag<IdV>);

    /**
     * @brief Call a function for each non-null entry in the registry.
     * @tparam Func Callable type
     * @param func Function to call with (string_view id, const T& value)
     */
    template <typename Func>
    void forEach(Func&& func) const;

    /**
     * @brief Get the number of non-null entries.
     * @return Number of non-null entries
     */
    std::size_t size() const;
    /**
     * @brief Check if the registry is empty (all entries null).
     * @return True if empty, false otherwise
     */
    bool empty() const;
    /**
     * @brief Get all entry names (including null entries).
     * @return Vector of all entry names
     */
    std::vector<std::string> getAllIds() const;
    /**
     * @brief Get names of non-null entries only.
     * @return Vector of active entry names
     */
    std::vector<std::string> getActiveIds() const;

private:
    void init_from_map(const std::unordered_map<std::string, std::any>& tmp);
    template <std::size_t... Is>
    void init_each(const std::unordered_map<std::string, std::any>& tmp, std::index_sequence<Is...>);
    template <std::size_t I>
    void assign_one(const std::unordered_map<std::string, std::any>& tmp);
    template <typename Func, std::size_t... Is>
    void forEach_impl(Func&& func, std::index_sequence<Is...>) const;
    template <std::size_t I, typename Func>
    void forEach_one(Func&& func) const;
};

/**
 * @brief Specialization for empty Slots pack (no slots).
 */
template <>
class RegistryFluent<> : public RegistryBase {
public:
    using Entry = typename RegistryBase::Entry;
    RegistryFluent();
    RegistryFluent(std::initializer_list<Entry>);

    template <fixed_string IdV>
    auto& Get() const;
    template <fixed_string IdV>
    auto& Get(id_tag<IdV>) const;
    template <fixed_string IdV, typename U>
    void Set(U&);
    template <fixed_string IdV, typename U>
    void Set(id_tag<IdV>, U&);
    template <fixed_string IdV, typename U>
    void SetPtr(U*);
    template <fixed_string IdV, typename U>
    void SetPtr(id_tag<IdV>, U*);
    template <fixed_string IdV>
    bool Contains() const;
    template <fixed_string IdV>
    bool Contains(id_tag<IdV>) const;
    template <fixed_string IdV>
    void Unset();
    template <fixed_string IdV>
    void Unset(id_tag<IdV>);
    template <typename Func>
    void forEach(Func&&) const;
    std::size_t size() const;
    bool empty() const;
};

/**
 * @brief Factory: Build a RegistryFluent from references.
 *
 * Usage: auto reg = MakeRegistry<"rho", "phi">(rho, phi);
 *
 * @tparam Ids Compile-time string IDs
 * @tparam Ts Types of the objects
 * @param objs References to the objects
 * @return Unique pointer to the constructed registry
 */
template <fixed_string... Ids, typename... Ts>
std::unique_ptr<RegistryFluent<Slot<Ids, std::remove_reference_t<Ts>>...>> MakeRegistry(Ts&... objs) {
    using Reg = RegistryFluent<Slot<Ids, std::remove_reference_t<Ts>>...>;
    return std::make_unique<Reg>(
        std::initializer_list<typename RegistryBase::Entry>{
            typename RegistryBase::Entry{std::string(Ids.sv()), &objs}...
        }
    );
}

/**
 * @brief Factory: Build a RegistryFluent from pointers.
 *
 * Usage: auto reg = MakeRegistryPtrs<"rho", double>(rho_ptr);
 *
 * @tparam Ids Compile-time string IDs
 * @tparam Ts Types of the objects
 * @param ptrs Pointers to the objects
 * @return Unique pointer to the constructed registry
 */
template <fixed_string... Ids, typename... Ts>
std::unique_ptr<RegistryFluent<Slot<Ids, Ts>...>> MakeRegistryPtrs(Ts*... ptrs) {
    using Reg = RegistryFluent<Slot<Ids, Ts>...>;
    return std::make_unique<Reg>(
        std::initializer_list<typename RegistryBase::Entry>{
            typename RegistryBase::Entry{std::string(Ids.sv()), ptrs}...
        }
    );
}




    // template <std::size_t... Is>
    // std::size_t count_non_null(std::index_sequence<Is...>) const {
    //     return ((std::get<Is>(m_ptrs) != nullptr ? 1 : 0) + ...);
    // }

    // template <std::size_t... Is>
    // void collectIds_impl(std::vector<std::string>& result, std::index_sequence<Is...>) const {
    //     (result.emplace_back(std::string(IdAt<Is>.sv())), ...);
    // }

    // template <std::size_t... Is>
    // void collectActiveIds_impl(std::vector<std::string>& result, std::index_sequence<Is...>) const {
    //     (collectActiveId_one<Is>(result), ...);
    // }

    // template <std::size_t I>
    // void collectActiveId_one(std::vector<std::string>& result) const {
    //     if (std::get<I>(m_ptrs) != nullptr) {
    //         result.emplace_back(std::string(IdAt<I>.sv()));
    //     }
    // }

#include "Stream/Registry/VisRegistry.hpp"