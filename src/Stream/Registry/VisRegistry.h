#pragma once

// #include "Stream/Registry/RegistryHelper.h"

#include "RegistryHelper.h"

#include "Types/Vector.h"

#include "Utility/IpplException.h"



// Base registry entry container
class RegistryBase{
public:
    virtual ~RegistryBase() = default;
    struct Entry {
        std::string name;
        std::any ptr_any;
        template <typename T>
        Entry(const std::string& entry_name, T* ptr)
            : name(entry_name), ptr_any(ptr) {}
    };
};



// Typed registry keyed by compile-time IDs; supports late binding.
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

    RegistryFluent() = default;

    RegistryFluent(std::initializer_list<Entry> entries) {
        std::unordered_map<std::string, std::any> tmp;
        tmp.reserve(entries.size());
        for (const auto& e : entries) tmp[e.name] = e.ptr_any;
        init_from_map(tmp);
    }

    template <fixed_string IdV>
    auto& Get() const {
        constexpr std::size_t I = index_of_v<IdV>;
        static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent");
        using T = TypeAt<I>;
        T* ptr = std::get<I>(m_ptrs);
        if (!ptr) throw std::runtime_error("Null entry for ID");
        return *ptr;
    }

    // Tag-based overloads (avoid 'template' at call sites)
    template <fixed_string IdV>
    auto& Get(id_tag<IdV>) { return this->template Get<IdV>(); }

    template <fixed_string IdV>
    const auto& Get(id_tag<IdV>) const { return this->template Get<IdV>(); }

    template <fixed_string IdV, typename U>
    void Set(U& object) {
        constexpr std::size_t I = index_of_v<IdV>;
        static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent::Set");
        using T = TypeAt<I>;
        static_assert(std::is_same_v<std::remove_const_t<U>, T>,
                      "Type mismatch in RegistryFluent::Set for this ID");
        if (std::get<I>(m_ptrs) != nullptr) {
            std::cerr << "Warning: ID '" << std::string(IdAt<I>.sv())
                      << "' already has an object bound; rebinding to new object.\n";
        }
        std::get<I>(m_ptrs) = &object;
    }

    template <fixed_string IdV, typename U>
    void Set(id_tag<IdV>, U& object) { this->template Set<IdV>(object); }

    template <fixed_string IdV, typename U>
    void SetPtr(U* ptr) {
        constexpr std::size_t I = index_of_v<IdV>;
        static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent::SetPtr");
        using T = TypeAt<I>;
        static_assert(std::is_same_v<std::remove_const_t<U>, T>,
                      "Type mismatch in RegistryFluent::SetPtr for this ID");
        if (std::get<I>(m_ptrs) != nullptr) {
            std::cerr << "Warning: ID '" << std::string(IdAt<I>.sv())
                      << "' already has an object bound; rebinding to new pointer.\n";
        }
        std::get<I>(m_ptrs) = ptr;
    }

    template <fixed_string IdV, typename U>
    void SetPtr(id_tag<IdV>, U* ptr) { this->template SetPtr<IdV>(ptr); }

    template <fixed_string IdV>
    bool Contains() const {
        constexpr std::size_t I = index_of_v<IdV>;
        static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent::Contains");
        return std::get<I>(m_ptrs) != nullptr;
    }

    template <fixed_string IdV>
    bool Contains(id_tag<IdV>) const { return this->template Contains<IdV>(); }

    template <fixed_string IdV>
    void Unset() {
        constexpr std::size_t I = index_of_v<IdV>;
        std::get<I>(m_ptrs) = nullptr;
    }

    template <fixed_string IdV>
    void Unset(id_tag<IdV>) { this->template Unset<IdV>(); }

private:
    void init_from_map(const std::unordered_map<std::string, std::any>& tmp) {
        init_each(tmp, std::make_index_sequence<sizeof...(Slots)>{});
    }

    template <std::size_t... Is>
    void init_each(const std::unordered_map<std::string, std::any>& tmp, std::index_sequence<Is...>) {
        (assign_one<Is>(tmp), ...);
    }

    template <std::size_t I>
    void assign_one(const std::unordered_map<std::string, std::any>& tmp) {
        const auto name_sv = IdAt<I>.sv();
        auto it = tmp.find(std::string(name_sv));
        if (it == tmp.end()) { std::get<I>(m_ptrs) = nullptr; return; }
        try {
            std::get<I>(m_ptrs) = std::any_cast<TypeAt<I>*>(it->second);
        } catch (const std::bad_any_cast&) {
            throw std::invalid_argument("Type mismatch for ID: " + it->first);
        }
    }
};





// Specialization for empty Slots pack
template <>
class RegistryFluent<> : public RegistryBase {
public:
    using Entry = typename RegistryBase::Entry;
    RegistryFluent() = default;
    RegistryFluent(std::initializer_list<Entry>) {}

    template <fixed_string IdV>
    auto& Get() const {
        static_assert(always_false_id<IdV>::value,
                      "RegistryFluent<> has no slots. Add a slot before calling Get.");
        throw std::logic_error("Get called on empty RegistryFluent");
    }

    template <fixed_string IdV>
    auto& Get(id_tag<IdV>) const { return this->template Get<IdV>(); }

    template <fixed_string IdV, typename U>
    void Set(U&) {
        static_assert(always_false_id<IdV>::value,
                      "RegistryFluent<> has no slots. Add a slot before calling Set.");
    }

    template <fixed_string IdV, typename U>
    void Set(id_tag<IdV>, U&) { this->template Set<IdV>(*(U*)nullptr); }

    template <fixed_string IdV, typename U>
    void SetPtr(U*) {
        static_assert(always_false_id<IdV>::value,
                      "RegistryFluent<> has no slots. Add a slot before calling SetPtr.");
    }

    template <fixed_string IdV, typename U>
    void SetPtr(id_tag<IdV>, U*) { this->template SetPtr<IdV>((U*)nullptr); }

    template <fixed_string IdV>
    bool Contains() const {
        static_assert(always_false_id<IdV>::value,
                      "RegistryFluent<> has no slots. Add a slot before calling Contains.");
        return false;
    }

    template <fixed_string IdV>
    bool Contains(id_tag<IdV>) const { return this->template Contains<IdV>(); }

    template <fixed_string IdV>
    void Unset() {
        static_assert(always_false_id<IdV>::value,
                      "RegistryFluent<> has no slots. Add a slot before calling Unset.");
    }

    template <fixed_string IdV>
    void Unset(id_tag<IdV>) { this->template Unset<IdV>(); }
};




// Factories
// Build a RegistryFluent from references
// Usage: auto reg = MakeRegistry<"rho", "phi">(rho, phi);
template <fixed_string... Ids, typename... Ts>
std::unique_ptr<RegistryFluent<Slot<Ids, std::remove_reference_t<Ts>>...>> MakeRegistry(Ts&... objs) {
    using Reg = RegistryFluent<Slot<Ids, std::remove_reference_t<Ts>>...>;
    return std::make_unique<Reg>(
        std::initializer_list<typename RegistryBase::Entry>{
            typename RegistryBase::Entry{std::string(Ids.sv()), &objs}...
        }
    );
}

// Build a RegistryFluent from pointers
// Usage: auto reg = MakeRegistryPtrs<"rho", double>(rho_ptr);
template <fixed_string... Ids, typename... Ts>
std::unique_ptr<RegistryFluent<Slot<Ids, Ts>...>> MakeRegistryPtrs(Ts*... ptrs) {
    using Reg = RegistryFluent<Slot<Ids, Ts>...>;
    return std::make_unique<Reg>(
        std::initializer_list<typename RegistryBase::Entry>{
            typename RegistryBase::Entry{std::string(Ids.sv()), ptrs}...
        }
    );
}
