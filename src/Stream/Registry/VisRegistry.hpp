#include <stdexcept>
#include <type_traits>
#include <utility>
#include "Stream/Registry/VisRegistry.h"
#include "Stream/Registry/RegistryHelper.h"

namespace ippl{

// RegistryFluent implementation
template <typename... Slots>
RegistryFluent<Slots...>::RegistryFluent() = default;

template <typename... Slots>
RegistryFluent<Slots...>::RegistryFluent(std::initializer_list<Entry> entries) {
    std::unordered_map<std::string, std::any> tmp;
    tmp.reserve(entries.size());
    for (const auto& e : entries) tmp[e.name] = e.ptr_any;
    init_from_map(tmp);
}

template <typename... Slots>
template <fixed_string IdV>
auto& RegistryFluent<Slots...>::Get() const {
    constexpr std::size_t I = index_of_v<IdV>;
    static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent");
    using T = TypeAt<I>;
    T* ptr = std::get<I>(m_ptrs);
    if (!ptr) throw std::runtime_error("Null entry for ID");
    return *ptr;
}

template <typename... Slots>
template <fixed_string IdV>
auto& RegistryFluent<Slots...>::Get(id_tag<IdV>) { return this->template Get<IdV>(); }

template <typename... Slots>
template <fixed_string IdV>
const auto& RegistryFluent<Slots...>::Get(id_tag<IdV>) const { return this->template Get<IdV>(); }

template <typename... Slots>
template <fixed_string IdV, typename U>
void RegistryFluent<Slots...>::Set(U& object) {
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

template <typename... Slots>
template <fixed_string IdV, typename U>
void RegistryFluent<Slots...>::Set(id_tag<IdV>, U& object) { this->template Set<IdV>(object); }

template <typename... Slots>
template <fixed_string IdV, typename U>
void RegistryFluent<Slots...>::SetPtr(U* ptr) {
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

template <typename... Slots>
template <fixed_string IdV, typename U>
void RegistryFluent<Slots...>::SetPtr(id_tag<IdV>, U* ptr) { this->template SetPtr<IdV>(ptr); }

template <typename... Slots>
template <fixed_string IdV>
bool RegistryFluent<Slots...>::Contains() const {
    constexpr std::size_t I = index_of_v<IdV>;
    static_assert(I != static_cast<std::size_t>(-1), "Unknown ID in RegistryFluent::Contains");
    return std::get<I>(m_ptrs) != nullptr;
}

template <typename... Slots>
template <fixed_string IdV>
bool RegistryFluent<Slots...>::Contains(id_tag<IdV>) const { return this->template Contains<IdV>(); }

template <typename... Slots>
template <fixed_string IdV>
void RegistryFluent<Slots...>::Unset() {
    constexpr std::size_t I = index_of_v<IdV>;
    std::get<I>(m_ptrs) = nullptr;
}

template <typename... Slots>
template <fixed_string IdV>
void RegistryFluent<Slots...>::Unset(id_tag<IdV>) { this->template Unset<IdV>(); }

template <typename... Slots>
template <typename Func>
void RegistryFluent<Slots...>::for_each(Func&& func) const {
    for_each_impl(std::forward<Func>(func), std::make_index_sequence<sizeof...(Slots)>{});
}

template <typename... Slots>
std::size_t RegistryFluent<Slots...>::size() const {
    return count_non_null(std::make_index_sequence<sizeof...(Slots)>{});
}

template <typename... Slots>
bool RegistryFluent<Slots...>::empty() const {
    return size() == 0;
}

template <typename... Slots>
std::vector<std::string> RegistryFluent<Slots...>::getAllIds() const {
    std::vector<std::string> result;
    result.reserve(sizeof...(Slots));
    collectIds_impl(result, std::make_index_sequence<sizeof...(Slots)>{});
    return result;
}

template <typename... Slots>
std::vector<std::string> RegistryFluent<Slots...>::getActiveIds() const {
    std::vector<std::string> result;
    collectActiveIds_impl(result, std::make_index_sequence<sizeof...(Slots)>{});
    return result;
}

template <typename... Slots>
void RegistryFluent<Slots...>::init_from_map(const std::unordered_map<std::string, std::any>& tmp) {
    init_each(tmp, std::make_index_sequence<sizeof...(Slots)>{});
}

template <typename... Slots>
template <std::size_t... Is>
void RegistryFluent<Slots...>::init_each(const std::unordered_map<std::string, std::any>& tmp, std::index_sequence<Is...>) {
    (assign_one<Is>(tmp), ...);
}

template <typename... Slots>
template <std::size_t I>
void RegistryFluent<Slots...>::assign_one(const std::unordered_map<std::string, std::any>& tmp) {
    const auto name_sv = IdAt<I>.sv();
    auto it = tmp.find(std::string(name_sv));
    if (it == tmp.end()) { std::get<I>(m_ptrs) = nullptr; return; }
    try {
        std::get<I>(m_ptrs) = std::any_cast<TypeAt<I>*>(it->second);
    } catch (const std::bad_any_cast&) {
        throw std::invalid_argument("Type mismatch for ID: " + it->first);
    }
}

template <typename... Slots>
template <typename Func, std::size_t... Is>
void RegistryFluent<Slots...>::for_each_impl(Func&& func, std::index_sequence<Is...>) const {
    (for_each_one<Is>(std::forward<Func>(func)), ...);
}

template <typename... Slots>
template <std::size_t I, typename Func>
void RegistryFluent<Slots...>::for_each_one(Func&& func) const {
    auto* ptr = std::get<I>(m_ptrs);
    if (ptr != nullptr) {
        const auto id_sv = IdAt<I>.sv();
        func(id_sv, *ptr);
    }
}

// Specialization for empty Slots pack
inline RegistryFluent<>::RegistryFluent() = default;
inline RegistryFluent<>::RegistryFluent(std::initializer_list<Entry>) {}

template <fixed_string IdV>
auto& RegistryFluent<>::Get() const {
    static_assert(always_false_id<IdV>::value,
                  "RegistryFluent<> has no slots. Add a slot before calling Get.");
    throw std::logic_error("Get called on empty RegistryFluent");
}

template <fixed_string IdV>
auto& RegistryFluent<>::Get(id_tag<IdV>) const { return this->template Get<IdV>(); }

template <fixed_string IdV, typename U>
void RegistryFluent<>::Set(U&) {
    static_assert(always_false_id<IdV>::value,
                  "RegistryFluent<> has no slots. Add a slot before calling Set.");
}

template <fixed_string IdV, typename U>
void RegistryFluent<>::Set(id_tag<IdV>, U&) { this->template Set<IdV>(*(U*)nullptr); }

template <fixed_string IdV, typename U>
void RegistryFluent<>::SetPtr(U*) {
    static_assert(always_false_id<IdV>::value,
                  "RegistryFluent<> has no slots. Add a slot before calling SetPtr.");
}

template <fixed_string IdV, typename U>
void RegistryFluent<>::SetPtr(id_tag<IdV>, U*) { this->template SetPtr<IdV>((U*)nullptr); }

template <fixed_string IdV>
bool RegistryFluent<>::Contains() const {
    static_assert(always_false_id<IdV>::value,
                  "RegistryFluent<> has no slots. Add a slot before calling Contains.");
    return false;
}

template <fixed_string IdV>
bool RegistryFluent<>::Contains(id_tag<IdV>) const { return this->template Contains<IdV>(); }

template <fixed_string IdV>
void RegistryFluent<>::Unset() {
    static_assert(always_false_id<IdV>::value,
                  "RegistryFluent<> has no slots. Add a slot before calling Unset.");
}

template <fixed_string IdV>
void RegistryFluent<>::Unset(id_tag<IdV>) { this->template Unset<IdV>(); }

template <typename Func>
void RegistryFluent<>::for_each(Func&&) const {
    // Empty registry - nothing to iterate over
}

inline std::size_t RegistryFluent<>::size() const { return 0; }
inline bool RegistryFluent<>::empty() const { return true; }

}