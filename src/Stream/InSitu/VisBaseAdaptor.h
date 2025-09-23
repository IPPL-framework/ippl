#pragma once

#include "Stream/Registry/VisRegistry"


// Fluent builder for compile-time slot accumulation
// Each add<"ID"> returns a new Builder with Slots... + Slot<"ID", T>

template <typename... Slots>
class VisBaseAdaptor{
    using registry_t = RegistryFluent<Slots...>;
    std::shared_ptr<registry_t> registry;

    // Clone existing bindings into a Next adaptor (with superset of Slots)
    template <typename Next, std::size_t... Is>
    void clone_into(Next& next, std::index_sequence<Is...>) const {
        (clone_one<typename nth<Is, Slots...>::type>(next), ...);
    }

    template <typename SlotT, typename Next>
    void clone_one(Next& next) const {
        constexpr auto Id = SlotT::Id;
        // Rebind into the next adaptor using the underlying registry API
        if (this->get_registry().template Contains<Id>()) {
            auto& ref = this->get_registry().template Get<Id>();
            next.get_registry().template SetPtr<Id>(&ref);
        }
    }

public:
    // Default: start with an empty registry for fluent building
    VisBaseAdaptor() : registry(std::make_shared<registry_t>()) {}

    // Constructor from object references (enabled only when there is at least one Slot)
    template <typename Dummy = void, typename = std::enable_if_t<(sizeof...(Slots) > 0), Dummy>>
    VisBaseAdaptor(typename Slots::type&... objs) {
        registry = std::make_shared<registry_t>(
            std::initializer_list<typename RegistryBase::Entry>{
                typename RegistryBase::Entry{std::string(Slots::Id.sv()), &objs}...
            }
        );
    }

    // Constructor from pointers (enabled only when there is at least one Slot)
    template <typename Dummy = void, typename = std::enable_if_t<(sizeof...(Slots) > 0), Dummy>, typename = Dummy>
    VisBaseAdaptor(typename Slots::type*... ptrs) {
        registry = std::make_shared<registry_t>(
            std::initializer_list<typename RegistryBase::Entry>{
                typename RegistryBase::Entry{std::string(Slots::Id.sv()), ptrs}...
            }
        );
    }

    // Construct by copying an existing registry object (creates a new shared instance)
    VisBaseAdaptor(registry_t& r_init) {
        registry = std::make_shared<registry_t>(r_init);
    }

    // Share ownership of an existing registry
    VisBaseAdaptor(std::shared_ptr<registry_t> reg)
        : registry(std::move(reg)) {}

    // Adopt from an existing unique_ptr (promote to shared_ptr)
    VisBaseAdaptor(std::unique_ptr<registry_t>& reg)
        : registry(std::move(reg)) {}

    // Accessors mirroring src_static
    registry_t& get_registry() noexcept { return *registry; }
    const registry_t& get_registry() const noexcept { return *registry; }
    std::shared_ptr<registry_t> get_registry_ptr() const noexcept { return registry; }

    template<fixed_string IdV, typename T>
    auto add(T& obj) && {
        // Compute at compile-time whether IdV is already in Slots...
        constexpr bool already_present = ((Slots::Id == IdV) || ... || false);
        if constexpr (already_present) {
            std::cerr << "Warning: ID '" << std::string(IdV.sv())
                      << "' already exists in this adaptor; ignoring add().\n";
            return std::move(*this);
        } else {
            using NewSlot = Slot<IdV, std::remove_reference_t<T>>;
            using NextAdaptor = VisBaseAdaptor<Slots..., NewSlot>;
            NextAdaptor next;  // starts with an empty registry
            // carry over current bindings
            this->clone_into(next, std::make_index_sequence<sizeof...(Slots)>{});
            // bind the new one via the registry API
            next.get_registry().template Set<IdV>(obj);
            return next;
        }
    }
};

// Convenience Factory (same as src_static)
template <fixed_string... Ids, typename... Ts>
auto MakeVisAdaptor(Ts&... objs)
    -> VisBaseAdaptor<Slot<Ids, std::remove_reference_t<Ts>>...>
{
    return VisBaseAdaptor<Slot<Ids, std::remove_reference_t<Ts>>...>(objs...);
}

// Overloads to build from existing registries (shared/unique)
template <typename... Slots>
auto MakeVisAdaptor(std::shared_ptr<RegistryFluent<Slots...>> reg)
    -> VisBaseAdaptor<Slots...>
{
    return VisBaseAdaptor<Slots...>(std::move(reg));
}

template <typename... Slots>
auto MakeVisAdaptor(std::unique_ptr<RegistryFluent<Slots...>>& reg)
    -> VisBaseAdaptor<Slots...>
{
    return VisBaseAdaptor<Slots...>(reg);
}

// === Helpers for easier fluent adds and name reuse ===

// 1) add_slot: functional helper returning the new adaptor type.
// Usage: auto vis = add_slot<"E2">(std::move(vis), fE);
// Still requires a new variable name because the type changes.
template <fixed_string IdV, typename Adaptor, typename Obj>
auto add_slot(Adaptor&& ad, Obj& obj) {
    return std::forward<Adaptor>(ad).template add<IdV>(obj);
}

// 2) with_added: continuation-style helper that lets you keep the same name
// inside the provided lambda parameter.
// Usage:
//   with_added<"E2">(std::move(vis), fE, [&](auto&& vis){
//       // use vis here (new type)
//   });
// Returns whatever the lambda returns.
template <fixed_string IdV, typename Adaptor, typename Obj, typename F>
auto with_added(Adaptor&& ad, Obj& obj, F&& fn) {
    auto next = std::forward<Adaptor>(ad).template add<IdV>(obj);
    return std::forward<F>(fn)(std::move(next));
}

// 3) VIS_REBIND: macro that creates a one-iteration scope where the new adaptor
// shadows the old variable name. This lets you reuse the same identifier.
// Usage:
//   VIS_REBIND(vis, "E2", fE) {
//       // here, vis is the new type
//       // you can chain more operations or introduce another VIS_REBIND
//   }
// Note: the shadowed name only exists inside the following block.
#ifndef VIS_REBIND
#define VIS_REBIND(VAR, ID, OBJ)                                             \
    for (bool _vis_once = true; _vis_once; _vis_once = false)                \
        if (auto _vis_new = std::move(VAR).add<ID>(OBJ); true)               \
            for (decltype(_vis_new)& VAR = _vis_new; _vis_once; _vis_once = false)
#endif




