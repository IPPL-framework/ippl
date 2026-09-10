#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <memory>

namespace ippl{

template<typename T>
void ViewRegistry::set(const std::string& name, T object) {
    storage_m[name] = object;
}

template<typename T>
std::string ViewRegistry::set(T object) {
    std::string generatedName = "__unnamed_" + std::to_string(unnamedCounter_m++);
    storage_m[generatedName] = object;
    return generatedName;
}

template<typename T>
std::shared_ptr<T> ViewRegistry::get(const std::string& name) const {
    auto it = storage_m.find(name);
    if (it == storage_m.end()) {
        return nullptr;
    }
    try {
        return std::any_cast<std::shared_ptr<T>>(it->second);
    } catch (const std::bad_any_cast&) {
        return nullptr;
    }
}

inline void ViewRegistry::unset(const std::string& name) {
    storage_m.erase(name);
}

inline void ViewRegistry::clear() {
    storage_m.clear();
}
}