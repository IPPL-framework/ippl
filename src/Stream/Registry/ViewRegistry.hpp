#pragma once
// #include "ViewRegistry.h"
#include <any>
#include <string>
#include <unordered_map>
#include <memory>

// Implementation of ViewRegistry methods

template<typename T>
void ViewRegistry::set(const std::string& name, T object) {
    m_storage[name] = object;
}

template<typename T>
std::string ViewRegistry::set(T object) {
    std::string generated_name = "__unnamed_" + std::to_string(m_unnamed_counter++);
    m_storage[generated_name] = object;
    return generated_name;
}

template<typename T>
std::shared_ptr<T> ViewRegistry::get(const std::string& name) const {
    auto it = m_storage.find(name);
    if (it == m_storage.end()) {
        return nullptr;
    }
    try {
        return std::any_cast<std::shared_ptr<T>>(it->second);
    } catch (const std::bad_any_cast&) {
        return nullptr;
    }
}

inline void ViewRegistry::unset(const std::string& name) {
    m_storage.erase(name);
}
