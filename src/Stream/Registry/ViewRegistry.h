#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace ippl{
/**
 * @class ViewRegistry
 * @brief A dynamic registry for storing and retrieving named objects using type erasure.
 *
 * The ViewRegistry class allows you to store objects of any type, typically as shared pointers,
 * and retrieve them by name. It uses std::any for type erasure and supports both named and unnamed
 * object registration. Unnamed objects are given a unique generated name.
 *
 * Typical usage is to store shared_ptr<T> objects, so that the registry keeps the objects alive
 * and accessible by name. Retrieval is type-safe and returns nullptr if the type or name does not match.
 */
class ViewRegistry {
private:
    /**
     * @brief Internal storage mapping names to objects (type-erased).
     */
    std::unordered_map<std::string, std::any> m_storage;
    /**
     * @brief Counter for generating unique names for unnamed objects.
     */
    size_t m_unnamed_counter = 0;

public:
    /**
     * @brief Store an object with a given name.
     *
     * @tparam T Type of the object to store (usually std::shared_ptr<T>).
     * @param name The name to associate with the object.
     * @param object The object to store.
     */
    template<typename T>
    void set(const std::string& name, T object);

    /**
     * @brief Store an object with an auto-generated name.
     *
     * @tparam T Type of the object to store (usually std::shared_ptr<T>).
     * @param object The object to store.
     * @return The generated unique name associated with the object.
     */
    template<typename T>
    std::string set(T object);

    /**
     * @brief Retrieve a stored object by name and type.
     *
     * @tparam T The type to cast the stored object to (usually the pointed-to type).
     * @param name The name associated with the object.
     * @return std::shared_ptr<T> if found and type matches, nullptr otherwise.
     */
    template<typename T>
    std::shared_ptr<T> get(const std::string& name) const;

    /**
     * @brief Remove an object from the registry by name.
     *
     * @param name The name of the object to remove.
     */
    void unset(const std::string& name);
};

}

#include "Stream/Registry/ViewRegistry.hpp"