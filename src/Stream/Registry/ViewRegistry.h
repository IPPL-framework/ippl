#include <any>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>


/* registry creates copies of objects passed... */
/* for us only meaningful with shared pointers to keep copies of views and maps
alive and not being deleted ... */


class ViewRegistry {
private:
    std::unordered_map<std::string, std::any> m_storage;
    size_t m_unnamed_counter = 0;

public:
    // Overload 1: Takes a name and an object. Returns void.
    template<typename T>
    void set(const std::string& name, T object) {
        m_storage[name] = object;
    }

    // Overload 2: Takes only an object. Returns the generated name.
    template<typename T>
    std::string set(T object) {
        std::string generated_name = "__unnamed_" + std::to_string(m_unnamed_counter++);
        m_storage[generated_name] = object;
        return generated_name;
    }

    template<typename T>
    std::shared_ptr<T> get(const std::string& name) const {
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

    void unset(const std::string& name) {
        m_storage.erase(name);
    }
};


// // A safe, modern dynamic registry
// class ViewRegistry {
// private:
//     std::unordered_map<std::string, std::any> m_storage;

// public:
//     // Store any object that can be copied into an 'any'
//     template<typename T>
//     void set(const std::string& name, T object) {
//         m_storage[name] = object;
//     }

//     // Retrieve a shared_ptr of a specific type
//     template<typename T>
//     std::shared_ptr<T> get(const std::string& name) const {
//         auto it = m_storage.find(name);
//         if (it == m_storage.end()) {
//             return nullptr;
//         }
//         // Try to cast the 'any' back to the requested type.
//         // Returns nullptr on failure.
//         try {
//             return std::any_cast<std::shared_ptr<T>>(it->second);
//         } catch (const std::bad_any_cast&) {
//             return nullptr;
//         }
//     }

//     void unset(const std::string& name) {
//         m_storage.erase(name);
//     }
// };