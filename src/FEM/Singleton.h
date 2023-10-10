// Class Singleton

#ifndef IPPL_SINGLETON_H
#define IPPL_SINGLETON_H

namespace ippl {
    /**
     * @brief Singleton base class that can be inherited to have child classes and their children
     * use the singleton design pattern
     *
     * @tparam T template type for the CRTP (curiously recurring template pattern)
     */
    template <typename T>
    class Singleton {
    public:
        static T& getInstance() {
            static T instance;
            return instance;
        }

        static T* getInstancePtr() {
            static T instance;
            return &instance;
        }

    protected:
        Singleton()  = default;
        ~Singleton() = default;
    };
}  // namespace ippl

#endif