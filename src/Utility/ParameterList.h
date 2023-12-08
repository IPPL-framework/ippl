//
// Class ParameterList
//   Ippl related parameters (e.g. tolerance in case of iterative solvers).
//   Example:
//      ippl::ParameterList params;
//      params.add<double>("tolerance", 1.0e-8);
//      params.get<double>("tolerance");
//
//
#ifndef IPPL_PARAMETER_LIST_H
#define IPPL_PARAMETER_LIST_H

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <variant>

#include "Utility/IpplException.h"

namespace ippl {

    /*!
     * @file ParameterList.h
     * @class ParameterList
     */
    class ParameterList {
    public:
        // allowed parameter types
        using variant_t =
            std::variant<double, float, bool, std::string, unsigned int, int, ParameterList>;

        ParameterList()                     = default;
        ParameterList(const ParameterList&) = default;

        /*!
         * Add a single parameter to this list.
         * @param key is the name of the parameter
         * @param value is the parameter value
         */
        template <typename T>
        void add(const std::string& key, const T& value) {
            if (params_m.contains(key)) {
                throw IpplException("ParameterList::add()",
                                    "Parameter '" + key + "' already exists.");
            }
            params_m[key] = value;
        }

        /*!
         * Obtain the value of a parameter. This function
         * throws an error if the key is not contained.
         * @param key the name of the parameter
         * @returns the value of a parameter
         */
        template <typename T>
        T get(const std::string& key) const {
            if (!params_m.contains(key)) {
                throw IpplException("ParameterList::get()",
                                    "Parameter '" + key + "' not contained.");
            }
            return std::get<T>(params_m.at(key));
        }

        /*!
         * Obtain the value of a parameter. If the key is
         * not contained, the default value is returned.
         * @param key the name of the parameter
         * @param defval the default value of the parameter
         * @returns the value of a parameter
         */
        template <typename T>
        T get(const std::string& key, const T& defval) const {
            if (!params_m.contains(key)) {
                return defval;
            }
            return std::get<T>(params_m.at(key));
        }

        /*!
         * Merge a parameter list into this parameter list.
         * @param p the parameter list to merge into this
         */
        void merge(const ParameterList& p) noexcept {
            for (const auto& [key, value] : p.params_m) {
                params_m[key] = value;
            }
        }

        /*!
         * Update the parameter values of this list with the
         * values provided in the input parameter list.
         * @param p the input parameter list with update parameter values
         */
        void update(const ParameterList& p) noexcept {
            for (const auto& [key, value] : p.params_m) {
                if (params_m.contains(key)) {
                    params_m[key] = value;
                }
            }
        }

        /*!
         * Update the single parameter value of this list.
         * @param key is the name of the parameter
         * @param value is the parameter value
         */
        template <typename T>
        void update(const std::string& key, const T& value) {
            if (!params_m.contains(key)) {
                throw IpplException("ParameterList::update()",
                                    "Parameter '" + key + "' does not exist.");
            }
            params_m[key] = value;
        }

        template <class Stream>
        friend Stream& operator<<(Stream& stream, const ParameterList& sp) {
            std::cout << "HI" << std::endl;
            for (const auto& [key, value] : sp.params_m) {
                const auto& keyLocal = key;
                std::visit(
                    [&](auto&& arg) {
                        stream << std::make_pair(keyLocal, arg);
                    },
                    value);
            }

            return stream;
        }

        /*!
         * Print this parameter list.
         */
        friend std::ostream& operator<<(std::ostream& os, const ParameterList& sp) {
            static int indent = -4;

            indent += 4;
            if (indent > 0) {
                os << '\n';
            }
            for (const auto& [key, value] : sp.params_m) {
                const auto& keyLocal = key;
                std::visit(
                    [&](auto&& arg) {
                        // 21 March 2021
                        // https://stackoverflow.com/questions/15884284/c-printing-spaces-or-tabs-given-a-user-input-integer
                        os << std::string(indent, ' ') << std::left << std::setw(20) << keyLocal
                           << " " << arg;
                    },
                    value);
                // 21 March 2021
                // https://stackoverflow.com/questions/289715/last-key-in-a-stdmap
                if (key != std::prev(sp.params_m.end())->first) {
                    os << '\n';
                }
            }
            indent -= 4;

            return os;
        }
     ParameterList& operator=(const ParameterList& other) {
        if (this != &other) {
            // Copy members from 'other' to 'this'
            params_m = other.params_m;
        }
        return *this;
    }

    protected:
        std::map<std::string, variant_t> params_m;
    };
}  // namespace ippl

#endif
