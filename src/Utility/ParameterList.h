//
// Class ParameterList
//   Ippl related parameters (e.g. tolerance in case of iterative solvers).
//   Example:
//      ippl::ParameterList params;
//      params.add<double>("tolerance", 1.0e-8);
//      params.get<double>("tolerance");
//
//
// Copyright (c) 2021, Matthias Frey, University of St Andrews, St Andrews, Scotland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_PARAMETER_LIST_H
#define IPPL_PARAMETER_LIST_H

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
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

        /*!
         * Add a single parameter to this list.
         * @param key is the name of the parameter
         * @param value is the parameter value
         */
        template <typename T>
        void add(const std::string& key, const T& value) {
#if __cplusplus > 201703L
            if (params_m.contains(key)) {
#else
            if (params_m.find(key) != params_m.end()) {
#endif
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
#if __cplusplus > 201703L
            if (!params_m.contains(key)) {
#else
            if (params_m.find(key) == params_m.end()) {
#endif
                throw IpplException("ParameterList::get()",
                                    "Parameter '" + key + "' not contained.");
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
#if __cplusplus > 201703L
                if (params_m.contains(key)) {
#else
                if (params_m.find(key) != params_m.end()) {
#endif
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
#if __cplusplus > 201703L
            if (!params_m.contains(key)) {
#else
            if (params_m.find(key) == params_m.end()) {
#endif
                throw IpplException("ParameterList::update()",
                                    "Parameter '" + key + "' does not exist.");
            }
            params_m[key] = value;
        }

        // The following commented portion has compiler errors with Intel and Clang
        // Disable parameter list printing for Cuda builds until
        // the lambda issue is resolved
        // #ifndef KOKKOS_ENABLE_CUDA
        //        /*!
        //         * Print this parameter list.
        //         */
        //        friend
        //        std::ostream& operator<<(std::ostream& os, const ParameterList& sp) {
        //            static int indent = -4;
        //
        //            indent += 4;
        //            if (indent > 0) {
        //                os << '\n';
        //            }
        //            for (const auto& [key, value] : sp.params_m) {
        //                std::visit([&](auto&& arg){
        //                    // 21 March 2021
        //                    //
        //                    https://stackoverflow.com/questions/15884284/c-printing-spaces-or-tabs-given-a-user-input-integer
        //                    os << std::string(indent, ' ')
        //                       << std::left << std::setw(20) << key
        //                       << " " << arg;
        //                }, value);
        //                // 21 March 2021
        //                // https://stackoverflow.com/questions/289715/last-key-in-a-stdmap
        //                if (key != std::prev(sp.params_m.end())->first) {
        //                    os << '\n';
        //                }
        //            }
        //            indent -= 4;
        //
        //            return os;
        //        }
        // #endif

    private:
        std::map<std::string, variant_t> params_m;
    };
}  // namespace ippl

#endif
