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

#include <iostream>
#include <iomanip>
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
        using variant_t = std::variant<double,
                                       float,
                                       bool,
                                       std::string,
                                       unsigned int,
                                       int>;

        template <typename T>
        void add(const std::string& key, const T& value) {
            params_m[key] = value;
        }

        /*!
         * Obtain the value of a parameter. This function
         * throws an error if the key is not contained.
         * @param key the name of the parameter
         * @returns the value of a parameter
         */
        template <typename T>
        T get(const std::string& key) {
            if (!params_m.contains(key)) {
                throw IpplException("ParameterList::get()",
                                    "Parameter '" + key + "' not contained.")
            }
            return std::get<T>(params_m[key]);
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
         * Print this parameter list.
         */
        friend
        std::ostream& operator<<(std::ostream& os, const ParameterList& sp) {
            for (const auto& [key, value] : sp.params_m) {
                std::visit([&](auto&& arg){
                    os << std::left << std::setw(20) << key
                       << " " << arg << '\n';
                }, value);
            }
            return os;
        }

    private:
        std::map<std::string, variant_t> params_m;
    };
}

#endif
