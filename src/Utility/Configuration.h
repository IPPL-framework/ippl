#ifndef IPPL_CONFIGURATION_H
#define IPPL_CONFIGURATION_H

#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#include "Utility/ParameterList.h"

namespace ippl {

    class Configuration : public ParameterList {
    public:
        Configuration() = default;

        void parse(std::string fname) {
            // read file into stringstream
            // 20 August 2023, https://stackoverflow.com/a/132394
            std::ifstream file(fname);
            std::stringstream ss;
            if (file) {
                ss << file.rdbuf();
                file.close();
            }

            const std::regex re("(.*)=(.*)");
            std::smatch match;

            std::string key, val;

            std::string line = "";
            while (std::getline(ss, line)) {
                std::regex_match(line, match, re);
                // remove leading and trailing white space
                // 20 August 2023
                // https://stackoverflow.com/a/21815483
                key = std::regex_replace(match[1].str(), std::regex("^ +| +$|( ) +"), "$1");
                val = std::regex_replace(match[2].str(), std::regex("^ +| +$|( ) +"), "$1");

                this->replace(key, val);
            }
        }

    private:
        void replace(const std::string& key, const std::string& val) {
            if (!key.empty() && this->params_m.contains(key)) {
                if (std::holds_alternative<std::string>(this->params_m[key])) {
                    this->update<std::string>(key, val);
                } else if (std::holds_alternative<int>(this->params_m[key])) {
                    this->update<int>(key, std::stoi(val));
                } else if (std::holds_alternative<double>(this->params_m[key])) {
                    this->update<double>(key, std::stod(val));
                } else if (std::holds_alternative<float>(this->params_m[key])) {
                    this->update<float>(key, std::stof(val));
                } else if (std::holds_alternative<bool>(this->params_m[key])) {
                    bool b = false;
                    std::istringstream is(val);
                    is >> std::boolalpha >> b;
                    this->update<bool>(key, b);
                }
            }
        }
    };
}  // namespace ippl

#endif
