#include <map>
#include <stdexcept>
#include <string>
#include <variant>

template <typename Config>
using ConfigItem = std::variant<int Config::*, bool Config::*, double Config::*, uint64_t Config::*,
                                std::string Config::*, void (Config::*)(const std::string&),
                                bool (Config::*)(const std::string&, const std::string&)>;

template <typename Config>
using ConfigParser = std::map<std::string, ConfigItem<Config>>;

template <typename Config>
bool checkLiteral(const ConfigItem<Config>& v, const std::string& value, Config& config) {
    if (std::holds_alternative<int Config::*>(v)) {
        auto p    = std::get<int Config::*>(v);
        config.*p = std::stoi(value);
    } else if (std::holds_alternative<double Config::*>(v)) {
        auto p    = std::get<double Config::*>(v);
        config.*p = std::stod(value);
    } else if (std::holds_alternative<uint64_t Config::*>(v)) {
        auto p    = std::get<uint64_t Config::*>(v);
        config.*p = std::stoll(value);
    } else if (std::holds_alternative<std::string Config::*>(v)) {
        auto p    = std::get<std::string Config::*>(v);
        config.*p = value;
    } else if (std::holds_alternative<bool Config::*>(v)) {
        auto p = std::get<bool Config::*>(v);
        if (value == "true" || value == "yes") {
            config.*p = true;
        } else if (value == "false" || value == "no") {
            config.*p = false;
        } else {
            throw std::runtime_error("Expected boolean value, found " + value);
        }
    } else {
        return false;
    }
    return true;
}

template <typename Config>
void parserFallback(const std::string& key, const std::string& value,
                    const ConfigParser<Config>& parser, Config& config) {
    if (parser.contains("")) {
        auto p = std::get<bool (Config::*)(const std::string&, const std::string&)>(parser.at(""));
        if (!(config.*p)(key, value)) {
            throw std::runtime_error("Invalid config: failed to parse value for key " + key);
        }
    } else {
        throw std::runtime_error("Invalid config: unknown key " + key);
    }
}

template <typename Config>
void parseConfig(std::istream& in, const ConfigParser<Config>& parser, Config& config) {
    std::string line;
    while (std::getline(in, line)) {
        bool found = false;
        auto split = line.find('=');
        if (split == std::string::npos) {
            throw std::runtime_error("Invalid config line: " + line);
        }
        auto key   = line.substr(0, split);
        auto value = line.substr(split + 1);
        for (const auto& [k, v] : parser) {
            if (key == k) {
                found = true;
                if (!checkLiteral(v, value, config)) {
                    if (std::holds_alternative<void (Config::*)(const std::string&)>(v)) {
                        auto p = std::get<void (Config::*)(const std::string&)>(v);
                        (config.*p)(value);
                    } else {
                        throw std::runtime_error("Invalid parser: wrong handler for key " + key);
                    }
                }
                break;
            }
        }
        if (!found) {
            parserFallback(key, value, parser, config);
        }
    }
}

template <typename Config>
Config parseConfig(std::istream& in, const ConfigParser<Config>& parser) {
    Config ret;
    parseConfig(in, parser, ret);
    return ret;
}
