#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <variant>

/*!
 * Value parser for configuration parsing; for a given, known parameter, parse the given value
 * @param value the value from the configuration file
 */
template <typename Config>
using ValueParser = void (Config::*)(const std::string& value);

/*!
 * Fallback handler for configuration parsing; if the key is not present in the configuration table,
 * attempt to parse the key and the value
 * @param key parameter name
 * @param value parameter value
 * @return Whether the parameter was successfully parsed
 */
template <typename Config>
using FallbackHandler = bool (Config::*)(const std::string& key, const std::string& value);

/*!
 * Accepted struct member types for parameter parsing
 * Literals: integers, booleans, reals, and unsigned (for large positive numbers)
 * Functions: value parser, fallback handler
 */
template <typename Config>
using ConfigItem =
    std::variant<int Config::*, bool Config::*, double Config::*, uint64_t Config::*,
                 std::string Config::*, ValueParser<Config>, FallbackHandler<Config>>;

/*!
 * Configuration parameter table; maps keys to struct members
 */
template <typename Config>
using ConfigParser = std::map<std::string, ConfigItem<Config>>;

/*!
 * Checks if a parameter should contain a literal value; if so, parse and store that value
 * @param param configuration parameter
 * @param value provided parameter value
 * @param config configuration struct
 */
template <typename Config>
bool checkLiteral(const ConfigItem<Config>& param, const std::string& value, Config& config) {
    if (std::holds_alternative<int Config::*>(param)) {
        auto p    = std::get<int Config::*>(param);
        config.*p = std::stoi(value);
    } else if (std::holds_alternative<double Config::*>(param)) {
        auto p    = std::get<double Config::*>(param);
        config.*p = std::stod(value);
    } else if (std::holds_alternative<uint64_t Config::*>(param)) {
        auto p    = std::get<uint64_t Config::*>(param);
        config.*p = std::stoll(value);
    } else if (std::holds_alternative<std::string Config::*>(param)) {
        auto p    = std::get<std::string Config::*>(param);
        config.*p = value;
    } else if (std::holds_alternative<bool Config::*>(param)) {
        auto p = std::get<bool Config::*>(param);
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

/*!
 * Fallback handler for unknown keys; throws an exception if a valid parameter cannot be read
 * @param key the unknown key
 * @param value provided parameter value
 * @param parser configuration parameter table
 * @param config configuration struct
 */
template <typename Config>
void parserFallback(const std::string& key, const std::string& value,
                    const ConfigParser<Config>& parser, Config& config) {
    if (parser.contains("")) {
        auto p = std::get<FallbackHandler<Config>>(parser.at(""));
        if (!(config.*p)(key, value)) {
            throw std::runtime_error("Invalid config: failed to parse value for key " + key);
        }
    } else {
        throw std::runtime_error("Invalid config: unknown key " + key);
    }
}

/*!
 * Parse configuration from stream, updating an existing configuration struct
 * @param in input stream
 * @param parser configuration parameter table
 * @param config configuration struct
 */
template <typename Config>
void parseConfig(std::istream& in, const ConfigParser<Config>& parser, Config& config) {
    static const auto trim = std::regex("^\\s+|\\s+$");

    std::string line;
    while (std::getline(in, line)) {
        bool found = false;
        auto split = line.find('=');
        if (split == std::string::npos) {
            throw std::runtime_error("Invalid config line: " + line);
        }
        auto key   = line.substr(0, split);
        key        = std::regex_replace(key, trim, "");
        auto value = line.substr(split + 1);
        value      = std::regex_replace(value, trim, "");
        for (const auto& [k, v] : parser) {
            if (key == k) {
                found = true;
                if (!checkLiteral(v, value, config)) {
                    if (std::holds_alternative<ValueParser<Config>>(v)) {
                        auto p = std::get<ValueParser<Config>>(v);
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

/*!
 * Parse configuration from stream
 * @param in input stream
 * @param parser configuration parameter table
 * @return Parsed configuration
 */
template <typename Config>
Config parseConfig(std::istream& in, const ConfigParser<Config>& parser) {
    Config ret;
    parseConfig(in, parser, ret);
    return ret;
}
