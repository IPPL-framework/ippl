/**
 * @file ProxyWriter.h
 * @brief Declarations for generating ParaView Catalyst proxy XML for steerable parameters.
 * This header contains declarations and small structs; templates are
 * defined in ProxyWriter.hpp and non-templates in ProxyWriter.cpp
 */
// ============= ProxyWriter: Declarations and Doxygen documentation =============

#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Conduit (for optional YAML-backed config of ranges/defaults)
#include <catalyst_conduit.hpp>

namespace ippl {
#define IPPL_PROXYWRITER_DECL 1

/**
 * @class ProxyWriter
 * @brief Generates ParaView Catalyst proxy XML for steerable parameters.
 *
 * - Call  initialize() to set output path and optional config
 * - Register channels via include()/includeVector()/includeBool()/includeEnum()/includeButton()
 * - Generate the XML file with produceUnified()
 *
 * YAML schema (implicit typing):
 * typeDefaults:
 *   scalar: { min: <num>, max: <num>, default: <num> }
 *   vector:
 *     component_defaults: { min: <num>, max: <num>, default: <num> }
 * steerParams:
 *   Efield: { min: <num>, max: <num>, default: <num> }
 *   Bfield:
 *     components:
 *       x: { min: <num>, max: <num>, default: <num> }
 *       y: { ... }
 *       z: { ... }
 *   # Or uniform vector defaults/range for all components
 *   V: { min: <num>, max: <num>, default: <num> }
 */
class ProxyWriter {
public:
  // ------------------------------- Channel model --------------------------------
  
  /**
   * @struct Channel
   * @brief Holds context and properties for a single registered steerable parameter or control.
   */
  struct Channel {
    
    // human label and suffix
    std::string label;
    
    // XML property name 
    // For arrays, this strips the 'array:' prefix.
    std::string propertyName;          

    // Namespace-group (struct or array name); for 'a.b' it's 'a', for 'a' it's 'a'
    std::string ns;


    bool        isArray{false};               // true when label starts with "array:"
    double      defaultValue{1.0};            // scalar default or seed for vectors
    bool        isVector{false};
    bool        isBool{false};
    bool        isButton{false};
    bool        isEnum{false};
    bool        isInteger{false};             // true for integral types excl. bool

    std::vector<std::pair<std::string,int>> enumEntries{};
    int         defaultInt{0};
    unsigned    vecDim{1};             // 1, 2, or 3 exposed components

    // Optional per-label overrides populated from YAML/Conduit config file
    bool        hasScalarRange{false};
    double      scalarMin{0.0};
    double      scalarMax{1.0};
    bool        hasScalarDefault{false};

    bool        hasVectorRanges{false};
    std::array<double,3> vecDefaults{ {1.0,1.0,1.0} };
    double      vecMin{0.0};
    double      vecMax{1.0};
  };

  // ---------------------------- Public API (decls) ------------------------------

  /**
   * @brief Initialize proxy writer.
   * If configYamlPath is empty, attempts to load default YAML (proxy_default_config.yaml).
   * No ranges are emitted by default; ranges will only be included in the prototype if the config provides them.
   * @param xmlOutputPath Path to the output XML directory or file.
   * @param configYamlPath Path to the YAML configuration file (optional).
   */
  void initialize(std::filesystem::path xmlOutputPath,
                  const std::string& configYamlPath = "");

  // Register steerable controls

  /**
   * @brief Register a scalar arithmetic channel.
   * @tparam T The arithmetic type of the parameter.
   * @param defaultValue The default simulation value.
   * @param label UI label and XML property name.
   */
  template <typename T>
  void include(const T& defaultValue, const std::string& label);


  /**
   * @brief Register a vector channel (up to 3 components exposed).
   * @tparam T The arithmetic type of the vector components.
   * @tparam Dim_V The number of dimensions to expose.
   * @param label UI label and XML property name.
   */
  template <typename T, unsigned Dim_V>
  void includeVector(const std::string& label);

  /**
   * @brief Register a boolean switch channel.
   * @param label UI label and XML property name.
   * @param defaultValue Initial value if not overridden by config.
   */
  void includeBool(const std::string& label, bool defaultValue = false);


  /**
   * @brief Register a momentary button channel (edge-triggered action).
   * @param label UI label and XML property name.
   */
  void includeButton(const std::string& label);


  /**
   * @brief Register an enum channel with named entries.
   * @param label UI label and XML property name.
   * @param entries Vector of (display name, integer value) pairs.
   * @param defaultValue Initial selected entry index or value, depending on consumer.
   */
  void includeEnum(const std::string& label,
                   const std::vector<std::pair<std::string,int>>& entries,
                   int defaultValue = 0);

  ///////////////////////////////
  // LMaybe it's best if we remove this functionality
  // in its entirety and simply stick with the one for
  // all proxy file options.
  // /**
  //  * @brief Produce one XML file per registered channel.
  //  * @return true on success, false otherwise.
  //  */
  // bool produce();


  /**
   * @brief Produce a single XML containing a unified source proxy encompassing all channels.
   * @param unifiedProxyName XML proxy name.
   * @param unifiedGroupLabel Group label in ParaView.
   * @return true on success, false otherwise.
   */
  bool produceUnified(const std::string& unifiedProxyName,
                      const std::string& unifiedGroupLabel);


   /**
    * @brief Record desired initial tuple count for a given array namespace (e.g., "LinMaps").
    * @param ns The namespace of the array elements.
    * @param n The initial size or count of elements.
    */
   void setArrayInitialSize(const std::string& ns, std::size_t n) { arrayInitialSize_m[ns] = static_cast<unsigned>(n); }

private:

  // Internal XML builders and helpers (implemented in ProxyWriter.cpp)

  /**
   * @brief Clears and resets all internal string streams used for XML generation.
   */
  void resetStreams();

  // only used by the deprecated produce() method
  // /**
  //  * @brief Appends a source proxy XML definition for a single registered channel.
  //  * @param ch The channel for which to generate the proxy.
  //  */
  // void appendSourceProxy(const Channel& ch);

  /**
   * @brief Generates the PropertyCollectionWidgetPrototype containing ranges, default values, and UI hints for all parameters.
   */
  void appendPrototype();

  /**
   * @brief Builds a single unified SourceProxy encompassing all non-array channels.
   * @param proxyName The name of the resulting Proxy.
   * @param groupLabel The UI label for the proxy.
   */
  void appendUnifiedSourceProxy(const std::string& proxyName,
                                const std::string& groupLabel);

  /**
   * @brief Builds one per-array source proxy for channels sharing the same namespace.
   * @param ns The namespace of the array elements.
   * @param chans Vector of pointers to the channels in this array namespace.
   */
  void appendArraySourceProxy(const std::string& ns,
                              const std::vector<const Channel*>& chans);

  /**
   * @brief Reads and parses steering configurations (e.g. ranges, defaults) from a YAML file.
   * @param path The path to the YAML file.
   * @return true if successfully loaded and parsed, false otherwise.
   */
  bool loadConfigFromYamlFile(const std::string& path);

  /**
   * @brief Parses steering configurations directly from a YAML formatted string.
   * @param yaml The YAML formatted string.
   * @return true if successfully parsed, false otherwise.
   */
  bool loadConfigFromYamlString(const std::string& yaml);

  /**
   * @brief Applies parsed scalar configuration data (min, max, defaults) to a specific channel.
   * @param ch The channel to update with configuration data.
   */
  void applyScalarConfig(Channel& ch) const;
  
  /**
   * @brief Applies parsed vector configuration data to a specific vector channel.
   * @tparam Dim_V The number of vector dimensions.
   * @param ch The channel to update with configuration data.
   */
  template<unsigned Dim_V>
  void applyVectorConfig(Channel& ch) const;

  // Parsed config cache (populated by YAML loader)
  struct ScalarCfg { bool has=false; double min=0, max=0, def=0; };
  struct VectorCfg {
    bool has=false; // at least one component or uniform present
    bool uniform=false; double umin=0, umax=0, udef=0;
    ScalarCfg comp[3];
  };

  // ------------------------------ Data members ---------------------------------
  std::filesystem::path outPath_m{};

  std::vector<Channel> channels_m{};
  std::ostringstream header_m{};
  std::ostringstream sources_m{};
  std::ostringstream misc_m{};
  std::ostringstream footer_m{};

  bool hasConfig_m{false};
  std::map<std::string, ScalarCfg> labelScalarCfg_m{};
  std::map<std::string, VectorCfg> labelVectorCfg_m{};

  // Initial lengths for array namespaces; used to pre-populate default_values count
  std::map<std::string, unsigned> arrayInitialSize_m{};
}; // PoxyyWriter
} // namespace ippl

// Provide template definitions (keep at global scope to avoid injecting std headers into namespace ippl)
#include "ProxyWriter.hpp"

