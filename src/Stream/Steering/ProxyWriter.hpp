// ProxyWriter.hpp - template definitions
#pragma once

#ifdef IPPL_PROXYWRITER_DECL

#include <algorithm>
#include <type_traits>

namespace ippl {

// Arithmetic scalar include
template <typename T>
inline void ProxyWriter::include(const T& defaultValue, const std::string& label) {
  static_assert(std::is_arithmetic_v<T>, "ProxyWriter::include requires a scalar arithmetic type");
  Channel ch; 
  ch.label = label; 
  // Parse label for array prefix and namespace; compute propertyName
  {
    ch.isArray = false;
    std::string work = label;
    if (work.rfind("array:", 0) == 0) { ch.isArray = true; work = work.substr(6); }
    ch.propertyName = work;
    auto dp = work.find('.');
    if (dp != std::string::npos) ch.ns = work.substr(0, dp); else ch.ns = work;
  }
  ch.defaultValue = static_cast<double>(defaultValue); ch.isVector = false; ch.vecDim = 1;
  // Mark integer scalars (exclude bool which is handled via includeBool)
  if constexpr (std::is_integral_v<T> && !std::is_same_v<std::decay_t<T>, bool>) {
    ch.isInteger = true;
  }
  // Heuristic: treat button-like labels as buttons (Int checkbox) even if numeric
  {
    auto endsWith = [](const std::string& s, const std::string& suf){ return s.size()>=suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf)==0; };
    if (endsWith(ch.propertyName, ".reset_btn") || endsWith(ch.propertyName, "_btn") || ch.propertyName.find("btn") != std::string::npos) {
      ch.isButton = true;
      ch.isVector = false;
    }
    // Heuristic: detect booleans when users accidentally call include<T> for bool arrays
    // Common patterns: names starting with 'bool_' or members ending with '.switch'
    if (!ch.isButton) {
      bool looksBool = false;
      if (ch.propertyName.rfind("bool_", 0) == 0) looksBool = true; // starts with bool_
      if (endsWith(ch.propertyName, ".switch")) looksBool = true;   // struct member switch
      if (looksBool) {
        ch.isBool = true;
        ch.isInteger = false; // treat as boolean domain rather than integer slider
      }
    }
  }
  if (hasConfig_m) {
    applyScalarConfig(ch);
  }
  channels_m.emplace_back(std::move(ch));
}

// Vector include (up to 3 components exposed)
template <typename T, unsigned Dim_V>
inline void ProxyWriter::includeVector(const std::string& label) {
  Channel ch; 
  ch.label = label; 
  // Parse label for array prefix and namespace; compute propertyName
  {
    ch.isArray = false;
    std::string work = label;
    if (work.rfind("array:", 0) == 0) { ch.isArray = true; work = work.substr(6); }
    ch.propertyName = work;
    auto dp = work.find('.');
    if (dp != std::string::npos) ch.ns = work.substr(0, dp); else ch.ns = work;
  }
  // Standard vector default changed from 1.0 to 0.0 per updated steering requirements
  ch.defaultValue = 0.0; ch.isVector = true; ch.vecDim = (Dim_V > 3 ? 3u : Dim_V);
  // Mark integer vectors when element type is integral (exclude bool)
  if constexpr (std::is_integral_v<T> && !std::is_same_v<std::decay_t<T>, bool>) {
    ch.isInteger = true;
  }
  if (hasConfig_m) {
    applyVectorConfig<Dim_V>(ch);
  }
  channels_m.emplace_back(std::move(ch));
}

// Apply vector config (ranges/defaults)
template <unsigned Dim_V>
inline void ProxyWriter::applyVectorConfig(Channel& ch) const {
  // Start with current defaults per component
  double d0 = ch.vecDefaults[0];
  double d1 = ch.vecDefaults[1];
  double d2 = ch.vecDefaults[2];

  // Only apply per-label overrides; do not emit global ranges by default
  auto it = labelVectorCfg_m.find(ch.label);
  if (it != labelVectorCfg_m.end()) {
    const VectorCfg& vc = it->second;
    if (vc.uniform) {
      d0 = vc.udef; d1 = vc.udef; d2 = vc.udef;
      ch.vecMin = vc.umin; ch.vecMax = vc.umax;
      ch.hasVectorRanges = true;
    }
    if (vc.comp[0].has) { d0 = vc.comp[0].def; ch.vecMin = vc.comp[0].min; ch.vecMax = vc.comp[0].max; ch.hasVectorRanges = true; }
    if (vc.comp[1].has) { d1 = vc.comp[1].def; ch.vecMin = vc.comp[1].min; ch.vecMax = vc.comp[1].max; ch.hasVectorRanges = true; }
    if (vc.comp[2].has) { d2 = vc.comp[2].def; ch.vecMin = vc.comp[2].min; ch.vecMax = vc.comp[2].max; ch.hasVectorRanges = true; }
  }

  ch.vecDefaults[0] = d0;
  ch.vecDefaults[1] = d1;
  ch.vecDefaults[2] = d2;
}

} // namespace ippl
#endif // IPPL_PROXYWRITER_DECL





