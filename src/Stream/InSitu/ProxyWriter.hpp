#pragma once

#include <algorithm>
#include <type_traits>

namespace ippl {

// Template and inline definitions for ProxyWriter

template <typename T>
void ProxyWriter::include(const T& defaultValue, const std::string& label) {
  static_assert(std::is_arithmetic_v<T>, "ProxyWriter::include requires a scalar arithmetic type");
  Channel ch; ch.label = label; ch.defaultValue = static_cast<double>(defaultValue); ch.isVector = false; ch.vecDim = 1;
  if (hasConfig_) {
    applyScalarConfig(ch);
  }
  channels_.emplace_back(std::move(ch));
}

template <typename T, unsigned Dim_v>
void ProxyWriter::includeVector(const std::string& label) {
  (void)sizeof(T); // T is ignored at runtime; only used for type-checking in potential extensions
  Channel ch; ch.label = label; ch.defaultValue = 1.0; ch.isVector = true; ch.vecDim = (Dim_v > 3 ? 3u : Dim_v);
  if (hasConfig_) {
    applyVectorConfig<Dim_v>(ch);
  }
  channels_.emplace_back(std::move(ch));
}

template <unsigned Dim_v>
void ProxyWriter::applyVectorConfig(Channel& ch) const {
  // Start from type defaults if present
  double d0 = typeDefaultVectorComp_.has ? typeDefaultVectorComp_.def : ch.defaultValue;
  double d1 = typeDefaultVectorComp_.has ? typeDefaultVectorComp_.def : ch.defaultValue;
  double d2 = typeDefaultVectorComp_.has ? typeDefaultVectorComp_.def : ch.defaultValue;
  double vmin = typeDefaultVectorComp_.has ? typeDefaultVectorComp_.min : rangeMin_;
  double vmax = typeDefaultVectorComp_.has ? typeDefaultVectorComp_.max : rangeMax_;

  auto it = labelVectorCfg_.find(ch.label);
  if (it != labelVectorCfg_.end()) {
    const VectorCfg& vc = it->second;
    if (vc.uniform) {
      d0 = vc.udef; d1 = vc.udef; d2 = vc.udef;
      vmin = vc.umin; vmax = vc.umax;
    }
    if (vc.comp[0].has) { d0 = vc.comp[0].def; vmin = std::min(vmin, vc.comp[0].min); vmax = std::max(vmax, vc.comp[0].max); }
    if (vc.comp[1].has) { d1 = vc.comp[1].def; vmin = std::min(vmin, vc.comp[1].min); vmax = std::max(vmax, vc.comp[1].max); }
    if (vc.comp[2].has) { d2 = vc.comp[2].def; vmin = std::min(vmin, vc.comp[2].min); vmax = std::max(vmax, vc.comp[2].max); }
  }

  // Apply
  ch.hasVectorRanges = true;
  ch.vecDefaults[0] = d0;
  ch.vecDefaults[1] = d1;
  ch.vecDefaults[2] = d2;
  ch.vecMin = vmin;
  ch.vecMax = vmax;
}

} // namespace ippl
