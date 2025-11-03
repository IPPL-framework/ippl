#include "ProxyWriter.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cctype>
#include <algorithm>

namespace {
inline std::string ltrim(std::string s) {
  size_t i = 0; while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; return s.substr(i);
}
inline std::string rtrim(std::string s) {
  size_t i = s.size(); while (i > 0 && std::isspace(static_cast<unsigned char>(s[i-1]))) --i; s.resize(i); return s;
}
inline std::string trim(std::string s) { return rtrim(ltrim(std::move(s))); }
inline bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

inline std::string strip_comment(const std::string& s) {
  // Very naive: drop everything after an unbraced '#'
  // We'll ignore '#' if it appears inside { } to keep inline maps intact
  int brace = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '{') ++brace; else if (s[i] == '}') brace = std::max(0, brace-1);
    else if (s[i] == '#' && brace == 0) return s.substr(0, i);
  }
  return s;
}

inline int leading_spaces(const std::string& s) {
  int n = 0; while (n < (int)s.size() && s[n] == ' ') ++n; return n;
}

inline std::string unquote(std::string s) {
  s = trim(std::move(s));
  if (!s.empty() && (s.front() == '"' || s.front() == '\'')) s.erase(s.begin());
  if (!s.empty() && (s.back() == '"' || s.back() == '\'')) s.pop_back();
  return s;
}

inline bool extract_number(const std::string& src, const std::string& key, double& out) {
  // find "key:" and parse following number until a delimiter (',' or '}' or end)
  const std::string pat = key + ":";
  auto pos = src.find(pat);
  if (pos == std::string::npos) return false;
  pos += pat.size();
  // skip spaces
  while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) ++pos;
  // capture sign and digits
  size_t end = pos;
  bool dotSeen = false; bool expSeen = false;
  if (end < src.size() && (src[end] == '+' || src[end] == '-')) ++end;
  while (end < src.size()) {
    char c = src[end];
    if (std::isdigit(static_cast<unsigned char>(c))) { ++end; continue; }
    if (c == '.' && !dotSeen) { dotSeen = true; ++end; continue; }
    if ((c == 'e' || c == 'E') && !expSeen) { expSeen = true; ++end; if (end < src.size() && (src[end] == '+' || src[end] == '-')) ++end; continue; }
    break;
  }
  if (end == pos) return false;
  try {
    out = std::stod(src.substr(pos, end - pos));
    return true;
  } catch (...) { return false; }
}

} // anonymous namespace

namespace ippl {

// -------------------------------- Initialization ---------------------------------

void ProxyWriter::initialize(std::filesystem::path xmlOutputPath,
                             double rangeMin,
                             double rangeMax,
                             std::string prototypeLabel) {
  outPath_ = std::move(xmlOutputPath);
  rangeMin_ = rangeMin;
  rangeMax_ = rangeMax;
  prototypeLabel_ = std::move(prototypeLabel);
  resetStreams();
}

void ProxyWriter::initialize(std::filesystem::path xmlOutputPath,
                             const std::string& configYamlPath,
                             double rangeMin,
                             double rangeMax,
                             std::string prototypeLabel) {
  initialize(std::move(xmlOutputPath), rangeMin, rangeMax, std::move(prototypeLabel));
  if (!configYamlPath.empty()) {
    loadConfigFromYamlFile(configYamlPath);
  }
}

// Convenience overload: only output path and YAML config path.
// Uses existing defaults for rangeMin_/rangeMax_ and prototypeLabel_.
void ProxyWriter::initialize(std::filesystem::path xmlOutputPath,
                             const std::string& configYamlPath) {
  // Keep previously configured or default ranges/label
  outPath_ = std::move(xmlOutputPath);
  resetStreams();
  if (!configYamlPath.empty()) {
    // Heuristic: if string looks like YAML content (contains newlines or typical keys),
    // treat as YAML text; otherwise treat as a file path.
    bool looksLikeYaml = (configYamlPath.find('\n') != std::string::npos) ||
                         (configYamlPath.find("type_defaults:") != std::string::npos) ||
                         (configYamlPath.find("steer_params:") != std::string::npos) ||
                         (configYamlPath.find(":") != std::string::npos && configYamlPath.size() > 20);
    if (looksLikeYaml) {
      loadConfigFromYamlString(configYamlPath);
    } else {
      loadConfigFromYamlFile(configYamlPath);
    }
  }
}

void ProxyWriter::initialize(std::filesystem::path xmlOutputPath,
                             const conduit_cpp::Node configNode,
                             double rangeMin,
                             double rangeMax,
                             std::string prototypeLabel) {
  initialize(std::move(xmlOutputPath), rangeMin, rangeMax, std::move(prototypeLabel));
  setConfigNode(configNode);
}

// --------------------------------- Inclusions ------------------------------------

void ProxyWriter::includeBool(const std::string& label, bool defaultValue) {
  Channel ch; ch.label = label; ch.defaultValue = defaultValue ? 1.0 : 0.0; ch.isVector = false; ch.isBool = true; ch.vecDim = 1;
  channels_.emplace_back(std::move(ch));
}

void ProxyWriter::includeButton(const std::string& label) {
  Channel ch; ch.label = label; ch.defaultValue = 0.0; ch.isVector = false; ch.isBool = false; ch.isButton = true; ch.vecDim = 1;
  channels_.emplace_back(std::move(ch));
}

void ProxyWriter::includeEnum(const std::string& label,
                              const std::vector<std::pair<std::string,int>>& entries,
                              int defaultValue) {
  Channel ch; ch.label = label; ch.isEnum = true; ch.defaultInt = defaultValue; ch.vecDim = 1;
  ch.enumEntries = entries;
  channels_.emplace_back(std::move(ch));
}

// ------------------------------- Produce (XML) -----------------------------------

bool ProxyWriter::produce() {
  resetStreams();

  // Header
  header_ << "<ServerManagerConfiguration>\n\n";

  // Sources group with one SourceProxy per channel
  sources_ << "    <ProxyGroup name='sources'>\n";
  for (const auto& ch : channels_) {
    appendSourceProxy(ch);
  }
  sources_ << "    </ProxyGroup>\n\n";

  // Misc group with a single prototype used by PropertyCollection
  misc_ << "    <ProxyGroup name='misc'>\n";
  appendPrototype();
  misc_ << "    </ProxyGroup>\n";

  footer_ << "</ServerManagerConfiguration>\n";

  // Stitch everything into a single buffer
  std::ostringstream full;
  full << header_.str() << sources_.str() << misc_.str() << footer_.str();

  // Ensure directory exists and write file
  std::error_code ec;
  std::filesystem::create_directories(outPath_.parent_path(), ec);
  std::ofstream ofs(outPath_);
  if (!ofs) return false;
  ofs << full.str();
  ofs.close();
  return ofs.good();
}

bool ProxyWriter::produceUnified(const std::string& unifiedProxyName,
                                 const std::string& unifiedGroupLabel) {
  resetStreams();

  // Header
  header_ << "<ServerManagerConfiguration>\n\n";

  sources_ << "    <ProxyGroup name='sources'>\n";
  appendUnifiedSourceProxy(unifiedProxyName, unifiedGroupLabel);
  sources_ << "    </ProxyGroup>\n\n";

  misc_ << "    <ProxyGroup name='misc'>\n";
  appendPrototype();
  misc_ << "    </ProxyGroup>\n";

  footer_ << "</ServerManagerConfiguration>\n";

  // Compose full XML
  std::ostringstream full;
  full << header_.str() << sources_.str() << misc_.str() << footer_.str();

  std::error_code ec;
  std::filesystem::create_directories(outPath_.parent_path(), ec);
  std::ofstream ofs(outPath_);
  if (!ofs) return false;
  ofs << full.str();
  ofs.close();
  return ofs.good();
}

// ---------------------------- Internal XML builders ------------------------------

void ProxyWriter::resetStreams() {
  header_.str(""); header_.clear();
  sources_.str(""); sources_.clear();
  misc_.str(""); misc_.clear();
  footer_.str(""); footer_.clear();
}

void ProxyWriter::appendSourceProxy(const Channel& ch) {
  const std::string& L = ch.label;
  sources_ << "        <SourceProxy class='vtkSteeringDataGenerator' name='SteerableParameters_" << L << "'>\n"
           << "            <IntVectorProperty name='PartitionType' command='SetPartitionType' number_of_elements='1' default_values='1' panel_visibility='never'>\n"
           << "            </IntVectorProperty>\n\n"
           << "            <IntVectorProperty name='FieldAssociation' command='SetFieldAssociation' number_of_elements='1' default_values='0' panel_visibility='never'>\n"
           << "            </IntVectorProperty>\n\n"
           << "            <DoubleVectorProperty name='scaleFactor'\n"
           << "                                  command='SetTuple1Double'\n"
           << "                                  clean_command='Clear'\n"
           << "                                  use_index='1'\n"
           << "                                  initial_string='steerable_field_b_" << L << "'\n"
           << "                                  number_of_elements_per_command='1'\n"
           << "                                  repeat_command='1'\n"
           << "                                  panel_widget='DoubleRange'>\n";
  sources_ << "              <DoubleRangeDomain name='range' min='" << (ch.hasScalarRange ? ch.scalarMin : rangeMin_)
           << "' max='" << (ch.hasScalarRange ? ch.scalarMax : rangeMax_) << "'/>\n";
  sources_ << "            </DoubleVectorProperty>\n\n";
  if(!ch.isBool){
    sources_ << "            <PropertyGroup label='SteerableParameters' panel_widget='PropertyCollection'>\n";
  }
  sources_ << "                <Hints>\n"
           << "                  <PropertyCollectionWidgetPrototype group='misc' name='SteerableParametersPrototype' />\n"
           << "                </Hints>\n"
           << "            </PropertyGroup>\n\n"
           << "            <Hints>\n"
           << "              <CatalystInitializePropertiesWithMesh mesh='steerable_channel_0D_mesh'>\n"
           << "                <Property name='scaleFactor' association='point' array='steerable_field_f_" << L << "' />\n"
           << "              </CatalystInitializePropertiesWithMesh>\n"
           << "            </Hints>\n"
           << "        </SourceProxy>\n\n";
}

void ProxyWriter::appendPrototype() {
  // Prototype for numeric steerables (scalars and vectors)
  misc_ << "      <Proxy name='SteerableNumericsPrototype' label=' Numerics-Collective-Prototype (do not cancel [x] or add new [+]!!)'> \n";
  for (const auto& ch : channels_) {
    if (ch.isBool || ch.isButton || ch.isEnum) continue;
    if (ch.isVector) {
      const double d0 = ch.hasVectorRanges ? ch.vecDefaults[0] : ch.defaultValue;
      const double d1 = ch.hasVectorRanges ? ch.vecDefaults[1] : ch.defaultValue;
      const double d2 = ch.hasVectorRanges ? ch.vecDefaults[2] : ch.defaultValue;
      const double vmin = ch.hasVectorRanges ? ch.vecMin : rangeMin_;
      const double vmax = ch.hasVectorRanges ? ch.vecMax : rangeMax_;
  misc_ << "        <DoubleVectorProperty name='vec3_" << ch.label << "' label='" << ch.label << "' number_of_elements='3' default_values='" << d0 << " " << d1 << " " << d2 << "'>\n";
  misc_ << "          <DoubleRangeDomain name='range' min='" << vmin << "' max='" << vmax << "'/>\n";
      misc_ << "          <Hints>\n";
      misc_ << "            <ShowComponentLabels>\n";
  misc_ << "              <ComponentLabel component='0' label='X'/>\n";
  misc_ << "              <ComponentLabel component='1' label='Y'/>\n";
  misc_ << "              <ComponentLabel component='2' label='Z'/>\n";
      misc_ << "            </ShowComponentLabels>\n";
      misc_ << "          </Hints>\n";
      misc_ << "        </DoubleVectorProperty>\n";
    } else {
      const double sdef = ch.defaultValue;
      const double smin = ch.hasScalarRange ? ch.scalarMin : rangeMin_;
      const double smax = ch.hasScalarRange ? ch.scalarMax : rangeMax_;
  misc_ << "        <DoubleVectorProperty name='scaleFactor_" << ch.label << "' label='" << ch.label << "' number_of_elements='1' default_values='" << sdef << "' panel_widget='DoubleRange'>\n";
  misc_ << "          <DoubleRangeDomain name='range' min='" << smin << "' max='" << smax << "'/>\n";
      misc_ << "        </DoubleVectorProperty>\n";
    }
  }
  misc_ << "      </Proxy>\n\n";

  // Prototype for enum steerables
  misc_ << "      <Proxy name='SteerableEnumsPrototype' label='Enums-Collective-Prototype (do not cancel [x] or add new [+]!!)'>\n";
  for (const auto& ch : channels_) {
    if (!ch.isEnum || ch.enumEntries.empty()) continue;
  misc_ << "        <IntVectorProperty name='PrototypeEnum_" << ch.label << "' label='" << ch.label << "' number_of_elements='1' default_values='" << ch.defaultInt << "' immediate_apply='1'>\n";
  misc_ << "          <EnumerationDomain name='enum'>\n";
    for (const auto& [text, val] : ch.enumEntries) {
  misc_ << "            <Entry text='" << text << "' value='" << val << "'/>\n";
    }
    misc_ << "          </EnumerationDomain>\n";
    misc_ << "        </IntVectorProperty>\n";
  }
  misc_ << "      </Proxy>\n";
}

void ProxyWriter::appendUnifiedSourceProxy(const std::string& proxyName,
                                           const std::string& groupLabel) {
  (void)groupLabel; // currently unused in XML
  sources_ << "        <SourceProxy class='vtkSteeringDataGenerator' name='" << proxyName << "'>\n"
           << "            <IntVectorProperty name='PartitionType' command='SetPartitionType' number_of_elements='1' default_values='1' panel_visibility='never'>\n"
           << "            </IntVectorProperty>\n\n"
           << "            <IntVectorProperty name='FieldAssociation' command='SetFieldAssociation' number_of_elements='1' default_values='0' panel_visibility='never'>\n"
           << "            </IntVectorProperty>\n\n";

  // Add properties per channel
  for (const auto& ch : channels_) {
    const std::string& L = ch.label;
    if (ch.isBool) {
      sources_ << "            <IntVectorProperty name='" << L << "' label='" << L << "'\n"
               << "                                  command='SetTuple1Int'\n"
               << "                                  clean_command='Clear'\n"
               << "                                  use_index='1'\n"
               << "                                  number_of_elements='1'\n"
               << "                                  initial_string='steerable_field_b_" << L << "'\n"
               << "                                  default_values='" << (ch.defaultValue != 0.0 ? 1 : 0) << "'\n"
               << "                                  number_of_elements_per_command='1'\n"
               << "                                  repeat_command='1'\n"
               << "                                  panel_widget='CheckBox'>\n"
               << "              <BooleanDomain name='bool'/>\n"
               << "            </IntVectorProperty>\n\n";
    } else if (ch.isButton) {
      sources_ << "            <IntVectorProperty name='" << L << "' label='" << L << " '\n"
               << "                                  command='SetTuple1Int'\n"
               << "                                  clean_command='Clear'\n"
               << "                                  use_index='1'\n"
               << "                                  number_of_elements='1'\n"
               << "                                  initial_string='steerable_field_b_" << L << "'\n"
               << "                                  default_values='0'\n"
               << "                                  number_of_elements_per_command='1'\n"
               << "                                  repeat_command='1'\n"
               << "                                  immediate_apply='1'\n"
               << "                                  panel_widget='CheckBox'>\n"
               << "              <BooleanDomain name='bool'/>\n"
               << "              <Documentation>\n"
               << "                Check this box to trigger the button action. The simulation will automatically uncheck it internally  after processing.\n"
               << "              </Documentation>\n"
               << "            </IntVectorProperty>\n\n";
    } else if (ch.isEnum) {
      sources_ << "            <IntVectorProperty name='" << L << "'\n"
               << "                                  command='SetTuple1Int'\n"
               << "                                  clean_command='Clear'\n"
               << "                                  use_index='1'\n"
               << "                                  initial_string='steerable_field_b_" << L << "'\n"
               << "                                  number_of_elements_per_command='1'\n"
               << "                                  repeat_command='1'\n"
               << "                                  \n"
               << "                                  number_of_elements='1'\n"
               << "                                  default_values='" << ch.defaultInt << "'\n"
               << "                                  immediate_apply='1'\n"
               << "                                  \n"
               << "                                  >\n"
               << "            </IntVectorProperty>\n\n";
    } else if (!ch.isVector) {
      const double smin = ch.hasScalarRange ? ch.scalarMin : rangeMin_;
      const double smax = ch.hasScalarRange ? ch.scalarMax : rangeMax_;
      const double sdef = ch.defaultValue;
      sources_ << "            <DoubleVectorProperty name='" << L << "' label='" << L << "'\n"
               << "                                  command='SetTuple1Double'\n"
               << "                                  clean_command='Clear'\n"
               << "                                  use_index='1'\n"
               << "                                  number_of_elements='1'\n"
               << "                                  initial_string='steerable_field_b_" << L << "'\n"
               << "                                  default_values='" << sdef << "'\n"
               << "                                  number_of_elements_per_command='1'\n"
               << "                                  repeat_command='1'\n"
               << "                                  panel_widget='DoubleRange'>\n"
               << "              <DoubleRangeDomain name='range' min='" << smin << "' max='" << smax << "'/>\n"
               << "            </DoubleVectorProperty>\n\n";
    } else {
      const double d0 = ch.hasVectorRanges ? ch.vecDefaults[0] : ch.defaultValue;
      const double d1 = ch.hasVectorRanges ? ch.vecDefaults[1] : ch.defaultValue;
      const double d2 = ch.hasVectorRanges ? ch.vecDefaults[2] : ch.defaultValue;
      const double vmin = ch.hasVectorRanges ? ch.vecMin : rangeMin_;
      const double vmax = ch.hasVectorRanges ? ch.vecMax : rangeMax_;
      sources_ << "            <DoubleVectorProperty name='" << L << "' label='" << L << "'\n"
               << "                                  command='SetTuple3Double'\n"
               << "                                  use_index='1'\n"
               << "                                  clean_command='Clear'\n"
               << "                                  initial_string='steerable_field_b_" << L << "'\n"
               << "                                  number_of_elements='3'\n"
               << "                                  default_values='" << d0 << " " << d1 << " " << d2 << "'\n"
               << "                                  number_of_elements_per_command='3'\n"
               << "                                  repeat_command='1'>\n"
               << "              <DoubleRangeDomain name='range' min='" << vmin << "' max='" << vmax << "'/>\n"
               << "              <Hints>\n"
               << "                <ShowComponentLabels>\n"
               << "                  <ComponentLabel component='0' label='X'/>\n"
               << "                  <ComponentLabel component='1' label='Y'/>\n"
               << "                  <ComponentLabel component='2' label='Z'/>\n"
               << "                </ShowComponentLabels>\n"
               << "              </Hints>\n"
               << "            </DoubleVectorProperty>\n\n";
    }
  }

  // Property groups
  bool hasNumerics = false;
  for (const auto& ch : channels_) {
    if (!ch.isBool && !ch.isButton && !ch.isEnum) { hasNumerics = true; break; }
  }
  if (hasNumerics) {
    sources_ << "            <PropertyGroup label='Numerics' panel_widget='PropertyCollection'>\n";
    sources_ << "                <Hints>\n";
    sources_ << "                  <PropertyCollectionWidgetPrototype group='misc' name='SteerableNumericsPrototype' />\n";
    sources_ << "                </Hints>\n";
    for (const auto& ch : channels_) {
      if (ch.isBool || ch.isButton || ch.isEnum) continue;
      if (ch.isVector) {
        sources_ << "                <Property name='" << ch.label << "' function='vec3_" << ch.label << "' label='" << ch.label << "'/>\n";
      } else {
        sources_ << "                <Property name='" << ch.label << "' function='scaleFactor_" << ch.label << "' label='" << ch.label << "'/>\n";
      }
    }
    sources_ << "            </PropertyGroup>\n\n";
  }

  bool hasEnums = false;
  for (const auto& ch : channels_) { if (ch.isEnum && !ch.enumEntries.empty()) { hasEnums = true; break; } }
  if (hasEnums) {
    sources_ << "            <PropertyGroup label='Enums' panel_widget='PropertyCollection'>\n";
    sources_ << "                <Hints>\n";
    sources_ << "                  <PropertyCollectionWidgetPrototype group='misc' name='SteerableEnumsPrototype' />\n";
    sources_ << "                </Hints>\n";
    for (const auto& ch : channels_) {
      if (!ch.isEnum || ch.enumEntries.empty()) continue;
      sources_ << "                <Property name='" << ch.label << "' function='PrototypeEnum_" << ch.label << "' label='" << ch.label << "'/>\n";
    }
    sources_ << "            </PropertyGroup>\n\n";
  }

  bool hasSwitches = false;
  for (const auto& ch : channels_) { if (ch.isBool) { hasSwitches = true; break; } }
  if (hasSwitches) {
    sources_ << "            <PropertyGroup label='Switches'>\n";
    for (const auto& ch : channels_) {
      if (!ch.isBool) continue;
      sources_ << "                <Property name='" << ch.label << "' function='bool' label='" << ch.label << "' />\n";
    }
    sources_ << "            </PropertyGroup>\n\n";
  }

  bool hasButtons = false;
  for (const auto& ch : channels_) { if (ch.isButton) { hasButtons = true; break; } }
  if (hasButtons) {
    sources_ << "            <PropertyGroup label='Buttons / Triggers'>\n";
    for (const auto& ch : channels_) {
      if (!ch.isButton) continue;
      sources_ << "                <Property name='" << ch.label <<  "' />\n";
    }
    sources_ << "            </PropertyGroup>\n\n";
  }

  sources_ << "            <Hints>\n";
  sources_ << "              <CatalystInitializePropertiesWithMesh mesh='steerable_channel_0D_mesh'>\n";
  for (const auto& ch : channels_) {
    if (ch.isButton) continue;
    const std::string& L = ch.label;
    sources_ << "                <Property name='" << L << "' association='point' array='steerable_field_f_" << L << "' />\n";
  }
  sources_ << "              </CatalystInitializePropertiesWithMesh>\n";
  sources_ << "            </Hints>\n";

  sources_ << "        </SourceProxy>\n\n";
}

// ------------------------------ Config management --------------------------------

void ProxyWriter::setConfigNode(const conduit_cpp::Node n) {
  // Deep copy via YAML serialization, then re-parse into our caches
  std::string ys;
  try {
    ys = n.to_yaml();
  } catch (...) {
    ys.clear();
  }
  if (!ys.empty()) {
    loadConfigFromYamlString(ys);
  }
}

void ProxyWriter::applyScalarConfig(Channel& ch) const {
  // Start from type default if present
  if (typeDefaultScalar_.has) {
    ch.hasScalarRange = true;
    ch.scalarMin = typeDefaultScalar_.min;
    ch.scalarMax = typeDefaultScalar_.max;
    ch.defaultValue = typeDefaultScalar_.def;
    ch.hasScalarDefault = true;
  }
  auto it = labelScalarCfg_.find(ch.label);
  if (it != labelScalarCfg_.end() && it->second.has) {
    ch.hasScalarRange = true;
    ch.scalarMin = it->second.min;
    ch.scalarMax = it->second.max;
    ch.defaultValue = it->second.def;
    ch.hasScalarDefault = true;
  }
}

static void parse_inline_map_into(const std::string& inlineMap,
                                  bool& hasAny,
                                  double& vmin, double& vmax, double& vdef) {
  hasAny = false;
  std::string s = inlineMap;
  // ensure braces removed
  auto lb = s.find('{'); if (lb != std::string::npos) s.erase(0, lb+1);
  auto rb = s.rfind('}'); if (rb != std::string::npos) s.erase(rb);
  s = trim(s);
  double tmp;
  if (extract_number(s, "min", tmp)) { vmin = tmp; hasAny = true; }
  if (extract_number(s, "max", tmp)) { vmax = tmp; hasAny = true; }
  if (extract_number(s, "default", tmp)) { vdef = tmp; hasAny = true; }
}

bool ProxyWriter::loadConfigFromYamlFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::ostringstream ss; ss << ifs.rdbuf();
  return loadConfigFromYamlString(ss.str());
}

bool ProxyWriter::loadConfigFromYamlString(const std::string& yaml) {
  // Reset caches
  hasConfig_ = false;
  typeDefaultScalar_ = {};
  typeDefaultVectorComp_ = {};
  labelScalarCfg_.clear();
  labelVectorCfg_.clear();

  enum class Mode { None, TypeDefaults, TypeVector, SteerParams, Label, Components };
  Mode mode = Mode::None;
  std::string currentLabel;

  std::istringstream iss(yaml);
  std::string rawLine;
  while (std::getline(iss, rawLine)) {
    std::string line = strip_comment(rawLine);
    line = rtrim(line);
    if (line.find_first_not_of(' ') == std::string::npos) continue; // empty
    int indent = leading_spaces(line);
    std::string t = trim(line);

    if (indent == 0 && t == "type_defaults:") { mode = Mode::TypeDefaults; continue; }
    if (indent == 0 && t == "steer_params:") { mode = Mode::SteerParams; currentLabel.clear(); continue; }

    // type_defaults subtree
    if (mode == Mode::TypeDefaults) {
      if (indent == 2 && starts_with(t, "scalar:")) {
        // inline map expected on same line
        auto pos = t.find(':');
        std::string rest = (pos != std::string::npos) ? trim(t.substr(pos+1)) : std::string();
        bool has=false; double mn=0,mx=0,df=0;
        parse_inline_map_into(rest, has, mn, mx, df);
        if (has) { typeDefaultScalar_.has = true; typeDefaultScalar_.min = mn; typeDefaultScalar_.max = mx; typeDefaultScalar_.def = df; hasConfig_ = true; }
        continue;
      }
      if (indent == 2 && starts_with(t, "vector:")) { mode = Mode::TypeVector; continue; }
      // fallthrough for other lines
    }
    if (mode == Mode::TypeVector) {
      if (indent == 4 && starts_with(t, "component_defaults:")) {
        auto pos = t.find(':'); std::string rest = (pos != std::string::npos) ? trim(t.substr(pos+1)) : std::string();
        bool has=false; double mn=0,mx=0,df=0;
        parse_inline_map_into(rest, has, mn, mx, df);
        if (has) { typeDefaultVectorComp_.has = true; typeDefaultVectorComp_.min = mn; typeDefaultVectorComp_.max = mx; typeDefaultVectorComp_.def = df; hasConfig_ = true; }
        continue;
      }
      // If we dedent back to 0, we're out
      if (indent == 0) { mode = Mode::None; }
    }

    // steer_params subtree
    if (mode == Mode::SteerParams) {
      if (indent == 2) {
        // label: maybe inline map
        auto pos = t.find(':'); if (pos == std::string::npos) continue;
        std::string key = unquote(t.substr(0, pos));
        std::string rest = trim(t.substr(pos+1));
        currentLabel = key;
        if (!rest.empty() && rest.find('{') != std::string::npos) {
          bool has=false; double mn=0,mx=0,df=0;
          parse_inline_map_into(rest, has, mn, mx, df);
          if (has) {
            // store both scalar and vector-uniform for this label; the include() / includeVector() will pick appropriately
            ScalarCfg sc; sc.has = true; sc.min = mn; sc.max = mx; sc.def = df;
            labelScalarCfg_[key] = sc;

            VectorCfg vc; vc.has = true; vc.uniform = true; vc.umin = mn; vc.umax = mx; vc.udef = df;
            labelVectorCfg_[key] = vc;
            hasConfig_ = true;
          }
        }
        continue;
      }
      if (indent == 4 && t == "components:") {
        // Expect following lines at indent 6: x:, y:, z:
        // We'll keep mode and rely on indent checks below
        continue;
      }
      if (indent == 6) {
        auto pos = t.find(':'); if (pos == std::string::npos) continue;
        std::string comp = unquote(t.substr(0, pos));
        std::string rest = trim(t.substr(pos+1));
        int idx = (comp == "x" ? 0 : comp == "y" ? 1 : comp == "z" ? 2 : -1);
        if (idx >= 0) {
          bool has=false; double mn=0,mx=0,df=0;
          parse_inline_map_into(rest, has, mn, mx, df);
          if (has) {
            VectorCfg& vc = labelVectorCfg_[currentLabel];
            vc.has = true; vc.uniform = false; // per-component
            vc.comp[idx].has = true; vc.comp[idx].min = mn; vc.comp[idx].max = mx; vc.comp[idx].def = df;
            hasConfig_ = true;
          }
        }
        continue;
      }
    }
  }

  return hasConfig_;
}

} // namespace ippl
