#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <utility>

namespace ippl {

class ProxyWriter {


  struct Channel {
    std::string label;
    double      defaultValue{1.0};
    bool        isVector{false};
    bool        isBool{false};
    bool        isButton{false};
    bool        isEnum{false};
    std::vector<std::pair<std::string,int>> enumEntries{}; // for enums: text/value pairs
    int         defaultInt{0};
    unsigned    vecDim{1}; // for vectors, clamp to 3
  };

  // Vector properties are driven directly with a 3-arg setter
  // (SetTuple3Double(arrayName, x, y, z)) on the generator.

  std::filesystem::path outPath_;
  double rangeMin_ {-99};
  double rangeMax_ {99};
  std::string prototypeLabel_ {"SteerableParameters"};

  std::vector<Channel> channels_;

  std::ostringstream header_;
  std::ostringstream sources_;
  std::ostringstream misc_;
  std::ostringstream footer_;

public:



    void initialize (std::filesystem::path xmlOutputPath,
                       double rangeMin = -99.0,
                       double rangeMax = 99.0,
                       std::string prototypeLabel = "SteerableParameters"){
    outPath_ = std::move(xmlOutputPath);
    rangeMin_ = rangeMin;
    rangeMax_ = rangeMax;
    prototypeLabel_ = std::move(prototypeLabel);
    resetStreams();
  }

  // Vector properties always use SetTuple3Double; no alternate backend.


  // Include a new steerable scalar channel with a default value and label.
  // T must be a scalar arithmetic type (double/float/int/...).
  template <typename T>
  void include(const T& defaultValue, const std::string& label) {
    static_assert(std::is_arithmetic_v<T>, "ProxyWriter::include requires a scalar arithmetic type");
    Channel ch; ch.label = label; ch.defaultValue = static_cast<double>(defaultValue); ch.isVector = false; ch.vecDim = 1;
    channels_.emplace_back(std::move(ch));
  }

  // Include a steerable vector channel; only first 3 components are exposed to GUI.
  template <typename T, unsigned Dim_v>
  void includeVector(const std::string& label) {
    Channel ch; ch.label = label; ch.defaultValue = 1.0; ch.isVector = true; ch.vecDim = (Dim_v > 3 ? 3u : Dim_v);
    channels_.emplace_back(std::move(ch));
  }

  void includeBool(const std::string& label, bool defaultValue = false) {
    Channel ch; ch.label = label; ch.defaultValue = defaultValue ? 1.0 : 0.0; ch.isVector = false; ch.isBool = true; ch.vecDim = 1;
    channels_.emplace_back(std::move(ch));
  }

  // Include a steerable push-button channel (momentary action)
  void includeButton(const std::string& label) {
    Channel ch; ch.label = label; ch.defaultValue = 0.0; ch.isVector = false; ch.isBool = false; ch.isButton = true; ch.vecDim = 1;
    channels_.emplace_back(std::move(ch));
  }

  // Include a steerable enum channel with choices and default value (by integer value)
  void includeEnum(const std::string& label,
                   const std::vector<std::pair<std::string,int>>& entries,
                   int defaultValue = 0) {
    Channel ch; ch.label = label; ch.isEnum = true; ch.defaultInt = defaultValue; ch.vecDim = 1;
    ch.enumEntries = entries;
    channels_.emplace_back(std::move(ch));
  }

  // Generate and write the XML file. Returns true on success.
  bool produce() {
    // Rebuild the XML each time produce is called (idempotent based on channels_ state).
    resetStreams();

    // Header
    header_ << "<ServerManagerConfiguration>\n\n";

    // Sources group with one SourceProxy per channel
    sources_ << "    <ProxyGroup name=\"sources\">\n";
    for (const auto& ch : channels_) {
      appendSourceProxy(ch);
    }
    sources_ << "    </ProxyGroup>\n\n";

    // Misc group with a single prototype used by PropertyCollection
    misc_ << "    <ProxyGroup name=\"misc\">\n";
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

  bool produceUnified(const std::string& unifiedProxyName = "SteerableParameters_ALL",
                      const std::string& unifiedGroupLabel = "SteerableParameters")
  {
    resetStreams();

    // Header
    header_ << "<ServerManagerConfiguration>\n\n";

    sources_ << "    <ProxyGroup name=\"sources\">\n";
    appendUnifiedSourceProxy(unifiedProxyName, unifiedGroupLabel);
    sources_ << "    </ProxyGroup>\n\n";

    misc_ << "    <ProxyGroup name=\"misc\">\n";
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

private:

  void resetStreams() {
    header_.str(""); header_.clear();
    sources_.str(""); sources_.clear();
    misc_.str(""); misc_.clear();
    footer_.str(""); footer_.clear();
  }

  void appendSourceProxy(const Channel& ch) {
    const std::string& L = ch.label;
    // SourceProxy for this steerable channel
    sources_ << "        <SourceProxy class=\"vtkSteeringDataGenerator\" name=\"SteerableParameters_" << L << "\">\n"
             << "            <IntVectorProperty name=\"PartitionType\" command=\"SetPartitionType\" number_of_elements=\"1\" default_values=\"1\" panel_visibility=\"never\">\n"
             << "            </IntVectorProperty>\n\n"
             << "            <IntVectorProperty name=\"FieldAssociation\" command=\"SetFieldAssociation\" number_of_elements=\"1\" default_values=\"0\" panel_visibility=\"never\">\n"
             << "            </IntVectorProperty>\n\n"
             << "            <DoubleVectorProperty name=\"scaleFactor\"\n"
             << "                                  command=\"SetTuple1Double\"\n"
             << "                                  clean_command=\"Clear\"\n"
             << "                                  use_index=\"1\"\n"
             << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
             << "                                  number_of_elements_per_command=\"1\"\n"
             << "                                  repeat_command=\"1\"\n"
             << "                                  panel_widget=\"DoubleRange\">\n"
             << "              <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
             << "            </DoubleVectorProperty>\n\n";
  if(!ch.isBool){
   sources_ << "            <PropertyGroup label=\"SteerableParameters\" panel_widget=\"PropertyCollection\">\n";
  }
   sources_ << "                <Hints>\n"
             << "                  <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableParametersPrototype\" />\n"
             << "                </Hints>\n"
             << "            </PropertyGroup>\n\n"
             << "            <Hints>\n"
             << "              <CatalystInitializePropertiesWithMesh mesh=\"steerable_channel_forward_" << L << "\">\n"
             << "                <Property name=\"scaleFactor\" association=\"point\" array=\"steerable_field_f_" << L << "\" />\n"
             << "              </CatalystInitializePropertiesWithMesh>\n"
             << "            </Hints>\n"
             << "        </SourceProxy>\n\n";
  }

  void appendPrototype() {
    // Single prototype used by all PropertyCollection groups above.
    misc_ << "      <Proxy name=\"SteerableParametersPrototype\" label=\"" << prototypeLabel_ << "\">\n"
          << "        <DoubleVectorProperty name=\"scaleFactor\" label=\"scaleFactor\" number_of_elements=\"1\" default_values=\"1\" panel_widget=\"DoubleRange\">\n"
          << "          <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
          << "        </DoubleVectorProperty>\n"
          << "      </Proxy>\n";

    // Vector prototype to drive 3-element vector editors in PropertyCollection
    misc_ << "      <Proxy name=\"SteerableVectorPrototype\" label=\"" << prototypeLabel_ << " (Vector)\">\n"
      << "        <DoubleVectorProperty name=\"vec3\" label=\"vec3\" number_of_elements=\"3\" default_values=\"1 1 1\" panel_widget=\"DoubleRange\">\n"
      << "          <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
      << "        </DoubleVectorProperty>\n"
      << "      </Proxy>\n";

    // Enum prototypes: one per enum channel so choices can differ per label
    for (const auto& ch : channels_) {
      if (!ch.isEnum || ch.enumEntries.empty()) continue;
      const std::string& L = ch.label;
      misc_ << "\n      <Proxy name=\"SteerableEnumPrototype_" << L << "\" label=\"" << prototypeLabel_ << " Enum\">\n"
            << "        <IntVectorProperty name=\"PrototypeEnum_" << L << "\"\n"
            << "                           label=\"" << L << "\"\n"
            << "                           number_of_elements=\"1\"\n"
            << "                           default_values=\"" << ch.defaultInt << "\"\n"
            << "                           immediate_apply=\"1\"\n"
            << "                           >\n"
            << "          <EnumerationDomain name=\"enum\">\n";
      for (const auto& [text, val] : ch.enumEntries) {
        misc_ << "                <Entry text=\"" << text << "\" value=\"" << val << "\"/>\n";
      }
      misc_ << "          </EnumerationDomain>\n"
            << "        </IntVectorProperty>\n"
            << "      </Proxy>\n";
    }

  // Bool prototype to drive checkbox in PropertyCollection (int-backed)
  // misc_ << "      <Proxy name=\"SteerableBoolPrototype\" label=\"" << prototypeLabel_ << " (Bool)\">\n"
  //   << "        <IntVectorProperty name=\"bool\" label=\"bool\" number_of_elements=\"1\" default_values=\"0\" panel_widget=\"CheckBox\">\n"
  //   << "          <BooleanDomain name=\"bool\"/>\n"
  //   << "        </IntVectorProperty>\n"
  //   << "      </Proxy>\n";
  }

  // Build a single SourceProxy with many per-label scaleFactor properties
  void appendUnifiedSourceProxy(const std::string& proxyName,
                                const std::string& groupLabel)
  {
    sources_ << "        <SourceProxy class=\"vtkSteeringDataGenerator\" name=\"" << proxyName << "\">\n"
             << "            <IntVectorProperty name=\"PartitionType\" command=\"SetPartitionType\" number_of_elements=\"1\" default_values=\"1\" panel_visibility=\"never\">\n"
             << "            </IntVectorProperty>\n\n"
             << "            <IntVectorProperty name=\"FieldAssociation\" command=\"SetFieldAssociation\" number_of_elements=\"1\" default_values=\"0\" panel_visibility=\"never\">\n"
             << "            </IntVectorProperty>\n\n";

    // Add properties per channel
    for (const auto& ch : channels_) {
      const std::string& L = ch.label;
      if (ch.isBool) {
        // // Checkbox boolean (0/1), int-backed and using SetTuple1Int
        // // Make it an explicit 1-element property so GUI checkbox reliably triggers updates.
        // sources_ << "            <IntVectorProperty name=\"" << L << "\" label=\"" << L << "\"\n"
        //          << "                                  command=\"SetTuple1Int\"\n"
        //          << "                                  clean_command=\"Clear\"\n"
        //          << "                                  use_index=\"1\"\n"
        //          << "                                  number_of_elements=\"1\"\n"
        //          << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
        //          << "                                  default_values=\"" << (ch.defaultValue != 0.0 ? 1 : 0) << "\"\n"
        //          << "                                  number_of_elements_per_command=\"1\"\n"
        //          << "                                  repeat_command=\"1\"\n"
        //          << "                                  panel_widget=\"CheckBox\">\n"
        //          << "              <BooleanDomain name=\"bool\"/>\n"
        //          << "            </IntVectorProperty>\n\n";

        // Checkbox boolean (0/1), int-backed and using SetTuple1Int
        // We define the panel_widget="CheckBox" right here.
        sources_ << "            <IntVectorProperty name=\"" << L << "\" label=\"" << L << "\"\n"
                 << "                                  command=\"SetTuple1Int\"\n"
                 << "                                  clean_command=\"Clear\"\n"
                 << "                                  use_index=\"1\"\n"
                 << "                                  number_of_elements=\"1\"\n"
                 << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
                 << "                                  default_values=\"" << (ch.defaultValue != 0.0 ? 1 : 0) << "\"\n"
                 << "                                  number_of_elements_per_command=\"1\"\n"
                 << "                                  repeat_command=\"1\"\n"
                 << "                                  panel_widget=\"CheckBox\">\n" // <-- This is key
                 << "              <BooleanDomain name=\"bool\"/>\n"
                 << "            </IntVectorProperty>\n\n";
      } else if (ch.isButton) {
        // Momentary button as checkbox: Check to trigger, simulation unchecks after processing
        // Uses CheckBox widget with IntVectorProperty (same as Switch but for momentary action)
        // command_button implementation did no seem to work...?...
        sources_ << "            <IntVectorProperty name=\"" << L << "\" label=\"" << L << " (Trigger)\"\n"
                 << "                                  command=\"SetTuple1Int\"\n"
                 << "                                  clean_command=\"Clear\"\n"
                 << "                                  use_index=\"1\"\n"
                 << "                                  number_of_elements=\"1\"\n"
                 << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
                 << "                                  default_values=\"0\"\n"
                 << "                                  number_of_elements_per_command=\"1\"\n"
                 << "                                  repeat_command=\"1\"\n"
                 << "                                  immediate_apply=\"1\"\n"
                 << "                                  panel_widget=\"CheckBox\">\n"
                 << "              <BooleanDomain name=\"bool\"/>\n"
                 << "              <Documentation>\n"
                 << "                Check this box to trigger the button action. The simulation will automatically uncheck it internally  after processing.\n"
                 << "              </Documentation>\n"
                 << "            </IntVectorProperty>\n\n";
      } else if (ch.isEnum) {
        // Integer-backed enum property in the source; the actual EnumerationDomain is provided via a misc prototype.
        sources_ << "            <IntVectorProperty name=\"" << L << "\"\n"
                 << "                                  command=\"SetTuple1Int\"\n"
                 << "                                  clean_command=\"Clear\"\n"
                 << "                                  use_index=\"1\"\n"
                 << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
                 << "                                  number_of_elements_per_command=\"1\"\n"
                 << "                                  repeat_command=\"1\"\n"
                 << "                                  \n\n\n"
                 << "                           number_of_elements=\"1\"\n"
                 << "                           default_values=\"" << ch.defaultInt << "\"\n"
                 << "                           immediate_apply=\"1\"\n"
                 << "                                  \n\n\n"
                 << "                                  >\n"
                 << "            </IntVectorProperty>\n\n";
      } else if (!ch.isVector) {
        sources_ << "            <DoubleVectorProperty name=\"" << L << "\" label=\"" << L << "\"\n"
                 << "                                  command=\"SetTuple1Double\"\n"
                 << "                                  clean_command=\"Clear\"\n"
                 << "                                  use_index=\"1\"\n"
                 << "                                  number_of_elements=\"1\"\n"
                 << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
                 << "                                  default_values=\"" << ch.defaultValue << "\"\n"
                 << "                                  number_of_elements_per_command=\"1\"\n"
                 << "                                  repeat_command=\"1\"\n"
                 << "                                  panel_widget=\"DoubleRange\">\n"
                 << "              <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
                 << "            </DoubleVectorProperty>\n\n";
      } else {
        // Direct tuple3 command path (generator must provide SetTuple3Double)
        sources_ << "            <DoubleVectorProperty name=\"" << L << "\" label=\"" << L << "\"\n"
                 << "                                  command=\"SetTuple3Double\"\n"
                 << "                                  use_index=\"1\"\n"
                 << "                                  clean_command=\"Clear\"\n"
                 << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
                 << "                                  number_of_elements=\"3\"\n"
                 << "                                  default_values=\"" << ch.defaultValue << " " << ch.defaultValue << " " << ch.defaultValue << "\"\n"
                 << "                                  number_of_elements_per_command=\"3\"\n"
                 << "                                  repeat_command=\"1\"\n"
                 << "                                  panel_widget=\"DoubleRange\">\n"
                 << "              <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
                 << "            </DoubleVectorProperty>\n\n";
      }
    }

    // Create one PropertyCollection group per label so the GUI shows separate sections
    for (const auto& ch : channels_) {
      // sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label
      //          << "\" panel_widget=\"PropertyCollection\">\n"
               ;
      if (ch.isBool) {

      sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label<< "\">\n";
        // sources_ << "                <Hints>\n"
                //  << "                  <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableBoolPrototype\" />\n"
                //  << "                </Hints>\n";
        sources_ << "                <Property name=\"" << ch.label << "\" function=\"bool\" label=\"" << ch.label << "\" />\n";
      } 
      else if (ch.isEnum) {
        // Use a PropertyCollection with the enum prototype specific to this label
        sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label << "\" panel_widget=\"PropertyCollection\">\n";
        sources_ << "             <Hints>\n";
        sources_ << "               <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableEnumPrototype_" << ch.label << "\" />\n";
        sources_ << "             </Hints>\n";
        sources_ << "             <Property name=\"" << ch.label << "\" function=\"PrototypeEnum_" << ch.label << "\"  label=\"" << ch.label << "\"/> \n";
      }
      else if (ch.isButton) {
        sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label<< "\">\n";
        // sources_ << "                <Property name=\"" << ch.label << "\" function=\"button\" label=\"" << ch.label << "\" />\n";
        sources_ << "                <Property name=\"" << ch.label <<  "\" />\n";
      } 
      else 
      if (!ch.isVector) {

        sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label
                 << "\" panel_widget=\"PropertyCollection\">\n";
        sources_ << "                <Hints>\n"
                 << "                  <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableParametersPrototype\" />\n"
                 << "                </Hints>\n";
        // Use scalar prototype for nicer slider
        sources_ << "                <Property name=\"" << ch.label << "\" function=\"scaleFactor\" label=\"" << ch.label << "\" />\n";
      } 
      else {
        // For vectors, use a dedicated vector prototype to present a 3-element editor

        sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label
                 << "\" panel_widget=\"PropertyCollection\">\n";
        sources_ << "                <Hints>\n"
                 << "                  <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableVectorPrototype\" />\n"
                 << "                </Hints>\n";
        sources_ << "                <Property name=\"" << ch.label << "\" function=\"vec3\" label=\"" << ch.label << "\" />\n";
      }
  sources_ << "            </PropertyGroup>\n\n";
    }

    // Initialization hints: pull defaults from forward channels per label
    // Skip buttons - they should not be initialized from simulation state
    sources_ << "            <Hints>\n";
    for (const auto& ch : channels_) {
      if (ch.isButton) continue; // Buttons are GUI-only, don't initialize from forward channel
      const std::string& L = ch.label;
      sources_ << "              <CatalystInitializePropertiesWithMesh mesh=\"steerable_channel_forward_" << L << "\">\n"
               << "                <Property name=\"" << L << "\" association=\"point\" array=\"steerable_field_f_" << L << "\" />\n"
               << "              </CatalystInitializePropertiesWithMesh>\n";

    }
    sources_ << "            </Hints>\n";

    sources_ << "        </SourceProxy>\n\n";
  }

private:
};

} // namespace ippl

