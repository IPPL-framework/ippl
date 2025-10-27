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
  };

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


  // Include a new steerable scalar channel with a default value and label.
  // T must be a scalar arithmetic type (double/float/int/...).
  template <typename T>
  void include(const T& defaultValue, const std::string& label) {
    static_assert(std::is_arithmetic_v<T>, "ProxyWriter::include requires a scalar arithmetic type");
    channels_.emplace_back(Channel{label, static_cast<double>(defaultValue)});
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
             << "            </DoubleVectorProperty>\n\n"
             << "            <PropertyGroup label=\"SteerableParameters\" panel_widget=\"PropertyCollection\">\n"
             << "                <Hints>\n"
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

    // Add one property per channel (unique name: scaleFactor_<label>)
    for (const auto& ch : channels_) {
      const std::string& L = ch.label;
      sources_ << "            <DoubleVectorProperty name=\"scaleFactor_" << L << "\"\n"
               << "                                  command=\"SetTuple1Double\"\n"
               << "                                  clean_command=\"Clear\"\n"
               << "                                  use_index=\"1\"\n"
               << "                                  initial_string=\"steerable_field_b_" << L << "\"\n"
               << "                                  number_of_elements_per_command=\"1\"\n"
               << "                                  repeat_command=\"1\"\n"
               << "                                  panel_widget=\"DoubleRange\">\n"
               << "              <DoubleRangeDomain name=\"range\" min=\"" << rangeMin_ << "\" max=\"" << rangeMax_ << "\"/>\n"
               << "            </DoubleVectorProperty>\n\n";
    }

    // Create one PropertyCollection group per label so the GUI shows separate sections
    for (const auto& ch : channels_) {
      sources_ << "            <PropertyGroup label=\"" << groupLabel << "_" << ch.label
               << "\" panel_widget=\"PropertyCollection\">\n"
               << "                <Hints>\n"
               << "                  <PropertyCollectionWidgetPrototype group=\"misc\" name=\"SteerableParametersPrototype\" />\n"
               << "                </Hints>\n"
               << "                <Property name=\"scaleFactor_" << ch.label << "\" function=\"scaleFactor\" />\n"
               << "            </PropertyGroup>\n\n";
    }

    // Initialization hints: pull defaults from forward channels per label
    sources_ << "            <Hints>\n";
    for (const auto& ch : channels_) {
      const std::string& L = ch.label;
      sources_ << "              <CatalystInitializePropertiesWithMesh mesh=\"steerable_channel_forward_" << L << "\">\n"
               << "                <Property name=\"scaleFactor_" << L << "\" association=\"point\" array=\"steerable_field_f_" << L << "\" />\n"
               << "              </CatalystInitializePropertiesWithMesh>\n";
    }
    sources_ << "            </Hints>\n";

    sources_ << "        </SourceProxy>\n\n";
  }

private:
};

} // namespace ippl
