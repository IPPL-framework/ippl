#ifndef IPPL_PARTICLE_ATTRIBUTE_SELECTION_H
#define IPPL_PARTICLE_ATTRIBUTE_SELECTION_H

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ippl {

/** Select custom attributes by get_name(); positions are always exported. */
class ParticleAttributeSelection {
public:
    enum class Mode { All, Minimal, AllExcept };

    ParticleAttributeSelection() = default;

    static ParticleAttributeSelection all() { return {}; }

    static ParticleAttributeSelection minimal(std::vector<std::string> names = {}) {
        return ParticleAttributeSelection(Mode::Minimal, std::move(names));
    }

    static ParticleAttributeSelection allExcept(std::vector<std::string> names) {
        return ParticleAttributeSelection(Mode::AllExcept, std::move(names));
    }

    bool selects(const std::string& name) const {
        const bool listed = std::binary_search(names_m.begin(), names_m.end(), name);
        return mode_m == Mode::All || (mode_m == Mode::Minimal ? listed : !listed);
    }

    ParticleAttributeSelection& withParticleIds(bool include = true) {
        particleIds_m = include;
        return *this;
    }

    ParticleAttributeSelection& withRankIds(bool include = true) {
        rankIds_m = include;
        return *this;
    }

    bool includesParticleIds() const { return particleIds_m; }
    bool includesRankIds() const { return rankIds_m; }
    Mode mode() const { return mode_m; }
    const std::vector<std::string>& names() const { return names_m; }

    /** Return an explanation for invalid names, or an empty string on success. */
    std::string validationError(const std::vector<std::string>& available) const {
        for (const auto& name : names_m) {
            const auto count = std::count(available.begin(), available.end(), name);
            if (count == 0) {
                return "Unknown custom particle attribute: '" + name + "'";
            }
            if (count > 1) {
                return "Ambiguous custom particle attribute: '" + name + "'";
            }
        }
        for (const auto& name : available) {
            if (!selects(name)) {
                continue;
            }
            if (name.empty() || name.find('/') != std::string::npos || name == "position"
                || name == "ParticleIDs" || name == "RankID") {
                return "Invalid or reserved particle attribute name: '" + name + "'";
            }
            if (std::count(available.begin(), available.end(), name) > 1) {
                return "Ambiguous custom particle attribute: '" + name + "'";
            }
        }
        return {};
    }

private:
    ParticleAttributeSelection(Mode mode, std::vector<std::string> names)
        : mode_m(mode)
        , names_m(std::move(names))
        , particleIds_m(mode != Mode::Minimal)
        , rankIds_m(mode != Mode::Minimal) {
        std::sort(names_m.begin(), names_m.end());
        names_m.erase(std::unique(names_m.begin(), names_m.end()), names_m.end());
    }

    Mode mode_m = Mode::All;
    std::vector<std::string> names_m;
    bool particleIds_m = true;
    bool rankIds_m = true;
};

}  // namespace ippl

#endif
