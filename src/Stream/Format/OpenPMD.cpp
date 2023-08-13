#include "Stream/Format/OpenPMD.h"

namespace ippl {

    void OpenPMD::header(ParameterList* param) const {
        // default parameter list
        ParameterList pl;
        pl.add<std::string>("version", version);
        pl.add<std::string>("author", "none");
        pl.add<std::string>("software", "none");
        pl.add<std::string>("softwareVersion", "none");
        pl.add<std::string>("softwareDependencies", "none");
        pl.add<std::string>("date", this->date());
        pl.add<std::string>("machine", "none");
        pl.add<std::string>("comment", "none");

        if (param != nullptr) {
            pl.merge(*param);
        }

        *param = pl;
    }

    std::string OpenPMD::date() const {
        std::time_t time = std::time({});
        char timeString[std::size("YYYY-MM-DD HH:mm:ss zzzzz")];
        std::strftime(std::data(timeString), std::size(timeString), "%F %T %z",
                      std::localtime(&time));
        return timeString;
    }
}  // namespace ippl
