/*
 * openPMD standard https://github.com/openPMD/openPMD-standard
 *
 */

#ifndef IPPL_STREAM_FORMAT_OPEN_PMD_H
#define IPPL_STREAM_FORMAT_OPEN_PMD_H

#include "Stream/Format/Format.h"

namespace ippl {

    class OpenPMD : public Format {
    public:
        OpenPMD() = default;

        void header(ParameterList* param) const override {
            std::cout << "This file is written in OpenPMD standard." << std::endl;

            param->add<std::string>("version", "1.1.0");
            param->add<std::string>("author", "none");
            param->add<std::string>("software", "Ippl");
            param->add<std::string>("softwareVersion", "2.1.0");
            param->add<std::string>("data", "YYYY-MM-DD HH:mm:ss tz");
            param->add<std::string>("machine", "machine");
            param->add<std::string>("comment", "");
        }
    };
}  // namespace ippl

#endif
