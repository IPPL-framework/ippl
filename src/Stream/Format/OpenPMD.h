/*
 * openPMD standard https://github.com/openPMD/openPMD-standard
 *
 */

#ifndef IPPL_STREAM_FORMAT_OPEN_PMD_H
#define IPPL_STREAM_FORMAT_OPEN_PMD_H

#include <ctime>
#include <string>

#include "Stream/Format/Format.h"

namespace ippl {

    class OpenPMD : public Format {
    public:
        OpenPMD() = default;

        void header(ParameterList* param) const override;

    private:
        const std::string version = "1.1.0";

        std::string date() const;
    };
}  // namespace ippl

#endif
