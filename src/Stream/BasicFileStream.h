#ifndef IPPL_BASIC_FILE_STREAM_H
#define IPPL_BASIC_FILE_STREAM_H

#include <filesystem>

#include "Stream/BasicStreams.h"

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

namespace ippl {

    namespace fs = std::filesystem;

    class BasicFileStream {
    public:
        BasicFileStream();

        virtual ~BasicFileStream() = default;

        virtual void create(const fs::path& path, const ParameterList& param);

        bool remove();

        virtual void open(const fs::path& path);

        virtual void close() = 0;

    protected:
        fs::path path_m;
        ParameterList param_m;
    };
}  // namespace ippl
#endif
