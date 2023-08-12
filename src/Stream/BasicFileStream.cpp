#include "Stream/BasicFileStream.h"

namespace ippl {

    BasicFileStream::BasicFileStream() {
        param_m.add<bool>("overwrite", false);
    }

    void BasicFileStream::create(const fs::path& path, const ParameterList& param) {
        if (!path.has_filename()) {
            throw IpplException("BasicFileStream::create", "No filename provided.");
        }

        param_m.merge(param);

        if (!param_m.get<bool>("overwrite") && fs::exists(path)) {
            throw IpplException("BasicFileStream::create", "File already exists.");
        }
    }

    bool BasicFileStream::remove() {
        return fs::remove(path_m);
    }

    void BasicFileStream::open(const fs::path& path) {
        if (!path.has_filename()) {
            throw IpplException("BasicFileStream::open", "No filename provided.");
        }
        path_m = path;
    }
}
