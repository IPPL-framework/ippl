#ifndef IPPL_BASE_STREAM_H
#define IPPL_BASE_STREAM_H

#include <filesystem>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

namespace ippl {

    namespace fs = std::filesystem;

    template <class Object>
    class BaseStream {
    public:
        BaseStream();

        virtual ~BaseStream() = default;

        virtual void create(const fs::path& path, const ParameterList& param);

        bool remove();

        virtual void open(const fs::path& path);

        virtual void close() = 0;

        virtual void operator<<(const Object& obj) = 0;

        virtual void operator>>(Object& obj) = 0;

    protected:
        fs::path path_m;
        ParameterList param_m;
    };

    template <class Object>
    BaseStream<Object>::BaseStream() {
        param_m.add<bool>("overwrite", false);
    }

    template <class Object>
    void BaseStream<Object>::create(const fs::path& path, const ParameterList& param) {
        if (!path.has_filename()) {
            throw IpplException("BaseStream::create", "No filename provided.");
        }

        param_m.merge(param);

        if (!param_m.get<bool>("overwrite") && fs::exists(path)) {
            throw IpplException("BaseStream::create", "File already exists.");
        }
    }

    template <class Object>
    bool BaseStream<Object>::remove() {
        return fs::remove(path_m);
    }

    template <class Object>
    void BaseStream<Object>::open(const fs::path& path) {
        if (!path.has_filename()) {
            throw IpplException("BaseStream::open", "No filename provided.");
        }
        path_m = path;
    }

}  // namespace ippl
#endif
