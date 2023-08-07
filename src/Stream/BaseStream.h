#ifndef IPPL_BASE_STREAM_H
#define IPPL_BASE_STREAM_H

#include "Utility/IpplException.h"

#include <filesystem>

namespace ippl {

    namespace fs = std::filesystem;

    template <class Object>
    class BaseStream {
    public:

        virtual void create(const fs::path& path, bool overwrite = false);

        bool remove();

        virtual void open(const fs::path& path, char access);

        virtual void close() = 0;

        virtual void operator<<(const Object& obj) = 0;

        virtual void operator>>(Object& obj) = 0;

    protected:
        fs::path path_m;
    };


    template <class Object>
    void BaseStream<Object>::create(const fs::path& path, bool overwrite) {
        if (!path.has_filename()) {
            throw IpplException("BaseStream::create", "No filename provided.");
        }
        if (!overwrite && fs::exists(path)) {
            throw IpplException("BaseStream::create", "File already exists.");
        }
    }

    template <class Object>
    bool BaseStream<Object>::remove() {
        return fs::remove(path_m);
    }


    template <class Object>
    void BaseStream<Object>::open(const fs::path& path, char /*access*/) {

        if (!path.has_filename()) {
            throw IpplException("BaseStream::open", "No filename provided.");
        }
        path_m = path;
    }


}  // namespace ippl
#endif
