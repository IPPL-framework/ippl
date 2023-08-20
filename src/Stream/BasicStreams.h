#ifndef IPPL_BASIC_STREAMS_H
#define IPPL_BASIC_STREAMS_H

#include "Utility/ParameterList.h"

namespace ippl {

    template <class Object>
    class basic_ostream {
    public:
        basic_ostream() = default;

        virtual ~basic_ostream() = default;

        virtual basic_ostream<Object>& operator<<(const Object& obj) = 0;
    };

    template <class Object>
    class basic_istream {
    public:
        basic_istream() = default;

        virtual ~basic_istream() = default;

        virtual basic_istream<Object>& operator>>(Object& obj) = 0;
    };

    template <class Object>
    class basic_iostream : public basic_istream<Object>, public basic_ostream<Object> {
    public:
        basic_iostream() = default;

        virtual ~basic_iostream() = default;
    };

}  // namespace ippl
#endif
