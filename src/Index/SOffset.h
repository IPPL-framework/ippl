//
// Class SOffset
//   Integer offset for sparse field index points.
//
#ifndef IPPL_SOFFSET_H
#define IPPL_SOFFSET_H

#include <ostream>
#include <type_traits>

#include "Index/NDIndex.h"
#include "Types/Vector.h"

namespace ippl {

    template <unsigned Dim>
    class SOffset {
    public:
        using point_type = Vector<int, Dim>;

        SOffset() = default;

        explicit SOffset(int value)
            : values_m(value) {}

        template <typename... Args,
                  typename std::enable_if<sizeof...(Args) == Dim, bool>::type = true>
        explicit SOffset(Args... args)
            : values_m(static_cast<int>(args)...) {}

        explicit SOffset(const point_type& values)
            : values_m(values) {}

        int& operator[](unsigned d) { return values_m[d]; }

        int operator[](unsigned d) const { return values_m[d]; }

        const point_type& values() const { return values_m; }

        SOffset& operator+=(const SOffset& rhs) {
            for (unsigned d = 0; d < Dim; ++d) {
                values_m[d] += rhs[d];
            }
            return *this;
        }

        SOffset& operator-=(const SOffset& rhs) {
            for (unsigned d = 0; d < Dim; ++d) {
                values_m[d] -= rhs[d];
            }
            return *this;
        }

        bool operator==(const SOffset& rhs) const {
            for (unsigned d = 0; d < Dim; ++d) {
                if (values_m[d] != rhs[d]) {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const SOffset& rhs) const { return !(*this == rhs); }

        bool inside(const NDIndex<Dim>& domain) const {
            for (unsigned d = 0; d < Dim; ++d) {
                if (!domain[d].contains(Index(values_m[d], values_m[d]))) {
                    return false;
                }
            }
            return true;
        }

    private:
        point_type values_m;
    };

    template <unsigned Dim>
    SOffset<Dim> operator+(SOffset<Dim> lhs, const SOffset<Dim>& rhs) {
        lhs += rhs;
        return lhs;
    }

    template <unsigned Dim>
    SOffset<Dim> operator-(SOffset<Dim> lhs, const SOffset<Dim>& rhs) {
        lhs -= rhs;
        return lhs;
    }

    template <unsigned Dim>
    SOffset<Dim> operator-(const SOffset<Dim>& value) {
        SOffset<Dim> result;
        for (unsigned d = 0; d < Dim; ++d) {
            result[d] = -value[d];
        }
        return result;
    }

    template <unsigned Dim>
    NDIndex<Dim> operator+(NDIndex<Dim> domain, const SOffset<Dim>& offset) {
        for (unsigned d = 0; d < Dim; ++d) {
            domain[d] = Index(domain[d].first() + offset[d], domain[d].last() + offset[d],
                              domain[d].stride());
        }
        return domain;
    }

    template <unsigned Dim>
    NDIndex<Dim> operator+(const SOffset<Dim>& offset, const NDIndex<Dim>& domain) {
        return domain + offset;
    }

    template <unsigned Dim>
    NDIndex<Dim> operator-(const NDIndex<Dim>& domain, const SOffset<Dim>& offset) {
        return domain + (-offset);
    }

    template <unsigned Dim>
    std::ostream& operator<<(std::ostream& out, const SOffset<Dim>& offset) {
        out << "[";
        for (unsigned d = 0; d < Dim; ++d) {
            out << offset[d];
            if (d + 1 < Dim) {
                out << ",";
            }
        }
        out << "]";
        return out;
    }

}  // namespace ippl

#endif
