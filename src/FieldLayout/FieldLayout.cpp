#include "Utility/IpplException.h"

namespace ippl {
    namespace detail {
        bool isUpper(unsigned int face) {
            for (; face > 0; face /= 3) {
                if (face % 3 == 1) {
                    return true;
                }
            }
            return false;
        }

        unsigned int getFaceDim(unsigned int face) {
            int dim        = -1;
            unsigned int d = 0;
            for (; face > 0; face /= 3, d++) {
                if (face % 3 != 2) {
                    if (dim == -1) {
                        dim = d;
                    } else {
                        throw IpplException(
                            "ippl::detail::getFaceDim",
                            "Argument corresponds to lower dimension hypercube than a facet");
                    }
                }
            }
            if (dim < 0) {
                return d;
            }
            return dim;
        }
    }  // namespace detail
}  // namespace ippl
