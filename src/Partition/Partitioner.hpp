//
// Class Partitioner
//   Partition a domain into subdomains.
//

#include <algorithm>
#include <numeric>
#include <vector>

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        template <typename view_type>
        void Partitioner<Dim>::split(const NDIndex<Dim>& domain, view_type& view, e_dim_tag* decomp,
                                     int nSplits) const {
            using NDIndex_t = NDIndex<Dim>;

            // Recursively split the domain until we have generated all the domains.
            std::vector<NDIndex_t> domains_c(nSplits);
            NDIndex_t leftDomain;

            // Start with the whole domain.
            domains_c[0] = domain;
            int v;
            unsigned int d = 0;

            int v1, v2, rm, vtot, vl, vr;
            double a, lmax, len;

            for (v = nSplits, rm = 0; v > 1; v /= 2) {
                rm += (v % 2);
            }

            if (rm == 0) {
                // nSplits is a power of 2

                std::vector<NDIndex_t> copy_c(nSplits);

                for (v = 1; v < nSplits; v *= 2) {
                    // Go to the next parallel dimension.
                    while (decomp[d] != PARALLEL)
                        if (++d == Dim)
                            d = 0;

                    // Split all the current nSplits.
                    int i, j;
                    for (i = 0, j = 0; i < v; ++i, j += 2) {
                        // Split to the left and to the right, saving both.
                        domains_c[i].split(copy_c[j], copy_c[j + 1], d);
                    }
                    // Copy back.
                    std::copy(copy_c.begin(), copy_c.begin() + v * 2, domains_c.begin());

                    // On to the next dimension.
                    if (++d == Dim)
                        d = 0;
                }

            } else {
                vtot = 1;  // count the number of nSplits to make sure that it worked
                           // nSplits is not a power of 2 so we need to do some fancy splitting
                           // sorry... this would be much cleaner with recursion
                /*
                    The way this works is to recursively split on the longest dimension.
                    Suppose you request 11 nSplits.  It will split the longest dimension
                    in the ratio 5:6 and put the new domains in node 0 and node 5.  Then
                    it splits the longest dimension of the 0 domain and puts the results
                    in node 0 and node 2 and then splits the longest dimension of node 5
                    and puts the results in node 5 and node 8. etc.
                    The logic is kind of bizarre, but it works.
                */
                for (v = 1; v < 2 * nSplits; ++v) {
                    // kind of reverse the bits of v
                    for (v2 = v, v1 = 1; v2 > 1; v2 /= 2) {
                        v1 = 2 * v1 + (v2 % 2);
                    }
                    vl = 0;
                    vr = nSplits;

                    while (v1 > 1) {
                        if ((v1 % 2) == 1) {
                            vl = vl + (vr - vl) / 2;
                        } else {
                            vr = vl + (vr - vl) / 2;
                        }
                        v1 /= 2;
                    }

                    v2 = vl + (vr - vl) / 2;

                    if (v2 > vl) {
                        a = v2 - vl;
                        a /= vr - vl;
                        vr         = v2;
                        leftDomain = domains_c[vl];
                        lmax       = 0;
                        d          = std::numeric_limits<unsigned int>::max();
                        for (unsigned int dd = 0; dd < Dim; ++dd) {
                            if (decomp[dd] == PARALLEL) {
                                if ((len = leftDomain[dd].length()) > lmax) {
                                    lmax = len;
                                    d    = dd;
                                }
                            }
                        }
                        NDIndex_t temp;
                        domains_c[vl].split(temp, domains_c[vr], d, a);
                        domains_c[vl] = temp;
                        ++vtot;
                    }
                }

                v = vtot;
            }

            // Make sure v is the same number of nSplits at this stage.
            PAssert_EQ(v, nSplits);

            for (size_t i = 0; i < domains_c.size(); ++i) {
                view(i) = domains_c[i];
            }
        }
    }  // namespace detail
}  // namespace ippl
