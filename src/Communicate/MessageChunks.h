#ifndef IPPL_MESSAGE_CHUNKS_H
#define IPPL_MESSAGE_CHUNKS_H

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "Types/IpplTypes.h"

namespace ippl::detail {
    /**
     * @brief Visit byte ranges representable by classic MPI int counts.
     * @note Zero bytes still produces one message. Peers must use the same chunk size and
     * post chunks in order with the same tag; MPI's nonovertaking rule preserves matching.
     */
    template <typename Post>
    void forEachMessageChunk(ippl::detail::size_type bytes, Post&& post,
                             int maxChunkBytes = INT_MAX) {
        if (maxChunkBytes <= 0) {
            throw std::invalid_argument("MPI message chunk size must be positive");
        }
        ippl::detail::size_type offset = 0;
        do {
            const int count = static_cast<int>(std::min<ippl::detail::size_type>(
                bytes - offset, static_cast<ippl::detail::size_type>(maxChunkBytes)));
            post(offset, count);
            offset += count;
        } while (offset < bytes);
    }
}  // namespace ippl::detail

#endif
