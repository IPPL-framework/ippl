#ifndef IPPL_INTERPOLATION_WIDTH_DISPATCHER_H
#define IPPL_INTERPOLATION_WIDTH_DISPATCHER_H

#include <stdexcept>
#include <string>

namespace ippl::Interpolation {

    /**
     * @brief Dispatcher for kernels templated on width with run-time width
     *
     *
     * @tparam W Current width being tested (starts at 1)
     * @tparam MaxW Maximum supported width
     */
    template <int W, int MaxW>
    struct WidthDispatcher {
        /**
         * @brief Dispatch to functor factory with compile-time width
         *
         * This function recursively checks if the runtime width matches the current
         * template parameter W. If it matches, it invokes the factory with W as a
         * template parameter. Otherwise, it recursively tries W+1.
         *
         * @tparam FunctorFactory Lambda or functor type with template operator()<int>()
         * @param runtime_width The runtime kernel width value
         * @param factory Functor factory that accepts compile-time width as template parameter
         * @throws std::runtime_error if runtime_width exceeds MaxW
         */
        template <typename FunctorFactory>
        static void dispatch(int runtime_width, FunctorFactory&& factory) {
            if constexpr (W <= MaxW) {
                if (runtime_width == W) {
                    // Found matching width - invoke factory with compile-time W
                    factory.template operator()<W>();
                } else {
                    // Try next width
                    WidthDispatcher<W + 1, MaxW>::dispatch(
                        runtime_width, std::forward<FunctorFactory>(factory));
                }
            } else {
                // Exceeded maximum width
                throw std::runtime_error(
                    "Kernel width " + std::to_string(runtime_width) +
                    " exceeds maximum supported width " + std::to_string(MaxW));
            }
        }
    };

}  // namespace ippl::Interpolation

#endif  // IPPL_INTERPOLATION_WIDTH_DISPATCHER_H
