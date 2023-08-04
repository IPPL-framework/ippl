#ifndef IPPL_GENERATOR_H
#define IPPL_GENERATOR_H

namespace ippl {
    namespace random {

        template <class DeviceType>
        class Generator {
        public:
            typedef DeviceType device_type;
            typedef Kokkos::Random_XorShift64_Pool<DeviceType> pool_type;

            KOKKOS_DEFAULTED_FUNCTION
            Generator(const Generator& g) = default;

            KOKKOS_FUNCTION
            Generator(int seed)
                : rand_pool64_m(seed) {}

            KOKKOS_FUNCTION
            ~Generator() {}

            template <typename T>
            KOKKOS_INLINE_FUNCTION T next() const {
                //         T operator()() const {
                // acquire the state of the random number generator engine
                auto rand_gen = rand_pool64_m.get_state();

                T result =
                    Kokkos::rand<typename pool_type::generator_type, T>::draw(rand_gen, T(0), T(1));

                // Give the state back, which will allow another thread to acquire it
                rand_pool64_m.free_state(rand_gen);

                return result;
            }

        private:
            pool_type rand_pool64_m;
        };
    }  // namespace random
}  // namespace ippl

#endif
