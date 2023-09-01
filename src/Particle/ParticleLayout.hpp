//
// Class ParticleLayout
//   Base class for all particle layout classes.
//
//   This class is used as the generic base class for all classes
//   which maintain the information on where all the particles are located
//   on a parallel machine.  It is responsible for performing particle
//   load balancing.
//
//   If more general layout information is needed, such as the global -> local
//   mapping for each particle, then derived classes must provide this info.
//
//   When particles are created or destroyed, this class is also responsible
//   for determining where particles are to be created, gathering this
//   information, and recalculating the global indices of all the particles.
//   For consistency, creation and destruction requests are cached, and then
//   performed all in one step when the update routine is called.
//
//   Derived classes must provide the following:
//     1) Specific version of update and loadBalance.  These are not virtual,
//        as this class is used as a template parameter (instead of being
//        assigned to a base class pointer).
//     2) Internal storage to maintain their specific layout mechanism
//     3) the definition of a class pair_iterator, and a function
//        void getPairlist(int, pair_iterator&, pair_iterator&) to get a
//        begin/end iterator pair to access the local neighbor of the Nth
//        local atom.  This is not a virtual function, it is a requirement of
//        the templated class for use in other parts of the code.
//

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, typename... Properties>
        void ParticleLayout<T, Dim, Properties...>::applyBC(const particle_position_type& R,
                                                            const NDRegion<T, Dim>& nr) {
            /* loop over all faces
             * 0: lower x-face
             * 1: upper x-face
             * 2: lower y-face
             * 3: upper y-face
             * etc...
             */
            Kokkos::RangePolicy<typename particle_position_type::execution_space> policy{
                0, (unsigned)R.getParticleCount()};
            for (unsigned face = 0; face < 2 * Dim; ++face) {
                // unsigned face = i % Dim;
                unsigned d   = face / 2;
                bool isUpper = face & 1;
                switch (bcs_m[face]) {
                    case BC::PERIODIC:
                        // Periodic faces come in pairs and the application of
                        // BCs checks both sides, so there is no reason to
                        // apply periodic conditions twice
                        if (isUpper) {
                            break;
                        }

                        Kokkos::parallel_for("Periodic BC", policy,
                                             PeriodicBC(R.getView(), nr, d, isUpper));
                        break;
                    case BC::REFLECTIVE:
                        Kokkos::parallel_for("Reflective BC", policy,
                                             ReflectiveBC(R.getView(), nr, d, isUpper));
                        break;
                    case BC::SINK:
                        Kokkos::parallel_for("Sink BC", policy,
                                             SinkBC(R.getView(), nr, d, isUpper));
                        break;
                    case BC::NO:
                    default:
                        break;
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
