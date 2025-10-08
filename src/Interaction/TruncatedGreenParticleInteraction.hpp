//
// Class TruncatedGreenParticleInteraction
//   This class implements the short range interaction part of the green function splitting obtained
//   by taking the gradient of [forceConstant * (1 - erf(alpha * r)) / r]. The long range part is
//   handled by FFTTruncatedGreenPeriodicSolver.
//
//   It assumes that ParticleContainer implements a function forAllPairs() to iterate over all
//   relevant particle pairs.
//

namespace ippl {
    template <typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    KOKKOS_INLINE_FUNCTION constexpr
        typename TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                                   VectorAttribute>::Vector_t
        TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                          VectorAttribute>::fieldFromPair(const Vector_t& dist,
                                                                          Scalar_t r2,
                                                                          Scalar_t alpha,
                                                                          Scalar_t forceConstant,
                                                                          Scalar_t qm) {
        const Scalar_t r = Kokkos::sqrt(r2);

        // F = - q * forceConstant grad [(1 - erf(alpha * r)) / r].
        return forceConstant * qm * (dist / r)
               * (2.0 * alpha * Kokkos::exp(-alpha * alpha * r2)
                      / (Kokkos::sqrt(Kokkos::numbers::pi) * r)
                  + (1.0 - Kokkos::erf(alpha * r)) / r2);
    }

    template <typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    void TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                           VectorAttribute>::solve() {
        static IpplTimings::TimerRef solveTimer =
            IpplTimings::getTimer("TruncatedGreenParticleInteraction::solve()");
        IpplTimings::startTimer(solveTimer);
        // get particle data
        auto& Field    = Field_m;
        const auto& R  = R_m;
        const auto& QM = QM_m;

        // get simulation specific data
        const auto rcut2 = std::pow<Scalar_t>(this->params_m.template get<Scalar_t>("rcut"), 2);
        const auto alpha = this->params_m.template get<Scalar_t>("alpha");
        const auto forceConstant = this->params_m.template get<Scalar_t>("force_constant");

        const auto& particleLayout = this->pc_m.getLayout();

        particleLayout.template forEachPair<execution_space>(
            KOKKOS_LAMBDA(const size_t& i, const size_t& j) {
                const Vector_t dist_ij = R(i) - R(j);
                const Scalar_t rsq_ij  = dist_ij.dot(dist_ij);

                if (rsq_ij >= rcut2) {
                    return;
                }

                const auto F_ij = fieldFromPair(dist_ij, rsq_ij, alpha, forceConstant, QM(j));

                // add force to particle i, don't do it for j as the ranges of i and j are
                // asymmetric
                Kokkos::atomic_sub(&Field(i), F_ij);
            });
        IpplTimings::stopTimer(solveTimer);
    }
}  // namespace ippl
