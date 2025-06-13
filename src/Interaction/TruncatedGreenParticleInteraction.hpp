

namespace ippl {
    template <typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    KOKKOS_INLINE_FUNCTION constexpr
        typename TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                                   VectorAttribute>::Vector_t
        TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                          VectorAttribute>::pairForce(const Vector_t& dist,
                                                                      Scalar_t r2, Scalar_t alpha,
                                                                      Scalar_t forceConstant,
                                                                      Scalar_t qm2) {
        const Scalar_t r = Kokkos::sqrt(r2);

        return forceConstant * qm2 * (dist / r)
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
        auto& F        = *F_m;
        const auto& R  = *R_m;
        const auto& QM = *QM_m;

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

                const auto F_ij = pairForce(dist_ij, rsq_ij, alpha, forceConstant);

                // TODO is energy nonetheless? in anycase its F/QM(i)
                Kokkos::atomic_sub(&F(i), F_ij * QM(j));
                // Kokkos::atomic_add(&F(j), F_ij * QM(i));
            });
        IpplTimings::stopTimer(solveTimer);
    }
}  // namespace ippl
