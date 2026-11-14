//
// Class TruncatedGreenParticleInteraction
//   This class implements the short-range part of the Ewald Green function split by evaluating the
//   gradient of forceConstant * erfc(alpha * r) / r. Below regularization_cutoff, the singular
//   Coulomb contribution is replaced by a linear force that matches continuously at the cutoff.
//   The long-range part is handled by the periodic or open-boundary P3M mesh solver.
//
//   It assumes that ParticleContainer implements a function forAllPairs() to iterate over all
//   relevant particle pairs.
//

namespace ippl {
    template <typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    KOKKOS_INLINE_FUNCTION constexpr typename TruncatedGreenParticleInteraction<
        ParticleContainer, ScalarAttribute, VectorAttribute>::Vector_t
    TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                      VectorAttribute>::fieldFromPair(const Vector_t& dist,
                                                                      Scalar_t r2, Scalar_t alpha,
                                                                      Scalar_t forceConstant,
                                                                      Scalar_t qm, Scalar_t reg) {
        const Scalar_t r = Kokkos::sqrt(r2);

        if (r < reg) {
            // Replace the singular Coulomb term with a linear force that is continuous at reg.
            const Scalar_t longRangeCorrection =
                (Kokkos::erf(alpha * r)
                 - 2.0 * alpha * r * Kokkos::exp(-alpha * alpha * r2)
                       / Kokkos::sqrt(Kokkos::numbers::pi))
                / (r2 * r);
            return (1.0 / (reg * reg * reg) - longRangeCorrection) * forceConstant * qm * dist;
        }

        // F = - q * forceConstant grad [(1 - erf(alpha * r)) / r].
        return forceConstant * qm * (dist / r)
               * (2.0 * alpha * Kokkos::exp(-alpha * alpha * r2)
                      / (Kokkos::sqrt(Kokkos::numbers::pi) * r)
                  + (1.0 - Kokkos::erf(alpha * r)) / r2);
    }

    template <typename ParticleContainer, typename VectorAttribute, typename ScalarAttribute>
    KOKKOS_INLINE_FUNCTION bool TruncatedGreenParticleInteraction<
        ParticleContainer, VectorAttribute,
        ScalarAttribute>::movingSourceFromPair(const Vector_t& dist, Scalar_t r2, const Vector_t& u,
                                               Scalar_t rcut2, Scalar_t alpha,
                                               Scalar_t forceConstant, Scalar_t qm, Scalar_t reg,
                                               Vector_t& e, Vector_t& b) {
        // Squared distance in the source rest frame: the component along u is stretched by gamma.
        const Scalar_t along = u.dot(dist);
        const Scalar_t rest2 = r2 + along * along;
        if (rest2 >= rcut2) {
            return false;
        }
        // The lab separation keeps the direction; only the radial term sees the rest distance.
        const Vector_t F = fieldFromPair(dist, rest2, alpha, forceConstant, qm, reg);
        e                = Kokkos::sqrt(1.0 + u.dot(u)) * F;
        b                = cross(u, F);
        return true;
    }

    template <typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    void TruncatedGreenParticleInteraction<ParticleContainer, ScalarAttribute,
                                           VectorAttribute>::solve() {
        // Short name: IpplTimings prints timer names in a 20-character column.
        static IpplTimings::TimerRef solveTimer = IpplTimings::getTimer("P3M: PP interaction");
        IpplTimings::startTimer(solveTimer);
        // get particle data
        auto& Field    = Field_m;
        const auto& R  = R_m;
        const auto& QM = QM_m;

        // get simulation specific data
        const auto rcut           = this->params_m.template get<Scalar_t>("rcut");
        const auto rcut2          = std::pow<Scalar_t>(rcut, 2);
        const auto alpha          = this->params_m.template get<Scalar_t>("alpha");
        const auto forceConstant  = this->params_m.template get<Scalar_t>("force_constant");
        const auto regularization = this->params_m.template get<Scalar_t>("regularization_cutoff");

        if (regularization <= 0.0) {
            IpplTimings::stopTimer(solveTimer);
            throw IpplException("TruncatedGreenParticleInteraction::solve",
                                "regularization_cutoff must be positive");
        }

        const auto& particleLayout = this->pc_m.getLayout();
        // The search cells and halos only cover the layout cutoff; partners beyond it are missed.
        if (rcut > particleLayout.getCutoff()) {
            IpplTimings::stopTimer(solveTimer);
            Inform m("TruncatedGreenParticleInteraction::solve");
            m << level1 << "rcut " << rcut << " exceeds the particle layout cutoff "
              << particleLayout.getCutoff() << "." << endl;

            throw IpplException("TruncatedGreenParticleInteraction::solve",
                                "rcut exceeds layout cutoff");
        }

        const bool cellPairs = this->params_m.template get<int>("pair_kernel") == CELL_PAIRS;
        if (P_m != nullptr) {
            // The constructor of this mode asserts three dimensions.
            if constexpr (Vector_t::dim == 3) {
                const auto speedOfLight = this->params_m.template get<Scalar_t>("speed_of_light");
                if (!(speedOfLight > 0.0)) {
                    IpplTimings::stopTimer(solveTimer);
                    throw IpplException("TruncatedGreenParticleInteraction::solve",
                                        "speed_of_light must be positive");
                }
                const Scalar_t inverseLight = 1.0 / speedOfLight;
                auto& B                     = *B_m;
                const auto& P               = *P_m;
                if (cellPairs) {
                    particleLayout.template forEachPair<execution_space>(
                        KOKKOS_LAMBDA(const size_t& i, const size_t& j) {
                            const Vector_t dist_ij = R(i) - R(j);
                            const Scalar_t rsq_ij  = dist_ij.dot(dist_ij);
                            if (rsq_ij >= rcut2 || rsq_ij == 0.0) {
                                return;
                            }
                            Vector_t e, b;
                            if (movingSourceFromPair(dist_ij, rsq_ij, P(j), rcut2, alpha,
                                                     forceConstant, QM(j), regularization, e, b)) {
                                Kokkos::atomic_sub(&Field(i), e);
                                const Vector_t magnetic = inverseLight * b;
                                Kokkos::atomic_sub(&B(i), magnetic);
                            }
                        });
                } else {
                    using Sums_t = detail::MovingSourceSums<Vector_t>;
                    particleLayout.template forEachOwnedParticle<execution_space, Sums_t>(
                        KOKKOS_LAMBDA(const size_t& i, const size_t& j, Sums_t& acc) {
                            const Vector_t dist_ij = R(i) - R(j);
                            const Scalar_t rsq_ij  = dist_ij.dot(dist_ij);
                            if (rsq_ij >= rcut2 || rsq_ij == 0.0) {
                                return;
                            }
                            Vector_t e, b;
                            if (movingSourceFromPair(dist_ij, rsq_ij, P(j), rcut2, alpha,
                                                     forceConstant, QM(j), regularization, e, b)) {
                                acc.e += e;
                                acc.b += b;
                            }
                        },
                        KOKKOS_LAMBDA(const size_t& i, const Sums_t& acc) {
                            Field(i) -= acc.e;
                            B(i) -= inverseLight * acc.b;
                        });
                }
            }
        } else if (cellPairs) {
            particleLayout.template forEachPair<execution_space>(
                KOKKOS_LAMBDA(const size_t& i, const size_t& j) {
                    const Vector_t dist_ij = R(i) - R(j);
                    const Scalar_t rsq_ij  = dist_ij.dot(dist_ij);

                    // Distinct particles at the same position contribute no force by convention.
                    if (rsq_ij >= rcut2 || rsq_ij == 0.0) {
                        return;
                    }

                    const auto F_ij =
                        fieldFromPair(dist_ij, rsq_ij, alpha, forceConstant, QM(j), regularization);

                    // add force to particle i, don't do it for j as the ranges of i and j are
                    // asymmetric
                    Kokkos::atomic_sub(&Field(i), F_ij);
                });
        } else {
            particleLayout.template forEachOwnedParticle<execution_space, Vector_t>(
                KOKKOS_LAMBDA(const size_t& i, const size_t& j, Vector_t& acc) {
                    const Vector_t dist_ij = R(i) - R(j);
                    const Scalar_t rsq_ij  = dist_ij.dot(dist_ij);
                    if (rsq_ij >= rcut2 || rsq_ij == 0.0) {
                        return;
                    }
                    acc +=
                        fieldFromPair(dist_ij, rsq_ij, alpha, forceConstant, QM(j), regularization);
                },
                KOKKOS_LAMBDA(const size_t& i, const Vector_t& acc) { Field(i) -= acc; });
        }
        IpplTimings::stopTimer(solveTimer);
    }
}  // namespace ippl
