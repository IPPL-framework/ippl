

namespace ippl {
    template<typename ParticleContainer, typename ScalarAttribute, typename VectorAttribute>
    void TruncatedGreenInteraction<ParticleContainer, ScalarAttribute, VectorAttribute>::solve() {
        // get particle data
        auto R = *R_m;
        auto F = *F_m;
        auto QM = *QM_m;

        // get simulation specific data
        auto rcut = this->params_m.template get<Scalar_t>("rcut");
        auto alpha = this->params_m.template get<Scalar_t>("alpha");
        auto forceConstant = this->params_m.template get<Scalar_t>("force_constant");

        const auto nloc = this->pc_m.getLocalNum();
        // TODO where to get this info from? Should it take PC as template param instead?
        const auto &particleLayout = this->pc_m.getLayout();
        auto data = particleLayout.getNeighborData();

        using team_policy_t = Kokkos::TeamPolicy<typename ScalarAttribute::execution_space>;
        using team_t = typename team_policy_t::member_type;

        // calculate interaction force
        Kokkos::parallel_for("Particle-Particle", team_policy_t(nloc, Kokkos::AUTO()),
                             KOKKOS_LAMBDA(const team_t &team) {
                                 const size_type particleIndex = team.league_rank();


                                 typename ParticleContainer::NList_t neighborList;

                                 particleLayout.getNeighbors(R(particleIndex), data, neighborList);

                                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, neighborList.extent(0)),
                                                      [&](const int &neighborIdx) {
                                                          if (neighborIdx == particleIndex) { return; }
                                                          Vector_t dist_ij = R(particleIndex) - R(neighborIdx);
                                                          const auto rsq_ij = dist_ij.dot(dist_ij);
                                                          double r_ij = Kokkos::sqrt(rsq_ij);

                                                          if (r_ij >= rcut) { return; }

                                                          // TODO make this KOKKOS inline function?
                                                          // calculate and apply force
                                                          Vector_t F_ij =
                                                                  forceConstant * (dist_ij / r_ij) * (2.0 * alpha * Kokkos::exp(
                                                                          -alpha * alpha * rsq_ij) /
                                                                      (Kokkos::sqrt(Kokkos::numbers::pi) * r_ij) + (
                                                                          1.0 - Kokkos::erf(alpha * r_ij)) / rsq_ij);
                                                          Kokkos::atomic_sub(&F(particleIndex), F_ij * QM(neighborIdx));
                                                          // Kokkos::atomic_add(&E(neighborList), F_ij * Q(particleIndex));
                                                      }
                                 );
                             });

        // Kokkos::fence();
        // ippl::Comm->barrier();

        // std::cerr << "Particle-Particle Interaction finished" << std::endl;
    }
}
