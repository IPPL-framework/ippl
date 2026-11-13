//
// TestP3MPairKernels
//   Compares the two traversals of TruncatedGreenParticleInteraction on the same overlap layout:
//   the per-cell pair kernel with one atomic per pair (pair_kernel = CELL_PAIRS) and the
//   per-owned-particle kernel with register accumulation (pair_kernel = OWNED_PARTICLE). The
//   fields must agree up to summation order. With larger N the program is the GPU benchmark of
//   the two kernels: it reports the time per solve and the directed candidate pairs per second.
//
//   With "moving", every particle gets a random normalized momentum and both kernels run the
//   moving-source mode, which also computes B. In both modes the moving-source mode must give the
//   static field (up to rounding) and exactly zero B when all momenta are zero; the program prints
//   both times for this case, which is the cost of the moving pair term.
//
//   Usage:
//     srun -n 1 ./TestP3MPairKernels [N=20000] [timed solves=3] [uniform|clustered] [rcut=0.08]
//                                    [static|moving]
//
//   Particles fill the unit box (uniform) or a Gaussian of rms 0.06 around its centre
//   (clustered, dense search cells). Exit code 0 when the relative max differences are below
//   1e-12 and the zero-momentum check is exact.
//

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>

#include "Interaction/TruncatedGreenParticleInteraction.h"
#include "Particle/ParticleSpatialOverlapLayout.h"

namespace {
    constexpr unsigned Dim = 3;
    using Scalar_t         = double;
    using Mesh_t           = ippl::UniformCartesian<Scalar_t, Dim>;
    using ParticleLayout_t = ippl::ParticleSpatialOverlapLayout<Scalar_t, Dim, Mesh_t>;
    using Vector_t         = ippl::Vector<Scalar_t, Dim>;

    class Particles : public ippl::ParticleBase<ParticleLayout_t> {
    public:
        explicit Particles(ParticleLayout_t& layout)
            : ippl::ParticleBase<ParticleLayout_t>(layout) {
            this->addAttribute(Q);
            this->addAttribute(P);
            this->addAttribute(E);
            this->addAttribute(B);
            this->setParticleBC(ippl::BC::NO);
        }
        ippl::ParticleAttrib<Scalar_t> Q;
        typename ippl::ParticleBase<ParticleLayout_t>::particle_position_type P;
        typename ippl::ParticleBase<ParticleLayout_t>::particle_position_type E;
        typename ippl::ParticleBase<ParticleLayout_t>::particle_position_type B;
    };

    using Interaction_t =
        ippl::TruncatedGreenParticleInteraction<Particles, Particles::particle_position_type,
                                                ippl::ParticleAttrib<Scalar_t>>;
    using exec_space  = Particles::particle_position_type::execution_space;
    using FieldView_t = Kokkos::View<Vector_t*, typename exec_space::memory_space>;

    double maxOverRanks(double value) {
        double global = 0;
        ippl::Comm->allreduce(value, global, 1, std::greater<double>());
        return global;
    }

    FieldView_t copyOwned(const Particles::particle_position_type& attribute, const char* label) {
        FieldView_t copy(label, attribute.getView().extent(0));
        Kokkos::deep_copy(copy, attribute.getView());
        return copy;
    }

    // Largest component of |a - b| and of |b| over the owned particles of all ranks.
    std::array<double, 2> maxDifference(const FieldView_t& a, const FieldView_t& b) {
        double diffLocal = 0;
        double refLocal  = 0;
        Kokkos::parallel_reduce(
            "compare fields", Kokkos::RangePolicy<exec_space>(0, a.extent(0)),
            KOKKOS_LAMBDA(const std::size_t i, double& diff, double& ref) {
                for (unsigned d = 0; d < Dim; ++d) {
                    diff = Kokkos::max(diff, Kokkos::abs(a(i)[d] - b(i)[d]));
                    ref  = Kokkos::max(ref, Kokkos::abs(b(i)[d]));
                }
            },
            Kokkos::Max<double>(diffLocal), Kokkos::Max<double>(refLocal));
        return {maxOverRanks(diffLocal), maxOverRanks(refLocal)};
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        Inform msg("TestP3MPairKernels");
        const std::size_t N             = argc > 1 ? std::stoull(argv[1]) : 20000;
        const int timedSolves           = argc > 2 ? std::stoi(argv[2]) : 3;
        const bool clustered            = argc > 3 && std::string(argv[3]) == "clustered";
        const Scalar_t rcut             = argc > 4 ? std::stod(argv[4]) : 0.08;
        const bool moving               = argc > 5 && std::string(argv[5]) == "moving";
        const ippl::Vector<int, Dim> nr = {16, 16, 16};

        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<Dim> fieldLayout(MPI_COMM_WORLD, owned, isParallel);
        Mesh_t mesh(owned, Vector_t(1.0 / nr[0]), Vector_t(0.0));
        ParticleLayout_t layout(fieldLayout, mesh, rcut);
        Particles particles(layout);

        // every rank creates its share anywhere in the box; update() moves them to their owners
        const std::size_t nranks = ippl::Comm->size();
        const std::size_t local  = N / nranks + (ippl::Comm->rank() < N % nranks ? 1 : 0);
        particles.create(local);
        {
            auto R = particles.R.getView();
            auto Q = particles.Q.getView();
            auto P = particles.P.getView();
            Kokkos::Random_XorShift64_Pool<exec_space> pool(12345 + ippl::Comm->rank());
            const Scalar_t charge = 1.0 / N;
            Kokkos::parallel_for(
                "init particles", Kokkos::RangePolicy<exec_space>(0, local),
                KOKKOS_LAMBDA(const std::size_t i) {
                    auto gen = pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        Scalar_t x = clustered ? 0.5 + 0.06 * gen.normal() : gen.drand(0.0, 1.0);
                        R(i)[d]    = Kokkos::min(Kokkos::max(x, 1e-9), 1.0 - 1e-9);
                    }
                    Q(i) = charge;
                    P(i) = Vector_t(0.0);
                    pool.free_state(gen);
                });
        }
        particles.update();

        // Directed candidate pairs (all j != i of the 27 neighbouring cells). Each owned particle
        // stores its count in its own entry and a reduction sums them. An earlier version added
        // every count to one unsigned long long with Kokkos::atomic_add, which made this program
        // appear to hang above about 2^22 particles on A100 and GH200. On CUDA (compute capability
        // >= 7.0), desul in Kokkos 5.2 has native atomic add only for (u)int32_t, (u)int64_t,
        // float and double. On Linux uint64_t is unsigned long, so unsigned long long falls back to
        // a compare-and-swap loop. With all threads on one address only one CAS succeeds per
        // round, so the count took 30 s for 2^20 particles on an A100, while one solve took 30 ms.
        Kokkos::View<unsigned long long*, typename exec_space::memory_space> perParticle(
            "candidates per particle", particles.getLocalNum());
        layout.forEachOwnedParticle<exec_space, unsigned long long>(
            KOKKOS_LAMBDA(const std::size_t, const std::size_t, unsigned long long& n) { ++n; },
            KOKKOS_LAMBDA(const std::size_t i, const unsigned long long& n) {
                perParticle(i) = n;
            });
        unsigned long long localCandidates = 0;
        Kokkos::parallel_reduce(
            "sum candidates", Kokkos::RangePolicy<exec_space>(0, perParticle.extent(0)),
            KOKKOS_LAMBDA(const std::size_t i, unsigned long long& sum) { sum += perParticle(i); },
            localCandidates);
        unsigned long long totalCandidates = 0;
        ippl::Comm->allreduce(localCandidates, totalCandidates, 1, std::plus<unsigned long long>());

        auto parameters = [&](int kernel) {
            ippl::ParameterList params;
            params.add("rcut", rcut);
            params.add("alpha", 2.0 / rcut);
            params.add("force_constant", 1.0);
            params.add("pair_kernel", kernel);
            params.add("speed_of_light", 1.0);
            return params;
        };
        // One untimed warm-up solve, then the mean time of the timed solves.
        auto timeSolves = [&](Interaction_t& interaction) {
            double seconds = 0;
            for (int s = -1; s < timedSolves; ++s) {
                particles.E = Vector_t(0.0);
                particles.B = Vector_t(0.0);
                Kokkos::fence();
                Kokkos::Timer timer;
                interaction.solve();
                Kokkos::fence();
                if (s >= 0) {
                    seconds += timer.seconds() / timedSolves;
                }
            }
            return seconds;
        };
        auto run = [&](int kernel, bool movingSources) {
            if (movingSources) {
                Interaction_t interaction(particles, particles.E, particles.B, particles.R,
                                          particles.P, particles.Q, parameters(kernel));
                return timeSolves(interaction);
            }
            Interaction_t interaction(particles, particles.E, particles.R, particles.Q,
                                      parameters(kernel));
            return timeSolves(interaction);
        };

        // All momenta are zero here, ghosts included, so the moving-source term reduces to the
        // static one and u x F vanishes. E may still differ in the last bits, because the
        // compiler can fuse the accumulation into a multiply-add differently in the two kernels.
        const double staticAtRest = maxOverRanks(run(Interaction_t::OWNED_PARTICLE, false));
        const auto staticE        = copyOwned(particles.E, "static E");
        const double movingAtRest = maxOverRanks(run(Interaction_t::OWNED_PARTICLE, true));
        const auto restE          = maxDifference(copyOwned(particles.E, "moving E"), staticE);
        const auto restB          = maxDifference(copyOwned(particles.B, "moving B"),
                                                  FieldView_t("zero", staticE.extent(0)));
        const bool restAgrees     = restE[0] <= 1e-13 * restE[1] && restB[0] == 0.0;

        if (moving) {
            auto P = particles.P.getView();
            Kokkos::Random_XorShift64_Pool<exec_space> pool(54321 + ippl::Comm->rank());
            Kokkos::parallel_for(
                "init momenta", Kokkos::RangePolicy<exec_space>(0, particles.getLocalNum()),
                KOKKOS_LAMBDA(const std::size_t i) {
                    auto gen = pool.get_state();
                    for (unsigned d = 0; d < Dim; ++d) {
                        P(i)[d] = gen.drand(-2.0, 2.0);
                    }
                    pool.free_state(gen);
                });
            // The ghosts must carry the new momenta; the positions and owners do not change.
            particles.update();
        }

        double seconds[2] = {0, 0};
        FieldView_t referenceE, referenceB;
        for (int kernel : {Interaction_t::CELL_PAIRS, Interaction_t::OWNED_PARTICLE}) {
            seconds[kernel] = maxOverRanks(run(kernel, moving));
            if (kernel == Interaction_t::CELL_PAIRS) {
                referenceE = copyOwned(particles.E, "reference E");
                referenceB = copyOwned(particles.B, "reference B");
            }
        }
        const auto differenceE = maxDifference(copyOwned(particles.E, "E"), referenceE);
        const auto differenceB = maxDifference(copyOwned(particles.B, "B"), referenceB);
        const double relative  = differenceE[0] / differenceE[1];
        const double relativeB = moving ? differenceB[0] / differenceB[1] : 0.0;

        const double cellPairs = seconds[Interaction_t::CELL_PAIRS];
        const double ownedPart = seconds[Interaction_t::OWNED_PARTICLE];
        msg << (clustered ? "clustered" : "uniform") << ", N = " << N << ", rcut = " << rcut << ", "
            << nranks << " ranks, " << totalCandidates << " directed candidates ("
            << static_cast<double>(totalCandidates) / N << " per particle), "
            << (moving ? "moving" : "static") << " sources" << endl;
        msg << "CELL_PAIRS     " << cellPairs * 1e3 << " ms per solve, "
            << totalCandidates / cellPairs / nranks << " candidates/s per rank" << endl;
        msg << "OWNED_PARTICLE " << ownedPart * 1e3 << " ms per solve, "
            << totalCandidates / ownedPart / nranks << " candidates/s per rank, speedup "
            << cellPairs / ownedPart << endl;
        msg << "relative max field difference " << relative << endl;
        if (moving) {
            msg << "relative max B difference " << relativeB << endl;
        }
        msg << "zero momenta, OWNED_PARTICLE: moving-source mode " << movingAtRest * 1e3
            << " ms per solve, static mode " << staticAtRest * 1e3 << " ms per solve, ratio "
            << movingAtRest / staticAtRest << ", relative max E difference " << restE[0] / restE[1]
            << ", max |B| " << restB[0] << endl;
        if (!(relative < 1e-12) || !(relativeB < 1e-12)) {
            msg << "FAIL: the kernels disagree" << endl;
            status = 1;
        }
        if (!restAgrees) {
            msg << "FAIL: zero momenta do not reproduce the static field" << endl;
            status = 1;
        }
    }
    ippl::finalize();
    return status;
}
