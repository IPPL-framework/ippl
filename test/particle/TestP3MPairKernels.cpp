//
// TestP3MPairKernels
//   Compares the two traversals of TruncatedGreenParticleInteraction on the same overlap layout:
//   the per-cell pair kernel with one atomic per pair (pair_kernel = CELL_PAIRS) and the
//   per-owned-particle kernel with register accumulation (pair_kernel = OWNED_PARTICLE). The
//   fields must agree up to summation order. With larger N the program is the GPU benchmark of
//   the two kernels: it reports the time per solve and the directed candidate pairs per second.
//
//   Usage:
//     srun -n 1 ./TestP3MPairKernels [N=20000] [timed solves=3] [uniform|clustered] [rcut=0.08]
//
//   Particles fill the unit box (uniform) or a Gaussian of rms 0.06 around its centre
//   (clustered, dense search cells). Exit code 0 when the relative max difference is below 1e-12.
//

#include "Ippl.h"

#include <Kokkos_Random.hpp>

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
            this->addAttribute(E);
            this->setParticleBC(ippl::BC::NO);
        }
        ippl::ParticleAttrib<Scalar_t> Q;
        typename ippl::ParticleBase<ParticleLayout_t>::particle_position_type E;
    };

    using Interaction_t =
        ippl::TruncatedGreenParticleInteraction<Particles, Particles::particle_position_type,
                                                ippl::ParticleAttrib<Scalar_t>>;
    using exec_space = Particles::particle_position_type::execution_space;

    double maxOverRanks(double value) {
        double global = 0;
        ippl::Comm->allreduce(value, global, 1, std::greater<double>());
        return global;
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        Inform msg("TestP3MPairKernels");
        const std::size_t N        = argc > 1 ? std::stoull(argv[1]) : 20000;
        const int timedSolves      = argc > 2 ? std::stoi(argv[2]) : 3;
        const bool clustered       = argc > 3 && std::string(argv[3]) == "clustered";
        const Scalar_t rcut        = argc > 4 ? std::stod(argv[4]) : 0.08;
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
                    pool.free_state(gen);
                });
        }
        particles.update();

        // directed candidate pairs (all j != i of the 27 neighbouring cells)
        Kokkos::View<unsigned long long, typename exec_space::memory_space> candidates("cand");
        layout.forEachOwnedParticle<exec_space, unsigned long long>(
            KOKKOS_LAMBDA(const std::size_t, const std::size_t, unsigned long long& n) { ++n; },
            KOKKOS_LAMBDA(const std::size_t, const unsigned long long& n) {
                Kokkos::atomic_add(&candidates(), n);
            });
        unsigned long long localCandidates = 0;
        Kokkos::deep_copy(localCandidates, candidates);
        unsigned long long totalCandidates = 0;
        ippl::Comm->allreduce(localCandidates, totalCandidates, 1,
                              std::plus<unsigned long long>());

        Kokkos::View<Vector_t*, typename exec_space::memory_space> reference("reference",
                                                                            particles.getLocalNum());
        double seconds[2] = {0, 0};
        for (int kernel : {Interaction_t::CELL_PAIRS, Interaction_t::OWNED_PARTICLE}) {
            ippl::ParameterList params;
            params.add("rcut", rcut);
            params.add("alpha", 2.0 / rcut);
            params.add("force_constant", 1.0);
            params.add("pair_kernel", kernel);
            Interaction_t interaction(particles, particles.E, particles.R, particles.Q, params);
            for (int s = -1; s < timedSolves; ++s) {  // one untimed warm-up solve
                particles.E = Vector_t(0.0);
                Kokkos::fence();
                Kokkos::Timer timer;
                interaction.solve();
                Kokkos::fence();
                if (s >= 0) {
                    seconds[kernel] += timer.seconds() / timedSolves;
                }
            }
            if (kernel == Interaction_t::CELL_PAIRS) {
                Kokkos::deep_copy(reference,
                                  Kokkos::subview(particles.E.getView(),
                                                  Kokkos::make_pair<std::size_t>(
                                                      0, particles.getLocalNum())));
            }
        }

        const auto field = particles.E.getView();
        double local2[2] = {0, 0};
        Kokkos::parallel_reduce(
            "compare kernels", Kokkos::RangePolicy<exec_space>(0, particles.getLocalNum()),
            KOKKOS_LAMBDA(const std::size_t i, double& diff, double& ref) {
                for (unsigned d = 0; d < Dim; ++d) {
                    diff = Kokkos::max(diff, Kokkos::abs(field(i)[d] - reference(i)[d]));
                    ref  = Kokkos::max(ref, Kokkos::abs(reference(i)[d]));
                }
            },
            Kokkos::Max<double>(local2[0]), Kokkos::Max<double>(local2[1]));
        const double relative = maxOverRanks(local2[0]) / maxOverRanks(local2[1]);

        const double cellPairs = maxOverRanks(seconds[Interaction_t::CELL_PAIRS]);
        const double ownedPart = maxOverRanks(seconds[Interaction_t::OWNED_PARTICLE]);
        msg << (clustered ? "clustered" : "uniform") << ", N = " << N << ", rcut = " << rcut
            << ", " << nranks << " ranks, " << totalCandidates << " directed candidates ("
            << static_cast<double>(totalCandidates) / N << " per particle)" << endl;
        msg << "CELL_PAIRS     " << cellPairs * 1e3 << " ms per solve, "
            << totalCandidates / cellPairs / nranks << " candidates/s per rank" << endl;
        msg << "OWNED_PARTICLE " << ownedPart * 1e3 << " ms per solve, "
            << totalCandidates / ownedPart / nranks << " candidates/s per rank, speedup "
            << cellPairs / ownedPart << endl;
        msg << "relative max field difference " << relative << endl;
        if (!(relative < 1e-12)) {
            msg << "FAIL: the kernels disagree" << endl;
            status = 1;
        }
    }
    ippl::finalize();
    return status;
}
