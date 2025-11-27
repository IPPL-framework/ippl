//
// TestNUFFTParamSweep
//   Test to sweep through TiledScatter parameters and measure performance
//
#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "Utility/IpplTimings.h"

#include "FFT/FFT.h"
#include "FFT/NUFFT/NativeNUFFT.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleLayout.h"

constexpr unsigned Dim = 3;
using T                = double;

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

template <typename Vec, class GeneratorPool, unsigned Dim>
struct generate_random_particles_with_charges {
    using view_type        = typename ippl::detail::ViewType<Vec, 1>::view_type;
    using value_type       = typename Vec::value_type;
    using view_type_scalar = typename ippl::detail::ViewType<value_type, 1>::view_type;

    view_type x;
    view_type_scalar Q;
    GeneratorPool rand_pool;
    Vec minU, maxU;

    generate_random_particles_with_charges(view_type x_, view_type_scalar Q_,
                                           GeneratorPool rand_pool_, Vec& minU_, Vec& maxU_)
        : x(x_)
        , Q(Q_)
        , rand_pool(rand_pool_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        for (unsigned d = 0; d < Dim; ++d) {
            x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
        }
        Q(i) = rand_gen.drand(0.5, 1.5);

        rand_pool.free_state(rand_gen);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("TestNUFFTParamSweep");

        using Mesh_t      = ippl::UniformCartesian<double, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;

        using playout_type = ippl::ParticleSpatialLayout<double, Dim>;
        using bunch_type   = Bunch<playout_type>;
        using size_type    = ippl::detail::size_type;

        const double pi = std::acos(-1.0);

        std::array<size_t, Dim> pt = {32, 32, 32};

        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<Dim> owned(I, J, K);

        std::array<bool, Dim> isParallel;
        isParallel.fill(false);

        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        typedef ippl::Vector<double, 3> Vector_t;
        Vector_t minU = {-pi, -pi, -pi};
        Vector_t maxU = {pi, pi, pi};

        std::array<double, Dim> dx = {
            (maxU[0] - minU[0]) / double(pt[0]),
            (maxU[1] - minU[1]) / double(pt[1]),
            (maxU[2] - minU[2]) / double(pt[2]),
        };

        Vector_t hx     = {dx[0], dx[1], dx[2]};
        Vector_t origin = {minU[0], minU[1], minU[2]};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl(layout, mesh);
        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::PERIODIC);

        size_type Np = std::pow(32, 3);

        msg << "Number of particles: " << Np << endl;
        msg << "Grid size: " << pt[0] << " x " << pt[1] << " x " << pt[2] << endl;

        // Initialize particles
        bunch.create(Np);

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));
        Kokkos::parallel_for(
            Np,
            generate_random_particles_with_charges<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                bunch.R.getView(), bunch.Q.getView(), rand_pool64, minU, maxU));

        typedef ippl::Field<Kokkos::complex<double>, Dim, Mesh_t, Centering_t>::uniform_type
            field_type;
        field_type field(mesh, layout);

        msg << "Testing with 32^3 grid and 32^3 particles" << endl;

        // Parameter sweep configurations
        std::vector<int> tile_sizes = {8, 16, 24, 32};
        std::vector<int> team_sizes = {32, 64, 128, 256};
        std::vector<int> z_tiles    = {1, 2, 4, 8};

        // Output file for results
        std::ofstream outfile("nufft_param_sweep.csv");
        outfile << "tile_size,team_size,z_tiles,time_ms,throughput_Mpts_s" << std::endl;

        msg << "\n=== Parameter Sweep ===" << endl;
        msg << "Testing " << tile_sizes.size() * team_sizes.size() * z_tiles.size()
            << " configurations..." << endl;

        int config_count = 0;
        for (int tile_size : tile_sizes) {
            for (int team_size : team_sizes) {
                for (int z_tile : z_tiles) {
                    config_count++;

                    // Create FFT with tiled scatter configuration
                    ippl::ParameterList fftParams;
                    fftParams.add("tolerance", 1e-10);
                    fftParams.add("use_finufft_defaults", false);
                    fftParams.add("use_kokkos_nufft", false);
                    fftParams.add("spread_method", "tiled");
                    fftParams.add("tile_size_3d", tile_size);
                    fftParams.add("z_tiles", z_tile);
                    fftParams.add("team_size", team_size);
                    fftParams.add("sort", true);

                    typedef ippl::Field<double, Dim, Mesh_t, Centering_t>::uniform_type
                        real_field_type;
                    typedef ippl::FFT<ippl::NUFFTransform, real_field_type> FFT_type;

                    auto nufft = std::make_unique<FFT_type>(layout, Np, 1, fftParams);

                    double avg_time_ms;
                    double throughput;

                    try {
                        // Warm-up runs
                        for (int i = 0; i < 2; ++i) {
                            nufft->transform(bunch.R, bunch.Q, field);
                            Kokkos::fence();
                        }
                        // Timed runs
                        constexpr int num_runs = 10;
                        auto start             = std::chrono::high_resolution_clock::now();
                        for (int run = 0; run < num_runs; ++run) {
                            nufft->transform(bunch.R, bunch.Q, field);
                        }
                        Kokkos::fence();
                        auto end = std::chrono::high_resolution_clock::now();

                        std::chrono::duration<double> elapsed = end - start;
                        double total_time                     = elapsed.count();

                        double avg_time_s = total_time / num_runs;
                        avg_time_ms       = avg_time_s * 1000.0;

                        // Calculate throughput
                        throughput = (Np / 1e6) / avg_time_s;  // Mpts/s
                    } catch (std::runtime_error& e) {
                        // Catch kernel launch OOM
                        avg_time_ms = std::nan("");
                        throughput  = std::nan("");
                    }

                    msg << "Config " << config_count << ": "
                        << "tile=" << tile_size << " "
                        << "team=" << team_size << " "
                        << "z=" << z_tile << " -> " << avg_time_ms << " ms "
                        << "(" << throughput << " Mpts/s)" << endl;

                    outfile << tile_size << "," << team_size << "," << z_tile << "," << avg_time_ms
                            << "," << throughput << std::endl;
                }
            }
        }

        outfile.close();
        msg << "\nResults written to nufft_param_sweep.csv" << endl;

        // Find and report best configuration
        std::ifstream infile("nufft_param_sweep.csv");
        std::string line;
        std::getline(infile, line);  // skip header

        double best_time = std::numeric_limits<double>::max();
        int best_tile = 0, best_team = 0, best_z = 0;

        while (std::getline(infile, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            if (tokens.size() >= 4) {
                int tile    = std::stoi(tokens[0]);
                int team    = std::stoi(tokens[1]);
                int z       = std::stoi(tokens[2]);
                double time = std::stod(tokens[3]);

                if (time < best_time) {
                    best_time = time;
                    best_tile = tile;
                    best_team = team;
                    best_z    = z;
                }
            }
        }
        infile.close();

        msg << "\n=== Best Configuration ===" << endl;
        msg << "Tile size: " << best_tile << endl;
        msg << "Team size: " << best_team << endl;
        msg << "Z tiles: " << best_z << endl;
        msg << "Time: " << best_time << " ms" << endl;
    }
    ippl::finalize();

    return 0;
}
