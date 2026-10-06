/** @file CompareCosmologyForce.cpp
 * @brief Frozen-particle adapter using the production IPPL CIC/FFT/gather path.
 * @ingroup cosmology_diagnostics
 * @see cosmology_contracts cosmology_validation cosmology_references
 */
// Frozen-particle adapter for independent particle-mesh force comparisons.
// All force operations use Simulation::solveForce; this file supplies only
// validated host input, MPI distribution, and diagnostic output.
#include "../CosmologySimulation.h"

#include <array>
#include <charconv>
#include <limits>
#include <sstream>
#include <string>

namespace {

template <class Number>
/**
 * @brief Parse one host scalar without accepting trailing characters.
 * @see cosmology_contracts cosmology_validation
 * @tparam Number Parsed scalar type.
 * @param text Scalar input string; the complete value must parse without trailing characters.
 * @param name Scalar/diagnostic filename or label used in error reporting and output identity.
 * @return Parsed Number value; malformed input raises std::invalid_argument.
 */
Number parseNumber(const std::string& text, const std::string& name) {
    std::istringstream stream(text);
    Number value;
    if (!(stream >> value)) throw std::invalid_argument("Invalid " + name + ": " + text);
    stream >> std::ws;
    if (!stream.eof()) throw std::invalid_argument("Trailing characters in " + name + ": " + text);
    return value;
}

/**
 * @brief Validate an unsigned particle label in the complete expected global ID interval.
 * @see cosmology_contracts cosmology_validation
 * @param text Scalar input string; the complete value must parse without trailing characters.
 * @param total Exact expected particle count; valid IDs occupy [0,total).
 * @return Global uint64 particle ID after range validation.
 */
std::uint64_t parseId(const std::string& text, std::uint64_t total) {
    std::uint64_t value;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || value >= total)
        throw std::invalid_argument("Particle ID must be an integer in [0,N^3): " + text);
    return value;
}

/**
 * @brief Wrap finite comoving positions into the half-open periodic box, including rounded endpoints.
 * @see cosmology_contracts cosmology_validation
 * @param value Serialized scalar argument, validated according to the parser's strict range.
 * @param box Positive periodic side in comoving Mpc/h.
 * @return Comoving coordinate in [0,box).
 */
double wrapPosition(double value, double box) {
    if (!std::isfinite(value)) throw std::invalid_argument("Particle positions must be finite");
    value = std::fmod(value, box);
    if (value < 0.0) value += box;
    // For a tiny negative value, adding box can round to exactly box.
    if (value >= box) value = 0.0;
    return value;
}

/**
 * @brief Broadcast root input failure text so all communicator ranks take the same error path.
 * @see cosmology_contracts cosmology_validation
 * @param error Root-owned diagnostic text; empty means root validation succeeded.
 * @param communicator Communicator entered by all participating ranks in the same order.
 */
void broadcastRootError(std::string& error, MPI_Comm communicator) {
    int length = int(error.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, communicator);
    error.resize(length);
    if (length) {
        MPI_Bcast(error.data(), length, MPI_CHAR, 0, communicator);
        throw std::runtime_error(error);
    }
}

/**
 * @brief Read and validate the exact frozen-force unit-mass CSV, ordered into a host global-ID array.
 * @see cosmology_contracts cosmology_validation
 * @param path Input CSV or comparison output path following the exact file contract.
 * @param total Exact expected particle count; valid IDs occupy [0,total).
 * @param box Positive periodic side in comoving Mpc/h.
 * @return Host xyz coordinates ordered by global particle ID, in Mpc/h.
 */
std::vector<double> readPositions(const std::string& path, std::uint64_t total, double box) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open frozen particle CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::invalid_argument("Empty frozen particle CSV");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "id,x,y,z,mass")
        throw std::invalid_argument("Frozen particle CSV header must be exactly id,x,y,z,mass");
    std::vector<double> positions(3 * total);
    std::vector<bool> seen(total, false);
    std::uint64_t rows = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (++rows > total) throw std::invalid_argument("Frozen particle CSV has more than N^3 rows");
        std::array<std::string, 5> fields;
        std::istringstream row(line);
        for (auto& field : fields)
            if (!std::getline(row, field, ','))
                throw std::invalid_argument("Frozen particle CSV row requires five fields");
        // Exactly four commas are required; reject extra fields and a trailing comma.
        if (std::count(line.begin(), line.end(), ',') != 4)
            throw std::invalid_argument("Frozen particle CSV row requires exactly five fields");
        const auto id = parseId(fields[0], total);
        if (seen[id]) throw std::invalid_argument("Duplicate frozen particle ID: " + fields[0]);
        seen[id] = true;
        const double mass = parseNumber<double>(fields[4], "particle mass");
        if (!std::isfinite(mass) || mass != 1.0)
            throw std::invalid_argument("Frozen force comparison requires each mass to equal 1");
        for (int d = 0; d < 3; ++d)
            positions[3 * id + d] = wrapPosition(parseNumber<double>(fields[d + 1], "position"), box);
    }
    if (input.bad()) throw std::runtime_error("Frozen particle CSV read failed");
    if (rows != total || !std::all_of(seen.begin(), seen.end(), [](bool present) { return present; }))
        throw std::invalid_argument("Frozen particle CSV must contain every ID in [0,N^3) exactly once");
    return positions;
}

}  // namespace

namespace cosmology {

void Simulation::compareFrozenForce(const std::string& inputCsv, const std::string& outputDirectory) {
    const auto communicator = ippl::Comm->getCommunicator();
    const int rank = ippl::Comm->rank(), ranks = ippl::Comm->size();
    const auto total = totalParticles();
    if (total > std::uint64_t(std::numeric_limits<int>::max() / 3))
        throw std::invalid_argument("Frozen diagnostic input exceeds MPI_Scatterv count range");
    if (particles_m->getLocalNum() != 0)
        throw std::invalid_argument("Frozen force comparison requires a fresh Simulation");

    // This small validation adapter deliberately uses root host I/O, followed
    // by one scatter. Production fields/particles remain distributed on the
    // configured Kokkos memory/execution space; no physics kernel is duplicated.
    std::vector<double> globalPositions;
    std::string rootError;
    if (rank == 0) {
        try {
            globalPositions = readPositions(inputCsv, total, config_m.boxSize);
        } catch (const std::exception& error) {
            rootError = error.what();
        }
    }
    // Every rank rejects invalid input together before entering Scatterv.
    broadcastRootError(rootError, communicator);
    std::vector<int> positionCounts(ranks), positionOffsets(ranks);
    std::uint64_t offset = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        const auto count = total / ranks + (std::uint64_t(peer) < total % ranks ? 1 : 0);
        positionCounts[peer] = int(3 * count);
        positionOffsets[peer] = int(3 * offset);
        offset += count;
    }
    const std::size_t localCount = positionCounts[rank] / 3;
    const std::uint64_t firstId = positionOffsets[rank] / 3;
    std::vector<double> localPositions(positionCounts[rank]);
    MPI_Scatterv(globalPositions.data(), positionCounts.data(), positionOffsets.data(), MPI_DOUBLE,
                 localPositions.data(), positionCounts[rank], MPI_DOUBLE, 0, communicator);
    globalPositions.clear();
    globalPositions.shrink_to_fit();

    particles_m->create(localCount);
    particles_m->mass = 1.0;
    particles_m->momentum = 0.0;
    particles_m->force = 0.0;
    auto positions = particles_m->R.getHostMirror();
    auto lagrangian = particles_m->lagrangian.getHostMirror();
    auto ids = particles_m->globalId.getHostMirror();
    for (std::size_t index = 0; index < localCount; ++index) {
        positions(index) = Vector(localPositions[3 * index], localPositions[3 * index + 1],
                                  localPositions[3 * index + 2]);
        lagrangian(index) = positions(index);
        ids(index) = firstId + index;
    }
    Kokkos::deep_copy(particles_m->R.getView(), positions);
    Kokkos::deep_copy(particles_m->lagrangian.getView(), lagrangian);
    Kokkos::deep_copy(particles_m->globalId.getView(), ids);
    updateParticles();
    solveForce();

    // Check that spatial migration preserved the validated ID permutation.
    ids = particles_m->globalId.getHostMirror();
    Kokkos::deep_copy(ids, particles_m->globalId.getView());
    const int migratedCount = int(particles_m->getLocalNum());
    std::vector<std::uint64_t> localIds(migratedCount), allIds;
    for (int index = 0; index < migratedCount; ++index) localIds[index] = ids(index);
    std::vector<int> counts(ranks), offsets(ranks);
    MPI_Gather(&migratedCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, communicator);
    if (rank == 0) {
        for (int peer = 1; peer < ranks; ++peer) offsets[peer] = offsets[peer - 1] + counts[peer - 1];
        allIds.resize(total);
    }
    MPI_Gatherv(localIds.data(), migratedCount, MPI_UINT64_T, allIds.data(), counts.data(),
                offsets.data(), MPI_UINT64_T, 0, communicator);
    if (rank == 0) {
        try {
            std::sort(allIds.begin(), allIds.end());
            for (std::uint64_t index = 0; index < total; ++index)
                if (allIds[index] != index)
                    throw std::runtime_error("Frozen particle IDs changed during migration");
            const std::filesystem::path output(outputDirectory);
            if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
                throw std::runtime_error("Frozen force output directory must be new or empty");
            std::filesystem::create_directories(output);
        } catch (const std::exception& error) {
            rootError = error.what();
        }
    }
    broadcastRootError(rootError, communicator);
    ippl::Comm->barrier();

    positions = particles_m->R.getHostMirror();
    auto forces = particles_m->force.getHostMirror();
    Kokkos::deep_copy(positions, particles_m->R.getView());
    Kokkos::deep_copy(forces, particles_m->force.getView());
    const auto output = std::filesystem::path(outputDirectory);
    std::ofstream particleOutput(output / ("forces_rank" + std::to_string(rank) + ".csv"));
    if (!particleOutput) throw std::runtime_error("Cannot create frozen particle force output");
    particleOutput << std::setprecision(17) << "id,x,y,z,fx,fy,fz\n";
    for (int index = 0; index < migratedCount; ++index) {
        particleOutput << ids(index);
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(positions(index)[d]))
                throw std::runtime_error("Nonfinite frozen particle position");
            particleOutput << ',' << positions(index)[d];
        }
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(forces(index)[d])) throw std::runtime_error("Nonfinite frozen particle force");
            particleOutput << ',' << forces(index)[d];
        }
        particleOutput << '\n';
    }
    particleOutput.close();
    if (!particleOutput) throw std::runtime_error("Frozen particle force output write failed");

    const auto density = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_m.getView());
    const auto meshForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), force_m.getView());
    const auto localDomain = layout_m->getLocalNDIndex();
    const int ng = density_m.getNghost();
    std::ofstream densityOutput(output / ("density_rank" + std::to_string(rank) + ".csv"));
    if (!densityOutput) throw std::runtime_error("Cannot create frozen density output");
    densityOutput << std::setprecision(17) << "ix,iy,iz,delta,fx,fy,fz\n";
    for (int iz = 0; iz < localDomain[2].length(); ++iz)
        for (int iy = 0; iy < localDomain[1].length(); ++iy)
            for (int ix = 0; ix < localDomain[0].length(); ++ix) {
                const double value = density(ix + ng, iy + ng, iz + ng);
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite frozen density");
                densityOutput << ix + localDomain[0].first() << ',' << iy + localDomain[1].first()
                              << ',' << iz + localDomain[2].first() << ',' << value;
                for (int d = 0; d < 3; ++d) {
                    const double component = meshForce(ix + ng, iy + ng, iz + ng)[d];
                    if (!std::isfinite(component)) throw std::runtime_error("Nonfinite frozen mesh force");
                    densityOutput << ',' << component;
                }
                densityOutput << '\n';
            }
    densityOutput.close();
    if (!densityOutput) throw std::runtime_error("Frozen density output write failed");
    if (!std::isfinite(maxImaginary_m) || maxImaginary_m > 1.0e-10)
        throw std::runtime_error("Frozen inverse Fourier force is not real");
    ippl::Comm->barrier();
    if (rank == 0) {
        std::ofstream metadata(output / "metadata.txt");
        metadata << std::setprecision(17)
                 << "mode=frozen particles; production Simulation::solveForce\n";
        writeExecutionMetadata(metadata);
        metadata << "ranks=" << ranks
                 << "\nn_grid=" << config_m.nGrid << "\nbox_size=" << config_m.boxSize
                 << "\nomega_m=" << config_m.omegaMatter << "\nparticles=" << total
                 << "\nmass_error=" << massError_m << "\nmax_inverse_imaginary=" << maxImaginary_m
                 << "\nposition_unit=Mpc/h\nmass=one unit per particle; N^3 particles\n"
                 << "mesh_origin=cell centers at (index+0.5)*L/N; zero-based indices\n"
                 << "assignment=CIC scatter and CIC gather; no deconvolution\n"
                 << "density=deposited unit mass per cell minus 1\n"
                 << "force=F0=-grad(phi0); laplacian(phi0)=1.5*Omega_m*delta\n"
                 << "fft=forward 1/N^3; inverse unnormalized\n"
                 << "kernel=i*k_d/k^2 times 1.5*Omega_m; full k^2; DC zero\n"
                 << "nyquist=differentiated Nyquist component zero\n"
                 << "periodic=finite input positions wrapped with fmod into [0,L)\n";
        metadata.close();
        if (!metadata) throw std::runtime_error("Frozen metadata write failed");
        std::cout << "Frozen force comparison output: " << outputDirectory << '\n';
    }
}

}  // namespace cosmology

/**
 * @brief Run frozen-particle adapter using the production ippl cic/fft/gather path.
 * @see cosmology_contracts cosmology_validation
 * @param argc Program argument count; this executable checks its own exact usage.
 * @param argv Program argument vector; see the file/workflow contract for scalar and path units.
 * @return Zero on successful completion; malformed/native fatal errors return nonzero or abort the communicator.
 * Fatal distributed failures must terminate communicator peers; the host-only test uses ordinary process status.
 */
int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    try {
        if (argc != 6)
            throw std::invalid_argument("Usage: CompareCosmologyForce N L Omega_m input.csv output_dir");
        cosmology::Config config;
        config.nGrid = parseNumber<int>(argv[1], "N");
        config.boxSize = parseNumber<double>(argv[2], "L");
        config.omegaMatter = parseNumber<double>(argv[3], "Omega_m");
        config.omegaBaryon = 0.0; // Irrelevant to this density-to-force diagnostic.
        config.icMode = "uniform";
        {
            cosmology::Simulation simulation(config);
            simulation.compareFrozenForce(argv[4], argv[5]);
        }
    } catch (const std::exception& error) {
        std::cerr << "Frozen force rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
        MPI_Abort(ippl::Comm->getCommunicator(), 1);
        return 1;
    }
    ippl::finalize();
    return 0;
}
