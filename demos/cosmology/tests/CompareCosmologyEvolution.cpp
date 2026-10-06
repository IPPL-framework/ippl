/** @file CompareCosmologyEvolution.cpp
 * @brief Shared phase-space adapter using production IPPL force/KDK/migration.
 * @ingroup cosmology_diagnostics
 * @see cosmology_contracts cosmology_validation cosmology_references
 */
// Matched-particle evolution diagnostic. All kicks, drifts, migration and
// force solves use the production Simulation methods; no ICs are generated.
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
 * @brief Validate NP and compute its cube within the diagnostic MPI/int allocation bounds.
 * @see cosmology_contracts cosmology_validation
 * @param grid Particle-lattice side NP; the diagnostic import creates exactly NP^3 labels.
 * @return Validated exact NP^3 as uint64.
 */
std::uint64_t particleCount(int grid) {
    if (grid < 1) throw std::invalid_argument("NP must be a positive integer");
    const auto n = std::uint64_t(grid);
    if (n > std::numeric_limits<std::uint64_t>::max() / n / n)
        throw std::invalid_argument("NP cubed exceeds the particle-count integer range");
    const auto total = n * n * n;
    if (total > std::uint64_t(std::numeric_limits<int>::max() / 6))
        throw std::invalid_argument("Evolution input exceeds MPI_Scatterv count range");
    return total;
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
        throw std::invalid_argument("Particle ID must be an integer in [0,NP^3): " + text);
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
 * @brief Read and validate canonical phase-space CSV, ordered into a host global-ID array.
 * @see cosmology_contracts cosmology_validation
 * @param path Input CSV or comparison output path following the exact file contract.
 * @param total Exact expected particle count; valid IDs occupy [0,total).
 * @param box Positive periodic side in comoving Mpc/h.
 * @return Host xyz and canonical momentum ordered by global particle ID, in Mpc/h.
 */
std::vector<double> readPhaseSpace(const std::string& path, std::uint64_t total, double box) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open evolution particle CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::invalid_argument("Empty evolution particle CSV");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "id,x,y,z,px,py,pz,mass")
        throw std::invalid_argument("Evolution CSV header must be exactly id,x,y,z,px,py,pz,mass");
    std::vector<double> phaseSpace(6 * total);
    std::vector<bool> seen(total, false);
    std::uint64_t rows = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (++rows > total) throw std::invalid_argument("Evolution particle CSV has more than NP^3 rows");
        std::array<std::string, 8> fields;
        std::istringstream row(line);
        for (auto& field : fields)
            if (!std::getline(row, field, ','))
                throw std::invalid_argument("Evolution particle CSV row requires eight fields");
        if (std::count(line.begin(), line.end(), ',') != 7)
            throw std::invalid_argument("Evolution particle CSV row requires exactly eight fields");
        const auto id = parseId(fields[0], total);
        if (seen[id]) throw std::invalid_argument("Duplicate evolution particle ID: " + fields[0]);
        seen[id] = true;
        const double mass = parseNumber<double>(fields[7], "particle mass");
        if (!std::isfinite(mass) || mass != 1.0)
            throw std::invalid_argument("Evolution comparison requires each mass to equal 1");
        for (int d = 0; d < 3; ++d) {
            phaseSpace[6 * id + d] = wrapPosition(parseNumber<double>(fields[d + 1], "position"), box);
            const double momentum = parseNumber<double>(fields[d + 4], "momentum");
            if (!std::isfinite(momentum)) throw std::invalid_argument("Particle momenta must be finite");
            phaseSpace[6 * id + 3 + d] = momentum;
        }
    }
    if (input.bad()) throw std::runtime_error("Evolution particle CSV read failed");
    if (rows != total || !std::all_of(seen.begin(), seen.end(), [](bool present) { return present; }))
        throw std::invalid_argument("Evolution particle CSV must contain every ID in [0,NP^3) exactly once");
    return phaseSpace;
}

/**
 * @brief Reject unordered epochs, nonpositive intervals and a checkpoint count not dividing the step count.
 * @see cosmology_contracts cosmology_validation
 * @param aInitial Initial scale factor, strictly positive and below aFinal.
 * @param aFinal Final scale factor, ordered after aInitial and at most one.
 * @param steps Positive integration step count.
 * @param checkpoints Positive output interval count dividing steps.
 */
void validateSchedule(double aInitial, double aFinal, int steps, int checkpoints) {
    if (!std::isfinite(aInitial) || !std::isfinite(aFinal)
        || !(0.0 < aInitial && aInitial < aFinal && aFinal <= 1.0))
        throw std::invalid_argument("Evolution requires 0 < a_initial < a_final <= 1");
    if (steps <= 0 || checkpoints <= 0 || checkpoints > steps || steps % checkpoints != 0)
        throw std::invalid_argument("n_steps must be positive and divisible by n_checkpoints");
}

}  // namespace

namespace cosmology {

void Simulation::compareImportedEvolution(int particleGrid, const std::string& inputCsv,
                                         const std::string& outputDirectory, double aInitial,
                                         double aFinal, int checkpoints) {
    const auto communicator = ippl::Comm->getCommunicator();
    const int rank = ippl::Comm->rank(), ranks = ippl::Comm->size();
    const auto total = particleCount(particleGrid);
    validateSchedule(aInitial, aFinal, config_m.nSteps, checkpoints);
    if (particles_m->getLocalNum() != 0 || expectedParticleCount_m != 0)
        throw std::invalid_argument("Imported evolution requires a fresh Simulation");
    // Core HaloCells explicitly rejects one-cell local axes in dimensions >1.
    // Check collectively before scatter so an unsupported tiny diagnostic mesh
    // exits clearly instead of throwing inside an MPI halo exchange.
    const auto localDomain = layout_m->getLocalNDIndex();
    int supported = 1, allSupported;
    for (int d = 0; d < 3; ++d)
        if (localDomain[d].length() < 2) supported = 0;
    MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, communicator);
    if (!allSupported)
        throw std::invalid_argument("Evolution requires at least two local mesh cells per axis; reduce MPI ranks or increase NM");
    expectedParticleCount_m = total;
    config_m.output = outputDirectory;
    config_m.writeParticles = true;

    // Diagnostic-only root host I/O and Scatterv. Attributes are then copied to
    // the configured Kokkos memory space; migration and physics stay distributed.
    std::vector<double> globalPhaseSpace;
    std::string rootError;
    if (rank == 0) {
        try {
            globalPhaseSpace = readPhaseSpace(inputCsv, total, config_m.boxSize);
            const std::filesystem::path output(outputDirectory);
            if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
                throw std::runtime_error("Evolution output directory must be new or empty");
            std::filesystem::create_directories(output);
        } catch (const std::exception& error) {
            rootError = error.what();
        }
    }
    broadcastRootError(rootError, communicator);
    std::vector<int> phaseCounts(ranks), phaseOffsets(ranks);
    std::uint64_t offset = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        const auto count = total / ranks + (std::uint64_t(peer) < total % ranks ? 1 : 0);
        phaseCounts[peer] = int(6 * count);
        phaseOffsets[peer] = int(6 * offset);
        offset += count;
    }
    const std::size_t localCount = phaseCounts[rank] / 6;
    const std::uint64_t firstId = phaseOffsets[rank] / 6;
    std::vector<double> localPhaseSpace(phaseCounts[rank]);
    MPI_Scatterv(globalPhaseSpace.data(), phaseCounts.data(), phaseOffsets.data(), MPI_DOUBLE,
                 localPhaseSpace.data(), phaseCounts[rank], MPI_DOUBLE, 0, communicator);
    globalPhaseSpace.clear();
    globalPhaseSpace.shrink_to_fit();

    particles_m->create(localCount);
    particles_m->mass = 1.0;
    particles_m->force = 0.0;
    auto positions = particles_m->R.getHostMirror();
    auto momenta = particles_m->momentum.getHostMirror();
    auto lagrangian = particles_m->lagrangian.getHostMirror();
    auto ids = particles_m->globalId.getHostMirror();
    for (std::size_t index = 0; index < localCount; ++index) {
        positions(index) = Vector(localPhaseSpace[6 * index], localPhaseSpace[6 * index + 1],
                                  localPhaseSpace[6 * index + 2]);
        momenta(index) = Vector(localPhaseSpace[6 * index + 3], localPhaseSpace[6 * index + 4],
                                localPhaseSpace[6 * index + 5]);
        lagrangian(index) = positions(index);
        ids(index) = firstId + index;
    }
    Kokkos::deep_copy(particles_m->R.getView(), positions);
    Kokkos::deep_copy(particles_m->momentum.getView(), momenta);
    Kokkos::deep_copy(particles_m->lagrangian.getView(), lagrangian);
    Kokkos::deep_copy(particles_m->globalId.getView(), ids);
    updateParticles();
    solveForce();

    std::ofstream checkpointOutput;
    if (rank == 0) {
        checkpointOutput.open(std::filesystem::path(outputDirectory) / "checkpoints.csv");
        if (!checkpointOutput) rootError = "Cannot create checkpoints.csv";
        checkpointOutput << std::setprecision(17)
                         << "checkpoint,step,a,mass_error,max_inverse_imaginary\n";
    }
    broadcastRootError(rootError, communicator);
    double maximumMassError = massError_m;
    const auto writeCheckpoint = [&](int checkpoint, int step, double a) {
        // Validate complete IDs after migration as well as finite phase space.
        auto currentIds = particles_m->globalId.getHostMirror();
        auto currentPositions = particles_m->R.getHostMirror();
        auto currentMomenta = particles_m->momentum.getHostMirror();
        Kokkos::deep_copy(currentIds, particles_m->globalId.getView());
        Kokkos::deep_copy(currentPositions, particles_m->R.getView());
        Kokkos::deep_copy(currentMomenta, particles_m->momentum.getView());
        const int count = int(particles_m->getLocalNum());
        std::vector<std::uint64_t> localIds(count), allIds;
        int finite = 1, allFinite;
        for (int index = 0; index < count; ++index) {
            localIds[index] = currentIds(index);
            for (int d = 0; d < 3; ++d)
                if (!std::isfinite(currentPositions(index)[d])
                    || !std::isfinite(currentMomenta(index)[d])) finite = 0;
        }
        MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, communicator);
        if (!allFinite) throw std::runtime_error("Nonfinite imported-evolution phase space");
        std::vector<int> counts(ranks), offsets(ranks);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, communicator);
        if (rank == 0) {
            for (int peer = 1; peer < ranks; ++peer) offsets[peer] = offsets[peer - 1] + counts[peer - 1];
            allIds.resize(total);
        }
        MPI_Gatherv(localIds.data(), count, MPI_UINT64_T, allIds.data(), counts.data(),
                    offsets.data(), MPI_UINT64_T, 0, communicator);
        if (rank == 0) {
            std::sort(allIds.begin(), allIds.end());
            for (std::uint64_t index = 0; index < total; ++index)
                if (allIds[index] != index) rootError = "Evolution particle IDs changed during migration";
        }
        broadcastRootError(rootError, communicator);
        if (!std::isfinite(maxImaginary_m) || maxImaginary_m > 1.0e-10)
            throw std::runtime_error("Evolution inverse Fourier force is not real");
        std::ostringstream name;
        name << "checkpoint" << std::setw(4) << std::setfill('0') << checkpoint;
        snapshot(name.str());
        ippl::Comm->barrier();
        if (rank == 0) {
            checkpointOutput << checkpoint << ',' << step << ',' << a << ',' << massError_m
                             << ',' << maxImaginary_m << '\n';
            checkpointOutput.flush();
            if (!checkpointOutput) rootError = "Checkpoint table write failed";
        }
        broadcastRootError(rootError, communicator);
    };
    writeCheckpoint(0, 0, aInitial);
    const double logStep = std::log(aFinal / aInitial) / config_m.nSteps;
    const int checkpointEvery = config_m.nSteps / checkpoints;
    for (int step = 0; step < config_m.nSteps; ++step) {
        const double a0 = aInitial * std::exp(step * logStep);
        const double a1 = aInitial * std::exp((step + 1) * logStep);
        advanceStep(a0, a1);
        maximumMassError = std::max(maximumMassError, massError_m);
        if ((step + 1) % checkpointEvery == 0)
            writeCheckpoint((step + 1) / checkpointEvery, step + 1, a1);
    }
    if (rank == 0) {
        checkpointOutput.close();
        if (!checkpointOutput) rootError = "Checkpoint table close failed";
        std::ofstream metadata(std::filesystem::path(outputDirectory) / "metadata.txt");
        metadata << std::setprecision(17)
                 << "mode=imported particles; production Simulation::advanceStep and solveForce\n";
        writeExecutionMetadata(metadata);
        metadata << "ranks=" << ranks
                 << "\nn_particles_grid=" << particleGrid << "\nn_grid=" << config_m.nGrid
                 << "\nn_steps=" << config_m.nSteps << "\nn_checkpoints=" << checkpoints
                 << "\nbox_size=" << config_m.boxSize << "\nomega_m=" << config_m.omegaMatter
                 << "\na_initial=" << aInitial << "\na_final=" << aFinal << "\nparticles=" << total
                 << "\nmaximum_mass_error=" << maximumMassError
                 << "\nmax_inverse_imaginary=" << maxImaginary_m
                 << "\nposition_unit=Mpc/h\nmomentum=p=a^2 dx/d(H0 t)\nparticle_mass=1\n"
                 << "ic_generation=none; imported momenta retained without growth rescaling\n"
                 << "mesh_origin=cell centers at (index+0.5)*L/Nmesh\n"
                 << "assignment=CIC scatter and CIC gather; no deconvolution\n"
                 << "density=deposited unit mass per cell divided by (Nparticles/Ncells), minus 1\n"
                 << "force=F0=-grad(phi0); laplacian(phi0)=1.5*Omega_m*delta\n"
                 << "fft=forward 1/Nmesh^3; inverse unnormalized\n"
                 << "nyquist=differentiated Nyquist component zero; full k^2\n"
                 << "integrator=production KDK; canonical positions and momenta synchronized at checkpoints\n"
                 << "endpoint_spacing=uniform in log(a); a_i=a_initial*exp(i*log(a_final/a_initial)/n_steps)\n"
                 << "kick_midpoint=sqrt(a_i*a_(i+1))\n"
                 << "kick_factor=integral da/(a^2 E(a)); drift_factor=integral da/(a^3 E(a))\n"
                 << "background=flat LCDM without radiation; E(a)=sqrt(Omega_m/a^3+1-Omega_m)\n"
                 << "checkpoint_phase=initial before kicks; later after complete second kick\n";
        metadata.close();
        if (!metadata) rootError = "Evolution metadata write failed";
    }
    broadcastRootError(rootError, communicator);
}

}  // namespace cosmology

/**
 * @brief Run shared phase-space adapter using production ippl force/kdk/migration.
 * @see cosmology_contracts cosmology_validation
 * @param argc Program argument count; this executable checks its own exact usage.
 * @param argv Program argument vector; see the file/workflow contract for scalar and path units.
 * @return Zero on successful completion; malformed/native fatal errors return nonzero or abort the communicator.
 * Fatal distributed failures must terminate communicator peers; the host-only test uses ordinary process status.
 */
int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    try {
        if (argc != 11)
            throw std::invalid_argument("Usage: CompareCosmologyEvolution NP NM L Omega_m a_initial a_final n_steps n_checkpoints input.csv output_dir");
        const int particleGrid = parseNumber<int>(argv[1], "NP");
        particleCount(particleGrid);
        cosmology::Config config;
        config.nGrid = parseNumber<int>(argv[2], "NM");
        config.boxSize = parseNumber<double>(argv[3], "L");
        config.omegaMatter = parseNumber<double>(argv[4], "Omega_m");
        const double aInitial = parseNumber<double>(argv[5], "a_initial");
        const double aFinal = parseNumber<double>(argv[6], "a_final");
        config.nSteps = parseNumber<int>(argv[7], "n_steps");
        const int checkpoints = parseNumber<int>(argv[8], "n_checkpoints");
        validateSchedule(aInitial, aFinal, config.nSteps, checkpoints);
        config.zInitial = 1.0 / aInitial - 1.0;
        config.zFinal = 1.0 / aFinal - 1.0;
        config.omegaBaryon = 0.0; // Background expansion only; no generated spectrum.
        config.icMode = "uniform";
        {
            cosmology::Simulation simulation(config);
            simulation.compareImportedEvolution(particleGrid, argv[9], argv[10],
                                                aInitial, aFinal, checkpoints);
        }
    } catch (const std::exception& error) {
        std::cerr << "Imported evolution rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
        MPI_Abort(ippl::Comm->getCommunicator(), 1);
        return 1;
    }
    ippl::finalize();
    return 0;
}
