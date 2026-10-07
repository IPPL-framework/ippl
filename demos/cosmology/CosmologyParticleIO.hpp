/** @file CosmologyParticleIO.hpp
 * @brief Serial bounded-buffer import, MPI distribution and per-rank binary snapshots.
 * @ingroup cosmology_core
 * @see cosmology_quijote cosmology_parallel
 * Host I/O copies each rank's imported x,p once to Kokkos execution memory. Evolution remains distributed.
 */
#ifndef IPPL_COSMOLOGY_PARTICLE_IO_HPP
#define IPPL_COSMOLOGY_PARTICLE_IO_HPP
namespace cosmology {
/** @brief Import canonical x,p without growth/amplitude rescaling; IDs are contiguous and ordered on disk.
 * The initial equal-mass force uses unit weights with mean Nparticles/Ncells.
 * File I/O is serial on rank zero; bounded byte messages use a separate MPI communicator.
 * @throws std::runtime_error If epoch/background, exact record count, IDs or finite phase space disagree.
 */
inline void Simulation::initializeExternal() {
    const int rank = ippl::Comm->rank(), ranks = ippl::Comm->size();
    expectedParticleCount_m = config_m.importedParticleCount;
    const std::uint64_t total = totalParticles();
    std::ifstream input;
    if (rank == 0) {
        input.open(config_m.icFile, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open external IC " + config_m.icFile);
        const auto header = readPhaseHeader(input);
        if (header.flags != 1 || header.count != total || header.total != total)
            throw std::runtime_error("External IC must be complete, ordered and match particle_count");
        const auto same = [](double x, double y) {
            return std::abs(x - y) <= 1.e-10 * std::max({std::abs(x), std::abs(y), 1.e-30});
        };
        if (!same(header.a, config_m.aInitial()) || !same(header.box, config_m.boxSize)
            || !same(header.omegaMatter, config_m.omegaMatter)
            || !same(header.omegaLambda, 1 - config_m.omegaMatter) || !same(header.hubble, config_m.hubble))
            throw std::runtime_error("External IC epoch/background does not match configuration");
        if (std::filesystem::file_size(config_m.icFile) != PhaseHeaderBytes + PhaseRecordBytes * total)
            throw std::runtime_error("External IC file length does not match header");
        particleMass_m = header.particleMass;
    }
    MPI_Bcast(&particleMass_m, 1, MPI_DOUBLE, 0, ippl::Comm->getCommunicator());
    const std::uint64_t localCount = total / ranks + (std::uint64_t(rank) < total % ranks);
    const std::uint64_t firstId = (total / ranks) * rank + std::min(std::uint64_t(rank), total % ranks);
    if (localCount > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Local particle count exceeds addressable memory");
    particles_m->create(static_cast<std::size_t>(localCount));
    particles_m->mass = 1.0;
    particles_m->force = 0.0;
    auto positions = particles_m->R.getHostMirror();
    auto momenta = particles_m->momentum.getHostMirror();
    auto lagrangian = particles_m->lagrangian.getHostMirror();
    auto ids = particles_m->globalId.getHostMirror();
    MPI_Comm transfer;
    MPI_Comm_dup(ippl::Comm->getCommunicator(), &transfer);
    std::vector<char> bytes(PhaseChunkRecords * PhaseRecordBytes);
    for (int peer = 0; peer < ranks; ++peer) {
        if (rank != 0 && rank != peer) continue;
        const std::uint64_t peerCount = total / ranks + (std::uint64_t(peer) < total % ranks);
        for (std::uint64_t offset = 0; offset < peerCount; offset += PhaseChunkRecords) {
            const auto count = std::min(std::uint64_t(PhaseChunkRecords), peerCount - offset);
            const int byteCount = static_cast<int>(count * PhaseRecordBytes);
            if (rank == 0) {
                if (!input.read(bytes.data(), byteCount)) throw std::runtime_error("Truncated external IC records");
                if (peer != 0) MPI_Send(bytes.data(), byteCount, MPI_BYTE, peer, 0, transfer);
            } else MPI_Recv(bytes.data(), byteCount, MPI_BYTE, 0, 0, transfer, MPI_STATUS_IGNORE);
            if (rank == peer) {
                for (std::uint64_t j = 0; j < count; ++j) {
                    const char* record = bytes.data() + j * PhaseRecordBytes;
                    const std::uint64_t index = offset + j;
                    const auto id = phaseReadInteger(record);
                    if (id != firstId + index) throw std::runtime_error("External IC has duplicate, missing or unordered IDs");
                    ids(index) = id;
                    for (int d = 0; d < 3; ++d) {
                        double x = phaseReadDouble(record + 8 + 8 * d);
                        const double p = phaseReadDouble(record + 32 + 8 * d);
                        if (!std::isfinite(x) || !std::isfinite(p) || x < 0 || x > config_m.boxSize)
                            throw std::runtime_error("External IC phase space is nonfinite or outside the box");
                        if (x == config_m.boxSize) x = 0;
                        positions(index)[d] = lagrangian(index)[d] = x;
                        momenta(index)[d] = p;
                    }
                }
            }
        }
    }
    MPI_Comm_free(&transfer);
    Kokkos::deep_copy(particles_m->R.getView(), positions);
    Kokkos::deep_copy(particles_m->momentum.getView(), momenta);
    Kokkos::deep_copy(particles_m->lagrangian.getView(), lagrangian);
    Kokkos::deep_copy(particles_m->globalId.getView(), ids);
    updateParticles();
}

/** @brief Reduce two modulo-2^64 label checksums plus range/count checks without root ID storage.
 * @param initialize Establish the initial signature; false compares against it after migration.
 * These signatures detect corruption but are not a mathematical proof of uniqueness; ordered input validation is exact.
 */
inline void Simulation::verifyParticleLabels(bool initialize) {
    const auto ids = particles_m->globalId.getView();
    const auto total = totalParticles();
    std::uint64_t sum = 0, hash = 0, invalid = 0;
    Kokkos::parallel_reduce("Cosmology label signatures", particles_m->getLocalNum(),
        KOKKOS_LAMBDA(const std::size_t i, std::uint64_t& s, std::uint64_t& h, std::uint64_t& bad) {
            const auto id = ids(i);
            s += id;
            h += mixBits(id);
            bad += (id >= total);
        }, sum, hash, invalid);
    const std::uint64_t local[3] = {sum, hash, invalid};
    std::uint64_t global[3];
    MPI_Allreduce(local, global, 3, MPI_UINT64_T, MPI_SUM, ippl::Comm->getCommunicator());
    checkParticleCount();
    if (global[2] != 0) throw std::runtime_error("Particle ID outside the original label range");
    if (initialize) { idSum_m = global[0]; idHash_m = global[1]; }
    else if (global[0] != idSum_m || global[1] != idHash_m)
        throw std::runtime_error("Particle ID signatures changed after migration");
}

/** @brief Write local x,p,IDs at synchronized epoch a with fixed-width records and atomic shard publication.
 * @param name Stable epoch token used by snapshots.csv.
 * @param a Actual integration endpoint; no time interpolation is performed.
 * Host mirrors synchronize execution memory only at requested snapshots. Physical masses are metadata, not PM weights.
 */
inline void Simulation::binarySnapshot(const std::string& name, double a) {
    auto positions = particles_m->R.getHostMirror(), momenta = particles_m->momentum.getHostMirror();
    auto ids = particles_m->globalId.getHostMirror();
    Kokkos::deep_copy(positions, particles_m->R.getView());
    Kokkos::deep_copy(momenta, particles_m->momentum.getView());
    Kokkos::deep_copy(ids, particles_m->globalId.getView());
    const auto path = std::filesystem::path(config_m.output) /
        ("particles_" + name + "_rank" + std::to_string(ippl::Comm->rank()) + ".bin");
    const auto partial = path.string() + ".partial";
    std::ofstream output(partial, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open snapshot " + partial);
    PhaseSpaceHeader header;
    header.count = particles_m->getLocalNum();
    header.total = totalParticles();
    header.a = a; header.box = config_m.boxSize;
    header.omegaMatter = config_m.omegaMatter; header.omegaLambda = 1 - config_m.omegaMatter;
    header.hubble = config_m.hubble; header.particleMass = particleMass_m;
    writePhaseHeader(output, header);
    std::vector<char> bytes(PhaseChunkRecords * PhaseRecordBytes);
    for (std::uint64_t first = 0; first < header.count; first += PhaseChunkRecords) {
        const auto count = std::min(std::uint64_t(PhaseChunkRecords), header.count - first);
        for (std::uint64_t j = 0; j < count; ++j) {
            char* record = bytes.data() + j * PhaseRecordBytes;
            phaseWriteInteger(record, ids(first + j));
            for (int d = 0; d < 3; ++d) {
                const double x = positions(first + j)[d], p = momenta(first + j)[d];
                if (!std::isfinite(x) || !std::isfinite(p)) throw std::runtime_error("Nonfinite snapshot particle");
                phaseWriteDouble(record + 8 + 8 * d, x);
                phaseWriteDouble(record + 32 + 8 * d, p);
            }
        }
        output.write(bytes.data(), static_cast<std::streamsize>(count * PhaseRecordBytes));
        if (!output) throw std::runtime_error("Particle snapshot write failed");
    }
    output.close();
    if (!output) throw std::runtime_error("Particle snapshot close failed");
    std::filesystem::rename(partial, path);
}
} // namespace cosmology
#endif
