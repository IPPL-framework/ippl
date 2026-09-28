/**
 * @file mesh-1.cpp
 * @brief Distributed FEL-layout mesh and staircased dielectric-radiator inspection.
 *
 * Positions are laboratory-frame SI metres internally. The static material is
 * sampled at lower + (globalIndex + 1/2)*h. No Maxwell update or particle source
 * is applied. E, B, J and three four-potential levels use the existing FEL
 * aliases/container; epsilon adds one scalar field on the same mesh/layout.
 *
 * @ingroup chdr_mesh
 * @details All ranks read and validate the same YAML, instantiate either
 * chdr::PrismGeometry or chdr::BrickGeometry, and enter construct(). Only rank
 * zero writes diagnostics after collective sampling. Consult the student guide
 * for the global/local index convention and the distinction between a geometric
 * material mask and a future dielectric Maxwell operator.
 */
#include "Ippl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "BrickGeometry.h"
#include "ChdrMeshConfig.h"
#include "FELFieldContainer.hpp"
#include "MaterialViewer.h"
#include "PrismGeometry.h"

#ifndef CHDR_DEFAULT_CONFIG
/// Default YAML path; CMake normally supplies an absolute path in the build tree.
#define CHDR_DEFAULT_CONFIG "config.yaml"
#endif

namespace {
    /// Number of Cartesian spatial coordinates, ordered x, y, z.
    constexpr unsigned Dim = 3;
    /// E(3) + B(3) + J(4) + three potential levels(12) + epsilon_r(1).
    constexpr std::uint64_t BytesPerCell = 23 * sizeof(double);
    /// Monotonic host clock used for the construction interval.
    using Clock = std::chrono::steady_clock;
    /// FEL-compatible three-component vector for positions and E/B values.
    using Vec = ippl::Vector<double, Dim>;
    /// Scalar IPPL field used for relative permittivity on the shared layout.
    using ScalarField = Field<double, Dim>;

    /** @brief Bounded preview point, expressed in global integer cell indices. */
    struct Sample {
        int index[3];  ///< Global owned-cell indices (x, y, z); no halo offset.
        int plane;     ///< Plane identifier: 0=xz, 1=xy, 2=yz.
    };

    /** @brief Count global owned cells, checking the core storage estimate for overflow.
     * @param cfg Validated configuration with positive cell counts.
     * @return Product of the three cell counts (halos excluded).
     * @throws std::runtime_error If cells times BytesPerCell exceeds uint64_t.
     */
    std::uint64_t cellCount(const chdr::MeshConfig& cfg) {
        std::uint64_t result = 1;
        for (int n : cfg.cells) {
            if (result > std::numeric_limits<std::uint64_t>::max() / n / BytesPerCell)
                throw std::runtime_error("Mesh is too large for a 64-bit byte count");
            result *= n;
        }
        return result;
    }

    /** @brief Choose reproducible sample indices for three material-field slices.
     *
     * Each plane passes through the cell containing the analytic radiator centroid.
     * The two in-plane strides are ceil(N / previewMaxPointsPerAxis). Therefore
     * each plane holds at most the configured cap squared samples. Every rank
     * constructs the same list, including ranks that own none of those points.
     * This function chooses indices; it does not evaluate material values.
     *
     * @param cfg Validated laboratory-frame geometry and global mesh.
     * @param[out] sliceCoordinates Plane positions in mm, ordered by normal axis
     * (x for yz, y for xz, z for xy), not by Sample::plane.
     * @return Global indices for the xz, xy, and yz planes in that order.
     */
    std::vector<Sample> previewSamples(const chdr::MeshConfig& cfg,
                                       std::array<double, 3>& sliceCoordinates) {
        std::array<int, 3> cut{};
        for (int d = 0; d < 3; ++d) {
            double centroid = cfg.brickLower[d] + cfg.brickSize[d] / 2;
            if (cfg.geometryType == chdr::GeometryType::Prism) {
                centroid = cfg.axis[d] * cfg.height / 2;
                for (const auto& vertex : cfg.vertices)
                    centroid += vertex[d] / 3;
            }
            const double h = cfg.size[d] / cfg.cells[d];
            cut[d] =
                std::clamp(static_cast<int>((centroid - cfg.lower[d]) / h), 0, cfg.cells[d] - 1);
            sliceCoordinates[d] = (cfg.lower[d] + (cut[d] + 0.5) * h) * 1e3;
        }
        const int axes[3][2] = {{0, 2}, {0, 1}, {1, 2}};
        std::vector<Sample> samples;
        for (int plane = 0; plane < 3; ++plane) {
            const int u = axes[plane][0], v = axes[plane][1];
            const int du = 1 + (cfg.cells[u] - 1) / cfg.previewMaxPointsPerAxis;
            const int dv = 1 + (cfg.cells[v] - 1) / cfg.previewMaxPointsPerAxis;
            for (int i = 0; i < cfg.cells[u]; i += du)
                for (int j = 0; j < cfg.cells[v]; j += dv) {
                    Sample sample{{cut[0], cut[1], cut[2]}, plane};
                    sample.index[u] = i;
                    sample.index[v] = j;
                    samples.push_back(sample);
                }
        }
        return samples;
    }

    /** @brief Serialize the first three array entries with an optional scale factor.
     * @tparam Values Indexable numeric array or vector type.
     * @param out Output stream receiving a JSON array.
     * @param values Three coordinates, counts, or components.
     * @param factor Multiplicative conversion; use 1000 for metres to millimetres.
     */
    template <typename Values>
    void triple(std::ostream& out, const Values& values, double factor = 1.0) {
        out << '[' << values[0] * factor << ',' << values[1] * factor << ',' << values[2] * factor
            << ']';
    }

    /** @brief Write root-only metadata, material slices, and a self-contained viewer.
     *
     * mesh.json uses output schema 2 and millimetres; input YAML remains schema 1.
     * The CSV contains actual reduced field values. After both streams close,
     * chdr::writeMaterialViewer embeds their contents in the generated Python.
     * That script produces the PDF when run; C++ does not invoke Python.
     *
     * @tparam Geometry Host-readable radiator with volume() in cubic metres.
     * @param cfg Validated configuration; relative output paths resolve against process cwd.
     * @param radiator Analytic shape matching cfg.geometryType.
     * @param rankBounds Six integers per rank: first global indices, then lengths.
     * @param samples Global preview indices from previewSamples().
     * @param values One actual relative-permittivity value per sample, assembled on root.
     * @param cuts Slice positions in mm, ordered x, y, z by normal axis.
     * @param dielectricCells Global count of owned dielectric cells.
     * @param checksum Unsigned global sum of hashed occupied-cell IDs.
     * @param allocatedBytes Sum of field allocation spans, including rank halos.
     * @param seconds Maximum construction time across ranks, excluding previews and I/O.
     * @param rankCount MPI communicator size.
     * @pre Call on rank zero after successful material/halo verification.
     * @throws std::exception On directory creation, diagnostic I/O, or viewer failure;
     * construct() broadcasts the failure so all ranks return consistently.
     */
    template <class Geometry>
    void writeOutput(const chdr::MeshConfig& cfg, const Geometry& radiator,
                     const std::vector<int>& rankBounds, const std::vector<Sample>& samples,
                     const std::vector<double>& values, const std::array<double, 3>& cuts,
                     std::uint64_t dielectricCells, std::uint64_t checksum,
                     std::uint64_t allocatedBytes, double seconds, int rankCount) {
        std::filesystem::create_directories(cfg.outputPath);
        const auto output = std::filesystem::path(cfg.outputPath);
        std::ofstream meta(output / "mesh.json");
        meta.exceptions(std::ios::failbit | std::ios::badbit);
        meta << std::setprecision(17);
        std::array<double, 3> h{};
        for (int d = 0; d < 3; ++d)
            h[d] = cfg.size[d] / cfg.cells[d];
        meta << "{\n\"schema_version\":2,\"units\":\"mm\",\n\"domain\":{\"lower\":";
        triple(meta, cfg.lower, 1e3);
        meta << ",\"size\":";
        triple(meta, cfg.size, 1e3);
        meta << ",\"cells\":";
        triple(meta, cfg.cells);
        meta << ",\"spacing\":";
        triple(meta, h, 1e3);
        meta << "},\n\"radiator\":{";
        if (cfg.geometryType == chdr::GeometryType::Prism) {
            meta << "\"type\":\"prism\",\"vertices\":[";
            for (int v = 0; v < 3; ++v) {
                if (v)
                    meta << ',';
                triple(meta, cfg.vertices[v], 1e3);
            }
            meta << "],\"axis\":";
            triple(meta, cfg.axis);
            meta << ",\"height\":" << cfg.height * 1e3;
        } else {
            meta << "\"type\":\"brick\",\"lower\":";
            triple(meta, cfg.brickLower, 1e3);
            meta << ",\"size\":";
            triple(meta, cfg.brickSize, 1e3);
        }
        meta << ",\"epsilon_r\":" << cfg.radiatorEpsilon
             << ",\"analytic_volume_mm3\":" << radiator.volume() * 1e9 << "},\n"
             << "\"background_epsilon_r\":" << cfg.backgroundEpsilon << ",\n\"ranks\":[";
        for (int rank = 0; rank < rankCount; ++rank) {
            std::array<double, 3> low{}, size{};
            for (int d = 0; d < 3; ++d) {
                low[d]  = cfg.lower[d] + rankBounds[6 * rank + d] * h[d];
                size[d] = rankBounds[6 * rank + 3 + d] * h[d];
            }
            if (rank)
                meta << ',';
            meta << "{\"rank\":" << rank << ",\"lower\":";
            triple(meta, low, 1e3);
            meta << ",\"size\":";
            triple(meta, size, 1e3);
            meta << '}';
        }
        meta << "],\n\"diagnostics\":{\"dielectric_cells\":" << dielectricCells
             << ",\"total_cells\":" << cellCount(cfg)
             << ",\"voxel_volume_mm3\":" << dielectricCells * h[0] * h[1] * h[2] * 1e9
             << ",\"core_field_bytes\":" << cellCount(cfg) * BytesPerCell
             << ",\"allocated_field_bytes_with_halos\":" << allocatedBytes
             << ",\"material_index_checksum\":\"" << checksum << "\""
             << ",\"rank_count\":" << rankCount << ",\"halo_mismatches\":0"
             << ",\"construction_seconds_max\":" << seconds << "},\n"
             << "\"preview\":{\"max_points_per_axis\":" << cfg.previewMaxPointsPerAxis
             << ",\"sampled\":true,\"slice_coordinates_mm\":{\"xz\":" << cuts[1]
             << ",\"xy\":" << cuts[2] << ",\"yz\":" << cuts[0] << "}}\n}\n";
        meta.close();
        std::ofstream csv(output / "slices.csv");
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv << std::setprecision(17) << "plane,u_mm,v_mm,epsilon_r\n";
        const char* names[]  = {"xz", "xy", "yz"};
        const int axes[3][2] = {{0, 2}, {0, 1}, {1, 2}};
        for (std::size_t s = 0; s < samples.size(); ++s) {
            const auto& sample = samples[s];
            const int u = axes[sample.plane][0], v = axes[sample.plane][1];
            csv << names[sample.plane] << ','
                << (cfg.lower[u] + (sample.index[u] + 0.5) * h[u]) * 1e3 << ','
                << (cfg.lower[v] + (sample.index[v] + 0.5) * h[v]) * 1e3 << ',' << values[s]
                << '\n';
        }
        csv.close();
        chdr::writeMaterialViewer(output);
    }

    /** @brief Construct and inspect one static distributed material field.
     *
     * The geometry object is copied by value into Kokkos kernels; it must store
     * device-safe state. Global indices, rather than local partition indices,
     * determine cell-centre coordinates. Changing MPI decomposition must preserve
     * the occupied-cell count, checksum, and exported slice values.
     *
     * Each rank allocates FEL E/B/J, three four-potential levels, and epsilon_r.
     * All electromagnetic values remain zero. Only epsilon_r is rasterized and
     * halo-exchanged. The all-cell/halo check reuses the production predicate;
     * independent geometric references live in verify_mesh_output.py.
     *
     * @tparam Geometry Trivially copyable type exposing a const device-callable
     * contains(x,y,z) in metres and a host volume() in cubic metres.
     * @param cfg Validated configuration identical on all participating ranks.
     * @param radiator Analytic shape selected from cfg.geometryType.
     * @return 0 on success; 1 if rank-zero output fails (broadcast to all ranks).
     * @pre IPPL/MPI and Kokkos are initialized; all ranks call this collectively.
     * @throws std::runtime_error On unsuitable local cell counts or halo mismatch.
     * Allocation/other exceptions propagate to main(), which aborts the communicator.
     */
    template <class Geometry>
    int construct(const chdr::MeshConfig& cfg, const Geometry& radiator) {
        const auto start        = Clock::now();
        const auto communicator = ippl::Comm->getCommunicator();
        const int rank = ippl::Comm->rank(), ranks = ippl::Comm->size();
        Vec h, lower, upper;
        ippl::NDIndex<Dim> domain;
        for (int d = 0; d < 3; ++d) {
            domain[d] = ippl::Index(cfg.cells[d]);
            h[d]      = cfg.size[d] / cfg.cells[d];
            lower[d]  = cfg.lower[d];
            upper[d]  = lower[d] + cfg.size[d];
        }
        FELFieldContainer<double, Dim> fields(h, lower, upper, cfg.decompose, domain, lower, false);
        const auto local = fields.getFL().getLocalNDIndex();
        int minimumLocal =
            static_cast<int>(std::min({local[0].length(), local[1].length(), local[2].length()}));
        int minimumGlobal = 0;
        MPI_Allreduce(&minimumLocal, &minimumGlobal, 1, MPI_INT, MPI_MIN, communicator);
        if (minimumGlobal < 2)
            throw std::runtime_error(
                "IPPL halo exchange needs at least two owned cells per local axis; reduce ranks or "
                "change decomposition");
        fields.initializeFields();
        fields.getE() = 0.0;
        fields.getB() = 0.0;
        fields.getJ() = 0.0;
        std::array<SourceField_t<double, Dim>, 3> potentials;
        for (auto& potential : potentials) {
            potential.initialize(fields.getMesh(), fields.getFL());
            potential = 0.0;
        }
        ScalarField epsilon(fields.getMesh(), fields.getFL());
        epsilon         = cfg.backgroundEpsilon;
        const auto view = epsilon.getView();
        const int ghost = epsilon.getNghost();
        const ippl::Vector<int, 3> first{local[0].first(), local[1].first(), local[2].first()};
        const ippl::Vector<int, 3> length{static_cast<int>(local[0].length()),
                                          static_cast<int>(local[1].length()),
                                          static_cast<int>(local[2].length())};
        const ippl::Vector<int, 3> count{cfg.cells[0], cfg.cells[1], cfg.cells[2]};
        const double epsVacuum = cfg.backgroundEpsilon, epsRadiator = cfg.radiatorEpsilon;
        std::uint64_t localCount = 0, localChecksum = 0;
        Kokkos::parallel_reduce(
            "ChDR material rasterization", epsilon.getFieldRangePolicy(),
            KOKKOS_LAMBDA(int i, int j, int k, std::uint64_t& occupied, std::uint64_t& checksum) {
                const int gi = i + first[0] - ghost;
                const int gj = j + first[1] - ghost;
                const int gk = k + first[2] - ghost;
                const bool inside =
                    radiator.contains(lower[0] + (gi + 0.5) * h[0], lower[1] + (gj + 0.5) * h[1],
                                      lower[2] + (gk + 0.5) * h[2]);
                view(i, j, k) = inside ? epsRadiator : epsVacuum;
                if (inside) {
                    ++occupied;
                    // Commutative unsigned checksum of global material-cell IDs.
                    std::uint64_t id = (std::uint64_t(gi) * count[1] + gj) * count[2] + gk;
                    id += UINT64_C(0x9e3779b97f4a7c15);
                    id = (id ^ (id >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                    id = (id ^ (id >> 27)) * UINT64_C(0x94d049bb133111eb);
                    checksum += id ^ (id >> 31);
                }
            },
            localCount, localChecksum);
        epsilon.fillHalo();
        std::uint64_t badHalos = 0;
        using Policy           = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "ChDR verify material including internal halos",
            Policy({0, 0, 0},
                   {length[0] + 2 * ghost, length[1] + 2 * ghost, length[2] + 2 * ghost}),
            KOKKOS_LAMBDA(int i, int j, int k, std::uint64_t& bad) {
                const int gi = i + first[0] - ghost, gj = j + first[1] - ghost,
                          gk = k + first[2] - ghost;
                if (gi < 0 || gj < 0 || gk < 0 || gi >= count[0] || gj >= count[1]
                    || gk >= count[2])
                    return;
                const bool inside =
                    radiator.contains(lower[0] + (gi + 0.5) * h[0], lower[1] + (gj + 0.5) * h[1],
                                      lower[2] + (gk + 0.5) * h[2]);
                if (view(i, j, k) != (inside ? epsRadiator : epsVacuum))
                    ++bad;
            },
            badHalos);
        std::uint64_t localStats[3] = {localCount, localChecksum, badHalos}, globalStats[3]{};
        MPI_Allreduce(localStats, globalStats, 3, MPI_UINT64_T, MPI_SUM, communicator);
        if (globalStats[2])
            throw std::runtime_error("Material field/halo verification failed");
        // Include halo storage and the actual vector sizeof in the measured allocation.
        std::uint64_t allocated =
            epsilon.getView().span() * sizeof(double) + fields.getE().getView().span() * sizeof(Vec)
            + fields.getB().getView().span() * sizeof(Vec)
            + fields.getJ().getView().span() * sizeof(ippl::Vector<double, 4>);
        for (auto& potential : potentials)
            allocated += potential.getView().span() * sizeof(ippl::Vector<double, 4>);
        std::uint64_t totalAllocated = 0;
        MPI_Reduce(&allocated, &totalAllocated, 1, MPI_UINT64_T, MPI_SUM, 0, communicator);
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        double maxSeconds    = 0;
        MPI_Reduce(&seconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
        const int localBounds[6] = {first[0], first[1], first[2], length[0], length[1], length[2]};
        std::vector<int> rankBounds(rank == 0 ? 6 * ranks : 0);
        MPI_Gather(localBounds, 6, MPI_INT, rankBounds.data(), 6, MPI_INT, 0, communicator);

        // Every rank owns the same small index list. Device values are zero on
        // nonowners, so MPI_SUM assembles one owner value per sample on rank zero.
        // Only bounded preview samples reach the host; the full field stays local.
        std::array<double, 3> cuts{};
        const auto samples = previewSamples(cfg, cuts);
        Kokkos::View<Sample*> sampleView("ChDR preview indices", samples.size());
        auto sampleHost = Kokkos::create_mirror_view(sampleView);
        for (std::size_t s = 0; s < samples.size(); ++s)
            sampleHost(s) = samples[s];
        Kokkos::deep_copy(sampleView, sampleHost);
        Kokkos::View<double*> sampleValues("ChDR preview values", samples.size());
        Kokkos::parallel_for(
            "ChDR sample actual material field", samples.size(), KOKKOS_LAMBDA(int s) {
                const auto sample = sampleView(s);
                const int i = sample.index[0] - first[0], j = sample.index[1] - first[1],
                          k = sample.index[2] - first[2];
                sampleValues(s) =
                    (i >= 0 && j >= 0 && k >= 0 && i < length[0] && j < length[1] && k < length[2])
                        ? view(i + ghost, j + ghost, k + ghost)
                        : 0.0;
            });
        const auto valuesHost =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sampleValues);
        std::vector<double> combined(rank == 0 ? samples.size() : 0);
        MPI_Reduce(valuesHost.data(), combined.data(), static_cast<int>(samples.size()), MPI_DOUBLE,
                   MPI_SUM, 0, communicator);
        int writeFailed = 0;
        if (rank == 0) {
            try {
                writeOutput(cfg, radiator, rankBounds, samples, combined, cuts, globalStats[0],
                            globalStats[1], totalAllocated, maxSeconds, ranks);
                std::cout << "mesh-1: " << cellCount(cfg) << " cells, " << globalStats[0]
                          << " dielectric cells; " << ranks << " MPI rank(s)\n"
                          << "Core fields: " << cellCount(cfg) * BytesPerCell / 1e9
                          << " GB; allocated including halos: " << totalAllocated / 1e9 << " GB\n"
                          << "Material/halo verification passed. Output: " << cfg.outputPath
                          << '\n';
            } catch (const std::exception& error) {
                std::cerr << "mesh-1 output: " << error.what() << '\n';
                writeFailed = 1;
            }
        }
        MPI_Bcast(&writeFailed, 1, MPI_INT, 0, communicator);
        return writeFailed;
    }
}  // namespace

/** @brief Parse the CLI, coordinate validation errors, and select the radiator type.
 * @ingroup chdr_mesh
 * @param argc Number of command-line arguments.
 * @param argv Arguments: optional YAML path, --output DIRECTORY, and --dry-run.
 * @return 0 for help/dry-run/success; 1 for coordinated configuration/output errors.
 * @details --dry-run validates input and estimates owned field storage without
 * allocating the mesh. --help is handled before IPPL initialization. Unexpected
 * rank-local failures abort MPI instead of leaving peers blocked in collectives.
 */
int main(int argc, char* argv[]) {
    // IPPL consumes --help itself; provide the mini-app help before initialization.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout
                << "Usage: mesh-1 [config.yaml] [--output DIRECTORY] [--dry-run]\n"
                   "Creates FEL-compatible fields and a static prism or brick; no field solve.\n";
            return 0;
        }
    }
    ippl::initialize(argc, argv);
    int result = 0;
    try {
        chdr::MeshConfig cfg;
        bool dryRun = false, havePath = false;
        std::string path = CHDR_DEFAULT_CONFIG, output;
        int invalid      = 0;
        try {
            for (int i = 1; i < argc; ++i) {
                const std::string arg(argv[i]);
                if (arg == "--dry-run")
                    dryRun = true;
                else if (arg == "--output" && i + 1 < argc)
                    output = argv[++i];
                else if ((arg == "--info" || arg == "-i" || arg == "--overallocate" || arg == "-b")
                         && i + 1 < argc)
                    ++i;
                else if (!arg.empty() && arg[0] != '-' && !havePath) {
                    path     = arg;
                    havePath = true;
                } else
                    throw std::runtime_error("Unknown or incomplete argument: " + arg);
            }
            cfg = chdr::readMeshConfig(path);
            if (!output.empty())
                cfg.outputPath = output;
            cellCount(cfg);
            std::uint64_t possibleRanks = 1;
            for (int d = 0; d < 3; ++d)
                if (cfg.decompose[d])
                    possibleRanks *= cfg.cells[d];
            if (static_cast<std::uint64_t>(ippl::Comm->size()) > possibleRanks)
                throw std::runtime_error("Too many MPI ranks for enabled decomposition axes");
        } catch (const std::exception& error) {
            std::cerr << "mesh-1 rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
            invalid = 1;
        }
        int anyInvalid = 0;
        MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, ippl::Comm->getCommunicator());
        if (anyInvalid)
            result = 1;
        else if (dryRun) {
            if (ippl::Comm->rank() == 0)
                std::cout << "Dry run: " << cellCount(cfg) << " cells, "
                          << cellCount(cfg) * BytesPerCell / 1e9
                          << " GB core fields before halos; no mesh allocated.\n";
        } else if (cfg.geometryType == chdr::GeometryType::Prism)
            result = construct(cfg, chdr::PrismGeometry(cfg));
        else
            result = construct(cfg, chdr::BrickGeometry(cfg));
    } catch (const std::exception& error) {
        std::cerr << "mesh-1 rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
        // A rank-local allocation/kernel failure must not leave peers in collectives.
        ippl::Comm->abort(1);
        result = 1;
    }
    ippl::finalize();
    return result;
}
