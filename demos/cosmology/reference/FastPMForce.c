/** @file FastPMForce.c
 * @brief Pinned native FastPM frozen-force harness retaining its own Nyquist/precision conventions.
 * @ingroup cosmology_reference
 * @see cosmology_contracts cosmology_validation cosmology_references
 */
/* Frozen-particle diagnostic using the pinned, unmodified FastPM force routine.
 * This adapter is not a FastPM time integrator. See build_fastpm.sh for the pin.
 * FastPM's native mesh is node-centred; inputs/outputs use IPPL cell centres.
 */
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <mpi.h>
#include <fastpm/libfastpm.h>
#include <fastpm/gravity.h>
#include "pmpfft.h"

#ifndef FASTPM_REFERENCE_COMMIT
#error "Build with build_fastpm.sh to record the upstream source pin"
#endif

/** @brief Native host particle/run context; units and output conventions follow the file contract. */
typedef struct {
    uint64_t id; ///< Global immutable particle label.
    double physical[3]; ///< Imported IPPL-frame comoving position in Mpc/h.
    double native[3]; ///< Native node-centered position after half-cell rebase in Mpc/h.
} InputParticle;

/**
 * @brief Report a native-reference error and abort MPI_COMM_WORLD before exiting.
 * @see cosmology_contracts cosmology_validation
 * @param message Failure text retained for the fixed-budget check.
 */
static void fail(const char *message) {
    int rank; ///< Current MPI rank.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "FastPMForce rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(2);
}

/**
 * @brief Allocate zeroed host memory after checking multiplication overflow and allocation success.
 * @see cosmology_contracts cosmology_validation
 * @param n Allocation element count; zero requests still yield a valid minimal allocation.
 * @param size Size in bytes of each element.
 * @return Zeroed valid host allocation; failures abort the native communicator.
 */
static void *checkedAlloc(size_t n, size_t size) {
    if (size && n > SIZE_MAX / size) fail("allocation size overflow");
    void *result = calloc(n ? n : 1, size);
    if (!result) fail("allocation failed");
    return result;
}

/**
 * @brief Wrap a finite reference-coordinate value into its periodic interval.
 * @see cosmology_contracts cosmology_validation
 * @param x Finite periodic coordinate in the caller's comoving length unit.
 * @param length Positive periodic box side in the same coordinate unit.
 * @return Coordinate in [0,length).
 */
static double wrap(double x, double length) {
    double result = fmod(x, length);
    if (result < 0) result += length;
    return result >= length ? 0 : result;
}

/**
 * @brief Order native particle records by global label for deterministic export.
 * @see cosmology_contracts cosmology_validation
 * @param lhs First native particle record for global-ID ordering.
 * @param rhs Second native particle record for global-ID ordering.
 * @return Negative/zero/positive C comparator result for the two IDs.
 */
static int compareIds(const void *lhs, const void *rhs) {
    const uint64_t a = *(const uint64_t *)lhs;
    const uint64_t b = *(const uint64_t *)rhs;
    return (a > b) - (a < b);
}

/**
 * @brief Open one diagnostic output file below the selected run directory.
 * @see cosmology_contracts cosmology_validation
 * @param directory Fresh/empty selected diagnostic output directory.
 * @param name Scalar/diagnostic filename or label used in error reporting and output identity.
 * @return Host FILE stream ready for diagnostic output.
 */
static FILE *openOutput(const char *directory, const char *name) {
    char path[4096];
    const int pathLength = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (pathLength < 0 || (size_t)pathLength >= sizeof(path))
        fail("output path too long");
    FILE *file = fopen(path, "wx");
    if (!file) fail("cannot create output file (existing files are never overwritten)");
    return file;
}

/**
 * @brief Close a diagnostic output stream and reject a flush/close failure.
 * @see cosmology_contracts cosmology_validation
 * @param file Open native diagnostic stream; closing errors are fatal.
 */
static void closeOutput(FILE *file) {
    if (ferror(file) || fclose(file)) fail("writing output failed");
}

/**
 * @brief Run pinned native fastpm frozen-force harness retaining its own nyquist/precision conventions.
 * @see cosmology_contracts cosmology_validation
 * @param argc Program argument count; this executable checks its own exact usage.
 * @param argv Program argument vector; see the file/workflow contract for scalar and path units.
 * @return Zero on successful completion; malformed/native fatal errors return nonzero or abort the communicator.
 * Fatal distributed failures must terminate communicator peers; the host-only test uses ordinary process status.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; ///< Current MPI rank.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 6)
        fail("usage: FastPMForce N L Omega_m input.csv output_directory");
    char *end;
    errno = 0;
    long parsedN = strtol(argv[1], &end, 10);
    if (errno || *end || parsedN < 4 || parsedN > INT_MAX || parsedN % 2)
        fail("N must be an even integer >= 4");
    const size_t n = (size_t)parsedN;
    if (n > SIZE_MAX / n || n * n > SIZE_MAX / n) fail("N cubed overflows");
    const size_t count = n * n * n;
    const double length = strtod(argv[2], &end);
    if (*end || !isfinite(length) || length <= 0) fail("L must be positive and finite");
    const double omegaM = strtod(argv[3], &end);
    if (*end || !isfinite(omegaM) || omegaM <= 0 || omegaM > 1)
        fail("Omega_m must be in (0,1]");
    const double forcePrefactor = -1.5 * omegaM;
    const double halfCell = length / (2.0 * n);

    libfastpm_init();
    PM pm[1];
    /* Native FFTW MPI backend and the same transposed setting as the normal
     * FastPM solver. No OpenMP in this diagnostic build. */
    PMInit pmInit = {(ptrdiff_t)n, length, 1, 1, 1};
    pm_init(pm, &pmInit, MPI_COMM_WORLD);
    if (pm_i_region(pm)->total == 0)
        fail("empty native FFTW slab; reduce MPI rank count");

    /* Each rank reads this deliberately small diagnostic fixture. Ownership,
     * ghost exchange, CIC painting/gathering and the FFT are native FastPM.
     */
    FILE *input = fopen(argv[4], "r");
    if (!input) fail("cannot open input CSV");
    char line[2048];
    if (!fgets(line, sizeof(line), input)) fail("empty input CSV");
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, "id,x,y,z,mass")) fail("expected CSV header id,x,y,z,mass");
    InputParticle *local = checkedAlloc(count, sizeof(*local));
    uint64_t *ids = checkedAlloc(count, sizeof(*ids));
    size_t rows = 0, localCount = 0;
    while (fgets(line, sizeof(line), input)) {
        if (rows == count) fail("input has more than N cubed particles");
        InputParticle particle;
        double mass;
        int consumed = 0;
        if (!isdigit((unsigned char)line[0]) ||
            sscanf(line, "%" SCNu64 ",%lf,%lf,%lf,%lf%n", &particle.id,
                   &particle.physical[0], &particle.physical[1],
                   &particle.physical[2], &mass, &consumed) != 5)
            fail("malformed CSV row");
        for (char *tail = line + consumed; *tail; ++tail)
            if (!isspace((unsigned char)*tail)) fail("extra data in CSV row");
        if (mass != 1.0) fail("this diagnostic requires equal unit masses");
        ids[rows++] = particle.id;
        for (int d = 0; d < 3; ++d) {
            if (!isfinite(particle.physical[d])) fail("nonfinite position");
            particle.physical[d] = wrap(particle.physical[d], length);
            particle.native[d] = wrap(particle.physical[d] - halfCell, length);
        }
        if (pm_pos_to_rank(pm, particle.native) == rank) local[localCount++] = particle;
    }
    if (ferror(input)) fail("reading input failed");
    fclose(input);
    if (rows != count) fail("input must contain exactly N cubed particles");
    qsort(ids, count, sizeof(*ids), compareIds);
    for (size_t i = 0; i < count; ++i)
        if (ids[i] != i) fail("input IDs must be a permutation of 0 through N cubed minus 1");
    free(ids);
    uint64_t localTotal = localCount, globalTotal = 0;
    MPI_Allreduce(&localTotal, &globalTotal, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (globalTotal != count) fail("native particle ownership is not exhaustive");

    FastPMStore particles[1];
    fastpm_store_init(particles, "frozen", localCount ? localCount : 1,
                      COLUMN_POS | COLUMN_ID | COLUMN_ACC, FASTPM_MEMORY_HEAP);
    particles->np = localCount;
    particles->meta.M0 = 1;
    for (size_t i = 0; i < localCount; ++i) {
        particles->id[i] = local[i].id;
        for (int d = 0; d < 3; ++d) particles->x[i][d] = local[i].native[d];
    }
    /* Only the force routine is invoked: no IC generation, evolution, growth
     * approximation, COLA, neutrinos, PGD or cosmological timestep factors.
     */
    FastPMSolver solver = {0};
    solver.comm = MPI_COMM_WORLD;
    solver.NTask = ranks;
    solver.ThisTask = rank;
    fastpm_solver_add_species(&solver, FASTPM_SPECIES_CDM, particles);
    FastPMPainter painter[1];
    fastpm_painter_init(painter, pm, FASTPM_PAINTER_CIC, 2);
    FastPMFloat *deltaK = pm_alloc(pm);
    fastpm_solver_compute_force(&solver, pm, painter, FASTPM_SOFTENING_NONE,
                               FASTPM_KERNEL_NAIVE, deltaK, 1.0);

    if (rank == 0 && mkdir(argv[5], 0777) && errno != EEXIST)
        fail("cannot create output directory");
    MPI_Barrier(MPI_COMM_WORLD);
    char name[128];
    snprintf(name, sizeof(name), "forces_rank%d.csv", rank);
    FILE *output = openOutput(argv[5], name);
    fprintf(output, "id,x,y,z,fx,fy,fz\n");
    for (size_t i = 0; i < localCount; ++i)
        fprintf(output, "%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                particles->id[i], local[i].physical[0], local[i].physical[1],
                local[i].physical[2], forcePrefactor * particles->acc[i][0],
                forcePrefactor * particles->acc[i][1], forcePrefactor * particles->acc[i][2]);
    closeOutput(output);

    /* Reuse native Fourier density and native kernel to expose the actual mesh
     * operator, including FastPM's unchanged mixed-Nyquist behavior. */
    FastPMFloat *density = pm_alloc(pm);
    pm_assign(pm, deltaK, density);
    pm_c2r(pm, density);
    FastPMFloat *meshForce[3];
    for (int d = 0; d < 3; ++d) {
        meshForce[d] = pm_alloc(pm);
        const FastPMFieldDescr field = {COLUMN_ACC, d};
        gravity_apply_kernel_transfer(FASTPM_KERNEL_NAIVE, pm, deltaK, meshForce[d], field);
        pm_c2r(pm, meshForce[d]);
    }
    snprintf(name, sizeof(name), "density_rank%d.csv", rank);
    output = openOutput(argv[5], name);
    fprintf(output, "ix,iy,iz,delta,fx,fy,fz\n");
    PMXIter iter;
    pm_xiter_init(pm, &iter);
    for (; !pm_xiter_stop(&iter); pm_xiter_next(&iter)) {
        const double delta = density[iter.ind] - 1;
        fprintf(output, "%td,%td,%td,%.17g,%.17g,%.17g,%.17g\n",
                iter.iabs[0], iter.iabs[1], iter.iabs[2], delta,
                forcePrefactor * meshForce[0][iter.ind],
                forcePrefactor * meshForce[1][iter.ind],
                forcePrefactor * meshForce[2][iter.ind]);
    }
    closeOutput(output);
    if (rank == 0) {
        output = openOutput(argv[5], "metadata.txt");
        fprintf(output, "reference=FastPM\nupstream_commit=%s\nranks=%d\nthreads=1\n"
                "n_grid=%zu\nbox_size=%.17g\nomega_m=%.17g\nparticle_count=%zu\n"
                "particle_mass=1\nfft_precision_bits=%zu\nposition_precision_bits=%zu\n"
                "particle_acc_precision_bits=%zu\nmesh_force_precision_bits=%zu\n"
                "wavevector_precision_bits=32\nwavevector_square_precision_bits=32\n"
                "cic_weights_precision_bits=64\nkernel=FASTPM_KERNEL_NAIVE\n"
                "softening=FASTPM_SOFTENING_NONE\npainter=FASTPM_PAINTER_CIC\n"
                "cic_deconvolution=none\nfft_backend=FFTW_MPI\nfft_transposed=1\nr2c_compressed_axis=z\n"
                "nyquist_sign=negative\nnyquist_zeroing=fully_self_conjugate_corners_only\n"
                "input_frame=IPPL_cell_centred\nnative_frame=FastPM_node_centred\n"
                "native_position=wrap(input_position-L/(2*N))\n"
                "output_position=wrap(input_position)\n"
                "mesh_index_frame=IPPL_cell_centred_indices\n"
                "force_prefactor=%.17g\nforce_convention=F=-1.5*Omega_m*native_acc=-grad(phi0)\n"
                "poisson_convention=laplacian(phi0)=1.5*Omega_m*delta\n"
                "scale_factor_factor=none\ntime_integration=none\n"
                "density_export=inverse_native_delta_k_minus_one\n"
                "mesh_force_export=native_kernel_then_inverse_fft_then_force_prefactor\n",
                FASTPM_REFERENCE_COMMIT, ranks, n, length, omegaM, count,
                sizeof(FastPMFloat) * CHAR_BIT, sizeof(particles->x[0][0]) * CHAR_BIT,
                sizeof(particles->acc[0][0]) * CHAR_BIT, sizeof(FastPMFloat) * CHAR_BIT,
                forcePrefactor);
        closeOutput(output);
    }
    for (int d = 2; d >= 0; --d) pm_free(pm, meshForce[d]);
    pm_free(pm, density);
    pm_free(pm, deltaK);
    fastpm_store_destroy(particles);
    free(local);
    pm_destroy(pm);
    libfastpm_cleanup();
    MPI_Finalize();
    return 0;
}
