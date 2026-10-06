/** @file FastPMEvolution.c
 * @brief Native plain-PM evolution harness with imported phase space and synchronized CSV exports.
 * @ingroup cosmology_reference
 * @see cosmology_contracts cosmology_validation cosmology_references
 */
/* Imported-particle evolution with unmodified native fastpm_solver_evolve.
 * Only initialization and synchronized CSV output are adapted. No kick, drift,
 * force, ghost exchange, migration, or time scheduling is reimplemented here.
 * Compile against the pinned source using build_fastpm_evolution.sh.
 */
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <mpi.h>
#include <fastpm/libfastpm.h>
#include "pmpfft.h"
#include "vpm.h"

#ifndef FASTPM_REFERENCE_COMMIT
#error "Build with build_fastpm_evolution.sh to record the upstream source pin"
#endif

/** @brief Native host particle/run context; units and output conventions follow the file contract. */
typedef struct {
    size_t particleGrid; ///< Particle lattice NP per dimension.
    size_t meshGrid; ///< Force mesh NM per dimension.
    size_t particleCount; ///< Exact expected global count NP^3.
    double length; ///< Comoving periodic box side in Mpc/h.
    double omegaM; ///< Matter fraction at a=1.
    double aInitial; ///< Initial scale factor.
    double aFinal; ///< Final scale factor.
    int steps; ///< Positive integration step count.
    int checkpoints; ///< Requested synchronized output intervals.
    int rank; ///< Current MPI rank.
    int ranks; ///< Communicator size.
    const char *outputDirectory; ///< Fresh native diagnostic output directory.
    double *timeSteps; ///< Host array of uniform-log(a) native endpoint scale factors.
    FILE *checkpointFile; ///< Root-owned checkpoint diagnostic stream.
    int lastCheckpoint; ///< Last exported synchronized checkpoint index.
} RunContext;

/**
 * @brief Report a native-reference error and abort MPI_COMM_WORLD before exiting.
 * @see cosmology_contracts cosmology_validation
 * @param message Failure text retained for the fixed-budget check.
 */
static void fail(const char *message) {
    int rank; ///< Current MPI rank.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "FastPMEvolution rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(2);
}

/**
 * @brief Allocate zeroed host memory after checking multiplication overflow and allocation success.
 * @see cosmology_contracts cosmology_validation
 * @param count Allocation element count checked against SIZE_MAX overflow.
 * @param size Size in bytes of each element.
 * @return Zeroed valid host allocation; failures abort the native communicator.
 */
static void *checkedAlloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) fail("allocation size overflow");
    void *result = calloc(count ? count : 1, size);
    if (!result) fail("allocation failed");
    return result;
}

/**
 * @brief Parse a strictly positive int without overflow or trailing input.
 * @see cosmology_contracts cosmology_validation
 * @param value Serialized scalar argument, validated according to the parser's strict range.
 * @return Positive scalar int within INT_MAX.
 */
static int parsePositiveInt(const char *value) {
    char *end;
    errno = 0;
    long result = strtol(value, &end, 10);
    if (errno || end == value || *end || result < 1 || result > INT_MAX)
        fail("invalid positive integer argument");
    return (int)result;
}

/**
 * @brief Parse a finite strictly positive floating scalar without trailing input.
 * @see cosmology_contracts cosmology_validation
 * @param value Serialized scalar argument, validated according to the parser's strict range.
 * @return Finite positive double.
 */
static double parsePositiveDouble(const char *value) {
    char *end;
    errno = 0;
    double result = strtod(value, &end);
    if (errno || end == value || *end || !isfinite(result) || result <= 0)
        fail("invalid positive real argument");
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
 * @brief Open one diagnostic output file below the selected run directory.
 * @see cosmology_contracts cosmology_validation
 * @param run Native RunContext with consistent particle/mesh sizes, epochs and output controls.
 * @param name Scalar/diagnostic filename or label used in error reporting and output identity.
 * @return Host FILE stream ready for diagnostic output.
 */
static FILE *openOutput(const RunContext *run, const char *name) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s", run->outputDirectory, name); ///< Comoving periodic box side in Mpc/h.
    if (length < 0 || (size_t)length >= sizeof(path)) fail("output path too long");
    FILE *file = fopen(path, "wx");
    if (!file) fail("cannot create output file; existing files are never overwritten");
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
 * @brief Create a native-reference output location without overwriting previous evidence.
 * @see cosmology_contracts cosmology_validation
 * @param run Native RunContext with consistent particle/mesh sizes, epochs and output controls.
 */
static void createOutputDirectory(const RunContext *run) {
    if (run->rank == 0) {
        if (mkdir(run->outputDirectory, 0777) && errno != EEXIST)
            fail("cannot create output directory");
        DIR *directory = opendir(run->outputDirectory);
        if (!directory) fail("cannot inspect output directory");
        struct dirent *entry;
        while ((entry = readdir(directory)))
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
                fail("output directory must be new or empty");
        closedir(directory);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Validate unit-mass phase space and populate native FastPM particle stores with the half-cell coordinate rebase.
 * @see cosmology_contracts cosmology_validation
 * @param path Input CSV or comparison output path following the exact file contract.
 * @param run Native RunContext with consistent particle/mesh sizes, epochs and output controls.
 * @param pm Native PM geometry whose node-centered positions require the documented half-cell rebase.
 * @param particles Native FastPMStore owning imported IDs, positions and float32 canonical momenta.
 */
static void importParticles(const char *path, RunContext *run, PM *pm, FastPMStore *particles) {
    FILE *input = fopen(path, "r");
    if (!input) fail("cannot open input CSV");
    char line[2048];
    if (!fgets(line, sizeof(line), input)) fail("empty input CSV");
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, "id,x,y,z,px,py,pz,mass")) fail("wrong input CSV header");
    unsigned char *seen = checkedAlloc(run->particleCount, sizeof(*seen));
    const double halfCell = run->length / (2 * run->meshGrid);
    size_t rows = 0;
    while (fgets(line, sizeof(line), input)) {
        uint64_t id;
        double physical[3], native[3], momentum[3], mass;
        int consumed = 0;
        if (!isdigit((unsigned char)line[0]) ||
            sscanf(line, "%" SCNu64 ",%lf,%lf,%lf,%lf,%lf,%lf,%lf%n", &id,
                   &physical[0], &physical[1], &physical[2],
                   &momentum[0], &momentum[1], &momentum[2], &mass, &consumed) != 8)
            fail("malformed input CSV row");
        for (char *tail = line + consumed; *tail; ++tail)
            if (!isspace((unsigned char)*tail)) fail("extra data in CSV row");
        if (rows++ >= run->particleCount || id >= run->particleCount || seen[id])
            fail("IDs must be a permutation of 0 through NP cubed minus 1");
        seen[id] = 1;
        if (mass != 1) fail("equal unit particle masses required");
        for (int d = 0; d < 3; ++d) {
            if (!isfinite(physical[d]) || !isfinite(momentum[d]) || fabs(momentum[d]) > FLT_MAX)
                fail("nonfinite position or unrepresentable native momentum");
            physical[d] = wrap(physical[d], run->length);
            native[d] = wrap(physical[d] - halfCell, run->length);
        }
        if (pm_pos_to_rank(pm, native) == run->rank) {
            size_t index = particles->np++;
            particles->id[index] = id;
            for (int d = 0; d < 3; ++d) {
                particles->x[index][d] = native[d];
                /* Native v is canonical p, with the upstream float32 storage. */
                particles->v[index][d] = (float)momentum[d];
            }
        }
    }
    if (ferror(input)) fail("input read failed");
    fclose(input);
    if (rows != run->particleCount) fail("input must contain NP cubed particles");
    free(seen);
    particles->meta.M0 = 1;
    particles->meta.a_x = particles->meta.a_v = run->aInitial;
}

/**
 * @brief Export the actual native full-step state only when positions and momenta are synchronized.
 * @see cosmology_contracts cosmology_validation
 * @param context Native solver event context supplied to the callback.
 * @param baseEvent Native event whose endpoint/synchronization state is checked before output.
 * @param userdata RunContext pointer retaining checkpoint cadence and output state.
 * @return Native event callback status; output is gated by actual synchronization and cadence.
 */
static int writeSynchronizedCheckpoint(void *context, FastPMEvent *baseEvent, void *userdata) {
    FastPMSolver *solver = context;
    FastPMTransitionEvent *event = (FastPMTransitionEvent *)baseEvent;
    RunContext *run = userdata;
    const FastPMState *state = event->transition->end;
    /* Intermediate drift checkpoints have x=v, but not force=x. Only export
     * the naturally synchronized full-step endpoints, never interpolations. */
    if (state->x != state->v || state->x != state->force || state->x < 0 || state->x % 2)
        return 0;
    const int step = state->x / 2;
    const int stride = run->steps / run->checkpoints;
    if (step % stride) return 0;
    const int checkpoint = step / stride;
    if (checkpoint != run->lastCheckpoint + 1 || checkpoint > run->checkpoints)
        fail("unexpected native synchronized checkpoint order");
    FastPMStore *particles = fastpm_solver_get_species(solver, FASTPM_SPECIES_CDM);
    const double a = run->timeSteps[step];
    if (particles->meta.a_x != a || particles->meta.a_v != a)
        fail("native positions and momenta are not synchronized at requested endpoint");
    uint64_t localCount = particles->np, totalCount = 0;
    MPI_Allreduce(&localCount, &totalCount, 1, MPI_UINT64_T, MPI_SUM, solver->comm);
    if (totalCount != run->particleCount) fail("particle count changed during evolution");
    char name[128];
    snprintf(name, sizeof(name), "particles_checkpoint%04d_rank%d.csv", checkpoint, run->rank);
    FILE *output = openOutput(run, name);
    fprintf(output, "id,x,y,z,px,py,pz\n");
    const double halfCell = run->length / (2 * run->meshGrid);
    double localMomentum[3] = {0}, totalMomentum[3];
    for (size_t i = 0; i < particles->np; ++i) {
        double physical[3];
        const uint64_t id = particles->id[i];
        if (id >= run->particleCount) fail("native evolution produced invalid particle ID");
        for (int d = 0; d < 3; ++d) {
            if (!isfinite(particles->x[i][d]) || !isfinite(particles->v[i][d]))
                fail("native evolution produced nonfinite phase space");
            /* Export the actual native state even at checkpoint zero, so the
             * initial-state check can detect import/origin/migration errors. */
            physical[d] = wrap(particles->x[i][d] + halfCell, run->length);
            localMomentum[d] += particles->v[i][d];
        }
        fprintf(output, "%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                id, physical[0], physical[1], physical[2],
                (double)particles->v[i][0], (double)particles->v[i][1], (double)particles->v[i][2]);
    }
    closeOutput(output);
    MPI_Allreduce(localMomentum, totalMomentum, 3, MPI_DOUBLE, MPI_SUM, solver->comm);
    if (run->rank == 0) {
        fprintf(run->checkpointFile, "%d,%d,%.17g,%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g\n",
                checkpoint, step, a, totalCount,
                fabs((double)totalCount / run->particleCount - 1),
                totalMomentum[0] / totalCount, totalMomentum[1] / totalCount,
                totalMomentum[2] / totalCount, HubbleEa(a, solver->cosmology));
        if (fflush(run->checkpointFile)) fail("checkpoint diagnostics flush failed");
    }
    run->lastCheckpoint = checkpoint;
    return 0;
}

/**
 * @brief Record native plain-PM background kick/drift factors without replacing native integration.
 * @see cosmology_contracts cosmology_validation
 * @param solver Initialized native plain-PM solver; its own factors/operators are recorded, not reimplemented.
 * @param run Native RunContext with consistent particle/mesh sizes, epochs and output controls.
 */
static void writeNativeFactorProbe(FastPMSolver *solver, const RunContext *run) {
    FILE *output = run->rank == 0 ? openOutput(run, "factors.csv") : NULL;
    if (output) fprintf(output, "step,a0,ah,a1,drift,canonical_kick0,canonical_kick1\n");
    /* All ranks invoke the native initializers because their logging API uses
     * collective barriers. This is a read-only probe, not the evolution loop:
     * these factors are never applied to a particle by the adapter. */
    for (int step = 0; step < run->steps; ++step) {
        const double a0 = run->timeSteps[step], a1 = run->timeSteps[step + 1];
        const double ah = exp(.5 * log(a0) + .5 * log(a1));
        FastPMDriftFactor drift0, drift1;
        FastPMKickFactor kick0, kick1;
        fastpm_drift_init(&drift0, solver, a0, ah, ah);
        fastpm_drift_init(&drift1, solver, ah, ah, a1);
        fastpm_kick_init(&kick0, solver, a0, a0, ah);
        fastpm_kick_init(&kick1, solver, ah, a1, a1);
        const double drift = drift0.dyyy[drift0.nsamples - 1] - drift0.dyyy[0]
                           + drift1.dyyy[drift1.nsamples - 1] - drift1.dyyy[0];
        const double canonicalKick0 = (kick0.dda[kick0.nsamples - 1] - kick0.dda[0]) / (-1.5 * run->omegaM);
        const double canonicalKick1 = (kick1.dda[kick1.nsamples - 1] - kick1.dda[0]) / (-1.5 * run->omegaM);
        if (!isfinite(drift) || !isfinite(canonicalKick0) || !isfinite(canonicalKick1))
            fail("nonfinite native time-factor probe");
        if (output) fprintf(output, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                            step, a0, ah, a1, drift, canonicalKick0, canonicalKick1);
    }
    if (output) closeOutput(output);
}

/**
 * @brief Run native plain-pm evolution harness with imported phase space and synchronized csv exports.
 * @see cosmology_contracts cosmology_validation
 * @param argc Program argument count; this executable checks its own exact usage.
 * @param argv Program argument vector; see the file/workflow contract for scalar and path units.
 * @return Zero on successful completion; malformed/native fatal errors return nonzero or abort the communicator.
 * Fatal distributed failures must terminate communicator peers; the host-only test uses ordinary process status.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    RunContext run = {0};
    MPI_Comm_rank(MPI_COMM_WORLD, &run.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &run.ranks);
    if (argc != 11)
        fail("usage: FastPMEvolution NP NM L Omega_m ai af n_steps n_checkpoints input.csv output_dir");
    run.particleGrid = parsePositiveInt(argv[1]);
    run.meshGrid = parsePositiveInt(argv[2]);
    if (run.particleGrid < 4 || run.meshGrid < 4 || run.particleGrid % 2 || run.meshGrid % 2)
        fail("NP and NM must be even integers >= 4");
    if (run.particleGrid > SIZE_MAX / run.particleGrid ||
        run.particleGrid * run.particleGrid > SIZE_MAX / run.particleGrid)
        fail("particle count overflow");
    run.particleCount = run.particleGrid * run.particleGrid * run.particleGrid;
    if (run.particleCount > INT_MAX)
        fail("local reference particle count exceeds native signed-int loops");
    run.length = parsePositiveDouble(argv[3]);
    run.omegaM = parsePositiveDouble(argv[4]);
    run.aInitial = parsePositiveDouble(argv[5]);
    run.aFinal = parsePositiveDouble(argv[6]);
    run.steps = parsePositiveInt(argv[7]);
    run.checkpoints = parsePositiveInt(argv[8]);
    if (run.omegaM > 1 || run.aInitial >= run.aFinal || run.aFinal > 1)
        fail("require 0<Omega_m<=1 and 0<ai<af<=1");
    if (run.steps > (INT_MAX - 3) / 5 || run.steps % run.checkpoints)
        fail("n_steps must be divisible by n_checkpoints and fit native state table");
    run.outputDirectory = argv[10];
    run.lastCheckpoint = -1;
    run.timeSteps = checkedAlloc((size_t)run.steps + 1, sizeof(*run.timeSteps));
    const double logStep = log(run.aFinal / run.aInitial) / run.steps;
    for (int i = 0; i <= run.steps; ++i) run.timeSteps[i] = run.aInitial * exp(i * logStep);

    libfastpm_init();
    FastPMSolver solver = {0};
    solver.comm = MPI_COMM_WORLD;
    solver.ThisTask = run.rank;
    solver.NTask = run.ranks;
    solver.config->nc = run.particleGrid;
    solver.config->boxsize = run.length;
    solver.config->FORCE_TYPE = FASTPM_FORCE_PM;
    solver.config->KERNEL_TYPE = FASTPM_KERNEL_NAIVE;
    solver.config->SOFTENING_TYPE = FASTPM_SOFTENING_NONE;
    solver.config->PAINTER_TYPE = FASTPM_PAINTER_CIC;
    solver.config->painter_support = 2;
    solver.config->UseFFTW = 1;
    solver.config->NprocY = 1;
    /* The unused nonstandard branch is still evaluated in native Sphi. A
     * nonzero nLPT avoids its otherwise discarded 0/0; FORCE_PM returns the
     * ordinary background quadrature irrespective of this setting. */
    solver.config->nLPT = -2.5;
    solver.cosmology->h = .675;
    solver.cosmology->Omega_m = run.omegaM;
    solver.cosmology->w0 = -1;
    solver.cosmology->growth_mode = FASTPM_GROWTH_MODE_LCDM;
    /* Zero initialization explicitly disables radiation, curvature, neutrinos,
     * evolving dark energy, PGD and all non-PM acceleration modifications. */
    fastpm_cosmology_init(solver.cosmology);

    /* Import initialization only: no generated lattice or LPT fields. Native
     * solver_init/vpm_create unnecessarily require N divisible by MPI ranks.
     * A fixed native VPM descriptor supports the same tested uneven FFTW slabs
     * as the frozen-force adapter, including three ranks. Evolution itself is
     * the actual unchanged fastpm_solver_evolve implementation. */
    VPM meshes[2] = {0};
    PMInit pmInit = {(ptrdiff_t)run.meshGrid, run.length, 1, 1, 1};
    pm_init(&meshes[0].pm, &pmInit, solver.comm);
    if (pm_i_region(&meshes[0].pm)->size[0] == 0) fail("empty native FFTW slab; reduce MPI ranks");
    meshes[0].a_start = run.aInitial;
    meshes[0].pm_nc_factor = (double)run.meshGrid / run.particleGrid;
    meshes[1].end = 1;
    solver.vpm_list = meshes;
    /* Deliberately generous local-diagnostic capacity: any rank can own every
     * imported particle after shell crossing, with no allocation model fit. */
    fastpm_store_init(solver.cdm, "imported", run.particleCount,
                      COLUMN_POS | COLUMN_VEL | COLUMN_ID | COLUMN_ACC, FASTPM_MEMORY_HEAP);
    fastpm_solver_add_species(&solver, FASTPM_SPECIES_CDM, solver.cdm);
    importParticles(argv[9], &run, &meshes[0].pm, solver.cdm);
    createOutputDirectory(&run);
    if (run.rank == 0) {
        run.checkpointFile = openOutput(&run, "checkpoints.csv");
        fprintf(run.checkpointFile, "checkpoint,step,a,particle_count,mass_error,mean_px,mean_py,mean_pz,E\n");
    }
    fastpm_add_event_handler(&solver.event_handlers, FASTPM_EVENT_TRANSITION,
                            FASTPM_EVENT_STAGE_AFTER, writeSynchronizedCheckpoint, &run);
    writeNativeFactorProbe(&solver, &run);
    const double start = MPI_Wtime();
    fastpm_solver_evolve(&solver, run.timeSteps, run.steps + 1);
    if (run.lastCheckpoint != run.checkpoints) fail("native evolution missed requested checkpoint");
    double seconds = MPI_Wtime() - start, maximumSeconds;
    MPI_Allreduce(&seconds, &maximumSeconds, 1, MPI_DOUBLE, MPI_MAX, solver.comm);
    if (run.rank == 0) {
        closeOutput(run.checkpointFile);
        FILE *metadata = openOutput(&run, "metadata.txt");
        fprintf(metadata, "reference=FastPM\nupstream_commit=%s\nn_particles_grid=%zu\nn_grid=%zu\n"
                "particle_count=%zu\nn_steps=%d\nn_checkpoints=%d\nbox_size=%.17g\nomega_m=%.17g\n"
                "a_initial=%.17g\na_final=%.17g\nranks=%d\nthreads=1\nseconds=%.17g\n"
                "force_type=FASTPM_FORCE_PM\nintegrator=fastpm_solver_evolve\n"
                "scheduler=logarithmic_scale_factor_endpoints; native_geometric_midpoints\n"
                "native_drift_schedule=two_subdrifts_at_fixed_half_step_momentum\n"
                "snapshot=synchronized_full_step_TRANSITION_AFTER; no interpolation\n"
                "snapshot_velocity_conversion=none\ncanonical_momentum=p=a^2*dx/d(H0*t)\n"
                "momentum_import=cast_to_native_float32\nposition_precision_bits=64\n"
                "momentum_precision_bits=32\nparticle_acc_precision_bits=32\nfft_precision_bits=64\n"
                "wavevector_precision_bits=32\nwavevector_square_precision_bits=32\ncic_weights_precision_bits=64\n"
                "background=flat_matter_plus_Lambda\nhubble=0.675\nOmega_k=0\nT_cmb=0\nN_eff=0\nN_nu=0\nN_ncdm=0\n"
                "w0=-1\nwa=0\nbackground_quad_relative_tolerance=1e-8\n"
                "factors_csv=read_only_native_initializer_probe; not_applied_by_adapter\n"
                "factor_drift=sum_of_native_subdrift_endpoint_differences\n"
                "factor_canonical_kick=native_dda_endpoint_difference/(-1.5*Omega_m)\n"
                "native_growth_approximation=LCDM; unused_by_plain_PM_with_imported_momenta\n"
                "nLPT=-2.5; unused_nonstandard_branch_only\n"
                "kernel=FASTPM_KERNEL_NAIVE\npainter=FASTPM_PAINTER_CIC\nsoftening=FASTPM_SOFTENING_NONE\n"
                "deconvolution_for_force=none\nnyquist_zeroing=fully_self_conjugate_corners_only\n"
                "nyquist_sign=negative\nr2c_compressed_axis=z\nfft_backend=FFTW_MPI\nfft_transposed=1\n"
                "native_position=wrap(input_position-L/(2*NM))\noutput_position=wrap(native_position+L/(2*NM))\n"
                "initial_output_position=actual_native_state_rebased_to_IPPL_frame\nforce_conversion=native_kick_includes_minus_1.5*Omega_m\n"
                "initialization_adapter=direct_native_PM_store_cosmology_fixed_VPM\n"
                "initialization_bypass=lattice_IC_generation_and_divisibility_precheck_only\n"
                "upstream_source_modifications=none\nper_rank_particle_capacity=%zu\n"
                "mass_error_definition=abs(global_unit_particle_mass/imported_unit_particle_mass-1)\n",
                FASTPM_REFERENCE_COMMIT, run.particleGrid, run.meshGrid, run.particleCount,
                run.steps, run.checkpoints, run.length, run.omegaM, run.aInitial, run.aFinal,
                run.ranks, maximumSeconds, run.particleCount);
        closeOutput(metadata);
    }
    fastpm_destroy_event_handlers(&solver.event_handlers);
    fastpm_store_destroy(solver.cdm);
    pm_destroy(&meshes[0].pm);
    fastpm_cosmology_destroy(solver.cosmology);
    free(run.timeSteps);
    libfastpm_cleanup();
    MPI_Finalize();
    return 0;
}
