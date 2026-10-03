#!/usr/bin/env bash
# Link the evolution adapter to the already-qualified, pinned FastPM libraries.
# This script never rebuilds or modifies the frozen-force reference artifacts.
# Usage: bash build_fastpm_evolution.sh FASTPM_BUILD_DIRECTORY [OUTPUT_DIRECTORY]
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: %s FASTPM_BUILD_DIRECTORY [OUTPUT_DIRECTORY]\n' "$0" >&2
    exit 2
fi
scriptDir=$(cd "$(dirname "$0")" && pwd -P)
referenceDir=$(cd "$1" && pwd -P)
outputArg=${2:-"$referenceDir/evolution"}
mkdir -p "$outputArg"
outputDir=$(cd "$outputArg" && pwd -P)
if [[ "$outputDir" == "$referenceDir" ]]; then
    printf 'Evolution output must not be the frozen-force build directory.\n' >&2
    exit 2
fi
manifest="$referenceDir/build-manifest.txt"
sourceDir="$referenceDir/source"
prefix="$referenceDir/install"
commit=15b6c4fd7502a81d99dd13f54fcc9cfa44be1331

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"
    else shasum -a 256 "$@"; fi
}
checkHashes() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum -c -
    else shasum -a 256 -c -; fi
}
manifestValue() {
    awk -v key="$1" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$manifest"
}
verifyReference() {
    [[ -f "$manifest" ]] || { printf 'Missing frozen-force build manifest.\n' >&2; exit 2; }
    [[ $(git -C "$sourceDir" rev-parse HEAD) == "$commit" ]] || { printf 'Wrong FastPM source pin.\n' >&2; exit 2; }
    [[ $(manifestValue upstream_commit) == "$commit" ]] || { printf 'Wrong manifest source pin.\n' >&2; exit 2; }
    git -C "$sourceDir" diff --exit-code HEAD --
    local count
    count=$(awk 'length($1) == 64 && $1 !~ /[^0-9a-f]/ {n++} END {print n+0}' "$manifest")
    [[ "$count" -ge 16 ]] || { printf 'Frozen-force manifest has insufficient artifact hashes.\n' >&2; exit 2; }
    awk 'length($1) == 64 && $1 !~ /[^0-9a-f]/' "$manifest" | checkHashes
    (cd "$sourceDir"; checkHashes < "$referenceDir/source.sha256")
}
verifyReference > "$outputDir/reference-verification-before.log" 2>&1
priorManifestSha=$(sha256 "$manifest" | awk '{print $1}')
cc=$(manifestValue cc)
mpicc=$(manifestValue mpicc)
fftwPrefix=$(manifestValue fftw_prefix)
[[ -n "$cc" && -n "$mpicc" && -n "$fftwPrefix" ]] || { printf 'Incomplete compiler/dependency provenance.\n' >&2; exit 2; }
export OMPI_CC="$cc"
cppFlags=(-DFASTPM_FFT_PRECISION=64 -I"$sourceDir/api" -I"$prefix/include" -I"$fftwPrefix/include")
libraries=("$sourceDir/libfastpm/libfastpm.a" "$prefix/lib/libmpsort-mpi.a"
           "$prefix/lib/libradixsort.a" "$prefix/lib/libbigfile-mpi.a"
           "$prefix/lib/libbigfile.a" "$prefix/lib/libkdcount.a"
           "$prefix/lib/libchealpix.a" "$prefix/lib/libpfft.a"
           "$fftwPrefix/lib/libfftw3_mpi.a" "$fftwPrefix/lib/libfftw3.a"
           "$prefix/lib/libgsl.a" "$prefix/lib/libgslcblas.a")
(
    # Preserve short __FILE__: native diagnostic memory tags have fixed size.
    cd "$scriptDir"
    "$mpicc" -O2 -std=gnu99 -Wall -Wextra "${cppFlags[@]}" \
        "-DFASTPM_REFERENCE_COMMIT=\"$commit\"" -I"$sourceDir/libfastpm" \
        FastPMEvolution.c "${libraries[@]}" -lm -o "$outputDir/FastPMEvolution" \
        > "$outputDir/harness-build.log" 2>&1
)
verifyReference > "$outputDir/reference-verification-after.log" 2>&1
[[ $(sha256 "$manifest" | awk '{print $1}') == "$priorManifestSha" ]] || { printf 'Frozen-force manifest changed during link.\n' >&2; exit 1; }
cp "$manifest" "$outputDir/frozen-force-build-manifest.txt"
{
    printf 'upstream_repository=https://github.com/fastpm/fastpm.git\nupstream_commit=%s\n' "$commit"
    printf 'source_directory=%s\noutput_directory=%s\n' "$sourceDir" "$outputDir"
    printf 'source_modifications=none\nlibraries_rebuilt=false\n'
    printf 'frozen_force_manifest=%s\nfrozen_force_manifest_sha256=%s\n' "$manifest" "$priorManifestSha"
    printf 'native_integrator=fastpm_solver_evolve; FASTPM_FORCE_PM\n'
    printf 'native_force=FASTPM_KERNEL_NAIVE; FASTPM_PAINTER_CIC; FASTPM_SOFTENING_NONE\n'
    printf 'initialization_adapter=imported native PM/store/cosmology and fixed native VPM descriptor\n'
    printf 'upstream_initialization_bypassed=lattice IC generation and mesh/rank divisibility precheck only\n'
    printf 'integration_modifications=none; native kick/drift/force/migration/scheduling unchanged\n'
    printf 'snapshot_adapter=direct synchronized full-step transition export; no snapshot unit conversion\n'
    printf 'position_precision=64\ncanonical_momentum_precision=32\nparticle_acc_precision=32\nfft_precision=64\n'
    printf 'cc=%s\nmpicc=%s\nflags=-O2 -std=gnu99 -Wall -Wextra\nopenmp=disabled\n' "$cc" "$mpicc"
    printf 'cpp_flags='; printf '%q ' "${cppFlags[@]}"; printf '\n'
    printf 'harness_compile_directory=%s\nharness_compile_filename=FastPMEvolution.c\n' "$scriptDir"
    "$cc" --version
    "$mpicc" --showme 2>/dev/null || true
    sha256 "$scriptDir/FastPMEvolution.c" "$scriptDir/build_fastpm_evolution.sh" \
        "$outputDir/FastPMEvolution" "$outputDir/frozen-force-build-manifest.txt" "${libraries[@]}"
} > "$outputDir/build-manifest.txt"
printf 'Evolution executable: %s/FastPMEvolution\nManifest: %s/build-manifest.txt\n' "$outputDir" "$outputDir"
