#!/usr/bin/env bash
# Build a pinned, unmodified native FastPM force reference, entirely locally.
# Usage: bash build_fastpm.sh BUILD_DIRECTORY
# Optional: FASTPM_CC, FASTPM_MPICC, FASTPM_FFTW_PREFIX, FASTPM_JOBS.
# The FFTW prefix must provide static double-precision MPI FFTW libraries.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s BUILD_DIRECTORY\n' "$0" >&2
    exit 2
fi
scriptDir=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$1"
buildDir=$(cd "$1" && pwd -P)
sourceDir="$buildDir/source"
prefix="$buildDir/install"
cc=${FASTPM_CC:-cc}
mpicc=${FASTPM_MPICC:-mpicc}
jobs=${FASTPM_JOBS:-4}
export OMPI_CC="$cc"
commit=15b6c4fd7502a81d99dd13f54fcc9cfa44be1331
repository=https://github.com/fastpm/fastpm.git
fftwVersion=3.3.10
fftwSha=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
pfftVersion=1.0.8-alpha3-fftw3
pfftSha=91d2229996b06b37b719e94ba50a2a813c744611eb936a193041575a407cca97
gslVersion=2.8
gslSha=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190
fftwPrefix=${FASTPM_FFTW_PREFIX:-"$buildDir/fftw-install"}
mkdir -p "$buildDir/downloads" "$buildDir/deps" "$prefix"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"
    else shasum -a 256 "$@"; fi
}
fetch() {
    local url=$1 archive=$2 expected=$3
    if [[ ! -f "$archive" ]]; then curl -fL --retry 2 "$url" -o "$archive"; fi
    local actual
    actual=$(sha256 "$archive" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        printf 'Dependency checksum mismatch for %s: %s\n' "$archive" "$actual" >&2
        exit 1
    fi
}

if [[ ! -d "$sourceDir" ]]; then
    git clone "$repository" "$sourceDir" > "$buildDir/acquisition.log" 2>&1
    git -C "$sourceDir" checkout --detach "$commit" >> "$buildDir/acquisition.log" 2>&1
fi
if [[ $(git -C "$sourceDir" rev-parse HEAD) != "$commit" ]]; then
    printf 'Existing FastPM checkout is not the pinned commit; use a fresh build directory.\n' >&2
    exit 1
fi
git -C "$sourceDir" diff --exit-code HEAD --
(
    cd "$sourceDir"
    while IFS= read -r -d '' file; do sha256 "$file"; done < <(git ls-files -z)
) > "$buildDir/source.sha256"

if [[ -n ${FASTPM_FFTW_PREFIX:-} ]]; then
    for item in include/fftw3.h include/fftw3-mpi.h lib/libfftw3.a lib/libfftw3_mpi.a; do
        [[ -f "$fftwPrefix/$item" ]] || { printf 'Missing FFTW dependency: %s\n' "$fftwPrefix/$item" >&2; exit 2; }
    done
else
    archive="$buildDir/downloads/fftw-$fftwVersion.tar.gz"
    fetch "https://www.fftw.org/fftw-$fftwVersion.tar.gz" "$archive" "$fftwSha"
    [[ -d "$buildDir/deps/fftw-$fftwVersion" ]] || tar -xzf "$archive" -C "$buildDir/deps"
    mkdir -p "$buildDir/fftw-build"
    (
        cd "$buildDir/fftw-build"
        env CC="$cc" MPICC="$mpicc" CFLAGS=-O2 \
            "../deps/fftw-$fftwVersion/configure" --prefix="$fftwPrefix" \
            --enable-mpi --disable-fortran --disable-shared --enable-static > configure.log 2>&1
        make -j "$jobs" > build.log 2>&1
        make install > install.log 2>&1
    )
fi
fftwPrefix=$(cd "$fftwPrefix" && pwd -P)

archive="$buildDir/downloads/pfft-$pfftVersion.tar.gz"
fetch "https://github.com/rainwoodman/pfft/releases/download/$pfftVersion/pfft-$pfftVersion.tar.gz" "$archive" "$pfftSha"
[[ -d "$buildDir/deps/pfft-$pfftVersion" ]] || tar -xzf "$archive" -C "$buildDir/deps"
mkdir -p "$buildDir/pfft-build"
(
    cd "$buildDir/pfft-build"
    # The PFFT release bundles older FFTW. Skip that subdirectory and link the
    # explicitly selected FFTW instead; this changes no PFFT source file.
    env CC="$mpicc" MPICC="$mpicc" CFLAGS=-O2 CPPFLAGS="-I$fftwPrefix/include" \
        LDFLAGS="-L$fftwPrefix/lib" "../deps/pfft-$pfftVersion/configure" \
        --prefix="$prefix" --disable-shared --enable-static --disable-fortran \
        --disable-doc --enable-mpi --no-recursion > configure.log 2>&1
    make -j "$jobs" 'SUBDIRS=kernel gcell util api .' > build.log 2>&1
    make 'SUBDIRS=kernel gcell util api .' install > install.log 2>&1
)

archive="$buildDir/downloads/gsl-$gslVersion.tar.gz"
fetch "https://ftp.gnu.org/gnu/gsl/gsl-$gslVersion.tar.gz" "$archive" "$gslSha"
[[ -d "$buildDir/deps/gsl-$gslVersion" ]] || tar -xzf "$archive" -C "$buildDir/deps"
mkdir -p "$buildDir/gsl-build"
(
    cd "$buildDir/gsl-build"
    env CC="$cc" CFLAGS=-O2 "../deps/gsl-$gslVersion/configure" --prefix="$prefix" \
        --disable-shared --enable-static > configure.log 2>&1
    make -j "$jobs" > build.log 2>&1
    make install > install.log 2>&1
)

for dep in mpsort kdcount bigfile chealpix; do
    make -B -C "$sourceDir/depends/$dep" -j "$jobs" install PREFIX="$prefix" \
        CC="$mpicc" MPICC="$mpicc" CFLAGS='-O2 -std=gnu99' > "$buildDir/$dep-build.log" 2>&1
done
cppFlags=(-DFASTPM_FFT_PRECISION=64 -I"$sourceDir/api" -I"$prefix/include" -I"$fftwPrefix/include")
# Command-line overrides of upstream build variables only; no source patches.
make -B -C "$sourceDir/libfastpm" -j "$jobs" CC="$mpicc" OPENMP= \
    OPTIMIZE='-O2 -std=gnu99' "CPPFLAGS=${cppFlags[*]}" \
    "DEPCMD=$mpicc -MG -MP -MM" > "$buildDir/fastpm-build.log" 2>&1
libraries=("$sourceDir/libfastpm/libfastpm.a" "$prefix/lib/libmpsort-mpi.a"
           "$prefix/lib/libradixsort.a" "$prefix/lib/libbigfile-mpi.a"
           "$prefix/lib/libbigfile.a" "$prefix/lib/libkdcount.a"
           "$prefix/lib/libchealpix.a" "$prefix/lib/libpfft.a"
           "$fftwPrefix/lib/libfftw3_mpi.a" "$fftwPrefix/lib/libfftw3.a"
           "$prefix/lib/libgsl.a" "$prefix/lib/libgslcblas.a")
(
    # FastPM embeds caller __FILE__ in an 80-byte diagnostic memory tag. Compile
    # by basename to keep that native diagnostic safe without a source patch.
    cd "$scriptDir"
    "$mpicc" -O2 -std=gnu99 -Wall -Wextra "${cppFlags[@]}" \
        "-DFASTPM_REFERENCE_COMMIT=\"$commit\"" -I"$sourceDir/libfastpm" \
        FastPMForce.c "${libraries[@]}" -lm -o "$buildDir/FastPMForce" \
        > "$buildDir/harness-build.log" 2>&1
)
git -C "$sourceDir" diff --exit-code HEAD --
(
    cd "$sourceDir"
    while IFS= read -r -d '' file; do sha256 "$file"; done < <(git ls-files -z)
) > "$buildDir/source-after-build.sha256"
cmp "$buildDir/source.sha256" "$buildDir/source-after-build.sha256"

{
    printf 'upstream_repository=%s\nupstream_commit=%s\n' "$repository" "$commit"
    printf 'source_directory=%s\nbuild_directory=%s\n' "$sourceDir" "$buildDir"
    printf 'source_modifications=none (tracked source hashes verified before/after build)\n'
    printf 'source_tree=%s\n' "$(git -C "$sourceDir" rev-parse HEAD^{tree})"
    printf 'gsl_version=%s\ngsl_archive_sha256=%s\n' "$gslVersion" "$gslSha"
    printf 'pfft_version=%s\npfft_archive_sha256=%s\n' "$pfftVersion" "$pfftSha"
    printf 'pfft_configure=--disable-shared --enable-static --disable-fortran --disable-doc --enable-mpi --no-recursion\n'
    printf 'pfft_make_override=SUBDIRS=kernel gcell util api . (external FFTW; no bundled FFTW build)\n'
    printf 'fftw_prefix=%s\n' "$fftwPrefix"
    if [[ -n ${FASTPM_FFTW_PREFIX:-} ]]; then
        printf 'fftw_source=external FASTPM_FFTW_PREFIX; exact library hashes recorded below\n'
    else
        printf 'fftw_version=%s\nfftw_archive_sha256=%s\n' "$fftwVersion" "$fftwSha"
    fi
    printf 'cc=%s\nmpicc=%s\nflags=-O2 -std=gnu99\nopenmp=disabled\n' "$cc" "$mpicc"
    printf 'harness_compile_directory=%s\nharness_compile_filename=FastPMForce.c (short native diagnostic memory tag)\n' "$scriptDir"
    printf 'cpp_flags='; printf '%q ' "${cppFlags[@]}"; printf '\n'
    printf 'fft_precision=64\nparticle_position_precision=64\nparticle_acc_precision=32\n'
    printf 'wavevector_and_individual_k_squared_precision=32\ncic_weight_precision=64\n'
    printf 'runtime_backend=FFTW_MPI; PFFT linked because native pm_module_init calls pfft_init\n'
    printf 'native_kernel=NAIVE; CIC; no softening; no deconvolution; no Nyquist modifications\n'
    printf 'force_conversion=canonical_F=-1.5*Omega_m*native_acc\n'
    printf 'origin_conversion=x_native=wrap(x_IPPL-L/(2*N))\n'
    uname -a
    "$cc" --version
    "$mpicc" --showme 2>/dev/null || true
    sha256 "$scriptDir/FastPMForce.c" "$scriptDir/build_fastpm.sh" "$buildDir/FastPMForce" \
        "$buildDir/source.sha256" "${libraries[@]}"
} > "$buildDir/build-manifest.txt"
printf 'Reference executable: %s/FastPMForce\nManifest: %s/build-manifest.txt\n' "$buildDir" "$buildDir"
