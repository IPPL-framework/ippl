#!/usr/bin/env bash
# Build the supplied legacy initializer without changing any source file.
# Usage: bash build_zarija.sh /path/to/zarija-source /path/to/build-area
# Optional: ZARIJA_CC, ZARIJA_CXX, ZARIJA_MPICC, ZARIJA_MPICXX,
#           ZARIJA_FFTW_PREFIX (existing MPI FFTW3), ZARIJA_JOBS.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'Usage: %s SOURCE_DIRECTORY BUILD_DIRECTORY\n' "$0" >&2
    exit 2
fi
sourceDir=$(cd "$1" && pwd -P)
scriptDir=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$2"
buildDir=$(cd "$2" && pwd -P)
if [[ "$sourceDir" == "$buildDir" ]]; then
    printf 'Build directory must differ from the reference source directory.\n' >&2
    exit 2
fi

cc=${ZARIJA_CC:-cc}
cxx=${ZARIJA_CXX:-c++}
mpicc=${ZARIJA_MPICC:-mpicc}
mpicxx=${ZARIJA_MPICXX:-mpicxx}
jobs=${ZARIJA_JOBS:-4}
fftwVersion=3.3.10
fftwSha=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
fftwUrl="https://www.fftw.org/fftw-${fftwVersion}.tar.gz"
fftwPrefix=${ZARIJA_FFTW_PREFIX:-"$buildDir/fftw-install"}
export OMPI_CC="$cc" OMPI_CXX="$cxx"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"
    else shasum -a 256 "$@"; fi
}

if [[ ! -f "$fftwPrefix/include/fftw3-mpi.h" || ! -f "$fftwPrefix/lib/libfftw3_mpi.a" ]]; then
    if [[ -n ${ZARIJA_FFTW_PREFIX:-} ]]; then
        printf 'ZARIJA_FFTW_PREFIX needs include/fftw3-mpi.h and lib/libfftw3_mpi.a\n' >&2
        exit 2
    fi
    archive="$buildDir/fftw-${fftwVersion}.tar.gz"
    if [[ ! -f "$archive" ]]; then curl -fL --retry 2 "$fftwUrl" -o "$archive"; fi
    actualSha=$(sha256 "$archive" | awk '{print $1}')
    if [[ "$actualSha" != "$fftwSha" ]]; then
        printf 'FFTW archive checksum mismatch: %s\n' "$actualSha" >&2
        exit 1
    fi
    if [[ ! -d "$buildDir/fftw-${fftwVersion}" ]]; then
        tar -xzf "$archive" -C "$buildDir"
    fi
    mkdir -p "$buildDir/fftw-build"
    (
        cd "$buildDir/fftw-build"
        env CC="$cc" MPICC="$mpicc" CFLAGS=-O2 \
            "../fftw-${fftwVersion}/configure" --prefix="$fftwPrefix" \
            --enable-mpi --disable-fortran --disable-shared --enable-static \
            > configure.log 2>&1
        make -j "$jobs" > build.log 2>&1
        make install > install.log 2>&1
    )
fi

cppSources=(Cosmology.cpp MT_Random.cpp DataBase.cpp InputParser.cpp Parallelization.cpp
            Output.cpp Initializer.cpp main.cpp)
cSources=(PerfMon.c distribution.c)
headers=(Cosmology.h DataBase.h MT_Random.h Parallelization.h InputParser.h PerfMon.h
         TypesAndDefs.h Initializer.h distribution.h Output.h)
files=("${cppSources[@]}" "${cSources[@]}" "${headers[@]}" README Makefile input.par cmb.tf)
objectDir="$buildDir/zarija"
mkdir -p "$objectDir"
for file in "${files[@]}"; do cp "$sourceDir/$file" "$objectDir/$file"; done

# Open MPI 5 retains these symbols but hides their declarations by default. The
# initializer contains an UNUSED MPI-1 datatype helper. Exposing the declarations
# permits its unmodified translation unit to compile; no algorithm is patched.
# This macro has no effect on MPI implementations that do not recognize it.
cppFlags=(-O2 -std=c++11 -DDOUBLE_REAL -DFFTW3 -DUSENAMESPACE
          -DOMPI_OMIT_MPI1_COMPAT_DECLS=0 -I"$fftwPrefix/include")
cFlags=(-O2 -std=c99 -DDOUBLE_REAL -I"$fftwPrefix/include")
objects=()
(
    cd "$sourceDir"
    sha256 "${files[@]}"
) > "$buildDir/source.sha256"
(
    cd "$objectDir"
    sha256 "${files[@]}"
) > "$buildDir/copied-source.sha256"
cmp "$buildDir/source.sha256" "$buildDir/copied-source.sha256"

for file in "${cSources[@]}"; do
    object="$objectDir/${file%.c}.o"
    "$mpicc" "${cFlags[@]}" -c "$objectDir/$file" -o "$object"
    objects+=("$object")
done
for file in "${cppSources[@]}"; do
    object="$objectDir/${file%.cpp}.o"
    "$mpicxx" "${cppFlags[@]}" -c "$objectDir/$file" -o "$object"
    objects+=("$object")
done
"$mpicxx" "${objects[@]}" "$fftwPrefix/lib/libfftw3_mpi.a" \
    "$fftwPrefix/lib/libfftw3.a" -lm -o "$buildDir/init"
"$mpicxx" "${cppFlags[@]}" -I"$objectDir" "$scriptDir/ReferenceABI.cpp" \
    -o "$buildDir/reference-abi"

{
    printf 'source_directory=%s\nbuild_directory=%s\n' "$sourceDir" "$buildDir"
    printf 'source_modifications=none (all copied files match source.sha256)\n'
    if [[ -z ${ZARIJA_FFTW_PREFIX:-} ]]; then
        printf 'fftw_url=%s\nfftw_archive_sha256=%s\n' "$fftwUrl" "$fftwSha"
    else
        printf 'fftw_source=external ZARIJA_FFTW_PREFIX (library hashes recorded below)\n'
    fi
    printf 'fftw_prefix=%s\n' "$fftwPrefix"
    printf 'fftw_configure=--enable-mpi --disable-fortran --disable-shared --enable-static\n'
    printf 'cc=%s\ncxx=%s\nmpicc=%s\nmpicxx=%s\n' "$cc" "$cxx" "$mpicc" "$mpicxx"
    printf 'c_flags='; printf '%q ' "${cFlags[@]}"; printf '\n'
    printf 'cpp_flags='; printf '%q ' "${cppFlags[@]}"; printf '\n'
    printf 'reference_precision=DOUBLE_REAL (double), default integer=int, IDtype=long\n'
    printf 'reference_test_macro=disabled\n'
    printf 'output_format_2=per-rank records: x,vx,y,vy,z,vz [real], id [sizeof(integer) bytes]\n'
    printf 'output_format_0=ASCII with 16 significant digits under DOUBLE_REAL\n'
    printf 'output_warning=serial binary MPI ID transfer has long/int type mismatch; prefer format 2\n'
    "$buildDir/reference-abi"
    uname -a
    "$cc" --version
    "$cxx" --version
    "$mpicc" --showme 2>/dev/null || true
    "$mpicxx" --showme 2>/dev/null || true
    sha256 "$buildDir/init" "$fftwPrefix/lib/libfftw3_mpi.a" "$fftwPrefix/lib/libfftw3.a"
} > "$buildDir/build-manifest.txt"

printf 'Reference executable: %s/init\nManifest: %s/build-manifest.txt\n' "$buildDir" "$buildDir"
