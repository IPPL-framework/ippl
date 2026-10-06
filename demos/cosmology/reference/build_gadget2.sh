#!/usr/bin/env bash
# Portable periodic TreePM build; cached executables are never rebuilt.
# Usage: build_gadget2.sh BUILD_DIRECTORY PMGRID GSL_PREFIX
# Optional: GADGET_CC, GADGET_MPICC, BUILD_JOBS.
set -euo pipefail
[[ $# == 3 ]] || { echo "Usage: $0 BUILD_DIRECTORY PMGRID GSL_PREFIX" >&2; exit 2; }
mkdir -p "$1"
buildDir=$(cd "$1" && pwd -P)
grid=$2
gslPrefix=$(cd "$3" && pwd -P)
[[ "$grid" =~ ^[0-9]+$ && "$grid" -ge 4 ]] || { echo "Invalid PMGRID" >&2; exit 2; }
executable="$buildDir/Gadget2-TreePM-$grid-double"
if [[ -x "$executable" ]]; then echo "Reuse $executable"; exit 0; fi
cc=${GADGET_CC:-cc}
mpicc=${GADGET_MPICC:-mpicc}
jobs=${BUILD_JOBS:-4}
export OMPI_CC="$cc"
[[ -f "$gslPrefix/include/gsl/gsl_math.h" ]] || { echo "Missing GSL headers" >&2; exit 2; }
mkdir -p "$buildDir/downloads" "$buildDir/source" "$buildDir/logs"
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
fetch() {
    local url=$1 archive=$2 expected=$3
    if [[ ! -f "$archive" ]]; then
        curl -fL --retry 2 "$url" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    [[ $(sha256 "$archive" | awk '{print $1}') == "$expected" ]] || {
        echo "Archive checksum mismatch: $archive" >&2; exit 2;
    }
}
fetch https://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz \
    "$buildDir/downloads/gadget-2.0.7.tar.gz" \
    8e321110b9fb2d05819f9cfcbffda19f56bb77c7cdd1ca21139b81b1fca4e3ed
fetch https://www.fftw.org/fftw-2.1.5.tar.gz \
    "$buildDir/downloads/fftw-2.1.5.tar.gz" \
    f8057fae1c7df8b99116783ef3e94a6a44518d49c72e2e630c24b689c6022630
fftwPrefix="$buildDir/fftw2-install"
if [[ ! -f "$fftwPrefix/lib/libdrfftw_mpi.a" ]]; then
    [[ -d "$buildDir/source/fftw-2.1.5" ]] || tar -xzf "$buildDir/downloads/fftw-2.1.5.tar.gz" -C "$buildDir/source"
    mkdir -p "$buildDir/fftw2-build"
    buildArgs=()
    # FFTW2 predates Apple Silicon; its generic ARM target compiles portable C.
    if [[ $(uname -s) == Darwin && $(uname -m) == arm64 ]]; then
        buildArgs=(--build=arm-apple-darwin)
    fi
    (
        cd "$buildDir/fftw2-build"
        env CC="$cc" MPICC="$mpicc" CFLAGS='-O2 -Wno-error=implicit-function-declaration -Wno-error=implicit-int' \
            ../source/fftw-2.1.5/configure --prefix="$fftwPrefix" --enable-mpi \
            --enable-type-prefix --disable-shared --enable-static "${buildArgs[@]}" > "$buildDir/logs/fftw2-configure.log" 2>&1
        make -j "$jobs" > "$buildDir/logs/fftw2-build.log" 2>&1
        make install > "$buildDir/logs/fftw2-install.log" 2>&1
    )
fi
[[ -d "$buildDir/source/Gadget-2.0.7" ]] || tar -xzf "$buildDir/downloads/gadget-2.0.7.tar.gz" -C "$buildDir/source"
work="$buildDir/work-$grid"
[[ -d "$work" ]] || cp -R "$buildDir/source/Gadget-2.0.7/Gadget2" "$work"
options="-DPERIODIC -DUNEQUALSOFTENINGS -DPEANOHILBERT -DWALLCLOCK -DPMGRID=$grid -DDOUBLEPRECISION -DDOUBLEPRECISION_FFTW -DSYNCHRONIZATION"
make -C "$work" -j "$jobs" SYSTYPE=Portable CC="$mpicc" OPT="$options" \
    OPTIMIZE='-O2 -std=gnu99' MPICHLIB= HDF5INCL= HDF5LIB= \
    GSL_INCL="-I$gslPrefix/include" GSL_LIBS="-L$gslPrefix/lib" \
    FFTW_INCL="-I$fftwPrefix/include" FFTW_LIBS="-L$fftwPrefix/lib" \
    > "$buildDir/logs/gadget2-$grid-build.log" 2>&1
install -m 0755 "$work/Gadget2" "$executable"
{
    printf 'source=GADGET-2.0.7\npmgrid=%s\ncc=%s\nmpicc=%s\noptions=%s\n' "$grid" "$cc" "$mpicc" "$options"
    printf 'source_modifications=none\n'
    sha256 "$executable" "$work/Makefile" "$buildDir/downloads/"*.tar.gz \
        "$gslPrefix/lib/libgsl.a" "$gslPrefix/lib/libgslcblas.a" "$fftwPrefix/lib/"*.a "$0"
} > "$executable.manifest.txt"
printf 'Built %s\n' "$executable"
