#!/usr/bin/env bash
set -euo pipefail

# Rebuild only the isolated 256^3 TreePM executable on Merlin. The source tree,
# patched Makefile, FFTW2 installation, and campaign artifacts remain under the
# campaign root; the separate 128^3 build is never touched.
campaign_root=/data/user/adelmann/gadget2-zeldovich-np256-20261006
legacy_root=/data/user/adelmann/gadget2-zeldovich-20261004
build_dir="$campaign_root/build/gadget-work-256"
build_makefile="$build_dir/Makefile"
build_log="$campaign_root/build/gadget2-pmgrid256-build.log"
patch_file="$campaign_root/source/pmgrid256.patch"
source_archive="$legacy_root/downloads/gadget-2.0.7.tar.gz"
fftw_archive="$legacy_root/downloads/fftw-2.1.5.tar.gz"
fftw_prefix="$legacy_root/prefix/fftw-2.1.5"
gsl_prefix=/opt/psi/Compiler/gsl/2.8/gcc/14.3.0
executable="$campaign_root/prefix/Gadget2-TreePM-256-double"
expected_makefile_sha256=c9ce6212fc3debd61252a3c1c8b3ffe33aec1a95bf530e91b1d941b643bf3796
expected_patch_sha256=326211f5f746652836fab53ce85a192f1ae869478d30337652deeebf24fdcac5
expected_gadget_archive_sha256=8e321110b9fb2d05819f9cfcbffda19f56bb77c7cdd1ca21139b81b1fca4e3ed
expected_fftw_archive_sha256=f8057fae1c7df8b99116783ef3e94a6a44518d49c72e2e630c24b689c6022630
expected_executable_sha256=3b1668629958329d3e5e9913717548e3d690dea3201f45816bddfafb6d90a0ac
build_jobs=${BUILD_JOBS:-8}

if [[ -e "$campaign_root/campaign.json" || -e "$campaign_root/runs/z99_gadget2_np256" \
      || -e "$campaign_root/outputs/z99_gadget2_np256" ]]; then
  echo "Refusing to rebuild after the 256^3 campaign has started." >&2
  exit 2
fi

for required in "$build_makefile" "$build_dir" "$patch_file" \
                "$source_archive" "$fftw_archive" "$fftw_prefix/include" \
                "$fftw_prefix/lib" "$gsl_prefix/include" "$gsl_prefix/lib"; do
  [[ -e "$required" ]] || { echo "Missing build input: $required" >&2; exit 2; }
done

[[ "$(sha256sum "$build_makefile" | awk '{print $1}')" == "$expected_makefile_sha256" ]] || {
  echo "The frozen PMGRID=256 Makefile has changed." >&2; exit 2;
}
[[ "$(sha256sum "$patch_file" | awk '{print $1}')" == "$expected_patch_sha256" ]] || {
  echo "The PMGRID patch has changed." >&2; exit 2;
}
[[ "$(sha256sum "$source_archive" | awk '{print $1}')" == "$expected_gadget_archive_sha256" ]] || {
  echo "The GADGET-2 source archive checksum differs." >&2; exit 2;
}
[[ "$(sha256sum "$fftw_archive" | awk '{print $1}')" == "$expected_fftw_archive_sha256" ]] || {
  echo "The FFTW2 source archive checksum differs." >&2; exit 2;
}
grep -Fq -- '-DPMGRID=256' "$build_makefile"
! grep -Fq -- '-DPMGRID=128' "$build_makefile"

module load gcc/14.3.0 openmpi/5.0.10_slurm
make -C "$build_dir" clean
make -C "$build_dir" -j"$build_jobs" 2>&1 | tee -a "$build_log"

built="$build_dir/Gadget2"
[[ -x "$built" ]] || { echo "The build produced no executable." >&2; exit 1; }
built_hash=$(sha256sum "$built" | awk '{print $1}')
[[ "$built_hash" == "$expected_executable_sha256" ]] || {
  echo "Rebuilt binary SHA256 $built_hash differs from the frozen run binary; not installing." >&2
  exit 1
}

temporary="$executable.tmp.$$"
trap 'rm -f "$temporary"' EXIT
install -m 0755 "$built" "$temporary"
[[ "$(sha256sum "$temporary" | awk '{print $1}')" == "$expected_executable_sha256" ]]
mv -f "$temporary" "$executable"
trap - EXIT
printf 'Installed verified GADGET-2 PMGRID=256 executable: %s\nSHA256 %s\n' \
  "$executable" "$expected_executable_sha256"
