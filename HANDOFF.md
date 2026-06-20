# FEL HDF5 Output Handoff

## Branch

- Remote branch: `IPPL-framework/ippl:fel-hdf5-output`

## How To Compile

Installed-HDF5 Release build used locally:

```bash
cmake -S /Users/adelmann/ippl-fel -B /Users/adelmann/ippl-fel/build-hdf5-gcc \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DIPPL_PLATFORMS=OPENMP \
  -DIPPL_ENABLE_FFT=ON \
  -DIPPL_ENABLE_SOLVERS=ON \
  -DIPPL_ENABLE_FEL=ON \
  -DIPPL_HDF5=ON \
  -DIPPL_USE_INSTALLED_HDF5=ON \
  -DHeffte_ENABLE_FFTW=ON

```

If an installed HDF5 is not available, omit `-DIPPL_USE_INSTALLED_HDF5=ON`; 
FetchContent will build HDF5 during configuration/build.

Example smoke run from the build directory:

```bash
cd /Users/adelmann/ippl-fel/build-hdf5-gcc/fel
mpirun -np 4 -x OMP_NUM_THREADS=2 ./FreeElectronLaser config_hdf5_smoke.json --info 5
```

Use `--info 5` when checking CSV diagnostics. With `--info 0`, the `Inform`-based
CSV streams are suppressed and the CSV files can remain zero length.

## HDF5 Output Layout

When `IPPL_HDF5=ON` and the config has a nonzero `output.rhythm`, the FEL run
writes one rank-0 HDF5 visualization file:

```text
<output.path>/fel_output_<nranks>.h5
```

For the production command run from `build-hdf5-gcc/fel` with `../../config.json`,
the current config path `../renderdata` resolves to:

```text
/Users/adelmann/ippl-fel/build-hdf5-gcc/renderdata/fel_output_<nranks>.h5
```

The file contains one group per visualization frame:

```text
/step_<iteration>
```

Each step group contains:

```text
poynting_magnitude  double[nx,nz]
poynting_z          double[nx,nz]
particle_xz_m       double[nparticles,2]
```

Attributes on each step group:

```text
iteration       int
time            double
unit_length_m   double
extents_m       double[3]
spacing_m       double[3]
resolution      int[3]
```

Notes:

- `poynting_magnitude` and `poynting_z` are lab-frame values sampled on the
  mid-y x-z plane and reduced to rank 0.
- `particle_xz_m[:,0]` is particle x in meters and `particle_xz_m[:,1]` is
  particle z in meters, gathered from all ranks.
- The HDF5 file is visualization-frame data and is written only on
  `output.rhythm`; the CSV files are separate scalar time-series diagnostics
  written every step when `Inform` output is enabled.

## Debug Renderer

The debug renderer script is:

```text
/Users/adelmann/ippl-fel/scripts/render_fel_hdf5.py
```

It renders PNG frames from the HDF5 file using the same visual convention as the
old ffmpeg path: lab-frame Poynting magnitude on the x-z mid-y plane, with
particles overlaid in green.

Python packages needed by the script:

```text
h5py
imageio
matplotlib
numpy
```

Render the first few frames:

```bash
cd /Users/adelmann/ippl-fel
python3 scripts/render_fel_hdf5.py \
  build-hdf5-gcc/renderdata/fel_output_4.h5 \
  --output-dir build-hdf5-gcc/renderdata/debug_frames \
  --limit 10
```

Render all frames and assemble a movie:

```bash
cd /Users/adelmann/ippl-fel
python3 scripts/render_fel_hdf5.py \
  build-hdf5-gcc/renderdata/fel_output_4.h5 \
  --output-dir build-hdf5-gcc/renderdata/debug_frames \
  --movie build-hdf5-gcc/renderdata/fel_debug.mp4 \
  --fps 30
```

Useful options:

```text
--scale <float>          multiplier before clipping into the turbo colormap
--max-particles <int>   particle draw limit per frame; 0 disables particles
--dpi <int>             PNG resolution control
--limit <int>           render only the first N HDF5 step groups
```

## Changed Files

- `fel/FreeElectronLaserManager.h`
  - Added comments explaining that CSV outputs are rank-0 scalar diagnostics and
    globally reduced across MPI ranks.
  - Changed CSV headers to no-space comma-separated names.
  - Changed CSV rows from space-separated to comma-separated.

## Verification

- Rebuilt Release FEL target:
  - `cmake --build /Users/adelmann/ippl-fel/build-hdf5-gcc --target FreeElectronLaser -j 8`
- Ran 4-rank HDF5 smoke case:
  - `mpirun -np 4 -x OMP_NUM_THREADS=2 ./FreeElectronLaser config_hdf5_smoke.json --info 5`
- Verified generated CSV files parse with Python's `csv` module:
  - `radiation_4.csv`: 156 rows, 2 fields per row.
  - `radiation_band_4.csv`: 156 rows, 2 fields per row.
  - `feldiag_4.csv`: 156 rows, 9 fields per row.

## Notes

- on LUMI on 8 GPUs, after 350 timesteps we get::

 :0: : Device-side assertion `Kokkos::View ERROR: out of bounds access label=("") with indices [18446744073709551615] but extents [16325]' failed.
Failed to allocate file: Bad file descriptor
GPU core dump failed
:0:rocdevice.cpp            :3020: 8279240781564d us:  Callback: Queue 0x154fad200000 aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. co
de: 0x1016
