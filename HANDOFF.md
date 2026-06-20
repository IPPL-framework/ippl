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
