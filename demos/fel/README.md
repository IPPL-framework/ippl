# FEL — Free Electron Laser mini-app

An electromagnetic PIC simulation of a free-electron laser: a relativistic
electron bunch is tracked through an undulator in a co-moving Lorentz frame,
with the self-consistent field advanced by an FDTD Maxwell solver. The radiated
power and FEL diagnostics are written to CSV files.

## Build

```sh
cmake -S . -B build -DIPPL_ENABLE_FEL=ON -DCMAKE_CXX_STANDARD=20
cmake --build build --target FreeElectronLaser
```

The executable is built at
`build/demos/fel/FreeElectronLaser`. The example configuration is staged beside
it as `build/demos/fel/config.json` whenever the target is built.

## Run

```sh
cd build
./demos/fel/FreeElectronLaser --info 5
```

An optional first argument can select another MITHRA-style JSON job file. By
default, the executable uses the staged `build/demos/fel/config.json`; see
[config.json](config.json) for the available keys. Run on multiple ranks with
`mpirun -np <N> ...`.

Output is written to the directory given by `output.path` in the config:
`radiation_<nranks>.csv` holds the radiated power. The directory is created
automatically; relative paths are resolved from the process working directory.

## Acknowledgements

The relativistic bunch initialization and the resonance-power diagnostic are directly ported from [MITHRA](https://github.com/aryafallahi/mithra), a full-wave free-electron-laser solver by A. Fallahi (GPL-licensed). It was also used as reference for much of the other parts. 
