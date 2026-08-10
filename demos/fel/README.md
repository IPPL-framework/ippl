# FEL — Free Electron Laser mini-app

An electromagnetic PIC simulation of a free-electron laser: a relativistic
electron bunch is tracked through an undulator in a co-moving Lorentz frame,
with the self-consistent field advanced by an FDTD Maxwell solver. The radiated
power is written to a CSV (and, optionally, a Poynting-flux video).

## Build

```sh
cmake -S . -B build -DIPPL_ENABLE_FEL=ON -DCMAKE_CXX_STANDARD=20
cmake --build build --target FreeElectronLaser
```

The executable is built at
`build/fel/FreeElectronLaser`.

## Run

```sh
cd build
./fel/FreeElectronLaser ../fel/config.json --info 5
```

The argument is a MITHRA-style JSON job file (defaults to `../fel/config.json`);
see [config.json](config.json) for the available keys. Run on multiple ranks
with `mpirun -np <N> ...`.

Output is written to the directory given by `output.path` in the config:
`radiation_<nranks>.csv` holds the radiated power. If `output.rhythm > 0`, a
Poynting-flux video is produced and requires **ffmpeg** on the `PATH`.

## Acknowledgements

The relativistic bunch initialization and the resonance-power diagnostic are directly ported from [MITHRA](https://github.com/aryafallahi/mithra), a full-wave free-electron-laser solver by A. Fallahi (GPL-licensed). It was also used as reference for much of the other parts. 
