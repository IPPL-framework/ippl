# FEL — Free Electron Laser mini-app

An electromagnetic PIC simulation of a free-electron laser: a relativistic
electron bunch is tracked through an undulator in a co-moving Lorentz frame,
with self-fields advanced through collocated four-potentials by the nonstandard
vacuum FDTD solver. The radiated power and FEL diagnostics are written to files.

Start with the [student guide](StudentGuide.md) for the actual timestep sequence,
configuration and units, class diagram, MPI/Kokkos data flow and validation limits.
The source comments form a focused API reference through the local Doxyfile.
The [mathematical model](MathematicalModel.md) gives the solved potential equations,
nonstandard stencil, dispersion relation, CIC sources, Boris substeps and Mur
boundaries, with MITHRA references and the current implementation's differences.

## Build

```sh
cmake -S . -B build -DIPPL_ENABLE_FEL=ON -DCMAKE_CXX_STANDARD=20
cmake --build build --target FreeElectronLaser
```

The FEL configuration is parsed with the Conduit API shipped by Catalyst. CMake
therefore finds or fetches Catalyst when the FEL demo is enabled, even when
`IPPL_ENABLE_CATALYST` itself is off.

The executable is built at
`build/demos/fel/FreeElectronLaser`. The example configuration is staged beside
it as `build/demos/fel/config.json` whenever the target is built.

## Run

```sh
./build/demos/fel/FreeElectronLaser ./demos/fel/config.json --info 5
```

An optional first argument can select another MITHRA-style JSON job file. By
default, the executable reads `config.json` in the **working directory**. To use
the staged copy without an argument, change to `build/demos/fel` first. See
[config.json](config.json) for the JSON keys. Run on multiple ranks with
`mpirun -np <N> ...`.

The shipped case has 27.6 million cells and is not a quick smoke test. Use a small,
short input for initial work, checking the boosted-grid spacing condition in the
guide. `timestep-ratio` is currently parsed but unused; `dt` comes from the solver.

Output is written to the directory given by `output.path` in the config:
`radiation_<nranks>.csv`, `radiation_band_<nranks>.csv` and
`feldiag_<nranks>.csv`. The directory is created automatically; relative paths
are resolved from the process working directory. Files are appended, so use a
fresh directory per run. Their headers use commas but data rows use whitespace;
the guide gives a reader example and explains the units and diagnostic limits.

## Generate the focused documentation

```sh
cd demos/fel
doxygen Doxyfile
```

Open `docs/html/index.html`. The local build needs Doxygen, Graphviz, LaTeX,
dvips and Ghostscript for diagrams and offline formula images. If Doxygen is
available at CMake configuration, `cmake --build build --target fel-docs` from
the repository root does the same. Documentation generation is opt-in; generated
`docs/` files are ignored. See [StudentGuide.md](StudentGuide.md) without Doxygen.

## Acknowledgements

The relativistic bunch initialization and the resonance-power diagnostic are directly ported from [MITHRA](https://github.com/aryafallahi/mithra), a full-wave free-electron-laser solver by A. Fallahi (GPL-licensed). It was also used as reference for much of the other parts. 
