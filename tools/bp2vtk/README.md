# bp2vtk — AdiosCatalyst BP → VTK for ParaView

AdiosCatalyst writes a *flattened Conduit "multimesh" tree* into ADIOS2
(`catalyst/channels/<ch>/data/block_*/...`). This is **not** a VTK/Fides/VTX
schema, so ParaView's ADIOS2 readers (`vtkADIOS2CoreImageReader`, VTX, Fides)
cannot open it — they misread the Conduit `assembly`/`type` string variables.
The AdiosCatalyst-native consumer is `AdiosReplay` (+ ParaView Catalyst); this
tool is a lightweight offline alternative that reconstructs ParaView-openable
files directly from the `.bp`.

## What it produces

For each ADIOS step it writes standard VTK XML plus a `.pvd` time series:

- **Uniform field blocks** → one global `vtkImageData` (`.vti`) with cell data.
  Fields are domain-decomposed: each MPI rank writes its subdomain as an ADIOS
  block with its own origin/dims (read back via `BlocksInfo`); the tool places
  each block into the correct spot of the reassembled global image.
- **Particle block** → one `vtkUnstructuredGrid` (`.vtu`) with all particles as
  points (one `VTK_VERTEX` cell each) and every attribute as point data
  (`position`, `velocity`, `charge`, `electric_field`, `RankID`, …).
- Helper blocks (particle bounding box) are skipped.

## Build

```bash
./build.sh                    # auto-detects ippl's in-tree ADIOS2 under ../../build
# or: ./build.sh /path/to/ipplADIOS/build
```

## Run

```bash
./_b/bp2vtk /path/to/alpine.bp [output_dir]     # default output_dir = <input>_vtk
```

Then open the `*.pvd` files in `output_dir` with ParaView (each is a time
series). Fields render as image volumes/slices; the particle `.pvd` renders as
points you can colour by any attribute or run through Glyph.

## Notes

- Values are written as ASCII (datasets here are small); switch to base64 if you
  scale up.
- Requires the ADIOS2 built by the ippl project (`-DIPPL_ENABLE_ADIOS2=ON`).
